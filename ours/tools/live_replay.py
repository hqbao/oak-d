#!/usr/bin/env python3
"""Offline replay of the LIVE display path over a recorded session.

This is the honest self-test that lets us iterate on the live translation
pipeline WITHOUT a device. It reproduces, frame-for-frame, exactly what
``oakd/sources/depthai_ours_vio.py`` does on the OAK-D:

  1. gyro preintegration between frames -> per-frame rotation prior ``R_prior``
  2. accelerometer averaged into the camera frame + gravity leveling
  3. ``RGBDVisualOdometry.process`` with ``lock_translation_to_rotation=True``
     (the gyro owns rotation; translation is solved with that rotation held
     fixed, so an in-place yaw cannot leak a phantom translation)
  4. ``InertialTranslationFilter`` predict(accel)+correct(vision) per frame

It then aligns the resulting trajectory to the recorded Basalt poses with a
rigid SE3 (and Sim3) Umeyama fit and reports the ATE -- the same scoring as
``tools/vio_run.py``. Two regime-specific numbers are also printed:

  * STILL sessions  -> our displayed path length + max drift (should be small;
    the device is stationary so any motion is phantom drift).
  * MOTION sessions -> ATE vs Basalt + our/Basalt path-length ratio (we should
    track the real forward/back/strafe motion, not lag or stall).

Usage::

    python tools/live_replay.py --session sessions/gold/still_15s
    python tools/live_replay.py --session sessions/gold/push_fwdback_20s
    python tools/live_replay.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.vio import (  # noqa: E402
    InertialFilterConfig,
    InertialTranslationFilter,
    OdometryConfig,
    RGBDVisualOdometry,
    SessionReader,
)
from ours.vio.imu import so3_exp, so3_log  # noqa: E402
from ours.tools.vio_run import load_basalt_positions, umeyama  # noqa: E402


_REST_MOTION_THRESH = 0.35   # m/s^2, same as the live source


def replay(session_dir: Path, max_frames: int = 0, verbose: bool = False,
           no_accel: bool = False, decimate: int = 1, lock_trans: bool = False,
           vel_damp: float = 0.9, resolve_disagree: bool = False,
           clamp_speed: float = 0.0, min_inliers: int = 0):
    reader = SessionReader(session_dir)
    K = reader.K
    imu = reader.load_imu()
    imu_ts = imu["ts_ns"]
    imu_gyro = imu["gyro"]
    imu_accel = imu["accel"]

    # IMU -> left-camera rotation (bring gyro/accel into the optical frame).
    if reader.calib.T_imu_left is not None:
        R_imu_cam = reader.calib.T_imu_left[:3, :3]
    else:
        R_imu_cam = np.eye(3)

    # --- startup: static window for gravity leveling + gyro bias ------------
    f0_ts = reader._frames[0]["ts_ns"]
    win_mask = (imu_ts >= f0_ts) & (imu_ts < f0_ts + int(0.4e9))
    if win_mask.any():
        gyro_bias = imu_gyro[win_mask].mean(axis=0)
        accel0_cam = R_imu_cam @ imu_accel[win_mask].mean(axis=0)
    else:
        gyro_bias = np.zeros(3)
        accel0_cam = None

    od_cfg = OdometryConfig(gyro_fuse=True,
                            lock_translation_to_rotation=lock_trans,
                            resolve_translation_on_disagree=resolve_disagree,
                            max_translation_speed=clamp_speed,
                            min_inliers_for_translation=min_inliers)
    vo = RGBDVisualOdometry(K, od_cfg)
    if accel0_cam is not None:
        vo.align_to_gravity(accel0_cam)

    tfilt = InertialTranslationFilter(InertialFilterConfig(vel_damp=vel_damp))
    tfilt.reset(vo.pose[:3, 3].copy())

    # frame indices actually processed (decimate>1 simulates the live loop
    # dropping its backlog when the host cannot keep up at the recorded fps).
    idx_all = list(range(len(reader)))
    idx_proc = idx_all[::max(1, decimate)]
    if max_frames > 0:
        idx_proc = idx_proc[:max_frames]
    positions: dict[int, np.ndarray] = {}

    accel_ema = accel0_cam.copy() if accel0_cam is not None else None
    prev_vo_t: np.ndarray | None = None
    prev_frame_ts: int | None = None
    j = 0   # IMU cursor
    n_fail = 0
    n_proc = 0

    for i in idx_proc:
        fr = reader.load_frame(i)
        gray = fr.gray_left
        depth = fr.depth_m

        # Accumulate IMU samples spanning (prev_frame_ts, fr.ts_ns]: integrate
        # the gyro into an inter-frame rotation, average the accel.
        R_imu_accum = np.eye(3)
        acc_sum = np.zeros(3)
        acc_cnt = 0
        gyro_cnt = 0
        last_ts = prev_frame_ts
        # advance cursor to first sample after prev_frame_ts
        while j < len(imu_ts) and imu_ts[j] <= fr.ts_ns:
            ts = int(imu_ts[j])
            if prev_frame_ts is not None and ts > prev_frame_ts:
                w = imu_gyro[j]
                if last_ts is not None:
                    dt = (ts - last_ts) * 1e-9
                    if 0.0 < dt < 0.1:
                        R_imu_accum = R_imu_accum @ so3_exp((w - gyro_bias) * dt)
                        gyro_cnt += 1
                last_ts = ts
                acc_sum += imu_accel[j]
                acc_cnt += 1
            j += 1

        accel_raw = None if acc_cnt == 0 else R_imu_cam @ (acc_sum / acc_cnt)
        R_prior = (R_imu_cam @ R_imu_accum @ R_imu_cam.T
                   if gyro_cnt > 0 else None)

        dt_f = ((fr.ts_ns - prev_frame_ts) * 1e-9
                if prev_frame_ts is not None else 1.0 / 20.0)
        prev_frame_ts = fr.ts_ns

        vo.process(gray, depth, R_prior=R_prior, dt_s=dt_f)
        n_proc += 1
        if not bool(vo.last_info.get("ok", False)):
            n_fail += 1

        # accel EMA + rest-gated leveling (same as live)
        accel_cam = None
        at_rest = False
        if accel_raw is not None:
            if accel_ema is None:
                accel_ema = accel_raw.copy()
            else:
                accel_ema += 0.2 * (accel_raw - accel_ema)
            accel_cam = accel_ema
            motion = float(np.linalg.norm(accel_raw - accel_ema))
            at_rest = motion < _REST_MOTION_THRESH
            if at_rest:
                na = float(np.linalg.norm(accel_cam))
                vo._g_ref = na if vo._g_ref is None else vo._g_ref + 0.05 * (na - vo._g_ref)
                vo.correct_tilt(accel_cam)

        # inertial filter step
        R_wc = vo.pose[:3, :3]
        gyro_deg = (float(np.degrees(np.linalg.norm(
            so3_log(R_imu_accum)))) if gyro_cnt > 0 else 0.0)
        vo_t_now = vo.pose[:3, 3].copy()
        vis_ok = bool(vo.last_info.get("ok", False))
        dp_vis = (vo_t_now - prev_vo_t) if (prev_vo_t is not None and vis_ok) else None
        prev_vo_t = vo_t_now
        accel_in = None if no_accel else accel_cam
        pos = tfilt.step(dt_f, R_wc, accel_in, dp_vis, gyro_deg).copy()
        positions[fr.seq] = pos

        if verbose and n_proc % 40 == 0:
            print(f"  f{i:4d} ok={vis_ok} gyro={gyro_deg:5.1f} "
                  f"|v|={np.linalg.norm(tfilt.v):.3f} p={pos.round(3)}")

    positions["_meta"] = {"n_proc": n_proc, "n_fail": n_fail}  # type: ignore
    return positions


def score(session_dir: Path, positions: dict[int, np.ndarray]):
    meta = positions.pop("_meta", {"n_proc": 0, "n_fail": 0})  # type: ignore
    basalt = load_basalt_positions(session_dir)
    seqs = sorted(set(positions) & set(basalt))
    if len(seqs) < 5:
        print("  not enough overlapping poses to score")
        return
    ours = np.array([positions[s] for s in seqs])
    ref = np.array([basalt[s] for s in seqs])

    # our + Basalt path length
    our_path = np.linalg.norm(np.diff(ours, axis=0), axis=1).sum()
    ref_path = np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()
    # net displacement (start->end straight line) -- jitter inflates path far
    # above this even when the endpoints are right.
    our_net = float(np.linalg.norm(ours[-1] - ours[0]))
    ref_net = float(np.linalg.norm(ref[-1] - ref[0]))

    # rigid Umeyama ATE
    R, t, s = umeyama(ours, ref, with_scale=False)
    aligned = (s * (R @ ours.T).T + t)
    err = np.linalg.norm(aligned - ref, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))

    Rs, ts, ss = umeyama(ours, ref, with_scale=True)
    aligned_s = (ss * (Rs @ ours.T).T + ts)
    rmse_s = float(np.sqrt((np.linalg.norm(aligned_s - ref, axis=1) ** 2).mean()))

    # still-vs-motion classification from Basalt's own path
    still = ref_path < 0.5
    nf = meta["n_fail"]; npc = meta["n_proc"]
    print(f"  frames proc/fail: {npc} / {nf} ({100*nf/max(npc,1):.0f}% KLT fail)")
    print(f"  Basalt path    : {ref_path*1000:7.1f} mm "
          f"({'STILL' if still else 'MOTION'})  net {ref_net*1000:6.1f} mm")
    print(f"  our path       : {our_path*1000:7.1f} mm "
          f"(ratio {our_path/max(ref_path,1e-6):.2f})  net {our_net*1000:6.1f} mm")
    print(f"  jitter index   : our_path/our_net = "
          f"{our_path/max(our_net,1e-6):.1f}  (Basalt {ref_path/max(ref_net,1e-6):.1f}; "
          f"1.0=clean, big=jitter)")
    if still:
        drift = np.linalg.norm(ours - ours[0], axis=1)
        print(f"  our drift max  : {drift.max()*1000:7.1f} mm   "
              f"(stationary -> want small)")
        print(f"  our drift final: {drift[-1]*1000:7.1f} mm")
    print(f"  ATE rigid RMSE : {rmse*1000:7.1f} mm "
          f"({100*rmse/max(ref_path,1e-6):.2f}% of path)")
    print(f"  ATE Sim3 RMSE  : {rmse_s*1000:7.1f} mm   "
          f"(our scale {ss:.3f} of Basalt)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="sessions/gold/push_fwdback_20s")
    ap.add_argument("--all", action="store_true",
                    help="run every gold session")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-accel", action="store_true",
                    help="vision-only filter (no accelerometer feed-forward)")
    ap.add_argument("--decimate", type=int, default=1,
                    help="process every Nth frame (simulate live frame drops)")
    ap.add_argument("--lock", action="store_true",
                    help="enable lock_translation_to_rotation (default: joint PnP,"
                         " matching the live source)")
    ap.add_argument("--vel-damp", type=float, default=0.9, dest="vel_damp",
                    help="velocity decay on vision-failure coast (1.0 = hold)")
    ap.add_argument("--resolve-disagree", action="store_true",
                    dest="resolve_disagree",
                    help="re-solve translation with the gyro rotation held fixed"
                         " when vision disagrees (anti-freeze under shake)")
    ap.add_argument("--clamp", type=float, default=0.0, dest="clamp_speed",
                    help="physical per-frame translation speed clamp (m/s); 0"
                         " = off. Caps non-physical phantom jumps (anti-wobble)")
    ap.add_argument("--min-inliers", type=int, default=0, dest="min_inliers",
                    help="freeze translation when PnP inliers < this (white-wall"
                         " / textureless freeze); 0 = off")
    args = ap.parse_args()

    if args.all:
        sessions = sorted(p.parent.parent for p in
                          Path("sessions/gold").glob("*/basalt/vio_pose.jsonl"))
    else:
        sessions = [Path(args.session)]

    for sd in sessions:
        print(f"=== {sd}  (no_accel={args.no_accel} decimate={args.decimate} "
              f"lock={args.lock} resolve_disagree={args.resolve_disagree} "
              f"clamp={args.clamp_speed} min_inliers={args.min_inliers}) ===")
        positions = replay(sd, max_frames=args.max_frames, verbose=args.verbose,
                            no_accel=args.no_accel, decimate=args.decimate,
                            lock_trans=args.lock, vel_damp=args.vel_damp,
                            resolve_disagree=args.resolve_disagree,
                            clamp_speed=args.clamp_speed,
                            min_inliers=args.min_inliers)
        score(sd, positions)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
