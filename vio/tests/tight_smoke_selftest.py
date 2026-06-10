#!/usr/bin/env python3
"""Phase 2 smoke test: the LIVE ``--tight`` backend produces a sane trajectory.

This exercises the EXACT engine-selection + snapshot path the ``--tight`` flag
wires up in the live VIO pipeline -- WITHOUT spinning up the multi-process IPC
graph -- so the gate is deterministic and fast:

    make_vi_engine(K, WindowedVIOConfig(imu_info_weight=True))   # the --tight engine
        -> InProcessEngine(WindowedVIOMap, vio_step, ...)
    per keyframe: engine.submit((T_cw, ids, px, depth, ts_ns, imu_seg))  # the 6-tuple
        -> vio_step -> WindowedVIOMap.add_keyframe(..., imu_seg=) -> run_ba

The front-end (per-frame RGB-D PnP + gyro prior) is driven exactly as the live
``OdometryModule`` drives it, and the inter-keyframe IMU block is built the way
``PreintegratePrior`` (rotate raw IMU into the camera frame) + ``EmitKeyframe``
(concatenate the per-frame segments since the last keyframe) build it. So a PASS
proves the tight path is wired correctly end-to-end and runs on real session
data, which is the Phase 2 bar.

GOAL of Phase 2 = it RUNS + produces a finite, non-trivial, non-exploding pose
trajectory (NOT necessarily beat loose -- that is Phase 3). The loose trajectory
is computed alongside for a ROUGH sanity comparison only.

Run::

    .venv/bin/python -m vio.tests.tight_smoke_selftest
    .venv/bin/python -m vio.tests.tight_smoke_selftest --session sessions/gold/push_straight_fast_15s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imu_camera.io.reader import SessionReader                       # noqa: E402
from vio.mathlib.imu.imu import GyroPreintegrator                    # noqa: E402
from vio.mathlib.odometry.odometry import (                          # noqa: E402
    OdometryConfig, RGBDVisualOdometry)
from vio.mathlib.backend.windowed import (                           # noqa: E402
    WindowedConfig, WindowedRGBDOdometry)
from vio.mathlib.backend.vio_window import WindowedVIOConfig         # noqa: E402
from vio.mathlib.engine import make_vi_engine                        # noqa: E402


def _slice_imu_seg(ts_all: np.ndarray, gyro_cam: np.ndarray,
                   accel_cam: np.ndarray, t0: int, t1: int):
    """Raw camera-frame IMU samples in ``(t0, t1]`` -> ``(ts, gyro, accel)``.

    Mirrors PreintegratePrior (per-sample ``R_imu_cam`` rotation, done by the
    caller) + EmitKeyframe (concatenate the inter-keyframe block). Returns None
    when fewer than two usable samples span the interval.
    """
    if t1 <= t0:
        return None
    m = (ts_all > t0) & (ts_all <= t1)
    if int(m.sum()) < 2:
        return None
    return (ts_all[m].astype(np.int64),
            gyro_cam[m].astype(np.float64),
            accel_cam[m].astype(np.float64))


def _trajectory_stats(name: str, est: dict[int, np.ndarray]) -> dict:
    seqs = sorted(est)
    pos = np.array([est[s] for s in seqs])
    finite = bool(np.all(np.isfinite(pos)))
    diffs = np.linalg.norm(np.diff(pos, axis=0), axis=1) if len(pos) > 1 else np.zeros(0)
    path = float(diffs.sum())
    max_step = float(diffs.max()) if diffs.size else 0.0
    span = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))) if len(pos) else 0.0
    print(f"  {name:6s} | poses={len(pos):4d} | finite={finite} | "
          f"path={path:6.2f} m | bbox-span={span:5.2f} m | "
          f"max-step={max_step*100:5.1f} cm")
    return {"pos": pos, "seqs": seqs, "finite": finite, "path": path,
            "span": span, "max_step": max_step, "n": len(pos)}


def run_tight_smoke(session_dir: Path, max_frames: int = 0,
                    kf_every: int = 5) -> int:
    reader = SessionReader(session_dir)
    n = len(reader) if max_frames <= 0 else min(max_frames, len(reader))
    if not reader.calib.has_imu_extrinsics:
        print(f"SKIP: {session_dir} has no IMU extrinsics (tight needs IMU).")
        return 0

    R_imu_cam = reader.calib.T_imu_left[:3, :3]
    imu = reader.load_imu()
    if imu["ts_ns"].size <= 1:
        print(f"SKIP: {session_dir} has no usable IMU stream.")
        return 0
    # Raw IMU pre-rotated into the camera optical frame (= PreintegratePrior).
    ts_all = imu["ts_ns"].astype(np.int64)
    gyro_cam = (R_imu_cam @ imu["gyro"].T).T
    accel_cam = (R_imu_cam @ imu["accel"].T).T
    # Static-startup window for the one-shot gravity align (the live --tight map
    # seeds gyro/accel bias internally, so no explicit bias seed is passed here).
    t0 = int(ts_all[0])
    win = ts_all <= t0 + int(0.3 * 1e9)

    odom_cfg = OdometryConfig(gyro_fuse=True)

    # --- LOOSE reference (the default backend), for the rough comparison ----
    loose_vo = WindowedRGBDOdometry(reader.K, cfg=WindowedConfig(),
                                    odom_cfg=odom_cfg)
    # --- TIGHT front-end + the EXACT --tight engine (imu_info_weight=True) ---
    tight_cfg = WindowedVIOConfig()
    tight_cfg.vio.imu_info_weight = True
    engine = make_vi_engine(reader.K, tight_cfg, worker=False)
    tight_fe = RGBDVisualOdometry(reader.K, odom_cfg)

    # Gyro rotation prior + gravity-align both front-ends identically.
    pre = GyroPreintegrator(imu["ts_ns"], imu["gyro"], reader.calib.T_imu_left)
    accel_imu0 = imu["accel"][win].mean(axis=0)
    loose_vo.align_to_gravity(R_imu_cam @ accel_imu0)
    tight_fe.align_to_gravity(R_imu_cam @ accel_imu0)

    est_tight: dict[int, np.ndarray] = {}
    est_loose: dict[int, np.ndarray] = {}
    tight_pose = tight_fe.pose.copy()
    prev_ts = None
    frames_since_kf = 0
    last_kf_ts = None
    n_kf = 0
    n_refined = 0
    last_info: dict = {}

    for i in range(n):
        f = reader.load_frame(i)
        R_prior = (pre.delta_rotation(prev_ts, f.ts_ns)
                   if prev_ts is not None else None)

        # LOOSE: the windowed-BA odometry advances itself per frame.
        pl = loose_vo.process(f.gray_left, f.depth_m, R_prior=R_prior)
        est_loose[f.seq] = pl[:3, 3].copy()

        # TIGHT: front-end per-frame pose (== live OdometryModule), then submit
        # the keyframe snapshot through the --tight engine at the kf cadence.
        tight_pose = tight_fe.process(f.gray_left, f.depth_m,
                                      R_prior=R_prior).copy()
        frames_since_kf += 1
        is_kf = (n_kf == 0) or (frames_since_kf >= kf_every)
        if is_kf:
            frames_since_kf = 0
            n_kf += 1
            tr = tight_fe.frontend.tracks
            ids = tr.ids.copy() if tr is not None and tr.ids is not None else None
            px = tr.points.copy() if tr is not None and tr.points is not None else None
            imu_seg = (None if last_kf_ts is None else
                       _slice_imu_seg(ts_all, gyro_cam, accel_cam,
                                      last_kf_ts, int(f.ts_ns)))
            last_kf_ts = int(f.ts_ns)
            T_cw = np.linalg.inv(tight_pose)
            # THE --tight snapshot: 6-tuple consumed by vio_step.
            engine.submit((T_cw, ids, px, f.depth_m, int(f.ts_ns), imu_seg))
            post = engine.poll()                    # refined latest T_cw or None
            if post is not None:
                tight_pose = np.linalg.inv(post)
                tight_fe.pose = tight_pose.copy()
                n_refined += 1
                last_info = dict(engine.map.last_info)
        est_tight[f.seq] = tight_pose[:3, 3].copy()
        prev_ts = f.ts_ns

    engine.close()

    print(f"\nsession : {reader.dir.name}   frames={n}")
    print(f"keyframes submitted={n_kf}  window-solves that refined={n_refined}")
    if last_info:
        print(f"last solve: kfs={last_info.get('vio_kfs')} "
              f"lms={last_info.get('vio_lms')} obs={last_info.get('vio_obs')} "
              f"imu_factors={last_info.get('vio_imu')} "
              f"iters={last_info.get('vio_iters')} "
              f"reproj_px={last_info.get('vio_reproj_px'):.3f}")
    print("trajectory:")
    st_tight = _trajectory_stats("TIGHT", est_tight)
    st_loose = _trajectory_stats("LOOSE", est_loose)

    # A few sample tight poses (proof it produces real numbers, not all-zero).
    sample_seqs = st_tight["seqs"][:: max(1, st_tight["n"] // 5)][:5]
    print("sample TIGHT poses (seq -> xyz m):")
    for s in sample_seqs:
        p = est_tight[s]
        print(f"  seq {s:4d} -> [{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}]")

    # ----------------------------- GATES -----------------------------------
    # Phase 2 bar: RUNS without crashing + finite + non-trivial + not exploding.
    fails = []
    if not st_tight["finite"]:
        fails.append("tight trajectory has NaN/Inf")
    if st_tight["n"] < 10:
        fails.append(f"too few tight poses ({st_tight['n']})")
    if n_refined == 0:
        fails.append("the tight window NEVER refined a keyframe (no poses out)")
    if st_tight["span"] < 1e-4:
        fails.append("tight trajectory is degenerate (zero spatial span)")
    # "not exploding": no single inter-frame jump beyond a generous bound (the
    # Basalt validity gate itself uses 1.0 m; allow 2.0 m for the f2f tip).
    if st_tight["max_step"] > 2.0:
        fails.append(f"tight trajectory explodes (max step {st_tight['max_step']:.2f} m)")
    # rough sanity vs loose (NOT an accuracy gate -- Phase 3): the tight path
    # length must be within a wide factor of the loose one (same trajectory).
    if st_loose["path"] > 1e-3:
        ratio = st_tight["path"] / st_loose["path"]
        print(f"\nrough tight/loose path ratio = {ratio:.2f} "
              f"(Phase 2 sanity only; ATE is Phase 3)")
        if not (0.1 <= ratio <= 10.0):
            fails.append(f"tight path wildly off loose (ratio {ratio:.2f})")

    if fails:
        print("\nFAIL:")
        for f_ in fails:
            print(f"  - {f_}")
        return 1
    print("\nPASS -- tight backend RUNS + produces a finite, sane trajectory "
          "on real session data (Phase 2 bar met).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--kf-every", type=int, default=5)
    args = ap.parse_args()
    return run_tight_smoke(Path(args.session), args.max_frames, args.kf_every)


if __name__ == "__main__":
    raise SystemExit(main())
