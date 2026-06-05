#!/usr/bin/env python3
"""Diagnostic: sweep windowed-VIO config and measure Sim3 scale vs Basalt.

The tight-coupled windowed VIO collapses metric scale on fast motion
(push_fwdback: scale 0.351 vs f2f 1.046). This probe runs the SAME vio path as
``vio_run.py --backend vio`` but lets us vary the WindowedVIOConfig (IMU factor
sigmas, window, kf cadence, min views) and reports the scale + ATE on a fast and
a slow session so we can find the root cause by measurement, not by guessing.

It also reports the SCALE TRAJECTORY: the running Sim3 scale of the keyframe
poses as the window slides, so we can see whether the shrink is a one-shot BA
artefact or a compounding feedback loop.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.vio import (  # noqa: E402
    GyroPreintegrator,
    OdometryConfig,
    SessionReader,
    WindowedVIORGBDOdometry,
)
from ours.vio.vio_window import VioConfig, WindowedVIOConfig  # noqa: E402
from ours.tools.vio_run import ate, load_basalt_positions  # noqa: E402


def run_vio(session_dir: Path, cfg: WindowedVIOConfig, max_frames: int = 0):
    reader = SessionReader(session_dir)
    n = len(reader) if max_frames <= 0 else min(max_frames, len(reader))

    imu = reader.load_imu()
    R_imu_cam = reader.calib.T_imu_left[:3, :3]
    gyro_cam = (R_imu_cam @ imu["gyro"].T).T
    accel_cam = (R_imu_cam @ imu["accel"].T).T
    t0 = imu["ts_ns"][0]
    win = imu["ts_ns"] <= t0 + int(0.3 * 1e9)
    bg0 = gyro_cam[win].mean(axis=0) if win.any() else np.zeros(3)

    od_cfg = OdometryConfig(gyro_fuse=True)
    vo = WindowedVIORGBDOdometry(
        reader.K, imu["ts_ns"], gyro_cam, accel_cam,
        bg0=bg0, ba0=np.zeros(3), odom_cfg=od_cfg, cfg=cfg)

    pre = GyroPreintegrator(imu["ts_ns"], imu["gyro"], reader.calib.T_imu_left)
    accel_imu = imu["accel"][win].mean(axis=0)
    vo.align_to_gravity(R_imu_cam @ accel_imu)

    est: dict[int, np.ndarray] = {}
    prev_ts = None
    for i in range(n):
        f = reader.load_frame(i)
        R_prior = None
        if prev_ts is not None:
            R_prior = pre.delta_rotation(prev_ts, f.ts_ns)
        pose = vo.process(f.gray_left, f.depth_m, f.ts_ns, R_prior=R_prior)
        prev_ts = f.ts_ns
        est[f.seq] = pose[:3, 3].copy()

    basalt = load_basalt_positions(reader.dir)
    common = sorted(set(est) & set(basalt))
    src = np.array([est[s] for s in common])
    dst = np.array([basalt[s] for s in common])
    rigid = ate(src, dst, with_scale=False)
    sim3 = ate(src, dst, with_scale=True)
    path_len = float(np.linalg.norm(np.diff(dst, axis=0), axis=1).sum())
    return {
        "ate_pct": 100.0 * rigid["rmse"] / max(path_len, 1e-9),
        "scale": sim3["scale"],
        "n": len(common),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--push-full", action="store_true",
                    help="run only the fast session at full length")
    ap.add_argument("--verify-fix", action="store_true",
                    help="run the proposed fix on slow/loop sessions vs baseline")
    ap.add_argument("--measure-b", action="store_true",
                    help="OLD (loose, 6-DoF) vs NEW default (tight IMU + "
                         "lock_tilt) on the fast push + 3 slow sessions")
    args = ap.parse_args()

    fast = Path("sessions/gold/push_fwdback_20s")
    slow = Path("sessions/gold/lab_loop_30s")

    base = WindowedVIOConfig()

    if args.measure_b:
        # OLD = the pre-(B) baseline: loose IMU, full 6-DoF pose (no tilt lock).
        old = replace(base, vio=replace(
            base.vio, sigma_vel=0.15, sigma_pos=0.15, lock_tilt=False))
        new = base   # current default: tight IMU + lock_tilt
        sessions = [
            ("push_fwdback_20s", fast),
            ("lab_loop_30s", Path("sessions/gold/lab_loop_30s")),
            ("corridor_60s", Path("sessions/gold/corridor_60s")),
            ("lab_straight_20s", Path("sessions/gold/lab_straight_20s")),
        ]
        print(f"{'session':18s} | {'OLD scale':>9s} {'OLD ATE%':>8s} | "
              f"{'NEW scale':>9s} {'NEW ATE%':>8s}", flush=True)
        print("-" * 62, flush=True)
        for name, d in sessions:
            ro = run_vio(d, old, args.max_frames)
            rn = run_vio(d, new, args.max_frames)
            print(f"{name:18s} | {ro['scale']:9.3f} {ro['ate_pct']:7.2f}% | "
                  f"{rn['scale']:9.3f} {rn['ate_pct']:7.2f}%", flush=True)
        return 0

    def vio_with(**vio_kw):
        return replace(base, vio=replace(base.vio, **vio_kw))

    if args.verify_fix:
        fix = vio_with(sigma_vel=0.03, sigma_pos=0.03, depth_sigma_coeff=0.1)
        sessions = [
            ("lab_loop_30s", Path("sessions/gold/lab_loop_30s")),
            ("corridor_60s", Path("sessions/gold/corridor_60s")),
            ("lab_straight_20s", Path("sessions/gold/lab_straight_20s")),
        ]
        print(f"{'session':18s} | {'base scale':>10s} {'base ATE%':>9s} | "
              f"{'fix scale':>9s} {'fix ATE%':>8s}", flush=True)
        print("-" * 64, flush=True)
        for name, d in sessions:
            rb = run_vio(d, base, args.max_frames)
            rf = run_vio(d, fix, args.max_frames)
            print(f"{name:18s} | {rb['scale']:10.3f} {rb['ate_pct']:8.2f}% | "
                  f"{rf['scale']:9.3f} {rf['ate_pct']:7.2f}%", flush=True)
        return 0

    configs = {
        "tight-imu (vel/pos 0.03)": vio_with(sigma_vel=0.03, sigma_pos=0.03),
        "loose-depth+tight-imu": vio_with(
            sigma_vel=0.03, sigma_pos=0.03, depth_sigma_coeff=0.1),
        "no-depth+tight-imu": vio_with(
            sigma_vel=0.03, sigma_pos=0.03, use_depth=False),
    }

    if args.push_full:
        print(f"{'config':28s} | {'push scale':>10s} {'push ATE%':>9s}",
              flush=True)
        print("-" * 52, flush=True)
        for name, cfg in configs.items():
            t0 = time.time()
            rf = run_vio(fast, cfg, 0)
            print(f"{name:28s} | {rf['scale']:10.3f} {rf['ate_pct']:8.2f}%"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
        return 0

    print(f"{'config':28s} | {'push scale':>10s} {'push ATE%':>9s} | "
          f"{'lab scale':>9s} {'lab ATE%':>8s}", flush=True)
    print("-" * 74, flush=True)
    for name, cfg in configs.items():
        t0 = time.time()
        rf = run_vio(fast, cfg, args.max_frames)
        print(f"{name:28s} | {rf['scale']:10.3f} {rf['ate_pct']:8.2f}% | "
              f"{'...':>9s} {'...':>8s}  push {time.time()-t0:.0f}s", flush=True)
        rs = run_vio(slow, cfg, args.max_frames)
        print(f"{name:28s} | {rf['scale']:10.3f} {rf['ate_pct']:8.2f}% | "
              f"{rs['scale']:9.3f} {rs['ate_pct']:7.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
