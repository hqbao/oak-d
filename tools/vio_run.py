#!/usr/bin/env python3
"""Run the from-scratch RGB-D VIO over a recorded session and score it.

This is the offline self-test: it runs our visual odometry on the recorded
left+depth stream, then compares the resulting trajectory against the *library*
(Basalt) poses that were recorded alongside the session
(``basalt/vio_pose.jsonl``). Because both trajectories are metric we align them
with a rigid SE3 Umeyama fit (rotation+translation, no scale) and report the
Absolute Trajectory Error (ATE). A with-scale fit is also reported so we can see
how close our metric scale is to Basalt's.

Usage::

    python tools/vio_run.py
    python tools/vio_run.py --session sessions/gold/lab_straight_20s
    python tools/vio_run.py --max-frames 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oakd.vio import (  # noqa: E402
    GyroPreintegrator,
    RGBDVisualOdometry,
    SessionReader,
    WindowedRGBDOdometry,
)


def load_basalt_positions(session_dir: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    path = session_dir / "basalt" / "vio_pose.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[int(d["seq"])] = np.asarray(d["pos"], dtype=np.float64)
    return out


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    """Least-squares similarity/rigid transform mapping src onto dst.

    src, dst: (N,3). Returns (R, t, s) such that dst ~= s*R@src + t.
    """
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc.T @ sc) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_s = (sc ** 2).sum() / src.shape[0]
        s = np.trace(np.diag(D) @ S) / var_s
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def ate(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    R, t, s = umeyama(src, dst, with_scale)
    aligned = (s * (R @ src.T)).T + t
    err = np.linalg.norm(aligned - dst, axis=1)
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
        "scale": float(s),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = all frames")
    ap.add_argument("--all", action="store_true",
                    help="run every gold session and print a summary table")
    ap.add_argument("--backend", choices=("f2f", "ba"), default="f2f",
                    help="f2f = frame-to-frame VO; ba = windowed bundle adjustment")
    ap.add_argument("--no-imu", action="store_true",
                    help="disable the gyro rotation prior (pure vision)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    use_imu = not args.no_imu
    if args.all:
        return run_all(use_imu, backend=args.backend)

    score_session(Path(args.session), args.max_frames, args.verbose,
                  use_imu=use_imu, backend=args.backend)
    return 0


# A recorded Basalt trajectory is only a valid reference if it didn't diverge.
# Occasionally Basalt loses tracking and teleports by hundreds of metres in one
# frame; such a run can't be used as ground truth. Detect this automatically
# from the data (max per-frame step) rather than hardcoding session names.
_MAX_VALID_STEP_M = 1.0


def basalt_ref_is_broken(positions: dict[int, np.ndarray]) -> bool:
    if len(positions) < 2:
        return True
    pos = np.array([positions[s] for s in sorted(positions)])
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return bool(steps.max() > _MAX_VALID_STEP_M)


def run_all(use_imu: bool = True, backend: str = "f2f") -> int:
    gold = Path("sessions/gold")
    rows = []
    for d in sorted(gold.iterdir()):
        if not (d / "basalt" / "vio_pose.jsonl").exists():
            continue
        broken = basalt_ref_is_broken(load_basalt_positions(d))
        note = "broken Basalt ref" if broken else ""
        res = None if broken else score_session(d, 0, False, quiet=True,
                                                use_imu=use_imu, backend=backend)
        rows.append((d.name, res, note))
        print(f"  {d.name:18s} done")

    print()
    print(f"backend: {backend}")
    print(f"{'session':18s} {'path(m)':>8s} {'ATE RMSE':>10s} {'%path':>7s} {'scale':>6s}")
    print("-" * 54)
    for name, res, note in rows:
        if res is None:
            tag = f"  <- {note}" if note else "  (too short / no overlap)"
            print(f"{name:18s} {'--':>8s} {'--':>10s} {'--':>7s} {'--':>6s}{tag}")
            continue
        print(f"{name:18s} {res['path']:8.2f} {res['rmse']*1000:8.1f}mm "
              f"{100*res['rmse']/res['path']:6.2f}% {res['scale']:6.3f}")
    return 0


def score_session(session_dir: Path, max_frames: int, verbose: bool,
                  quiet: bool = False, use_imu: bool = True,
                  backend: str = "f2f"):
    reader = SessionReader(session_dir)
    n = len(reader) if max_frames <= 0 else min(max_frames, len(reader))
    if backend == "ba":
        vo = WindowedRGBDOdometry(reader.K)
        use_imu = False  # BA backend does not use the gyro prior
    else:
        vo = RGBDVisualOdometry(reader.K)

    # Build a gyro preintegrator when the session has IMU extrinsics, so we can
    # feed a rotation prior to PnP. Sessions recorded before extrinsics were
    # captured (T_imu_left is None) silently fall back to pure vision.
    pre = None
    if use_imu and reader.calib.has_imu_extrinsics:
        imu = reader.load_imu()
        if imu["ts_ns"].size > 1:
            pre = GyroPreintegrator(imu["ts_ns"], imu["gyro"],
                                    reader.calib.T_imu_left)
            # Gravity-align the initial attitude from the static-startup accel,
            # so the world frame's "down" is real gravity (not the arbitrary
            # starting camera tilt). ATE is Umeyama-aligned, so this global
            # world rotation does not change the score -- it only makes the
            # reported/displayed attitude physically meaningful.
            R_imu_cam = reader.calib.T_imu_left[:3, :3]
            t0 = imu["ts_ns"][0]
            win = imu["ts_ns"] <= t0 + int(0.3 * 1e9)   # first ~0.3 s
            accel_imu = imu["accel"][win].mean(axis=0)
            vo.align_to_gravity(R_imu_cam @ accel_imu)

    if not quiet:
        print(f"session : {reader.dir}")
        print(f"frames  : {n}/{len(reader)}")
        print(f"imu     : {'gyro rotation prior ON' if pre else 'OFF (vision only)'}")
        print("running VO ...")

    est: dict[int, np.ndarray] = {}
    n_ok = 0
    prev_ts = None
    for i in range(n):
        f = reader.load_frame(i)
        if backend == "ba":
            pose = vo.process(f.gray_left, f.depth_m)
        else:
            R_prior = None
            if pre is not None and prev_ts is not None:
                R_prior = pre.delta_rotation(prev_ts, f.ts_ns)
            pose = vo.process(f.gray_left, f.depth_m, R_prior=R_prior)
        prev_ts = f.ts_ns
        est[f.seq] = pose[:3, 3].copy()
        if vo.last_info.get("ok"):
            n_ok += 1
        if verbose and i % 50 == 0:
            inf = vo.last_info
            print(f"  f{i:4d} tracks={inf.get('n_tracks', 0):3d} "
                  f"pnp={inf.get('n_pnp', 0):3d} "
                  f"inliers={inf.get('n_inliers', 0):3d} ok={inf.get('ok')} "
                  f"pos={pose[:3,3]}")

    basalt = load_basalt_positions(reader.dir)
    common = sorted(set(est) & set(basalt))
    if len(common) < 10:
        if not quiet:
            print(f"!! only {len(common)} common poses with Basalt -- cannot score")
        return None

    src = np.array([est[s] for s in common])       # our optical-frame traj
    dst = np.array([basalt[s] for s in common])     # Basalt FLU-world traj

    rigid = ate(src, dst, with_scale=False)
    sim = ate(src, dst, with_scale=True)
    traj_len = float(np.linalg.norm(np.diff(dst, axis=0), axis=1).sum())
    rigid["path"] = traj_len
    rigid["scale"] = sim["scale"]

    if not quiet:
        print(f"VO ok on {n_ok}/{n-1} motion steps")
        print()
        print(f"compared on {len(common)} poses | Basalt path length {traj_len:.2f} m")
        print("--- ATE vs Basalt (rigid SE3 align) ---")
        print(f"  RMSE   = {rigid['rmse']*1000:7.1f} mm")
        print(f"  median = {rigid['median']*1000:7.1f} mm")
        print(f"  max    = {rigid['max']*1000:7.1f} mm")
        print(f"  RMSE/path = {100*rigid['rmse']/traj_len:.2f}%")
        print("--- with scale (Sim3) ---")
        print(f"  RMSE   = {sim['rmse']*1000:7.1f} mm   "
              f"(our scale = {sim['scale']:.3f} of Basalt)")

    return rigid


if __name__ == "__main__":
    raise SystemExit(main())
