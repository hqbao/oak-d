#!/usr/bin/env python3
"""Diagnostic A/B for the tight-coupled VIO backend.

Runs WindowedVIORGBDOdometry with IMU factors ON vs OFF on the same session so
we can isolate whether the gold regression comes from the *optimizer* (vision
only, IMU off -> should match windowed BA) or from the *IMU factor* (gravity
leak / bias). Reports rigid ATE %path and Sim3 scale.

Usage::
    python ours/tools/vio_diag.py --session sessions/gold/lab_loop_30s --max-frames 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib import (  # noqa: E402
    OdometryConfig,
    SessionReader,
    WindowedRGBDOdometry,
    WindowedVIORGBDOdometry,
    WindowedVIOConfig,
)
from ours.lib.backend.vio_window import VioConfig  # noqa: E402


def load_basalt(d: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for line in (d / "basalt" / "vio_pose.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[int(r["seq"])] = np.asarray(r["pos"], dtype=np.float64)
    return out


def umeyama(src, dst, scale):
    ms, md = src.mean(0), dst.mean(0)
    sc, dc = src - ms, dst - md
    U, D, Vt = np.linalg.svd((dc.T @ sc) / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (np.trace(np.diag(D) @ S) / ((sc ** 2).sum() / len(src))) if scale else 1.0
    t = md - s * R @ ms
    a = (s * (R @ src.T)).T + t
    return float(np.sqrt(((a - dst) ** 2).sum(1).mean())), float(s)


def run(reader, imu, Ric, gc, ac, bg0, n, cfg) -> dict:
    vo = WindowedVIORGBDOdometry(
        reader.K, imu["ts_ns"], gc, ac, bg0=bg0, ba0=np.zeros(3),
        cfg=cfg, odom_cfg=OdometryConfig(gyro_fuse=True))
    t0 = imu["ts_ns"][0]
    win = imu["ts_ns"] <= t0 + int(0.3 * 1e9)
    vo.align_to_gravity(Ric @ imu["accel"][win].mean(0))
    est = {}
    for i in range(n):
        f = reader.load_frame(i)
        p = vo.process(f.gray_left, f.depth_m, f.ts_ns)
        est[f.seq] = p[:3, 3].copy()
    return est


def score(est, bas):
    common = sorted(set(est) & set(bas))
    src = np.array([est[s] for s in common])
    dst = np.array([bas[s] for s in common])
    r, _ = umeyama(src, dst, False)
    _, s = umeyama(src, dst, True)
    path = np.linalg.norm(np.diff(dst, axis=0), axis=1).sum()
    return r, 100 * r / path, s, path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()

    d = Path(args.session)
    reader = SessionReader(d)
    n = len(reader) if args.max_frames <= 0 else min(args.max_frames, len(reader))
    imu = reader.load_imu()
    Ric = reader.calib.T_imu_left[:3, :3]
    gc = (Ric @ imu["gyro"].T).T
    ac = (Ric @ imu["accel"].T).T
    t0 = imu["ts_ns"][0]
    win = imu["ts_ns"] <= t0 + int(0.3 * 1e9)
    bg0 = gc[win].mean(0)
    bas = load_basalt(d)

    print(f"session {d.name}  frames {n}/{len(reader)}")

    # reference: the mature windowed-BA backend (vision only) on same frames
    ba = WindowedRGBDOdometry(reader.K, odom_cfg=OdometryConfig(gyro_fuse=True))
    t0 = imu["ts_ns"][0]
    win = imu["ts_ns"] <= t0 + int(0.3 * 1e9)
    ba.align_to_gravity(Ric @ imu["accel"][win].mean(0))
    est = {}
    for i in range(n):
        f = reader.load_frame(i)
        p = ba.process(f.gray_left, f.depth_m)
        est[f.seq] = p[:3, 3].copy()
    r, pct, s, path = score(est, bas)
    print(f"  {'BA ref (bundle.optimize)':24s}  ATE={r*1000:6.1f}mm  "
          f"%path={pct:4.2f}  scale={s:.3f}")

    for tag, cfg in [
        ("imu OFF (vision only)", WindowedVIOConfig(use_imu=False)),
        ("imu ON  (tight)", WindowedVIOConfig(use_imu=True)),
    ]:
        est = run(reader, imu, Ric, gc, ac, bg0, n, cfg)
        r, pct, s, path = score(est, bas)
        print(f"  {tag:24s}  ATE={r*1000:6.1f}mm  %path={pct:4.2f}  scale={s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
