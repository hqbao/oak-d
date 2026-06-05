#!/usr/bin/env python3
"""Stage-1 sanity check for the from-scratch VIO.

Feed ONE recorded frame (+ its depth map) through the geometry and prove the
data path is correct: load calib, decode the left image + depth, back-project
to a 3D point cloud, and report what came out.

Everything runs offline on recorded gold data -- no OAK-D required.

Usage::

    python tools/vio_feed_test.py                       # default session, frame 0
    python tools/vio_feed_test.py --frame 200           # pick a denser frame
    python tools/vio_feed_test.py --session sessions/gold/lab_loop_30s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.vio import SessionReader, backproject, valid_mask  # noqa: E402


def describe_frame(reader: SessionReader, index: int) -> np.ndarray:
    frame = reader.load_frame(index)
    K = frame.K
    depth = frame.depth_m

    cloud = backproject(depth, K)
    mask = valid_mask(depth)
    pts = cloud[mask]

    h, w = depth.shape
    valid_pct = 100.0 * mask.sum() / mask.size

    print(f"--- frame {index} (seq={frame.seq}, ts={frame.ts_s:.3f}s) ---")
    print(f"  image     : {frame.gray_left.shape} {frame.gray_left.dtype}, "
          f"mean intensity={frame.gray_left.mean():.1f}")
    print(f"  depth      : {h}x{w}, valid {mask.sum()}/{mask.size} ({valid_pct:.1f}%)")
    if pts.shape[0]:
        z = pts[:, 2]
        print(f"  depth range: {z.min()*1000:.0f}..{z.max()*1000:.0f} mm "
              f"(median {np.median(z)*1000:.0f} mm)")
        print(f"  cloud      : {pts.shape[0]} points, "
              f"x[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
              f"y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
              f"z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}] m")
    else:
        print("  cloud      : EMPTY (no valid depth in this frame)")
    return frame.depth_m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    reader = SessionReader(args.session)
    print(f"session   : {reader.dir}")
    print(f"frames    : {len(reader)}  fps={reader.meta['params']['fps']}")
    print(f"calib     : left fx={reader.calib.left.fx:.2f} fy={reader.calib.left.fy:.2f} "
          f"cx={reader.calib.left.cx:.2f} cy={reader.calib.left.cy:.2f}")
    print(f"baseline  : {reader.calib.baseline_m*1000:.1f} mm")
    print()

    describe_frame(reader, args.frame)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
