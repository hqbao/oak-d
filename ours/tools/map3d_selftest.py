#!/usr/bin/env python3
"""Self-test for the SLAM 3D-map point cloud (``ours.lib.misc.geometry`` +
``ours.tools.slam_map3d``). Two parts:

1. SYNTHETIC unit check of :func:`keyframe_pointcloud` -- a known depth + pose
   must back-project to the exact world points (geometry is correct).
2. GOLD smoke -- build a map from a real session offline and assert it is sane
   (points exist, finite, room-scale, one camera per keyframe).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.misc.geometry import keyframe_pointcloud                 # noqa: E402
from ours.tools.slam_map3d import build_map                            # noqa: E402

_FAILS = 0


def _check(cond: bool, msg: str) -> None:
    global _FAILS
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        _FAILS += 1


def test_synthetic() -> None:
    print("keyframe_pointcloud synthetic geometry")
    K = np.array([[100.0, 0, 32.0], [0, 100.0, 24.0], [0, 0, 1.0]])
    h, w = 48, 64
    depth = np.full((h, w), 2.0, np.float32)        # flat wall at z=2 m
    gray = np.full((h, w), 128, np.uint8)

    # Identity pose -> world == camera frame; stride 1, all pixels valid.
    pts, col = keyframe_pointcloud([np.eye(4)], [depth], [gray], K,
                                   stride=1, max_depth=8.0)
    _check(pts.shape[0] == h * w, f"all {h*w} pixels kept ({pts.shape[0]})")
    _check(np.allclose(pts[:, 2], 2.0), "every point sits on the z=2 m wall")
    _check(np.allclose(col, 128 / 255.0), "colour = grey intensity (128/255)")
    # Principal-ray pixel (cx,cy) back-projects to (0,0,2).
    centre = pts[(h // 2) * w + (w // 2)]
    _check(np.allclose(centre, [0, 0, 2], atol=0.02),
           f"centre pixel -> optical axis point {centre.round(3).tolist()}")

    # A pure +1 m world translation (T_world_cam) must shift the whole cloud +1 m.
    T = np.eye(4); T[0, 3] = 1.0
    pts2, _ = keyframe_pointcloud([T], [depth], [gray], K, stride=1)
    _check(np.allclose(pts2 - pts, [1, 0, 0]), "pose translation shifts the cloud")

    # Depth range gate drops far points.
    far = np.full((h, w), 50.0, np.float32)
    pts3, _ = keyframe_pointcloud([np.eye(4)], [far], [gray], K, max_depth=6.0)
    _check(pts3.shape[0] == 0, "points beyond max_depth are dropped")


def test_gold_smoke() -> None:
    print("build_map on a gold session (offline)")
    m = build_map("sessions/gold/lab_loop_30s", kf_every=5, max_frames=80,
                  use_slam=False, stride=6, max_depth=6.0)
    pts = m["points"]
    _check(m["n_kf"] > 0, f"keyframes collected ({m['n_kf']})")
    _check(len(pts) > 1000, f"cloud has points ({len(pts)})")
    _check(bool(np.all(np.isfinite(pts))), "all points finite")
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(pts) else 0.0
    _check(0.1 < extent < 100.0, f"room-scale extent ({extent:.1f} m)")
    _check(m["cams"].shape[0] == m["n_kf"], "one camera position per keyframe")


def main() -> int:
    print("map3d_selftest")
    test_synthetic()
    test_gold_smoke()
    print("ALL MAP3D SELFTESTS PASSED" if _FAILS == 0
          else f"MAP3D SELFTEST FAILED ({_FAILS})")
    return 0 if _FAILS == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
