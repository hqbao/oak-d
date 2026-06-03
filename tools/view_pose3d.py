#!/usr/bin/env python3
"""Entry point — launch the OAK-D 3D pose viewer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication       # noqa: E402

from oakd.pose import PoseHistory              # noqa: E402
from oakd.sources import FakePoseSource        # noqa: E402
from oakd.ui.mainwindow import MainWindow      # noqa: E402


def _build_source(name: str, args):
    name = name.lower()
    if name == "fake":
        return FakePoseSource(rate_hz=100.0, radius_m=3.0, period_s=12.0)
    if name == "oak":
        from oakd.sources.depthai_vio import OakBasaltVioSource
        return OakBasaltVioSource()
    if name == "slam":
        from oakd.sources.depthai_slam import OakBasaltSlamSource
        return OakBasaltSlamSource()
    if name == "ours":
        from oakd.sources.depthai_ours_vio import OakOursVioSource
        return OakOursVioSource(fps=args.fps, backend="f2f",
                                use_own_klt=args.own_klt)
    if name == "ours-ba":
        from oakd.sources.depthai_ours_vio import OakOursVioSource
        return OakOursVioSource(
            fps=args.fps, backend="ba", use_own_klt=args.own_klt,
            ba_window=args.ba_window, ba_kf_every=args.ba_kf_every,
            ba_iters=args.ba_iters)
    if name == "ours-slam":
        from oakd.sources.depthai_ours_vio import OakOursVioSource
        return OakOursVioSource(
            fps=args.fps, backend="slam", use_own_klt=args.own_klt,
            slam_kf_every=args.slam_kf_every, slam_radius_m=args.slam_radius,
            slam_kf_min_trans=args.slam_kf_min_trans,
            slam_kf_min_rot=args.slam_kf_min_rot,
            slam_max_kf=args.slam_max_kf)
    raise SystemExit(f"unknown --source '{name}' "
                     f"(expected: fake|oak|slam|ours|ours-ba|ours-slam)")


def main() -> int:
    ap = argparse.ArgumentParser(description="OAK-D 3D pose viewer")
    ap.add_argument("--source", default="fake",
                    choices=("fake", "oak", "slam", "ours", "ours-ba",
                             "ours-slam"),
                    help="pose provider (ours = our f2f VO; ours-ba = windowed "
                         "BA; ours-slam = f2f + loop-closure SLAM)")
    ap.add_argument("--fps", type=int, default=20,
                    help="camera frame rate (ours/ours-ba/ours-slam) [20]")
    ap.add_argument("--own-klt", action="store_true", dest="own_klt",
                    help="use our own pure-NumPy KLT + corner detector for the "
                         "LIVE display (library-free). With Numba installed the "
                         "KLT inner loop is JIT-compiled and runs full quality at "
                         "~15ms/frame (real time); without Numba it falls back to "
                         "a lighter preset. Default OFF: live uses cv2 (~3ms). "
                         "Offline scoring (tools/vio_run.py) always uses our own.")
    # SLAM tuning (ours-slam)
    ap.add_argument("--slam-kf-every", type=int, default=5, dest="slam_kf_every",
                    help="SLAM update cadence: insert+loop-detect every N frames "
                         "(lower = more frequent loop closure) [5]")
    ap.add_argument("--slam-radius", type=float, default=0.0,
                    help="spatial gate (m): only loop-check keyframes within this "
                         "radius; bounds cost on very long runs. 0 = check all "
                         "(default; the appearance gate already rejects distant "
                         "keyframes cheaply) [0]")
    ap.add_argument("--slam-kf-min-trans", type=float, default=0.0,
                    dest="slam_kf_min_trans",
                    help="motion gate: skip a keyframe unless the camera moved "
                         ">= this many metres since the last one. Bounds the map "
                         "by trajectory length, not run time (a hovering drone "
                         "stops adding keyframes). 0 = disabled [0]")
    ap.add_argument("--slam-kf-min-rot", type=float, default=0.0,
                    dest="slam_kf_min_rot",
                    help="motion gate: skip a keyframe unless the camera rotated "
                         ">= this many degrees since the last one. 0 = disabled [0]")
    ap.add_argument("--slam-max-kf", type=int, default=0, dest="slam_max_kf",
                    help="hard cap on stored keyframes (drops the oldest when "
                         "exceeded; forgets old places so loops there can no "
                         "longer close). 0 = unlimited [0]")
    # BA tuning (ours-ba)
    ap.add_argument("--ba-window", type=int, default=6, dest="ba_window",
                    help="BA sliding-window size in keyframes [6]")
    ap.add_argument("--ba-kf-every", type=int, default=5, dest="ba_kf_every",
                    help="BA keyframe cadence: submit every N frames [5]")
    ap.add_argument("--ba-iters", type=int, default=5, dest="ba_iters",
                    help="BA iterations per solve [5]")
    args = ap.parse_args()

    history = PoseHistory(capacity=8192)
    source = _build_source(args.source, args)

    app = QApplication(sys.argv)
    win = MainWindow(history, source, source_name=args.source)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
