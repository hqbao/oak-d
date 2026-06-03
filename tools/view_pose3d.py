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
        return OakOursVioSource(fps=args.fps, backend="f2f")
    if name == "ours-ba":
        from oakd.sources.depthai_ours_vio import OakOursVioSource
        return OakOursVioSource(
            fps=args.fps, backend="ba",
            ba_window=args.ba_window, ba_kf_every=args.ba_kf_every,
            ba_iters=args.ba_iters)
    if name == "ours-slam":
        from oakd.sources.depthai_ours_vio import OakOursVioSource
        return OakOursVioSource(
            fps=args.fps, backend="slam",
            slam_kf_every=args.slam_kf_every, slam_radius_m=args.slam_radius)
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
    # SLAM tuning (ours-slam)
    ap.add_argument("--slam-kf-every", type=int, default=5, dest="slam_kf_every",
                    help="SLAM update cadence: insert+loop-detect every N frames "
                         "(lower = more frequent loop closure) [5]")
    ap.add_argument("--slam-radius", type=float, default=0.0,
                    help="spatial gate (m): only loop-check keyframes within this "
                         "radius; bounds cost on very long runs. 0 = check all "
                         "(default; the appearance gate already rejects distant "
                         "keyframes cheaply) [0]")
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
