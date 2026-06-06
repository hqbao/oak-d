#!/usr/bin/env python3
"""Entry point — launch the 3D pose viewer for *our* from-scratch VIO.

Standalone to the ``ours/`` pipeline: it wires our live RGB-D VIO source (the flow
graph from :mod:`ours.app`, bridged to the viewer by
:class:`ours.ui.live_source.FlowPoseSource`) into our own copy of the Qt 3D viewer
(:mod:`ours.ui`). It shares nothing with ``baseline/`` — the Basalt backends live in
``baseline/tools/view_pose3d.py``.

Sources: ``ours`` (bare f2f), ``ours-ba`` / ``ours-slam`` (same flow pipeline + an
out-of-process BA / loop-closure optimiser refining the map behind the responsive
marker), and ``fake`` (device-free figure-8 for UI bring-up).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# NOTE: keep module top import-light. The out-of-process BA/SLAM worker
# (``ours.lib.engine.subprocess``) is spawned with the ``spawn`` start method,
# whose bootstrap RE-IMPORTS this script as ``__mp_main__`` in the child. Importing
# PyQt6 / the Qt viewer there would pull a GUI toolkit into every worker process
# (slow, and Qt dislikes living in a spawned child). So the Qt / UI imports live
# inside the functions that need them, NOT at module top.


def _build_source(name: str, args):
    name = name.lower()
    if name == "fake":
        from ours.ui.source import FakePoseSource
        return FakePoseSource(rate_hz=100.0, radius_m=3.0, period_s=12.0)
    if name in ("ours", "ours-ba", "ours-slam"):
        # The live source is the flow pipeline (cam/imu_cam/odometry/ui flows over
        # a pub/sub bus). The displayed MARKER is always the realtime f2f pose
        # (pose.odom) with realtime-bounded inboxes. ``ours-ba`` / ``ours-slam``
        # additionally run the windowed-BA / loop-closure SLAM solve
        # OUT-OF-PROCESS (so it never holds the read loop's GIL) and refine the MAP
        # behind the marker: ours-ba draws the cyan BA-refined trajectory,
        # ours-slam the corrected keyframe dots + loop-closure flash.
        from ours.ui.live_source import FlowPoseSource
        mode = {"ours": "odom", "ours-ba": "ba", "ours-slam": "slam"}[name]
        return FlowPoseSource(width=args.width, height=args.height,
                              fps=args.fps,
                              recalibrate_bias=args.recalibrate_bias,
                              mode=mode)
    raise SystemExit(f"unknown --source '{name}' "
                     f"(expected: fake|ours|ours-ba|ours-slam)")


def main() -> int:
    ap = argparse.ArgumentParser(description="3D pose viewer (our VIO)")
    ap.add_argument("--source", default="ours",
                    choices=("fake", "ours", "ours-ba", "ours-slam"),
                    help="pose provider (ours = flow pipeline, f2f live; "
                         "ours-ba = + out-of-process windowed BA (cyan refined-map "
                         "line, marker stays responsive f2f); ours-slam = + "
                         "out-of-process loop-closure SLAM (keyframe dots + loop "
                         "flash); fake = device-free figure-8)")
    ap.add_argument("--fps", type=int, default=20,
                    help="camera frame rate (ours/ours-ba/ours-slam) [20]")
    ap.add_argument("--recalibrate-bias", action="store_true",
                    dest="recalibrate_bias",
                    help="live: ignore the cached gyro bias and re-measure it "
                         "(saved per device); otherwise it is calibrated once and "
                         "reused across runs")
    # Frame resolution. Lower = lighter on CPU; the flow pipeline auto-scales its
    # pixel-unit vision thresholds from the 640x400 baseline (docs/RESOLUTION_TUNING.md).
    ap.add_argument("--width", type=int, default=640,
                    help="capture width in px (lower = lighter) [640]")
    ap.add_argument("--height", type=int, default=400,
                    help="capture height in px [400]")
    args = ap.parse_args()

    # Qt / UI imports are local (see the module-top note: the spawned BA/SLAM
    # worker re-imports this script and must not pull in a GUI toolkit).
    from PyQt6.QtWidgets import QApplication
    from ours.lib.misc.pose import PoseHistory
    from ours.ui.mainwindow import MainWindow

    history = PoseHistory(capacity=8192)
    source = _build_source(args.source, args)

    # Shared Core-profile GL context: pyqtgraph caches shader programs per
    # process but GL program ids are per-context, so the app's two GL views
    # (pose Viewer3D + accel 3D) must share contexts or the second one throws
    # GLError(1281). Must be set before the QApplication. See theme docstring.
    from ours.ui import theme
    theme.ensure_gl_format()
    app = QApplication(sys.argv)
    # Live device sources show best top-down (drone seen from above); the
    # synthetic figure-8 reads better in the default iso view.
    default_view = "ISO" if args.source.lower() == "fake" else "TOP"
    win = MainWindow(history, source, source_name=args.source,
                     default_view=default_view,
                     cap_width=args.width, cap_height=args.height,
                     cap_fps=args.fps)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
