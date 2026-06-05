#!/usr/bin/env python3
"""Entry point — launch the 3D pose viewer for *our* from-scratch VIO.

Standalone to the ``ours/`` pipeline: it wires our live RGB-D VIO source
(:class:`ours.depthai_ours_vio.OakOursVioSource`) into our own copy of the Qt
3D viewer (:mod:`ours.ui`). It shares nothing with ``baseline/`` — the Basalt
backends live in ``baseline/tools/view_pose3d.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt6.QtWidgets import QApplication       # noqa: E402

from ours.lib.pose import PoseHistory              # noqa: E402
from ours.ui.source import FakePoseSource          # noqa: E402
from ours.ui.mainwindow import MainWindow      # noqa: E402


def _build_source(name: str, args):
    name = name.lower()
    if name == "fake":
        return FakePoseSource(rate_hz=100.0, radius_m=3.0, period_s=12.0)
    # Resolution + per-resolution vision overrides shared by every ours-* source.
    # Each override defaults to None -> the source auto-scales it from the
    # 640x400 baseline (see ours/vio/resolution.py + docs/RESOLUTION_TUNING.md).
    res_kw = dict(
        width=args.width, height=args.height,
        max_corners=args.max_corners, min_distance=args.min_distance,
        klt_win=args.klt_win, klt_levels=args.klt_levels,
        reproj_px=args.reproj_px, num_disparities=args.num_disparities,
        orb_features=args.orb_features,
    )
    if name == "ours":
        # Default live source = the new flow pipeline (capture/depth/odometry/
        # backend/slam/ui flows over a pub/sub bus). Displays the real-time f2f
        # trajectory; backend + SLAM flows run alongside.
        from ours.flows.live_source import FlowPoseSource
        return FlowPoseSource(width=args.width, height=args.height,
                              fps=args.fps,
                              recalibrate_bias=args.recalibrate_bias)
    if name == "ours-legacy":
        from ours.depthai_ours_vio import OakOursVioSource
        return OakOursVioSource(fps=args.fps, backend="f2f", **res_kw)
    if name == "ours-ba":
        from ours.depthai_ours_vio import OakOursVioSource
        return OakOursVioSource(
            fps=args.fps, backend="ba",
            ba_window=args.ba_window, ba_kf_every=args.ba_kf_every,
            ba_iters=args.ba_iters, **res_kw)
    if name == "ours-slam":
        from ours.depthai_ours_vio import OakOursVioSource
        return OakOursVioSource(
            fps=args.fps, backend="slam",
            slam_kf_every=args.slam_kf_every, slam_radius_m=args.slam_radius,
            slam_kf_min_trans=args.slam_kf_min_trans,
            slam_kf_min_rot=args.slam_kf_min_rot,
            slam_max_kf=args.slam_max_kf, **res_kw)
    if name == "ours-vio":
        from ours.depthai_ours_vio import OakOursVioSource
        return OakOursVioSource(
            fps=args.fps, backend="vio",
            ba_window=args.ba_window, ba_kf_every=args.ba_kf_every, **res_kw)
    raise SystemExit(f"unknown --source '{name}' "
                     f"(expected: fake|ours|ours-legacy|ours-ba|ours-slam|ours-vio)")


def main() -> int:
    ap = argparse.ArgumentParser(description="3D pose viewer (our VIO)")
    ap.add_argument("--source", default="ours",
                    choices=("fake", "ours", "ours-legacy", "ours-ba",
                             "ours-slam", "ours-vio"),
                    help="pose provider (ours = flow pipeline, f2f live; "
                         "ours-legacy = monolithic f2f source; ours-ba = "
                         "windowed BA; ours-slam = f2f + loop-closure SLAM; "
                         "ours-vio = tight-coupled visual+IMU VIO)")
    ap.add_argument("--fps", type=int, default=20,
                    help="camera frame rate (ours/ours-ba/ours-slam) [20]")
    ap.add_argument("--recalibrate-bias", action="store_true",
                    dest="recalibrate_bias",
                    help="live (--source ours): ignore the cached gyro bias and "
                         "re-measure it (saved per device); otherwise it is "
                         "calibrated once and reused across runs")
    # Frame resolution (any ours-* source). Lower = lighter on CPU; the pipeline
    # auto-scales its pixel-unit vision thresholds from the 640x400 baseline, and
    # the per-resolution overrides below let us co-tune (docs/RESOLUTION_TUNING.md).
    ap.add_argument("--width", type=int, default=640,
                    help="capture width in px (lower = lighter) [640]")
    ap.add_argument("--height", type=int, default=400,
                    help="capture height in px [400]")
    # Per-resolution vision overrides. Default None = auto-scale from baseline.
    ap.add_argument("--max-corners", type=int, default=None, dest="max_corners",
                    help="frontend: Shi-Tomasi corner budget "
                         "[auto: round(400*width/640)]")
    ap.add_argument("--min-distance", type=float, default=None,
                    dest="min_distance",
                    help="frontend: min px between corners [auto: 12*width/640]")
    ap.add_argument("--klt-win", type=int, default=None, dest="klt_win",
                    help="frontend: KLT window in px, odd [auto: 21*width/640]")
    ap.add_argument("--klt-levels", type=int, default=None, dest="klt_levels",
                    help="frontend: KLT pyramid levels [auto: 3 at 640, -1/halving]")
    ap.add_argument("--reproj-px", type=float, default=None, dest="reproj_px",
                    help="odometry: PnP RANSAC reprojection gate px "
                         "[auto: 2*width/640]")
    ap.add_argument("--num-disparities", type=int, default=None,
                    dest="num_disparities",
                    help="stereo: SGM disparity search range px [auto: 96*width/640]")
    ap.add_argument("--orb-features", type=int, default=None, dest="orb_features",
                    help="loop closure (ours-slam): ORB budget [auto: 800*width/640]")
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
    # Live device sources show best top-down (drone seen from above); the
    # synthetic figure-8 reads better in the default iso view.
    default_view = "ISO" if args.source.lower() == "fake" else "TOP"
    win = MainWindow(history, source, source_name=args.source,
                     default_view=default_view)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
