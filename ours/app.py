"""Wire and run the ``ours`` VIO as a graph of flows.

This is the live-pipeline assembler: it creates one :class:`~ours.lib.pubsub.Bus`,
constructs the six flows (capture, depth, odometry, backend, slam, ui) and starts
their threads. The flows talk only over the bus (see ``ours.lib.topics``).

Run it in **replay mode** over a recorded session -- the offline harness that
drives the whole graph without a camera, so the flow decomposition can be
validated against the same data the offline ``vio_run`` oracle uses::

    python -m ours.app --session sessions/gold/lab_straight_20s --depth-fast

The capture flow is the only device-specific piece; ``ReplayCaptureFlow`` feeds
the graph from disk. A live ``LiveCaptureFlow`` (OAK-D) publishes the identical
topics, so the depth/odometry/backend/slam/ui flows are unchanged on hardware
(live device validation is done on the bench, not here).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .flows.backend import BackendFlow
from .flows.capture import ReplayCaptureFlow
from .flows.depth import DepthFlow
from .flows.odometry import OdometryFlow
from .flows.slam import SlamFlow
from .flows.ui import UiCollectorFlow
from .lib.io.reader import SessionReader
from .lib.odometry.odometry import OdometryConfig
from .lib.loop.slam import SlamConfig
from .lib.pubsub import Bus
from .lib.stereo.stereo import SGMConfig, SGMStereoMatcher


def build_graph(bus: Bus, K, matcher, *, ui, kf_every: int = 5,
                use_gyro: bool = True, slam_cfg: SlamConfig | None = None):
    """Build the shared depth/odometry/backend/slam flows around a ``ui`` sink.

    The capture flow is built by the caller (replay vs live); everything
    downstream of ``frame.raw`` is identical, so it is constructed here once.
    Returns the list of reactive flows ``[depth, odom, backend, slam, ui]``.
    """
    depth = DepthFlow(bus, matcher)
    odom = OdometryFlow(bus, K, OdometryConfig(gyro_fuse=use_gyro),
                        kf_every=kf_every, use_gyro=use_gyro)
    backend = BackendFlow(bus, K, kf_every=1)
    slam = SlamFlow(bus, K, slam_cfg or SlamConfig(loop_max_odom_rot_deg=30.0))
    return [depth, odom, backend, slam, ui]


def build_replay(bus: Bus, reader: SessionReader, *, kf_every: int = 5,
                 use_gyro: bool = True, depth_fast: bool = False,
                 max_frames: int = 0,
                 slam_cfg: SlamConfig | None = None):
    """Construct the full 6-flow graph driven by a recorded session.

    Returns ``(capture, reactive_flows, ui)``. The reactive flows subscribe to
    their topics during construction, so they capture every message even if the
    capture flow starts publishing before their threads are running.
    """
    sgm = SGMConfig.live() if depth_fast else SGMConfig()
    matcher = SGMStereoMatcher.from_calib(reader.calib, sgm)

    capture = ReplayCaptureFlow(bus, reader, max_frames=max_frames,
                                use_gyro=use_gyro)
    ui = UiCollectorFlow(bus)
    flows = build_graph(bus, reader.K, matcher, ui=ui, kf_every=kf_every,
                        use_gyro=use_gyro, slam_cfg=slam_cfg)
    return capture, flows, ui


def build_live(bus: Bus, *, width: int = 640, height: int = 400, fps: int = 20,
               kf_every: int = 5, use_gyro: bool = True, depth_fast: bool = True,
               recalibrate_bias: bool = False,
               ui=None, slam_cfg: SlamConfig | None = None):
    """Construct the live OAK-D graph. Opens the device to read calibration.

    The live capture flow taps both raw cameras + IMU; its depth matcher
    rectifies BOTH (``rectify_left=True``) since the raw left is unrectified.
    Returns ``(capture, reactive_flows, ui)``. Caller starts the threads.
    """
    from .flows.capture import LiveCaptureFlow

    capture = LiveCaptureFlow(bus, width=width, height=height, fps=fps,
                              depth_fast=depth_fast, use_gyro=use_gyro,
                              recalibrate_bias=recalibrate_bias)
    calib = capture.open()                         # opens device, reads calib
    matcher = SGMStereoMatcher.from_calib(calib.calib, calib.sgm_cfg,
                                          rectify_left=True)
    ui = ui if ui is not None else UiCollectorFlow(bus)
    flows = build_graph(bus, calib.K, matcher, ui=ui, kf_every=kf_every,
                        use_gyro=use_gyro, slam_cfg=slam_cfg)
    return capture, flows, ui


def run_replay(session: str, *, kf_every: int = 5, use_gyro: bool = True,
               depth_fast: bool = False, max_frames: int = 0,
               timeout_s: float = 1800.0):
    """Run the graph over a session and return ``(ui, reader, elapsed_s)``."""
    reader = SessionReader(Path(session))
    bus = Bus()
    capture, flows, ui = build_replay(
        bus, reader, kf_every=kf_every, use_gyro=use_gyro,
        depth_fast=depth_fast, max_frames=max_frames)

    t0 = time.time()
    for f in flows:
        f.start()
    capture.start()
    capture.join()                 # produce all frames + emit END on frame.raw
    finished = ui.done.wait(timeout=timeout_s)   # all 3 ENDs => graph drained
    for f in flows:
        f.stop()
    if not finished:
        raise TimeoutError("flow graph did not drain within timeout")
    return ui, reader, time.time() - t0


def run_live(*, width: int = 640, height: int = 400, fps: int = 20,
             kf_every: int = 5, use_gyro: bool = True,
             depth_fast: bool = True, recalibrate_bias: bool = False) -> int:
    """Headless live run: stream the OAK-D through the graph until Ctrl-C."""
    bus = Bus()
    capture, flows, ui = build_live(
        bus, width=width, height=height, fps=fps, kf_every=kf_every,
        use_gyro=use_gyro, depth_fast=depth_fast,
        recalibrate_bias=recalibrate_bias)
    for f in flows:
        f.start()
    capture.start()
    print("[ours-flow] live running — Ctrl-C to stop")
    try:
        while capture.is_alive():
            time.sleep(2.0)
            n_loops = ui.corrections[-1].n_loops if ui.corrections else 0
            print(f"[ours-flow] poses={len(ui.odom)} refined={len(ui.refined)} "
                  f"loops={n_loops}")
    except KeyboardInterrupt:
        print("\n[ours-flow] stopping…")
    finally:
        capture.stop()
        ui.done.wait(timeout=10.0)
        for f in flows:
            f.stop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true",
                    help="run the live OAK-D device instead of a recorded session")
    ap.add_argument("--session", default="sessions/gold/lab_straight_20s")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--no-gyro", action="store_true")
    ap.add_argument("--depth-fast", action="store_true",
                    help="half-res SGM live preset (faster)")
    ap.add_argument("--fps", type=int, default=20, help="live camera fps")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--recalibrate-bias", action="store_true",
                    dest="recalibrate_bias",
                    help="live: ignore the cached gyro bias and re-measure it "
                         "(saved per device); otherwise it is calibrated once "
                         "and reused")
    args = ap.parse_args()

    if args.live:
        return run_live(width=args.width, height=args.height, fps=args.fps,
                        kf_every=args.kf_every, use_gyro=not args.no_gyro,
                        depth_fast=True,  # full-res SGM is too slow live
                        recalibrate_bias=args.recalibrate_bias)

    ui, reader, elapsed = run_replay(
        args.session, kf_every=args.kf_every, use_gyro=not args.no_gyro,
        depth_fast=args.depth_fast, max_frames=args.max_frames)

    n_loops = ui.corrections[-1].n_loops if ui.corrections else 0
    print(f"session  : {reader.dir}")
    print(f"frames   : {len(ui.odom)} poses on pose.odom")
    print(f"refined  : {len(ui.refined)} poses on pose.refined")
    print(f"loops    : {n_loops} closure(s) over {len(ui.corrections)} correction(s)")
    print(f"elapsed  : {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
