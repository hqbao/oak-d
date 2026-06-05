"""Wire and run the ``ours`` VIO as a graph of flows.

This is the live-pipeline assembler: it creates one :class:`~ours.lib.flow.pubsub.Bus`,
constructs the flows and starts their threads. The flows talk only over the bus
(see ``ours.lib.flow.topics``).

There is ONE acquisition front-end, shared by the real-time VIO and the
camera/IMU visualiser: the ``cam`` flow emits a ``cam.sync`` per scheduled
stereo pair and the ``imu_cam`` flow drains its inertial buffer up to that
timestamp, publishing the synced ``imucam.sample`` (frames + calibrated IMU) and
-- in the VIO path -- the ``frame.depth`` from its own depth task. The odometry
flow consumes that single stream -- there is no separate capture monolith.

Run it in **replay mode** over a recorded session -- the offline harness that
drives the whole graph without a camera::

    python -m ours.app --session sessions/gold/lab_straight_20s --depth-fast

``ReplayCamSource`` / ``ReplayImuSource`` feed the front-end from disk; the live
OAK-D sources (``LiveCamSource`` / ``LiveImuSource`` off one shared device)
publish the identical topics, so odometry/backend/slam/ui are unchanged on
hardware (live device validation is done on the bench, not here).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .flows.backend import BackendFlow
from .flows.cam import CamFlow
from .flows.cam.sources import ReplayCamSource
from .flows.imu_cam import ImuCamFlow
from .flows.imu_cam.sources import ReplayImuSource
from .flows.odometry import OdometryFlow
from .flows.slam import SlamFlow
from .flows.ui import UiCollectorFlow
from .lib.imu.imu_calib import ImuCalibration
from .lib.io.reader import SessionReader
from .lib.odometry.odometry import OdometryConfig
from .lib.loop.slam import SlamConfig
from .lib.flow.pubsub import Bus
from .lib.stereo.stereo import SGMConfig, SGMStereoMatcher


def build_graph(bus: Bus, K, *, ui, R_imu_cam=None, accel_align=None,
                kf_every: int = 5, use_gyro: bool = True,
                with_backend_slam: bool = True, realtime_latest: bool = False,
                slam_cfg: SlamConfig | None = None):
    """Build the shared odometry/backend/slam flows around a ``ui`` sink.

    The acquisition front-end (``cam`` + ``imu_cam``, the latter owning the depth
    task) is built by the caller (replay vs live); everything downstream of
    ``imucam.sample`` / ``frame.depth`` is identical, so it is constructed here
    once. ``R_imu_cam`` / ``accel_align`` seed the odometry flow's gyro prior and
    startup gravity-leveling.

    ``with_backend_slam`` (default ``True``) builds the windowed-BA back-end and
    the loop-closing SLAM flow. A pure visualiser that only needs the odometry
    front-end's output (e.g. the keypoint-depth view subscribing ``frame.tracks``)
    passes ``False`` to skip those two heavy flows -- they would otherwise compete
    for CPU and make the live stream fall seconds behind realtime.

    ``realtime_latest`` (default ``False``) builds the odometry flow with a
    coalescing latest-only inbox so a realtime visualiser never falls behind when
    its consumer is slower than the producer (the FIFO default is required for the
    VIO + deterministic replay, which must process every frame). Returns the list
    of reactive flows: ``[odom, backend, slam, ui]`` or ``[odom, ui]``.
    """
    odom = OdometryFlow(bus, K, R_imu_cam=R_imu_cam, accel_align=accel_align,
                        odom_cfg=OdometryConfig(gyro_fuse=use_gyro),
                        kf_every=kf_every, use_gyro=use_gyro,
                        latest_only=realtime_latest)
    if not with_backend_slam:
        return [odom, ui]
    backend = BackendFlow(bus, K, kf_every=1)
    slam = SlamFlow(bus, K, slam_cfg or SlamConfig(loop_max_odom_rot_deg=30.0))
    return [odom, backend, slam, ui]


def _replay_imu_startup(reader: SessionReader, use_gyro: bool):
    """Startup IMU references for a recorded session.

    Returns ``(R_imu_cam, accel_align, gyro_bias)`` mirroring what the live
    front-end measures once at boot:

    * ``R_imu_cam`` -- IMU->camera rotation (gyro prior conjugation), or ``None``
      when the session has no IMU extrinsics (pure-vision fallback).
    * ``accel_align`` -- mean startup accelerometer (camera frame) over the first
      ~0.3 s, the gravity-leveling reference.
    * ``gyro_bias`` -- mean gyro over the first ~1 s (near-static start), removed
      from every sample by the imu_cam flow so the rotation prior is unbiased.
    """
    if not (use_gyro and reader.calib.has_imu_extrinsics):
        return None, None, None
    imu = reader.load_imu()
    ts = imu["ts_ns"]
    if ts.size <= 1:
        return None, None, None
    R_imu_cam = reader.calib.T_imu_left[:3, :3]
    t0 = int(ts[0])
    gyro_bias = None
    gwin = ts <= t0 + int(1.0 * 1e9)                 # first ~1 s
    if gwin.any():
        gyro_bias = imu["gyro"][gwin].mean(axis=0)
    accel_align = None
    awin = ts <= t0 + int(0.3 * 1e9)                 # first ~0.3 s
    if awin.any():
        accel_align = R_imu_cam @ imu["accel"][awin].mean(axis=0)
    return R_imu_cam, accel_align, gyro_bias


def build_replay_frontend(bus: Bus, reader: SessionReader, *,
                          depth_fast: bool = False, max_frames: int = 0,
                          calibration: ImuCalibration | None = None,
                          latest_only: bool = False):
    """Wire ONLY the acquisition front-end (``cam`` + ``imu_cam`` with depth) from
    a recorded session.

    For visualisers that consume ``frame.depth`` / ``imucam.sample`` directly and
    need no odometry/backend/slam (e.g. the image|depth|IMU triplet). ``calibration``
    is the IMU correction the imu_cam flow applies (``None`` -> the published IMU is
    raw); ``latest_only`` makes the front-end coalescing so a realtime visualiser
    stays fresh. Returns ``(cam_flow, imu_flow)``.
    """
    sgm = SGMConfig.live() if depth_fast else SGMConfig()
    matcher = SGMStereoMatcher.from_calib(reader.calib, sgm)
    imu_flow = ImuCamFlow(bus, ReplayImuSource(reader), matcher=matcher,
                          calibration=calibration, latest_only=latest_only)
    cam_flow = CamFlow(
        bus, ReplayCamSource(reader, max_frames=max_frames), fps=20)
    return cam_flow, imu_flow


def build_replay(bus: Bus, reader: SessionReader, *, kf_every: int = 5,
                 use_gyro: bool = True, depth_fast: bool = False,
                 max_frames: int = 0, ui=None, with_backend_slam: bool = True,
                 realtime_latest: bool = False,
                 slam_cfg: SlamConfig | None = None):
    """Construct the full flow graph driven by a recorded session.

    Returns ``((cam_flow, imu_flow), reactive_flows, ui)``. The reactive flows
    subscribe to their topics during construction, so they capture every message
    even if the front-end starts publishing before their threads are running.

    ``ui`` lets a caller inject its own sink (e.g. the keypoint-depth tracker's
    :class:`~ours.flows.ui.tracks.UiTracksFlow`); it defaults to the offline
    :class:`~ours.flows.ui.UiCollectorFlow`. ``with_backend_slam=False`` skips the
    heavy back-end + SLAM flows for a pure odometry-output visualiser;
    ``realtime_latest=True`` makes the heavy flows latest-only (bounded latency).
    """
    R_imu_cam, accel_align, gyro_bias = _replay_imu_startup(reader, use_gyro)
    calibration = (ImuCalibration(gyro_bias=gyro_bias)
                   if gyro_bias is not None else None)

    cam_flow, imu_flow = build_replay_frontend(
        bus, reader, depth_fast=depth_fast, max_frames=max_frames,
        calibration=calibration, latest_only=realtime_latest)

    ui = ui if ui is not None else UiCollectorFlow(bus)
    flows = build_graph(bus, reader.K, ui=ui, R_imu_cam=R_imu_cam,
                        accel_align=accel_align, kf_every=kf_every,
                        use_gyro=use_gyro, with_backend_slam=with_backend_slam,
                        realtime_latest=realtime_latest, slam_cfg=slam_cfg)
    return (cam_flow, imu_flow), flows, ui


def build_live_frontend(bus: Bus, *, width: int = 640, height: int = 400,
                        fps: int = 20, use_gyro: bool = True,
                        depth_fast: bool = True, recalibrate_bias: bool = False,
                        latest_only: bool = False):
    """Open the OAK-D and wire ONLY the acquisition front-end (``cam`` +
    ``imu_cam`` with depth) off ONE shared device.

    The shared half of :func:`build_live`, exposed for visualisers that consume
    ``frame.depth`` / ``imucam.sample`` directly and need no odometry/backend/slam
    (e.g. the image|depth|IMU triplet). The depth matcher rectifies BOTH cameras
    (``rectify_left=True``) since the raw left is unrectified. Returns
    ``(device, cam_flow, imu_flow, cal)`` where ``cal`` is the live-calibration
    bundle (``cal.imu_calibration`` etc.); the caller starts the threads and
    releases ``device`` when the run ends. Hardware-only.
    """
    from .lib.oak_live import SharedLiveDevice
    from .lib.live_calib import read_live_calibration
    from .flows.cam.sources import LiveCamSource
    from .flows.imu_cam.sources import LiveImuSource

    device = SharedLiveDevice(width=width, height=height, fps=fps)
    cal = read_live_calibration(device, width=width, height=height,
                                use_gyro=use_gyro, depth_fast=depth_fast,
                                recalibrate_bias=recalibrate_bias)
    matcher = SGMStereoMatcher.from_calib(cal.calib, cal.sgm_cfg,
                                          rectify_left=True)
    imu_flow = ImuCamFlow(bus, LiveImuSource(device), matcher=matcher,
                          calibration=cal.imu_calibration,
                          latest_only=latest_only)
    cam_flow = CamFlow(bus, LiveCamSource(device), fps=fps, realtime=True)
    return device, cam_flow, imu_flow, cal


def build_live(bus: Bus, *, width: int = 640, height: int = 400, fps: int = 20,
               kf_every: int = 5, use_gyro: bool = True, depth_fast: bool = True,
               recalibrate_bias: bool = False, with_backend_slam: bool = True,
               realtime_latest: bool = False,
               ui=None, slam_cfg: SlamConfig | None = None):
    """Construct the live OAK-D graph off ONE shared device.

    Opens the device to read calibration + startup IMU references, then wires the
    SAME front-end the replay path uses (``cam`` + ``imu_cam``) onto the live
    sources. Returns ``(device, (cam_flow, imu_flow), reactive_flows, ui)``; the
    caller starts the threads and releases ``device`` when the run ends.

    Hardware-only: validated on the bench, not in the offline test harness.
    """
    device, cam_flow, imu_flow, cal = build_live_frontend(
        bus, width=width, height=height, fps=fps, use_gyro=use_gyro,
        depth_fast=depth_fast, recalibrate_bias=recalibrate_bias,
        latest_only=realtime_latest)

    ui = ui if ui is not None else UiCollectorFlow(bus)
    flows = build_graph(bus, cal.K, ui=ui, R_imu_cam=cal.R_imu_cam,
                        accel_align=cal.accel_align, kf_every=kf_every,
                        use_gyro=use_gyro, with_backend_slam=with_backend_slam,
                        realtime_latest=realtime_latest, slam_cfg=slam_cfg)
    return device, (cam_flow, imu_flow), flows, ui


def run_replay(session: str, *, kf_every: int = 5, use_gyro: bool = True,
               depth_fast: bool = False, max_frames: int = 0,
               timeout_s: float = 1800.0):
    """Run the graph over a session and return ``(ui, reader, elapsed_s)``."""
    reader = SessionReader(Path(session))
    bus = Bus()
    (cam_flow, imu_flow), flows, ui = build_replay(
        bus, reader, kf_every=kf_every, use_gyro=use_gyro,
        depth_fast=depth_fast, max_frames=max_frames)

    t0 = time.time()
    for f in flows:
        f.start()
    # Order matters: the consumers and the IMU reader (with its buffer-filling
    # source) must be live before the camera reader fires the first trigger, so
    # every frame's interval already has inertial samples to drain.
    imu_flow.start()
    cam_flow.start()
    cam_flow.join()                # produce all frames + emit END on cam.sync
    finished = ui.done.wait(timeout=timeout_s)   # all ENDs => graph drained
    imu_flow.stop()
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
    device, (cam_flow, imu_flow), flows, ui = build_live(
        bus, width=width, height=height, fps=fps, kf_every=kf_every,
        use_gyro=use_gyro, depth_fast=depth_fast,
        recalibrate_bias=recalibrate_bias)
    for f in flows:
        f.start()
    imu_flow.start()
    cam_flow.start()
    print("[ours-flow] live running — Ctrl-C to stop")
    try:
        while cam_flow.is_alive():
            time.sleep(2.0)
            n_loops = ui.corrections[-1].n_loops if ui.corrections else 0
            print(f"[ours-flow] poses={len(ui.odom)} refined={len(ui.refined)} "
                  f"loops={n_loops}")
    except KeyboardInterrupt:
        print("\n[ours-flow] stopping…")
    finally:
        cam_flow.stop()
        imu_flow.stop()
        ui.done.wait(timeout=10.0)
        for f in flows:
            f.stop()
        device.release()
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
