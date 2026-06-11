"""capture process: own the OAK-D (or a recorded session) + publish cam/IMU/depth.

Wires the same ``read_cam`` + ``imu_cam`` (with depth steps) front-end the
in-process :func:`imu_camera.modules.pipeline.build_live_frontend` /
:func:`~imu_camera.modules.pipeline.build_replay_frontend` builds, but adds an
:class:`~imu_camera.comms.IPCPublisher` that mirrors the local
:class:`~imu_camera.comms.LocalPubSub` topics onto an
:class:`~imu_camera.comms.IPCPubSub` server at the canonical endpoint
``"oak.capture"``. The calibration bundle is broadcast once on the **retained**
``calib.bundle`` topic so any subscriber that connects later (UI / SLAM / a calib
tool) immediately receives the latest copy.

Two modes share the same downstream wiring:

* ``--live`` -- :func:`~imu_camera.modules.pipeline.build_live_frontend` (real
  OAK-D). Hardware only.
* ``--session PATH`` (default) -- :func:`build_replay_frontend` over a recorded
  session, so the whole stack runs without a device on CI.

Run::

    python -m imu_camera.main --session sessions/gold/lab_loop_30s
    python -m imu_camera.main --live --width 640 --height 400 --fps 20
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imu_camera.comms import (                                    # noqa: E402
    IPCPublisher, IPCPubSub, LocalPubSub, RingRegistry, topics,
)
from imu_camera.comms.ring_registry import default_capture_specs  # noqa: E402
from imu_camera.comms.wire import WireCalibBundle                 # noqa: E402
from imu_camera.io.reader import SessionReader                    # noqa: E402

LOG = logging.getLogger("imu_camera.main")

#: Canonical endpoint name -- VIO / SLAM / UI / tools all connect here.
DEFAULT_ENDPOINT = "oak.capture"

#: Bridge-forwarded topics. Calibration travels on its own RETAINED topic so a
#: late subscriber boots with the bundle already cached.
_DATA_TOPICS = [
    topics.CAM_SYNC,
    topics.IMU_RAW,
    topics.IMUCAM_SAMPLE,
    topics.FRAME_DEPTH,
]
_CALIB_TOPIC = "calib.bundle"


def _build_calib_bundle_replay(reader: SessionReader) -> WireCalibBundle:
    """Wire-bundle from a recorded session's `calib.json`."""
    from imu_camera.modules.pipeline import _replay_imu_startup
    R_imu_cam, accel_align, gyro_bias = _replay_imu_startup(reader, use_gyro=True)
    T = reader.calib.T_imu_left if reader.calib.has_imu_extrinsics else None
    return WireCalibBundle(
        K=np.asarray(reader.K, dtype=np.float64),
        width=int(reader.calib.left.width),
        height=int(reader.calib.left.height),
        fps=20,
        T_imu_left=(None if T is None else np.asarray(T, dtype=np.float64)),
        R_imu_cam=(None if R_imu_cam is None
                   else np.asarray(R_imu_cam, dtype=np.float64)),
        accel_align=(None if accel_align is None
                     else np.asarray(accel_align, dtype=np.float64)),
        gyro_bias=(None if gyro_bias is None
                   else np.asarray(gyro_bias, dtype=np.float64)),
        # Replay has no live device -> the UI falls back to "default" when it
        # keys any IMU calibration it saves.
        device_id=None,
    )


def _scale_bundle_to_tof(bundle: WireCalibBundle, *,
                         src_w: int, src_h: int) -> WireCalibBundle:
    """Return a copy of ``bundle`` with K + dims scaled to the ToF grid.

    The ToF frame is a NON-uniform resize of the source (54/W_src != 42/H_src),
    so K is scaled ANISOTROPICALLY: fx, cx by ``TOF_W/src_w`` and fy, cy by
    ``TOF_H/src_h``. Depth metres are unchanged (the world distance a pixel sees
    does not change when the image is resized -- only the focal length in pixels
    does). Every other field (extrinsics, IMU calib, device id) carries through.
    """
    from imu_camera.modules.pipeline import TOF_W, TOF_H
    import dataclasses

    sx = TOF_W / float(src_w)
    sy = TOF_H / float(src_h)
    K = np.asarray(bundle.K, dtype=np.float64).copy()
    K[0, 0] *= sx          # fx
    K[0, 2] *= sx          # cx
    K[1, 1] *= sy          # fy
    K[1, 2] *= sy          # cy
    return dataclasses.replace(bundle, K=K, width=TOF_W, height=TOF_H)


def _build_calib_bundle_live(cal) -> WireCalibBundle:
    """Wire-bundle from a live `read_live_calibration` result."""
    T = cal.calib.T_imu_left if cal.calib.has_imu_extrinsics else None
    gyro_bias = (cal.imu_calibration.gyro_bias
                 if cal.imu_calibration is not None else None)
    return WireCalibBundle(
        K=np.asarray(cal.K, dtype=np.float64),
        width=int(cal.calib.left.width),
        height=int(cal.calib.left.height),
        fps=20,
        T_imu_left=(None if T is None else np.asarray(T, dtype=np.float64)),
        R_imu_cam=(None if cal.R_imu_cam is None
                   else np.asarray(cal.R_imu_cam, dtype=np.float64)),
        accel_align=(None if cal.accel_align is None
                     else np.asarray(cal.accel_align, dtype=np.float64)),
        gyro_bias=(None if gyro_bias is None
                   else np.asarray(gyro_bias, dtype=np.float64)),
        # Carry the live device id so the UI keys any saved IMU calib by the SAME
        # id capture/VIO use -> the saved calib takes effect on the next start.
        device_id=cal.device_id,
    )


# --------------------------------------------------------------------------- #
def run_capture_replay(session: Path, endpoint: str, *,
                       width: int, height: int,
                       max_frames: int = 0,
                       depth_fast: bool = True,
                       tof_sim: bool = False) -> int:
    """Replay-driven capture: SessionReader -> bridge -> IPC."""
    from imu_camera.modules.pipeline import (
        TOF_W, TOF_H, _replay_imu_startup, build_replay_frontend)
    from imu_camera.mathlib.imu.imu_calib import ImuCalibration

    reader = SessionReader(session)
    # Use the session's native resolution so the rings line up with the frames.
    width, height = int(reader.calib.left.width), int(reader.calib.left.height)

    # VL53L9CX simulation: depth is computed at the session SOURCE resolution
    # (width x height, where SGM works) but the PUBLISHED frames/depth are
    # downsampled to the fixed ToF grid by ToFDownsampleStep. The rings + the
    # broadcast calib bundle must therefore match the 54x42 grid, not the source.
    pub_w, pub_h = (TOF_W, TOF_H) if tof_sim else (width, height)

    rings = RingRegistry().create_all(default_capture_specs(
        endpoint=endpoint, width=pub_w, height=pub_h))
    # Live mode uses non-blocking publish (drop-oldest on stall so the OAK-D
    # firmware watchdog never fires). Replay mode below uses blocking=True so
    # every replayed frame reaches VIO.
    server = IPCPubSub(endpoint, role="server", retain_topics={_CALIB_TOPIC},
                       blocking=False)
    local = LocalPubSub()

    # In the ToF path cam.sync carries the SOURCE-res stereo pair (640x400) on the
    # local bus only; it must NOT cross the IPC boundary because the rings are
    # sized to 54x42 (writing a 640x400 array would raise). No IPC consumer needs
    # cam.sync (VIO subscribes only to imucam.sample + frame.depth), so drop it
    # from the bridged topics for the ToF run.
    data_topics = ([t for t in _DATA_TOPICS if t != topics.CAM_SYNC]
                   if tof_sim else _DATA_TOPICS)

    # Build the publisher BEFORE the front-end so subscribers connecting at any
    # time are wired in (the publisher subscribes to the local bus eagerly).
    pub = IPCPublisher(local, server, rings, data_topics, endpoint=endpoint)
    pub.start()

    # Replay startup IMU references for the imu_cam module's calibration.
    _, _, gyro_bias = _replay_imu_startup(reader, use_gyro=True)
    calibration = (ImuCalibration(gyro_bias=gyro_bias)
                   if gyro_bias is not None else None)

    cam_module, imu_module = build_replay_frontend(
        bus=local, reader=reader, depth_fast=depth_fast,
        max_frames=int(max_frames), calibration=calibration, tof_sim=tof_sim)

    # Broadcast the retained calibration bundle BEFORE starting the front-end
    # so any subscriber that connects mid-run gets the cached one immediately.
    # ToF: scale K to the 54x42 grid so a consumer that reads calib.bundle solves
    # against the SAME pixel grid the published frames/depth use.
    bundle = _build_calib_bundle_replay(reader)
    if tof_sim:
        bundle = _scale_bundle_to_tof(bundle, src_w=width, src_h=height)
    server.publish(_CALIB_TOPIC, bundle)
    LOG.info("capture[%s] replay session=%s frames=%d src=%dx%d pub=%dx%d%s",
             endpoint, session, len(reader), width, height, pub_w, pub_h,
             " (vl53l9cx ToF sim)" if tof_sim else "")

    # Install SIGTERM handler BEFORE starting the modules so the launcher's
    # SIGTERM is observed even if the producer is mid-frame. Without this the
    # `cam_module` join below blocks until the source is fully drained -- a
    # 30-second replay session would block shutdown for ~30 s and the launcher
    # would SIGKILL the process at the 10 s deadline, leaking every ring slot.
    stop = [False]
    def _on_sigterm(_signo, _frame):
        stop[0] = True
        # cam_module is a SourceModule; setting its _stop flag breaks out of the
        # produce() loop at the next item boundary so the join() returns.
        cam_module.stop()
    signal.signal(signal.SIGTERM, _on_sigterm)

    imu_module.start()
    cam_module.start()
    LOG.info("capture: cam + imu modules started, waiting for join ...")
    try:
        # Poll instead of pure join() so KeyboardInterrupt + signal handlers
        # are delivered promptly. The poll cadence (0.2 s) matches the live
        # path; cam_module.is_alive() flips False after produce() returns or
        # _stop is observed inside the loop.
        while not stop[0] and cam_module.is_alive():
            time.sleep(0.2)
        LOG.info("capture: cam loop exit (stop=%s, alive=%s)",
                 stop[0], cam_module.is_alive())
    except KeyboardInterrupt:
        LOG.info("capture: SIGINT -> stopping")
        stop[0] = True
    finally:
        # ReadCamModule (SourceModule) emits END on CAM_SYNC when produce()
        # returns; ImuCamModule forwards END from CAM_SYNC to IMU_RAW +
        # IMUCAM_SAMPLE + FRAME_DEPTH (see its `forwards_to`). The publisher
        # bridge converts those local-bus ENDs to WireEnd and sends them on the
        # IPC server.
        #
        # CRITICAL: wait for ImuCamModule's drain to chew through every queued
        # CAM_SYNC + the END BEFORE calling imu_module.stop(). stop() sets
        # `_stop`, which the drain checks at the TOP of every loop iteration --
        # so if we stop while CAM_SYNC items are still queued, we discard them
        # AND the END. `done` is set inside `_handle_end` after expected_ends
        # have been processed, which only happens when END is drained.
        #
        # Under SIGTERM the operator wants a fast exit, NOT a full drain --
        # END will never arrive from a half-killed producer, so cap the wait
        # at 2 s. Natural end-of-replay keeps the generous 120 s ceiling so a
        # busy backend can finish.
        cam_module.stop()
        drain_timeout = 2.0 if stop[0] else 120.0
        LOG.info("capture: waiting for imu module to drain (timeout=%.1fs) ...",
                 drain_timeout)
        ok = imu_module.done.wait(timeout=drain_timeout)
        LOG.info("capture: imu_module.done=%s", ok)
        imu_module.stop()
        # Give the bridge a brief window to flush the buffered WireEnds onto
        # the socket before we tear down the server.
        time.sleep(0.3)
        pub.stop()
        server.close()
        rings.unlink()
        rings.close()
        LOG.info("capture: shutdown complete")
    return 0


def run_capture_live(endpoint: str, *,
                     width: int, height: int, fps: int,
                     depth_fast: bool = True,
                     use_gyro: bool = True,
                     recalibrate_bias: bool = False,
                     use_camera_calib: bool = False,
                     tof_sim: bool = False) -> int:
    """Live OAK-D capture: device -> bridge -> IPC."""
    from imu_camera.modules.pipeline import TOF_W, TOF_H, build_live_frontend

    # ToF sim: depth runs at the SOURCE width x height (SGM), but the published
    # frames/depth + the rings + the calib bundle are the 54x42 ToF grid.
    pub_w, pub_h = (TOF_W, TOF_H) if tof_sim else (width, height)

    rings = RingRegistry().create_all(default_capture_specs(
        endpoint=endpoint, width=pub_w, height=pub_h))
    # Live: the IPC server is non-blocking (drop-oldest on stall) so a slow
    # downstream subscriber never stalls the OAK-D producer (firmware watchdog
    # ~1.5 s). The local front-end stays FIFO (`latest_only=False`): coalescing
    # imucam.sample / frame.depth here would drop frames BEFORE the bridge,
    # breaking VIO (the gyro continuity required by PreintegratePrior and the KLT
    # continuity required by TrackFeatures). Backpressure belongs at the IPC
    # boundary, not at the VIO inputs.
    server = IPCPubSub(endpoint, role="server", retain_topics={_CALIB_TOPIC},
                       blocking=False)
    local = LocalPubSub()
    # ToF: drop cam.sync from the bridge -- it carries the source-res pair, which
    # would not fit the 54x42 rings, and no IPC consumer reads it (see the replay
    # path for the full rationale).
    data_topics = ([t for t in _DATA_TOPICS if t != topics.CAM_SYNC]
                   if tof_sim else _DATA_TOPICS)
    pub = IPCPublisher(local, server, rings, data_topics, endpoint=endpoint)
    pub.start()

    try:
        device, cam_module, imu_module, cal = build_live_frontend(
            bus=local, width=width, height=height, fps=fps,
            use_gyro=use_gyro, depth_fast=depth_fast,
            recalibrate_bias=recalibrate_bias,
            use_camera_calib=use_camera_calib, latest_only=False,
            tof_sim=tof_sim)
    except Exception as e:                                         # noqa: BLE001
        LOG.error("capture: live build failed: %s", e)
        pub.stop()
        server.close()
        rings.unlink()
        rings.close()
        return 1

    bundle = _build_calib_bundle_live(cal)
    if tof_sim:
        bundle = _scale_bundle_to_tof(bundle, src_w=width, src_h=height)
    server.publish(_CALIB_TOPIC, bundle)
    LOG.info("capture[%s] live src=%dx%d pub=%dx%d@%d depth_fast=%s%s",
             endpoint, width, height, pub_w, pub_h, fps, depth_fast,
             " (vl53l9cx ToF sim)" if tof_sim else "")

    stop = [False]
    def _on_sigterm(_signo, _frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, _on_sigterm)

    imu_module.start()
    cam_module.start()
    try:
        while not stop[0] and cam_module.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        LOG.info("capture: SIGINT -> stopping")
    finally:
        # Close the device FIRST: the OAK-D firmware watchdog is only ~1.5s and
        # tearing down the bridge / IPC before release() risks tripping it.
        cam_module.stop()
        imu_module.stop()
        try:
            device.release()
        except Exception:                                          # noqa: BLE001
            pass
        # Same as the replay path: the front-end modules already forward END
        # via _emit_end / forwards_to on disconnect; the bridge mirrors them.
        time.sleep(0.3)
        pub.stop()
        server.close()
        rings.unlink()
        rings.close()
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Surface uncaught thread exceptions (otherwise a crashed module drain
    # silently leaves the process alive with no published data).
    def _excepthook(args):
        LOG.error("THREAD CRASH in %s: %s: %s", args.thread.name,
                  args.exc_type.__name__, args.exc_value, exc_info=(
                      args.exc_type, args.exc_value, args.exc_traceback))
    threading.excepthook = _excepthook
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help=f"IPCPubSub endpoint name (default: {DEFAULT_ENDPOINT!r})")
    ap.add_argument("--live", action="store_true",
                    help="open the OAK-D instead of replaying a session")
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s",
                    help="session directory (replay mode)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap frames in replay (0 = all)")
    ap.add_argument("--depth-fast", action="store_true", default=True,
                    help="half-res SGM preset (faster)")
    ap.add_argument("--no-gyro", action="store_true",
                    help="live: disable IMU gyro use in the calibration bundle")
    ap.add_argument("--recalibrate-bias", action="store_true",
                    help="live: ignore the cached gyro bias and re-measure it")
    ap.add_argument("--use-camera-calib", action="store_true",
                    help="live: apply the operator's SAVED per-device stereo calib "
                         "(from the wizard) instead of the FACTORY calib. Default "
                         "OFF -- factory is the trusted metrology reference; this "
                         "flag opts into the stored user calib if one exists.")
    ap.add_argument("--vl53l9cx", action="store_true",
                    help="simulate a VL53L9CX-class ToF camera: compute depth at "
                         "the source resolution then downsample gray + depth to "
                         "54x42 (accurate per-pixel ToF depth + intensity + IMU)")
    args = ap.parse_args()

    if args.live:
        return run_capture_live(
            endpoint=args.endpoint, width=args.width, height=args.height,
            fps=args.fps, depth_fast=args.depth_fast,
            use_gyro=not args.no_gyro,
            recalibrate_bias=args.recalibrate_bias,
            use_camera_calib=args.use_camera_calib,
            tof_sim=args.vl53l9cx)
    return run_capture_replay(
        session=Path(args.session), endpoint=args.endpoint,
        width=args.width, height=args.height,
        max_frames=args.max_frames, depth_fast=args.depth_fast,
        tof_sim=args.vl53l9cx)


if __name__ == "__main__":
    # Same os._exit pattern as the other split process mains -- prevent any
    # lingering non-daemon thread (depthai background thread, numba pool, etc.)
    # from holding the process past the launcher's 10 s SIGTERM deadline.
    import os as _os
    _rc = main()
    LOG.info("capture: main returned, calling os._exit(%d)", int(_rc))
    logging.shutdown()
    _os.sys.stdout.flush()
    _os.sys.stderr.flush()
    _os._exit(int(_rc))
