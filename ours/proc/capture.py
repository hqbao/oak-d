"""capture process: own the OAK-D (or a recorded session) + publish cam/IMU/depth.

Wires the same ``cam`` + ``imu_cam`` (with depth task) front-end the in-process
``ours.app.build_live_frontend`` / ``build_replay_frontend`` builds, but adds
an :class:`IpcPublisherFlow` that mirrors the local Bus topics onto an
:class:`IpcServerBus` at the canonical endpoint ``"oak.capture"``. The calibration
bundle is broadcast once on the **retained** ``calib.bundle`` topic so any
subscriber that connects later (UI / SLAM / a calib tool) immediately receives
the latest copy.

Two modes share the same downstream wiring:

* ``--live`` -- :func:`ours.app.build_live_frontend` (real OAK-D). Hardware only.
* ``--session PATH`` (default) -- :func:`ours.app.build_replay_frontend` over a
  recorded session, so the whole 4-process stack runs without a device on CI.

Run::

    python -m ours.proc.capture --session sessions/gold/lab_loop_30s
    python -m ours.proc.capture --live --width 640 --height 400 --fps 20
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.flow import Bus, topics                              # noqa: E402
from ours.lib.io.reader import SessionReader                       # noqa: E402
from ours.lib.ipc import IpcServerBus                              # noqa: E402
from ours.lib.ipc.messages import WireCalibBundle                  # noqa: E402
from ours.flows.bridge import (                                    # noqa: E402
    IpcPublisherFlow, RingRegistry,
)
from ours.flows.bridge.ring_registry import default_capture_specs  # noqa: E402

LOG = logging.getLogger("ours.proc.capture")

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
    from ours.app import _replay_imu_startup
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
    )


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
    )


# --------------------------------------------------------------------------- #
def run_capture_replay(session: Path, endpoint: str, *,
                       width: int, height: int,
                       max_frames: int = 0,
                       depth_fast: bool = True) -> int:
    """Replay-driven capture: SessionReader -> bridge -> IPC."""
    from ours.app import build_replay_frontend
    from ours.lib.imu.imu_calib import ImuCalibration

    reader = SessionReader(session)
    # Use the session's native resolution so the rings line up with the frames.
    width, height = int(reader.calib.left.width), int(reader.calib.left.height)

    rings = RingRegistry().create_all(default_capture_specs(
        endpoint=endpoint, width=width, height=height))
    # Live mode uses non-blocking publish (drop-oldest on stall so the OAK-D
    # firmware watchdog never fires). Replay mode below uses blocking=True so
    # every replayed frame reaches VIO.
    server = IpcServerBus(endpoint, retain_topics={_CALIB_TOPIC},
                          blocking=False)
    local = Bus()

    # Build the publisher BEFORE the front-end so subscribers connecting at any
    # time are wired in (the publisher subscribes to the local bus eagerly).
    pub_flow = IpcPublisherFlow(local, server, rings, _DATA_TOPICS,
                                endpoint=endpoint)
    pub_flow.start()

    # Replay startup IMU references for the imu_cam flow's calibration.
    from ours.app import _replay_imu_startup
    _, _, gyro_bias = _replay_imu_startup(reader, use_gyro=True)
    calibration = (ImuCalibration(gyro_bias=gyro_bias)
                   if gyro_bias is not None else None)

    cam_flow, imu_flow = build_replay_frontend(
        bus=local, reader=reader, depth_fast=depth_fast,
        max_frames=int(max_frames), calibration=calibration)

    # Broadcast the retained calibration bundle BEFORE starting the front-end
    # so any subscriber that connects mid-run gets the cached one immediately.
    bundle = _build_calib_bundle_replay(reader)
    server.publish(_CALIB_TOPIC, bundle)
    LOG.info("capture[%s] replay session=%s frames=%d %dx%d",
             endpoint, session, len(reader), width, height)

    # Install SIGTERM handler BEFORE starting the flows so the launcher's
    # SIGTERM is observed even if the producer is mid-frame. Without this the
    # `cam_flow.join()` below blocks until the source is fully drained -- a
    # 30-second replay session would block shutdown for ~30 s and the launcher
    # would SIGKILL the process at the 10 s deadline, leaking every ring slot.
    stop = [False]
    def _on_sigterm(_signo, _frame):
        stop[0] = True
        # cam_flow is a SourceFlow; setting its _stop flag breaks out of the
        # produce() loop at the next item boundary so the join() returns.
        cam_flow.stop()
    signal.signal(signal.SIGTERM, _on_sigterm)

    imu_flow.start()
    cam_flow.start()
    LOG.info("capture: cam_flow + imu_flow started, waiting for join ...")
    try:
        # Poll instead of pure join() so KeyboardInterrupt + signal handlers
        # are delivered promptly. The poll cadence (0.2 s) matches the live
        # path; cam_flow.is_alive() flips False after produce() returns or
        # _stop is observed inside the loop.
        while not stop[0] and cam_flow.is_alive():
            time.sleep(0.2)
        LOG.info("capture: cam_flow loop exit (stop=%s, alive=%s)",
                 stop[0], cam_flow.is_alive())
    except KeyboardInterrupt:
        LOG.info("capture: SIGINT -> stopping")
        stop[0] = True
    finally:
        # CamFlow (SourceFlow) emits END on CAM_SYNC when produce() returns;
        # ImuCamFlow forwards END from CAM_SYNC to IMU_RAW + IMUCAM_SAMPLE +
        # FRAME_DEPTH (see `imu_cam_flow.forwards_to`). The publisher bridge
        # converts those local-bus ENDs to WireEnd and sends them on the IPC
        # server.
        #
        # CRITICAL: wait for ImuCamFlow's drain to chew through every queued
        # CAM_SYNC + the END BEFORE calling imu_flow.stop(). Flow.stop sets
        # `_stop`, which the drain checks at the TOP of every loop iteration --
        # so if we stop while CAM_SYNC items are still queued, we discard them
        # AND the END. `done` is set inside `_handle_end` after expected_ends
        # have been processed, which only happens when END is drained.
        #
        # Under SIGTERM the operator wants a fast exit, NOT a full drain --
        # END will never arrive from a half-killed producer, so cap the wait
        # at 2 s. Natural end-of-replay keeps the generous 120 s ceiling so a
        # busy backend can finish.
        cam_flow.stop()
        drain_timeout = 2.0 if stop[0] else 120.0
        LOG.info("capture: waiting for imu_flow to drain (timeout=%.1fs) ...",
                 drain_timeout)
        ok = imu_flow.done.wait(timeout=drain_timeout)
        LOG.info("capture: imu_flow.done=%s", ok)
        imu_flow.stop()
        # Give the bridge a brief window to flush the buffered WireEnds onto
        # the socket before we tear down the server.
        time.sleep(0.3)
        pub_flow.stop()
        server.close()
        rings.unlink()
        rings.close()
        LOG.info("capture: shutdown complete")
    return 0


def run_capture_live(endpoint: str, *,
                     width: int, height: int, fps: int,
                     depth_fast: bool = True,
                     use_gyro: bool = True,
                     recalibrate_bias: bool = False) -> int:
    """Live OAK-D capture: device -> bridge -> IPC."""
    from ours.app import build_live_frontend

    rings = RingRegistry().create_all(default_capture_specs(
        endpoint=endpoint, width=width, height=height))
    # Live: the IPC server is non-blocking (drop-oldest on stall) so a slow
    # downstream subscriber never stalls the OAK-D producer (firmware watchdog
    # ~1.5 s -- see ours.ui.live_source). The local Bus front-end stays FIFO
    # (`latest_only=False`): coalescing imucam.sample / frame.depth here would
    # drop frames BEFORE the bridge, breaking VIO (the gyro continuity required
    # by PreintegratePrior and the KLT continuity required by TrackFeatures).
    # Backpressure belongs at the IPC boundary, not at the VIO inputs --
    # ARCHITECTURE.md section 3 ("VIO + deterministic replay require FIFO").
    server = IpcServerBus(endpoint, retain_topics={_CALIB_TOPIC},
                          blocking=False)
    local = Bus()
    pub_flow = IpcPublisherFlow(local, server, rings, _DATA_TOPICS,
                                endpoint=endpoint)
    pub_flow.start()

    try:
        device, cam_flow, imu_flow, cal = build_live_frontend(
            bus=local, width=width, height=height, fps=fps,
            use_gyro=use_gyro, depth_fast=depth_fast,
            recalibrate_bias=recalibrate_bias, latest_only=False)
    except Exception as e:                                         # noqa: BLE001
        LOG.error("capture: live build failed: %s", e)
        pub_flow.stop()
        server.close()
        rings.unlink()
        rings.close()
        return 1

    bundle = _build_calib_bundle_live(cal)
    server.publish(_CALIB_TOPIC, bundle)
    LOG.info("capture[%s] live %dx%d@%d depth_fast=%s",
             endpoint, width, height, fps, depth_fast)

    stop = [False]
    def _on_sigterm(_signo, _frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, _on_sigterm)

    imu_flow.start()
    cam_flow.start()
    try:
        while not stop[0] and cam_flow.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        LOG.info("capture: SIGINT -> stopping")
    finally:
        # Close the device FIRST: the OAK-D firmware watchdog is only ~1.5s and
        # tearing down the bridge / IPC before release() risks tripping it (the
        # same lesson as `ours.ui.live_source._run`).
        cam_flow.stop()
        imu_flow.stop()
        try:
            device.release()
        except Exception:                                          # noqa: BLE001
            pass
        # Same as the replay path: the front-end flows already forward END
        # via _emit_end / forwards_to on disconnect; the bridge mirrors them.
        time.sleep(0.3)
        pub_flow.stop()
        server.close()
        rings.unlink()
        rings.close()
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Surface uncaught thread exceptions (otherwise a crashed flow drain
    # silently leaves the process alive with no published data).
    def _excepthook(args):
        LOG.error("THREAD CRASH in %s: %s: %s", args.thread.name,
                  args.exc_type.__name__, args.exc_value, exc_info=(
                      args.exc_type, args.exc_value, args.exc_traceback))
    threading.excepthook = _excepthook
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help=f"IpcBus endpoint name (default: {DEFAULT_ENDPOINT!r})")
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
    args = ap.parse_args()

    if args.live:
        return run_capture_live(
            endpoint=args.endpoint, width=args.width, height=args.height,
            fps=args.fps, depth_fast=args.depth_fast,
            use_gyro=not args.no_gyro,
            recalibrate_bias=args.recalibrate_bias)
    return run_capture_replay(
        session=Path(args.session), endpoint=args.endpoint,
        width=args.width, height=args.height,
        max_frames=args.max_frames, depth_fast=args.depth_fast)


if __name__ == "__main__":
    # Same os._exit pattern as ours.proc.vio / ours.proc.slam / ours.proc.ui
    # -- prevent any lingering non-daemon thread (depthai background thread,
    # numba pool, etc.) from holding the process past the launcher's 10 s
    # SIGTERM deadline.
    import os as _os
    _rc = main()
    LOG.info("capture: main returned, calling os._exit(%d)", int(_rc))
    logging.shutdown()
    _os.sys.stdout.flush()
    _os.sys.stderr.flush()
    _os._exit(int(_rc))
