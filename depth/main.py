"""depth process: subscribe to raw stereo, run SGM, publish metric depth.

This is the STANDALONE depth-as-a-process harness. In the live topology depth
runs INLINE on the capture process's ``imu_cam`` thread (both share the one
canonical SGM matcher in :mod:`sky.depth.stereo`), so the launcher never spawns
this process. ``depth.main`` exists to prove the depth source tree runs as its
OWN independent project -- it is the promotable "depth as its own process".

This shell is deliberately PROCEDURAL: there is no reactive ``Module`` / ``Step``
graph. The raw ``cam.sync`` stereo arrives over IPC on the
:class:`~depth.comms.IPCSubscriber` recv thread, which is single-threaded and
strictly FIFO; the subscribed callback (:func:`_on_cam_sync`) runs the depth flow
straight through -- ``compute_depth(matcher, msg) -> publish_depth(bus, frame)``
-- one published ``frame.depth`` per consumed ``cam.sync``, in order. END from
``cam.sync`` is forwarded onto ``frame.depth`` by a second callback. The compute
runs ON the recv thread on purpose: it back-pressures the socket (the next
``recv_bytes`` only happens after this frame's depth is published), so a slow SGM
never drops a frame -- the FIFO 1:1 contract the harness proves.

Data flow::

    capture (oak.capture)                 depth (oak.depth)
    ---------------------                 -----------------
    cam.sync  (raw L/R) ----IPC--->  IPCSubscriber -> LocalPubSub
    imucam.sample / imu.raw ...           |  cam.sync
    calib.bundle (retained) --IPC-->      v
                                     _on_cam_sync: compute_depth -> publish_depth
                                          |  frame.depth (rectified-left + depth_m)
                                          v
                                     IPCPublisher -> IPCPubSub server (oak.depth)
                                     calib.bundle re-broadcast (retained)

The matcher's rectifiers need the FULL per-camera stereo calibration
(:class:`~depth.io.reader.StereoCalib`: ``K_left``/``K_right``/``dist`` +
``T_left_right``). That calibration is NOT carried on the wire
``calib.bundle`` (which broadcasts only the rectified-left intrinsic + the IMU
extrinsics, all VIO/SLAM need). So -- exactly as the capture project builds its
matcher from ``reader.calib`` / ``cal.calib`` -- this harness builds the matcher
from the recorded session's calibration (``--session``); the wire bundle is used
as the readiness barrier + to carry frame sizing, and is re-broadcast so any
``frame.depth`` consumer that connects to *this* endpoint gets it.

Run (replay pair; capture must already serve ``cam.sync`` on the capture
endpoint, e.g. ``python -m imu_camera.main --session sessions/gold/lab_loop_30s
--endpoint oak.capture``)::

    python -m depth.main --capture-endpoint oak.capture --endpoint oak.depth \\
        --session sessions/gold/lab_loop_30s --max-frames 20
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth.comms import (                                          # noqa: E402
    IPCPublisher, IPCPubSub, IPCSubscriber, LocalPubSub,
    RingRegistry, topics,
)
from depth.comms.messages import END                               # noqa: E402
from depth.comms.ring_registry import default_capture_specs        # noqa: E402
from depth.comms.wire import WireCalibBundle                       # noqa: E402
from depth.io.reader import SessionReader                          # noqa: E402
from sky.depth.stereo import SGMConfig, SGMStereoMatcher  # noqa: E402
from depth.modules import compute_depth, publish_depth             # noqa: E402

LOG = logging.getLogger("depth.main")

#: Canonical endpoints. depth subscribes to the capture endpoint for the raw
#: stereo + the retained calib, and serves its computed depth on its own one.
DEFAULT_CAPTURE_ENDPOINT = "oak.capture"
DEFAULT_DEPTH_ENDPOINT = "oak.depth"

#: The single raw-stereo topic depth consumes from capture. (The IMU streams
#: travel on imucam.sample / imu.raw + the retained calib.bundle alongside, but
#: computing depth needs only the synced stereo pair.)
_INPUT_TOPIC = topics.CAM_SYNC

#: The depth output. Re-broadcast the retained calib.bundle so a frame.depth
#: consumer connecting to THIS endpoint boots with the bundle already cached.
_OUTPUT_TOPIC = topics.FRAME_DEPTH
_CALIB_TOPIC = topics.CALIB_BUNDLE


# --------------------------------------------------------------------------- #
def _await_calib_bundle(endpoint: str, timeout_s: float) -> WireCalibBundle:
    """Open a dedicated client, block until the retained calib bundle arrives.

    Mirrors ``vio.main._await_calib_bundle``: a single IPCPubSub connection
    cannot mix the "wait for retained calib" phase with the "subscribe to data"
    phase, so the calib client is its own short-lived connection.
    """
    bundle: list[WireCalibBundle | None] = [None]
    got = threading.Event()

    def on_calib(wm: WireCalibBundle) -> None:
        bundle[0] = wm
        got.set()

    client = IPCPubSub(endpoint, role="client", connect_timeout_s=timeout_s)
    client.subscribe(_CALIB_TOPIC, on_calib)
    client.start()
    try:
        if not got.wait(timeout=timeout_s):
            raise TimeoutError(
                f"depth: no calib.bundle from {endpoint!r} in {timeout_s}s")
    finally:
        client.stop()
    assert bundle[0] is not None
    return bundle[0]


def _build_matcher(session: Path, depth_fast: bool) -> SGMStereoMatcher:
    """Build the SGM matcher from the recorded session's full stereo calibration.

    The capture project builds the SAME matcher from ``reader.calib`` (replay) /
    ``cal.calib`` (live) -- this harness reuses the replay construction because
    the per-camera calibration the rectifiers need is not on the wire bundle.
    ``rectify_left=False`` matches the replay path: the gold session's left is the
    chip's already-rectified left, only the raw right is rectified internally.
    """
    reader = SessionReader(session)
    sgm = SGMConfig.live() if depth_fast else SGMConfig()
    return SGMStereoMatcher.from_calib(reader.calib, sgm)


# --------------------------------------------------------------------------- #
def run_depth(*,
              capture_endpoint: str = DEFAULT_CAPTURE_ENDPOINT,
              endpoint: str = DEFAULT_DEPTH_ENDPOINT,
              session: Path,
              max_frames: int = 0,
              depth_fast: bool = True,
              calib_timeout_s: float = 30.0) -> int:
    """Run the standalone depth process until END / SIGTERM / Ctrl-C."""
    # 1. Block until the capture process publishes its (retained) calib bundle.
    #    This doubles as a readiness barrier: capture is up + its cam.sync rings
    #    already exist for us to attach below.
    LOG.info("depth: waiting for calib.bundle on %s ...", capture_endpoint)
    bundle = _await_calib_bundle(capture_endpoint, calib_timeout_s)
    width, height = int(bundle.width), int(bundle.height)
    LOG.info("depth: got calib %dx%d", width, height)

    # 2. Build the SGM matcher from the session's full stereo calibration. (The
    #    wire bundle carries only the rectified-left K, not the per-camera
    #    calibration the rectifiers need -- see module docstring.)
    matcher = _build_matcher(session, depth_fast)
    if (int(matcher.K.shape[0]), int(matcher.K.shape[1])) != (3, 3):
        LOG.error("depth: matcher K is not 3x3")
        return 1

    # 3. Attach the capture-side rings (consumer-attach) so the subscriber bridge
    #    can read cam.sync's raw left/right out of capture's shared memory.
    cap_rings = RingRegistry().attach_all(default_capture_specs(
        endpoint=capture_endpoint, width=width, height=height))

    # 4. Allocate depth's OWN rings for the frame.depth stream we publish. The
    #    DepthFrame converter writes the rectified-left into "<endpoint>.gray_left"
    #    and the metric depth into "<endpoint>.depth_m"; default_capture_specs
    #    provisions both (its gray_right ring is unused here, harmlessly).
    depth_rings = RingRegistry().create_all(default_capture_specs(
        endpoint=endpoint, width=width, height=height))

    # 5. Local bus -- pure plumbing between the two IPC bridges. The depth flow
    #    itself runs in the cam.sync callback below, NOT a reactive module.
    local = LocalPubSub()

    # 6. Output IPCPubSub server + publisher bridge. Non-blocking (drop-oldest on
    #    stall) so a slow frame.depth consumer never stalls the SGM thread.
    #    Retain calib.bundle and re-broadcast capture's bundle so a consumer
    #    connecting to THIS endpoint uses the calib arrival as a readiness barrier
    #    (proves depth is up AND depth's rings already exist).
    server = IPCPubSub(endpoint, role="server",
                       retain_topics={_CALIB_TOPIC}, blocking=False)
    pub = IPCPublisher(local, server, depth_rings, [_OUTPUT_TOPIC],
                       endpoint=endpoint, ring_endpoint=endpoint)
    pub.start()
    # Re-broadcast AFTER pub.start() opened the server socket so the retained slot
    # is cached for any late subscriber.
    server.publish(_CALIB_TOPIC, bundle)

    # 7. Procedural depth flow + END/shutdown bookkeeping. Both run on the
    #    IPCSubscriber recv thread (single-threaded, strictly FIFO), so the
    #    per-frame compute->publish is ordered AND END is forwarded only after the
    #    last data frame's depth has been published (it follows in the FIFO).
    finished = threading.Event()    # set when capture's END drains through us
    published = [0]                 # frame.depth count emitted (harness report)

    def _on_cam_sync(msg) -> None:
        """One raw stereo pair -> exactly one published frame.depth."""
        if msg is END:
            # END-forwarding: capture publishes WireEnd on cam.sync when the
            # replay session finishes; mirror it onto frame.depth so a downstream
            # consumer drains cleanly, then signal the run is done. (The Module
            # framework did this via forwards_to + _handle_end; here it is one
            # explicit branch.)
            local.publish(_OUTPUT_TOPIC, END)
            finished.set()
            return
        frame = compute_depth(matcher, msg)
        publish_depth(local, frame)
        published[0] += 1

    local.subscribe(_INPUT_TOPIC, _on_cam_sync)

    # 8. Input IPCPubSub client + subscriber bridge: capture's cam.sync -> local
    #    bus -> _on_cam_sync. Subscribe the flow BEFORE the bridge starts so no
    #    cam.sync published during boot is lost.
    in_client = IPCPubSub(capture_endpoint, role="client")
    in_bridge = IPCSubscriber(local, in_client, cap_rings, [_INPUT_TOPIC])

    LOG.info("depth[%s] subscribing to %s for %s (max_frames=%d)",
             endpoint, capture_endpoint, _INPUT_TOPIC, max_frames)

    in_bridge.start()

    stop = [False]

    def _on_sigterm(_signo, _frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        # Run until: (a) capture sends END on cam.sync -> finished, (b) the
        # max-frames cap is reached (harness), or (c) the operator interrupts.
        while not stop[0] and not finished.is_set():
            if max_frames > 0 and published[0] >= max_frames:
                LOG.info("depth: reached max_frames=%d", max_frames)
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        LOG.info("depth: SIGINT -> stopping")
    finally:
        # Drain order: stop the input bridge so no more cam.sync arrives (and so
        # the recv thread is no longer running _on_cam_sync), then tear down the
        # output side. Because compute->publish ran inline on the recv thread,
        # once the bridge has stopped every frame it delivered is already
        # published -- there is no separate worker inbox left to drain.
        in_bridge.stop()
        LOG.info("depth: input bridge stopped, finished=%s published=%d",
                 finished.is_set(), published[0])
        # Give the publisher bridge a brief window to flush buffered wire frames
        # (incl. the END forwarded onto frame.depth) before tearing down the
        # server.
        time.sleep(0.3)
        pub.stop()
        server.close()
        cap_rings.close()
        depth_rings.unlink()
        depth_rings.close()
        LOG.info("depth: shutdown complete (published %d frame.depth)",
                 published[0])
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Surface uncaught thread exceptions (otherwise a crashed callback on the IPC
    # recv thread is swallowed by the recv loop's try/except and the process
    # lingers with no published data).
    def _excepthook(args):
        LOG.error("THREAD CRASH in %s: %s: %s", args.thread.name,
                  args.exc_type.__name__, args.exc_value, exc_info=(
                      args.exc_type, args.exc_value, args.exc_traceback))
    threading.excepthook = _excepthook

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-endpoint", default=DEFAULT_CAPTURE_ENDPOINT,
                    help=f"capture IPC endpoint to subscribe to "
                         f"(default: {DEFAULT_CAPTURE_ENDPOINT!r})")
    ap.add_argument("--endpoint", default=DEFAULT_DEPTH_ENDPOINT,
                    help=f"this process's IPC endpoint "
                         f"(default: {DEFAULT_DEPTH_ENDPOINT!r})")
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s",
                    help="recorded session whose calib.json builds the matcher's "
                         "rectifiers (the wire bundle lacks per-camera calib)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after publishing this many frame.depth (0 = all)")
    ap.add_argument("--depth-fast", action="store_true", default=True,
                    help="half-res SGM preset (faster)")
    ap.add_argument("--calib-timeout", type=float, default=30.0,
                    help="seconds to wait for the calib.bundle on boot")
    args = ap.parse_args()

    return run_depth(
        capture_endpoint=args.capture_endpoint,
        endpoint=args.endpoint,
        session=Path(args.session),
        max_frames=args.max_frames,
        depth_fast=args.depth_fast,
        calib_timeout_s=args.calib_timeout,
    )


if __name__ == "__main__":
    # Same os._exit pattern as the other split process mains -- prevent any
    # lingering non-daemon thread (IPCSubscriber recv loop, numba thread pool)
    # from holding the process past a shutdown deadline.
    import os as _os
    _rc = main()
    LOG.info("depth: main returned, calling os._exit(%d)", int(_rc))
    logging.shutdown()
    _os.sys.stdout.flush()
    _os.sys.stderr.flush()
    _os._exit(int(_rc))
