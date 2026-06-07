"""slam process: subscribe to VIO's keyframes, run loop closure + pose graph.

Subscribes (over IPC) to the ``vio`` endpoint for ``keyframe`` (and to capture
for ``calib.bundle`` -- intrinsics only); runs the same ``SlamFlow`` the
in-process pipeline runs; republishes ``loop.correction`` on its own IPC
endpoint ``"oak.slam"`` for the UI.

This process owns the SLAM map (ORB feature index, pose-graph). The VIO map
lives in the VIO process (windowed BA); the two maps are independent by design
-- they consume different things and serve different views.

Calibration handshake: same as VIO -- a dedicated calib client blocks until the
retained ``calib.bundle`` arrives, then we build the local graph.

Run::

    python -m ours.proc.slam
    python -m ours.proc.slam --vio-endpoint oak.vio.test --capture-endpoint oak.capture.test
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.flow import Bus, topics                              # noqa: E402
from ours.lib.flow.messages import END                             # noqa: E402
from ours.lib.ipc import IpcClientBus, IpcServerBus                # noqa: E402
from ours.flows.bridge import (                                    # noqa: E402
    IpcPublisherFlow, IpcSubscriberFlow, RingRegistry,
)
from ours.flows.bridge.ring_registry import default_vio_specs      # noqa: E402
from ours.flows.slam import SlamFlow                               # noqa: E402
from ours.lib.loop.slam import SlamConfig                          # noqa: E402

from .vio import _await_calib_bundle                               # noqa: E402

LOG = logging.getLogger("ours.proc.slam")

DEFAULT_CAPTURE_ENDPOINT = "oak.capture"
DEFAULT_VIO_ENDPOINT = "oak.vio"
DEFAULT_SLAM_ENDPOINT = "oak.slam"

_INPUT_TOPICS = [topics.KEYFRAME]
_OUTPUT_TOPICS = [topics.LOOP_CORRECTION]


# --------------------------------------------------------------------------- #
def run_slam(*,
             capture_endpoint: str = DEFAULT_CAPTURE_ENDPOINT,
             vio_endpoint: str = DEFAULT_VIO_ENDPOINT,
             endpoint: str = DEFAULT_SLAM_ENDPOINT,
             worker: bool = False,
             calib_timeout_s: float = 30.0) -> int:
    """Run the SLAM process until END / SIGTERM / Ctrl-C."""
    # 1. Block until VIO's retained calib bundle arrives. VIO republishes the
    #    same calib it got from capture AFTER allocating its kf_* rings, so
    #    receiving it here proves (a) VIO is up, (b) intrinsics are known, and
    #    (c) the kf_gray / kf_depth rings we need to attach to already exist.
    #    (We deliberately don't subscribe to capture at all -- SLAM is a pure
    #    consumer of VIO's output.)
    LOG.info("slam: waiting for calib.bundle on %s ...", vio_endpoint)
    bundle = _await_calib_bundle(vio_endpoint, calib_timeout_s)
    width, height = int(bundle.width), int(bundle.height)
    LOG.info("slam: got calib %dx%d", width, height)

    # 2. Attach to VIO's keyframe rings (SLAM is a consumer of VIO's output).
    vio_rings = RingRegistry().attach_all(default_vio_specs(
        endpoint=vio_endpoint, width=width, height=height))

    # 3. Build local bus + the SLAM flow (loop closure + pose graph).
    local = Bus()
    slam = SlamFlow(local, bundle.K, SlamConfig(loop_max_odom_rot_deg=30.0),
                    latest_only=False, worker=worker)

    # 4. Open output IpcServerBus + publisher for the loop corrections.
    #    Retain `calib.bundle` and re-broadcast capture's bundle so consumers
    #    that talk to *this* endpoint (UI, smoke selftest) can use the calib
    #    arrival as a readiness barrier.
    server = IpcServerBus(endpoint, retain_topics={"calib.bundle"})
    pub_flow = IpcPublisherFlow(local, server, vio_rings, _OUTPUT_TOPICS,
                                endpoint=endpoint)
    pub_flow.start()
    server.publish("calib.bundle", bundle)

    # 5. Open input IpcClientBus + subscriber bridge: VIO keyframes -> local bus.
    in_client = IpcClientBus(vio_endpoint)
    in_bridge = IpcSubscriberFlow(local, in_client, vio_rings, _INPUT_TOPICS)

    # 6. END-watch: capture's END propagates through VIO to here.
    finished = threading.Event()
    def _end_watch(_msg) -> None:
        if _msg is END:
            finished.set()
    for t in _INPUT_TOPICS:
        local.subscribe(t, _end_watch)

    LOG.info("slam[%s] subscribing to %s for keyframes", endpoint, vio_endpoint)

    # 7. Start everything (slam first so its subscriptions are wired before
    #    the bridge starts pushing messages onto the local bus).
    slam.start()
    in_bridge.start()

    stop = [False]
    def _on_sigterm(_signo, _frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        while not stop[0] and not finished.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        LOG.info("slam: SIGINT -> stopping")
    finally:
        in_bridge.stop()
        # Same generous drain window as VIO -- a full inbox of buffered
        # keyframes can take seconds to chew through (loop closure is heavy).
        # Under SIGTERM the operator wants a fast exit and VIO is also
        # shutting down so END will never arrive -- cap the wait at 2 s and
        # let Flow.stop() force-kill the drain thread, otherwise the launcher
        # SIGKILLs us at its 10 s deadline (no SHM rings here, but a clean
        # shutdown still keeps the launcher logs free of SIGKILL noise).
        drain_timeout = 2.0 if stop[0] else 120.0
        slam.done.wait(timeout=drain_timeout)
        slam.stop()
        # SlamFlow forwards END to loop.correction via its `_emit_end`; the
        # publisher bridge mirrors that onto IPC. No explicit publish_end.
        time.sleep(0.3)
        pub_flow.stop()
        server.close()
        vio_rings.close()
        LOG.info("slam: shutdown complete")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-endpoint", default=DEFAULT_CAPTURE_ENDPOINT)
    ap.add_argument("--vio-endpoint", default=DEFAULT_VIO_ENDPOINT)
    ap.add_argument("--endpoint", default=DEFAULT_SLAM_ENDPOINT)
    ap.add_argument("--worker", action="store_true",
                    help="run pose-graph solve in a child process (release GIL)")
    ap.add_argument("--calib-timeout", type=float, default=30.0)
    args = ap.parse_args()

    return run_slam(
        capture_endpoint=args.capture_endpoint,
        vio_endpoint=args.vio_endpoint,
        endpoint=args.endpoint,
        worker=args.worker,
        calib_timeout_s=args.calib_timeout,
    )


if __name__ == "__main__":
    # Same os._exit pattern as ours.proc.vio / ours.proc.ui -- prevent any
    # lingering non-daemon thread from holding the process past the launcher's
    # 10 s SIGTERM deadline.
    import os as _os
    _rc = main()
    LOG.info("slam: main returned, calling os._exit(%d)", int(_rc))
    logging.shutdown()
    _os.sys.stdout.flush()
    _os.sys.stderr.flush()
    _os._exit(int(_rc))
