"""slam process: subscribe to VIO's keyframes, run loop closure + pose graph.

Subscribes (over IPC) to the ``vio`` endpoint for ``keyframe`` and the retained
``calib.bundle`` (intrinsics only); runs the same
:class:`~slam.modules.pipeline.SlamModule` the pre-split in-process graph ran;
then republishes ``loop.correction`` + ``slam.map`` on its own
:class:`~slam.comms.IPCPubSub` endpoint ``"oak.slam"`` for the UI / tools.

This process owns the SLAM map (ORB feature index, pose-graph). The VIO map
lives in the VIO process (windowed BA); the two maps are independent by design
-- they consume different things and serve different views.

Calibration handshake
---------------------
Same as VIO -- a dedicated calib client blocks until the retained
``calib.bundle`` arrives on the VIO endpoint. VIO republishes the same calib it
got from capture AFTER allocating its kf_* rings, so receiving it here proves
(a) VIO is up, (b) intrinsics are known, and (c) the kf_gray / kf_depth rings we
need to attach to already exist. (We deliberately don't subscribe to capture at
all -- SLAM is a pure consumer of VIO's output.)

SLAM still only PUBLISHES ``loop.correction`` here (it has no reverse channel and
does not subscribe to VIO's pose). The CLOSED-LOOP feedback is wired on the OTHER
side: when VIO runs with ``--tight`` it opens its own read-only client on THIS
endpoint and subscribes to ``loop.correction``, feeding the pose-graph correction
back into its live pose so accumulated drift is bounded on revisits (see
``vio.main`` / ``vio/modules/propagate_imu.py``). SLAM itself is unchanged from the
pre-split ``ours.proc.slam`` -- it just exposes the correction; VIO consumes it.

Run::

    python -m slam.main
    python -m slam.main --vio-endpoint oak.vio.test --endpoint oak.slam.test
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

from slam.comms import (                                            # noqa: E402
    IPCPublisher, IPCSubscriber, IPCPubSub, LocalPubSub, RingRegistry, topics,
)
from slam.comms.messages import END                                # noqa: E402
from slam.comms.wire import WireCalibBundle                        # noqa: E402
from slam.comms.ring_registry import default_vio_specs             # noqa: E402
from slam.modules import SlamModule                                # noqa: E402
from slam.mathlib.loop.slam import SlamConfig                      # noqa: E402

LOG = logging.getLogger("slam.main")

DEFAULT_VIO_ENDPOINT = "oak.vio"
DEFAULT_SLAM_ENDPOINT = "oak.slam"

#: Topic SLAM subscribes to from VIO.
_INPUT_TOPICS = [topics.KEYFRAME]
#: Topics SLAM republishes. SLAM_MAP is the continuous keyframe-map overlay and
#: SLAM_LOOP the per-candidate loop-match funnel for the UI's loop-closure view
#: (both pure POD, no shared-memory ring, LIVE-only); the publisher forwards them
#: alongside the loop-event loop.correction.
_OUTPUT_TOPICS = [topics.LOOP_CORRECTION, topics.SLAM_MAP, topics.SLAM_LOOP]


# --------------------------------------------------------------------------- #
def _await_calib_bundle(endpoint: str, timeout_s: float) -> WireCalibBundle:
    """Open a dedicated client, block until the retained calib bundle arrives."""
    bundle: list[WireCalibBundle | None] = [None]
    got = threading.Event()

    def on_calib(wm: WireCalibBundle) -> None:
        bundle[0] = wm
        got.set()

    client = IPCPubSub(endpoint, role="client", connect_timeout_s=timeout_s)
    client.subscribe("calib.bundle", on_calib)
    client.start()
    try:
        if not got.wait(timeout=timeout_s):
            raise TimeoutError(
                f"slam: no calib.bundle from {endpoint!r} in {timeout_s}s")
    finally:
        client.stop()
    assert bundle[0] is not None
    return bundle[0]


# --------------------------------------------------------------------------- #
def run_slam(*,
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

    # 3. Build local bus + the SLAM module (loop closure + pose graph).
    # latest_only=True: this is the LIVE viewer's SLAM, not the deterministic
    # scoring path. The ORB + pose-graph solve grows with the map, so a strict
    # FIFO inbox would back up without bound and the `slam.map` overlay would lag
    # further and further behind real time. A coalescing inbox drops the backlog
    # and always solves the FRESHEST keyframe, so the map stays current (it skips
    # intermediate keyframes only when overloaded). END is never coalesced, so
    # clean shutdown still propagates.
    local = LocalPubSub()
    # Motion-gated keyframe insertion (live-only; the offline SlamModule default
    # stays 0/0 = insert every keyframe). A new keyframe joins the pose graph
    # only when the camera moved >= 10 cm OR rotated >= 5 deg since the LAST
    # INSERTED keyframe, so a stationary / slowly-panning camera stops piling up
    # near-identical redundant keyframes (the main driver of unbounded memory and
    # the O(N^3) PGO cost on long live sessions). The odometry edge is still
    # taken between consecutive INSERTED keyframes, so the chain stays exact.
    slam = SlamModule(local, bundle.K,
                      SlamConfig(loop_max_odom_rot_deg=30.0,
                                 kf_min_trans_m=0.1, kf_min_rot_deg=5.0),
                      latest_only=True, worker=worker, publish_map=True)

    # 4. Open output IPCPubSub server + publisher for the loop corrections.
    #    Retain `calib.bundle` and re-broadcast VIO's bundle so consumers that
    #    talk to *this* endpoint (UI, smoke selftest) can use the calib arrival
    #    as a readiness barrier.
    server = IPCPubSub(endpoint, role="server", retain_topics={"calib.bundle"})
    pub = IPCPublisher(local, server, vio_rings, _OUTPUT_TOPICS,
                       endpoint=endpoint)
    pub.start()
    server.publish("calib.bundle", bundle)

    # 5. Open input IPCPubSub client + subscriber bridge: VIO keyframes -> local.
    in_client = IPCPubSub(vio_endpoint, role="client")
    in_bridge = IPCSubscriber(local, in_client, vio_rings, _INPUT_TOPICS)

    # 6. END-watch: capture's END propagates through VIO to here.
    finished = threading.Event()

    def _end_watch(_msg) -> None:
        if _msg is END:
            finished.set()
    for t in _INPUT_TOPICS:
        local.subscribe(t, _end_watch)

    LOG.info("slam[%s] subscribing to %s for keyframes", endpoint, vio_endpoint)

    # 7. Start everything (slam first so its subscriptions are wired before the
    #    bridge starts pushing messages onto the local bus).
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
        # let Module.stop() force-kill the drain thread, otherwise the launcher
        # SIGKILLs us at its 10 s deadline (no SHM rings here, but a clean
        # shutdown still keeps the launcher logs free of SIGKILL noise).
        drain_timeout = 2.0 if stop[0] else 120.0
        slam.done.wait(timeout=drain_timeout)
        slam.stop()
        # SlamModule forwards END to loop.correction via its `_emit_end`; the
        # publisher bridge mirrors that onto IPC. No explicit publish_end.
        time.sleep(0.3)
        pub.stop()
        server.close()
        vio_rings.close()
        LOG.info("slam: shutdown complete")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vio-endpoint", default=DEFAULT_VIO_ENDPOINT,
                    help=f"VIO IPC endpoint (default: {DEFAULT_VIO_ENDPOINT!r})")
    ap.add_argument("--endpoint", default=DEFAULT_SLAM_ENDPOINT,
                    help=f"this process's IPC endpoint (default: {DEFAULT_SLAM_ENDPOINT!r})")
    ap.add_argument("--worker", action="store_true",
                    help="run pose-graph solve in a child process (release GIL)")
    ap.add_argument("--calib-timeout", type=float, default=30.0,
                    help="seconds to wait for the calib.bundle on boot")
    args = ap.parse_args()

    return run_slam(
        vio_endpoint=args.vio_endpoint,
        endpoint=args.endpoint,
        worker=args.worker,
        calib_timeout_s=args.calib_timeout,
    )


if __name__ == "__main__":
    # Use os._exit (not SystemExit / return-from-main) so a lingering non-daemon
    # thread -- IPCSubscriber's recv loop, the InProcessEngine worker, etc --
    # cannot keep the process alive past `slam: shutdown complete`. Without this
    # the launcher waits its full 10 s deadline and SIGKILLs us. Mirrors the same
    # pattern in `imu_camera.main` / `vio.main`.
    import os as _os
    _rc = main()
    LOG.info("slam: main returned, calling os._exit(%d)", int(_rc))
    logging.shutdown()
    _os.sys.stdout.flush()
    _os.sys.stderr.flush()
    _os._exit(int(_rc))
