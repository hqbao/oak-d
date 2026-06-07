"""vio process: subscribe to capture, run odometry + windowed BA, republish.

Subscribes (over IPC) to the ``capture`` endpoint for ``calib.bundle``,
``imucam.sample`` and ``frame.depth``; runs the same ``OdometryFlow`` +
``BackendFlow`` the in-process ``ours.app.build_graph`` builds; then mirrors
``pose.odom``, ``keyframe``, ``frame.tracks``, ``frame.inliers`` and
``pose.refined`` onto its own IpcBus endpoint ``"oak.vio"`` for SLAM / UI / tools.

Calibration handshake
---------------------
VIO needs the camera intrinsics + IMU extrinsics + gyro-bias / accel-align
seeds BEFORE it can build the odometry flow. Two-client startup:

1. Open a **calib client** subscribed to the retained ``calib.bundle`` topic;
   wait (with timeout) for the first bundle (retained, so a late VIO boot still
   gets it instantly).
2. Build the local odometry / backend graph with the bundle.
3. Open a **data client** subscribed to ``imucam.sample`` + ``frame.depth`` and
   the bridge subscriber, then start everything.

Each client is one IpcClientBus connection -- the IPC bus API requires every
subscription to be registered BEFORE ``start``, so a single client cannot mix
the "wait for calib" + "subscribe to data" phases.

Run::

    python -m ours.proc.vio                                # default endpoints
    python -m ours.proc.vio --capture-endpoint oak.capture.test --endpoint oak.vio.test
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
from ours.lib.flow.messages import END                             # noqa: E402
from ours.lib.ipc import IpcClientBus, IpcServerBus                # noqa: E402
from ours.lib.ipc.messages import WireCalibBundle                  # noqa: E402
from ours.flows.bridge import (                                    # noqa: E402
    IpcPublisherFlow, IpcSubscriberFlow, RingRegistry,
)
from ours.flows.bridge.ring_registry import (                      # noqa: E402
    default_capture_specs, default_vio_specs,
)
from ours.flows.backend import BackendFlow                         # noqa: E402
from ours.flows.odometry import OdometryFlow                       # noqa: E402

LOG = logging.getLogger("ours.proc.vio")

DEFAULT_CAPTURE_ENDPOINT = "oak.capture"
DEFAULT_VIO_ENDPOINT = "oak.vio"

#: Topics VIO subscribes to from capture.
_INPUT_TOPICS = [topics.IMUCAM_SAMPLE, topics.FRAME_DEPTH]

#: Topics VIO republishes (downstream is SLAM + UI).
_OUTPUT_TOPICS = [
    topics.POSE_ODOM,
    topics.KEYFRAME,
    topics.POSE_REFINED,
    topics.FRAME_TRACKS,
    topics.FRAME_INLIERS,
]


# --------------------------------------------------------------------------- #
def _await_calib_bundle(endpoint: str, timeout_s: float) -> WireCalibBundle:
    """Open a dedicated client, block until the retained calib bundle arrives."""
    bundle: list[WireCalibBundle | None] = [None]
    got = threading.Event()

    def on_calib(wm: WireCalibBundle) -> None:
        bundle[0] = wm
        got.set()

    client = IpcClientBus(endpoint, connect_timeout_s=timeout_s)
    client.subscribe("calib.bundle", on_calib)
    client.start()
    try:
        if not got.wait(timeout=timeout_s):
            raise TimeoutError(
                f"vio: no calib.bundle from {endpoint!r} in {timeout_s}s")
    finally:
        client.stop()
    assert bundle[0] is not None
    return bundle[0]


# --------------------------------------------------------------------------- #
def run_vio(*,
            capture_endpoint: str = DEFAULT_CAPTURE_ENDPOINT,
            endpoint: str = DEFAULT_VIO_ENDPOINT,
            kf_every: int = 5,
            use_gyro: bool = True,
            worker: bool = False,
            calib_timeout_s: float = 30.0,
            backend_window: int = 6,
            backend_iters: int = 5) -> int:
    """Run the VIO process until END / SIGTERM / Ctrl-C."""
    # 1. Block until capture publishes its calibration bundle.
    LOG.info("vio: waiting for calib.bundle on %s ...", capture_endpoint)
    bundle = _await_calib_bundle(capture_endpoint, calib_timeout_s)
    width, height = int(bundle.width), int(bundle.height)
    LOG.info("vio: got calib %dx%d, T_imu=%s, gyro_bias=%s",
             width, height, bundle.T_imu_left is not None,
             bundle.gyro_bias is not None)

    # 2. Allocate the capture-side ring registry (consumer-attach) for the
    #    subscriber bridge to read frame data from shared memory.
    cap_rings = RingRegistry().attach_all(default_capture_specs(
        endpoint=capture_endpoint, width=width, height=height))

    # 3. Allocate VIO's OWN rings for the keyframe stream we republish (kf_gray,
    #    kf_depth -- SLAM picks them up here, not from capture).
    vio_rings = RingRegistry().create_all(default_vio_specs(
        endpoint=endpoint, width=width, height=height))

    # 4. Build the local bus + the odometry / backend graph using the bundle.
    local = Bus()
    odom = OdometryFlow(local, bundle.K,
                        R_imu_cam=bundle.R_imu_cam,
                        accel_align=bundle.accel_align,
                        kf_every=kf_every, use_gyro=use_gyro,
                        latest_only=False, level_tilt=False)
    backend = BackendFlow(local, bundle.K,
                          window=backend_window, iters=backend_iters,
                          latest_only=False, worker=worker)

    # 5. Open the OUTPUT IpcServerBus + publisher bridge. KEYFRAME is the only
    #    VIO output that needs shared memory (image + depth payload), so it gets
    #    a dedicated publisher backed by VIO's own kf_* rings. Everything else
    #    VIO republishes (POSE_ODOM, POSE_REFINED, FRAME_TRACKS, FRAME_INLIERS)
    #    is pure POD (poses + per-frame ids / pixels) -- no ring slots needed,
    #    so the second publisher's ring registry is effectively unused. The
    #    image + depth the keypoint visualiser pairs with FRAME_TRACKS arrive on
    #    capture's FRAME_DEPTH (capture is the SINGLE writer of those rings; VIO
    #    must not race it). Retain `calib.bundle` and republish capture's bundle
    #    so any consumer that connects to *this* endpoint (UI, the proc4
    #    selftest, ...) can use the calib arrival as a readiness barrier
    #    (proves VIO is up AND VIO's rings already exist).
    server = IpcServerBus(endpoint, retain_topics={"calib.bundle"})
    pub_kf = IpcPublisherFlow(local, server, vio_rings, [topics.KEYFRAME],
                              endpoint=endpoint, ring_endpoint=endpoint)
    pub_pose = IpcPublisherFlow(local, server, vio_rings,
                                [topics.POSE_ODOM, topics.POSE_REFINED,
                                 topics.FRAME_TRACKS, topics.FRAME_INLIERS],
                                endpoint=endpoint,
                                ring_endpoint=endpoint)
    pub_kf.start()
    pub_pose.start()
    # Re-broadcast the calib bundle onto VIO's endpoint AFTER pub_kf.start()
    # actually opened the server socket. The retained slot caches it for any
    # subscriber that connects later (UI / SLAM / smoke selftest).
    server.publish("calib.bundle", bundle)

    # 6. Open the INPUT IpcClientBus + subscriber bridge: capture topics ->
    #    local bus. Other flows (odom, backend) consume from the local bus.
    in_client = IpcClientBus(capture_endpoint)
    in_bridge = IpcSubscriberFlow(local, in_client, cap_rings, _INPUT_TOPICS)

    # 7. END-detection sink: when capture finishes the replay session it
    #    publishes WireEnd on its data topics; the bridge translates it to the
    #    local-bus END. We want to know when both data topics have ENDed so we
    #    can shut down cleanly. Two ENDs expected (imucam.sample + frame.depth),
    #    matching odometry's expected_ends = 2.
    ends_seen = [0]
    finished = threading.Event()

    def _end_watch(_msg) -> None:
        if _msg is END:
            ends_seen[0] += 1
            if ends_seen[0] >= 2:
                finished.set()
    for t in _INPUT_TOPICS:
        local.subscribe(t, _end_watch)

    LOG.info("vio[%s] subscribing to %s -> %d topics", endpoint,
             capture_endpoint, len(_OUTPUT_TOPICS))

    # 8. Start everything. Order matters: bridge consumers first (so messages
    #    published while odom/backend boot are not lost on the local bus).
    odom.start()
    backend.start()
    in_bridge.start()

    stop = [False]
    def _on_sigterm(_signo, _frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        # Run until: (a) replay capture sends END on every input -> finished, or
        # (b) operator interrupts -> stop[0] / KeyboardInterrupt.
        while not stop[0] and not finished.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        LOG.info("vio: SIGINT -> stopping")
    finally:
        # Drain order: stop input bridge so no more messages arrive, then wait
        # for odom + backend to finish their inboxes (END is already in flight),
        # then forward END on every output topic so downstream procs (SLAM, UI)
        # drain. NB: the wait must be long enough that even a full inbox of
        # buffered replay frames drains -- 30 frames at ~100 ms / frame can
        # easily take 3-5 s; allow a generous ceiling here, and let SIGTERM
        # short-circuit if the operator gives up.
        in_bridge.stop()
        odom.done.wait(timeout=120.0)
        odom.stop()
        backend.done.wait(timeout=120.0)
        backend.stop()
        # The flows already forward END on their declared downstream topics
        # via `_emit_end` (see `Flow._handle_end`), but those go onto the
        # LOCAL bus -- the publisher bridge then mirrors them onto the IPC
        # server. So no explicit `server.publish_end` is needed here.
        time.sleep(0.3)
        pub_kf.stop()
        pub_pose.stop()
        server.close()
        cap_rings.close()
        vio_rings.unlink()
        vio_rings.close()
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-endpoint", default=DEFAULT_CAPTURE_ENDPOINT,
                    help=f"capture IPC endpoint (default: {DEFAULT_CAPTURE_ENDPOINT!r})")
    ap.add_argument("--endpoint", default=DEFAULT_VIO_ENDPOINT,
                    help=f"this process's IPC endpoint (default: {DEFAULT_VIO_ENDPOINT!r})")
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--no-gyro", action="store_true")
    ap.add_argument("--worker", action="store_true",
                    help="run windowed BA solve in a child process (release GIL)")
    ap.add_argument("--calib-timeout", type=float, default=30.0,
                    help="seconds to wait for the calib.bundle on boot")
    ap.add_argument("--backend-window", type=int, default=6)
    ap.add_argument("--backend-iters", type=int, default=5)
    args = ap.parse_args()

    return run_vio(
        capture_endpoint=args.capture_endpoint,
        endpoint=args.endpoint,
        kf_every=args.kf_every,
        use_gyro=not args.no_gyro,
        worker=args.worker,
        calib_timeout_s=args.calib_timeout,
        backend_window=args.backend_window,
        backend_iters=args.backend_iters,
    )


if __name__ == "__main__":
    raise SystemExit(main())
