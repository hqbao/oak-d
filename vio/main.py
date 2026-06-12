"""vio process: subscribe to capture, run odometry + windowed BA, republish.

Subscribes (over IPC) to the ``capture`` endpoint for ``calib.bundle``,
``imucam.sample`` and ``frame.depth``; runs the same
:class:`~vio.modules.pipeline.OdometryModule` +
:class:`~vio.modules.pipeline.BackendModule` the pre-split in-process graph built;
then mirrors ``pose.odom``, ``pose.vo`` (pure-vision f2f line), ``keyframe``,
``frame.tracks``, ``frame.inliers`` and ``pose.refined`` onto its own
:class:`~vio.comms.IPCPubSub` endpoint ``"oak.vio"`` for SLAM / UI / tools.

Calibration handshake
---------------------
VIO needs the camera intrinsics + IMU extrinsics + gyro-bias / accel-align
seeds BEFORE it can build the odometry module. Two-client startup:

1. Open a **calib client** subscribed to the retained ``calib.bundle`` topic;
   wait (with timeout) for the first bundle (retained, so a late VIO boot still
   gets it instantly).
2. Build the local odometry / backend graph with the bundle.
3. Open a **data client** subscribed to ``imucam.sample`` + ``frame.depth`` and
   the bridge subscriber, then start everything.

Each client is one :class:`~vio.comms.IPCPubSub` connection -- the IPC API
requires every subscription to be registered BEFORE ``start``, so a single client
cannot mix the "wait for calib" + "subscribe to data" phases.

The worker-engine subprocess boundary (``BackendModule(worker=True)``) stays on
stdlib pickle (``multiprocessing.Queue`` over same-project classes) and is NOT
routed through the class-path-independent codec -- the codec is only for the
cross-process IPC wire contract.

Run::

    python -m vio.main                                # default endpoints
    python -m vio.main --capture-endpoint oak.capture.test --endpoint oak.vio.test
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

from vio.comms import (                                             # noqa: E402
    IPCPublisher, IPCSubscriber, IPCPubSub, LocalPubSub, RingRegistry, topics,
)
from vio.comms.messages import END                                 # noqa: E402
from vio.comms.wire import WireCalibBundle                         # noqa: E402
from vio.comms.ring_registry import (                              # noqa: E402
    default_capture_specs, default_vio_specs,
)
from vio.modules import BackendModule, OdometryModule              # noqa: E402
from sky.front.odometry import OdometryConfig           # noqa: E402
from vio.comms.lib.config.resolution import ResolutionProfile     # noqa: E402
from vio.mathlib.resolution_build import frontend_config          # noqa: E402

LOG = logging.getLogger("vio.main")

DEFAULT_CAPTURE_ENDPOINT = "oak.capture"
DEFAULT_VIO_ENDPOINT = "oak.vio"
DEFAULT_SLAM_ENDPOINT = "oak.slam"

#: Topics VIO subscribes to from capture.
_INPUT_TOPICS = [topics.IMUCAM_SAMPLE, topics.FRAME_DEPTH]

#: Topic VIO subscribes to from SLAM for the CLOSED-LOOP feedback (LIVE + --tight
#: only): the pose-graph loop correction, fed back into the live nav-state so
#: accumulated drift is BOUNDED on revisits (Basalt's realtime VIO has none).
_SLAM_FEEDBACK_TOPICS = [topics.LOOP_CORRECTION]

#: Topics VIO republishes (downstream is SLAM + UI). POSE_VO is the pure-vision
#: frame-to-frame trajectory (live "VO" line) -- pure POD pose, no ring, like
#: POSE_ODOM / POSE_REFINED.
_OUTPUT_TOPICS = [
    topics.POSE_ODOM,
    topics.POSE_VO,
    topics.KEYFRAME,
    topics.POSE_REFINED,
    topics.FRAME_TRACKS,
    topics.FRAME_INLIERS,
    topics.FRAME_GYROFUSE,
]


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
                f"vio: no calib.bundle from {endpoint!r} in {timeout_s}s")
    finally:
        client.stop()
    assert bundle[0] is not None
    return bundle[0]


# --------------------------------------------------------------------------- #
def run_vio(*,
            capture_endpoint: str = DEFAULT_CAPTURE_ENDPOINT,
            endpoint: str = DEFAULT_VIO_ENDPOINT,
            slam_endpoint: str | None = None,
            kf_every: int = 5,
            use_gyro: bool = True,
            worker: bool = False,
            calib_timeout_s: float = 30.0,
            backend_window: int = 6,
            backend_iters: int = 5,
            tight: bool = False,
            stabilize_velocity: bool = False,
            depth_icp: bool = False,
            ba_window: bool = False) -> int:
    """Run the VIO process until END / SIGTERM / Ctrl-C.

    ``tight`` selects the TIGHT-coupled VIO backend (the joint visual + IMU
    window optimiser, ``imu_info_weight=True``) instead of the default LOOSE
    windowed-BA backend. Opt-in: when False the path is byte-identical to before
    -- the front-end stops retaining raw IMU and the back-end builds the loose
    engine. When True the odometry module retains the per-frame raw IMU so each
    keyframe carries the inter-keyframe IMU block the tight backend preintegrates.

    ``slam_endpoint`` enables the CLOSED-LOOP feedback (``slam -> vio``): when set
    AND ``tight`` is True, VIO opens a read-only client on the SLAM endpoint,
    subscribes to ``loop.correction``, and feeds each pose-graph correction back
    into the live nav-state (``PropagateImu``) so accumulated drift is BOUNDED on
    revisits -- better than Basalt's realtime VIO (which has no loop closure and
    drifts unboundedly). This is LIVE + ``--tight`` ONLY: the offline / oracle /
    loose path never sets it, so those paths are byte-identical. Drift bounding is
    SMOOTH (the correction is bled over a few frames, never a hard snap).

    ``stabilize_velocity`` enables the Phase-4 velocity regularisation on the
    tight backend (the CV smoothness prior + excitation-gated ZUPT) to curb the
    54x42 / shake window-velocity divergence. Opt-in and ``--tight`` ONLY: it is
    forwarded straight to :class:`~vio.modules.pipeline.BackendModule`, which only
    honours it inside its ``tight`` branch. When False the tight config is the
    untouched (oracle-tuned) default and the loose path is unaffected -- so the
    byte-parity oracle stays gap=0.

    ``depth_icp`` enables the Phase-4 dense-ICP relative-pose factor on the tight
    backend: an IMU-seeded point-to-plane ICP between adjacent in-window keyframe
    depth clouds adds a measured TRANSLATION constraint, anchoring the
    inter-keyframe Delta-p that the feature-starved 54x42 frontend cannot observe.
    Opt-in and ``--tight`` ONLY (same contract as ``stabilize_velocity``); OFF
    leaves the tight config + oracle byte-identical.

    ``ba_window`` enables the BA-window visualiser snapshot stream (``--ba-window``):
    the LOOSE backend builds the capture-aware engine and republishes one
    ``ba.window`` :class:`~vio.comms.messages.BaWindow` per keyframe solve (the
    window keyframe poses + shared 3D landmarks + observation rays + reprojection
    error + the PRE-solve state) for the UI's "BA Window" view. Opt-in and
    LOOSE-only (ignored on ``--tight``); OFF by default and never set by the
    oracle, so the byte-parity oracle stays gap=0 (the capture step runs the SAME
    frozen solve; the snapshot only rides the existing overlay channel).
    """
    # Closed-loop feedback is --tight + LIVE only: a slam endpoint must be wired
    # AND the tight nav-state must exist (retain_imu, set by tight). The loose /
    # oracle path never reaches here with both, so it stays byte-identical.
    loop_correct = bool(tight and slam_endpoint)
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
    local = LocalPubSub()
    # Resolution-scaled frontend config: at the 640 baseline this is the
    # historical full-quality FrontendConfig (block_size=7, no bucketing); at a
    # low ToF resolution (e.g. the 54x42 VL53L9CX sim) the profile shrinks the
    # Shi-Tomasi window to 3px and turns on bucketed per-cell detection so the
    # frontend produces more, evenly-spread, consistent corners (the PnP no
    # longer flips LOST<->OK on clustered points). Numba availability only caps
    # the KLT window/pyramid/budget, never the detection geometry.
    try:
        from sky.front.klt_numba import HAVE_NUMBA
    except Exception:
        HAVE_NUMBA = False
    res = ResolutionProfile.for_resolution(width, height)
    fe_cfg = frontend_config(res, numba=HAVE_NUMBA)
    LOG.info("vio: frontend profile -> %s", res.describe())
    # The VIO process serves the interactive LIVE viewer (capture runs --live), so
    # VIO must self-level (level_tilt) and gyro-fuse rotation exactly like the
    # in-process live graph -- otherwise the body frame renders tilted and heading
    # under-rotates on fast turns. The byte-identical-pose constraint only applies
    # to the offline deterministic scoring harness (a separate entry point).
    odom = OdometryModule(local, bundle.K,
                          R_imu_cam=bundle.R_imu_cam,
                          accel_align=bundle.accel_align,
                          odom_cfg=OdometryConfig(gyro_fuse=use_gyro),
                          frontend_cfg=fe_cfg,
                          kf_every=kf_every, use_gyro=use_gyro,
                          latest_only=False, level_tilt=True,
                          publish_vo=True,   # live viewer's pure-vision "VO" line
                          retain_imu=tight,  # tight backend needs inter-KF IMU
                          loop_correct=loop_correct)  # closed-loop SLAM feedback
    if tight:
        LOG.info("vio: TIGHT-coupled VIO backend selected (--tight) "
                 "[imu_info_weight=True]")
    if loop_correct:
        LOG.info("vio: CLOSED-LOOP SLAM correction ENABLED (slam=%s -> live "
                 "pose.odom) -- drift bounded on revisits", slam_endpoint)
    backend = BackendModule(local, bundle.K,
                            window=backend_window, iters=backend_iters,
                            latest_only=False, worker=worker, tight=tight,
                            stabilize_velocity=stabilize_velocity,
                            depth_icp=depth_icp, capture_window=ba_window)
    # BA-window capture is LOOSE-only; --tight overrides it (the tight map has no
    # capture overlay). Publish ba.window only when the capture engine is actually
    # built, so a consumer never waits on a topic that will never emit.
    ba_window_on = bool(ba_window and not tight)

    # 5. Open the OUTPUT IPCPubSub server + publisher bridge. KEYFRAME is the only
    #    VIO output that needs shared memory (image + depth payload), so it gets
    #    a dedicated publisher backed by VIO's own kf_* rings. Everything else
    #    VIO republishes (POSE_ODOM, POSE_VO, POSE_REFINED, FRAME_TRACKS,
    #    FRAME_INLIERS) is pure POD (poses + per-frame ids / pixels) -- no ring
    #    slots needed, so the second publisher's ring registry is effectively
    #    unused. The image + depth the keypoint visualiser pairs with
    #    FRAME_TRACKS arrive on capture's FRAME_DEPTH (capture is the SINGLE
    #    writer of those rings; VIO must not race it). Retain `calib.bundle` and
    #    republish capture's bundle so any consumer that connects to *this*
    #    endpoint (UI, the pair selftest, ...) can use the calib arrival as a
    #    readiness barrier (proves VIO is up AND VIO's rings already exist).
    server = IPCPubSub(endpoint, role="server", retain_topics={"calib.bundle"})
    pub_kf = IPCPublisher(local, server, vio_rings, [topics.KEYFRAME],
                          endpoint=endpoint, ring_endpoint=endpoint)
    # Pure-POD republished topics (poses + per-frame ids / pixels -- no ring
    # slots). ``ba.window`` (the opt-in BA-window solve snapshot) is also pure POD
    # (window poses + landmarks + observation rays, no images), so it rides this
    # same publisher; appended ONLY when the capture engine is built.
    _pose_topics = [topics.POSE_ODOM, topics.POSE_VO, topics.POSE_REFINED,
                    topics.FRAME_TRACKS, topics.FRAME_INLIERS,
                    topics.FRAME_GYROFUSE]
    if ba_window_on:
        _pose_topics.append(topics.BA_WINDOW)
    pub_pose = IPCPublisher(local, server, vio_rings, _pose_topics,
                            endpoint=endpoint,
                            ring_endpoint=endpoint)
    pub_kf.start()
    pub_pose.start()
    # Re-broadcast the calib bundle onto VIO's endpoint AFTER pub_kf.start()
    # actually opened the server socket. The retained slot caches it for any
    # subscriber that connects later (UI / SLAM / smoke selftest).
    server.publish("calib.bundle", bundle)

    # 6. Open the INPUT IPCPubSub client + subscriber bridge: capture topics ->
    #    local bus. Other modules (odom, backend) consume from the local bus.
    in_client = IPCPubSub(capture_endpoint, role="client")
    in_bridge = IPCSubscriber(local, in_client, cap_rings, _INPUT_TOPICS)

    # 6b. CLOSED-LOOP feedback bridge (LIVE + --tight only): subscribe SLAM's
    #     loop.correction on the slam endpoint and re-hydrate it onto THIS local
    #     bus, where OdometryModule's loop-correction inbox picks it up. The
    #     correction carries only POD poses (no shared-memory rings), so it needs
    #     none of VIO's rings. SLAM boots AFTER VIO, so its endpoint may not exist
    #     yet -- the client retries on a generous timeout, and a failed connect is
    #     non-fatal (VIO still runs; closed-loop just stays off). The offline /
    #     loose path never sets loop_correct, so this whole block is skipped there.
    loop_bridge = None
    loop_client = None
    if loop_correct:
        try:
            loop_client = IPCPubSub(slam_endpoint, role="client",
                                    connect_timeout_s=calib_timeout_s)
            loop_bridge = IPCSubscriber(local, loop_client, vio_rings,
                                        _SLAM_FEEDBACK_TOPICS)
        except (TimeoutError, ConnectionError) as e:
            LOG.warning("vio: closed-loop feedback DISABLED -- could not connect "
                        "to slam endpoint %s (%s); live VIO runs uncorrected",
                        slam_endpoint, e)
            loop_bridge = None
            loop_client = None

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
    # Closed-loop feedback bridge last (its connect retries until SLAM is up; a
    # failed connect logs + exits its thread without affecting the rest of VIO).
    if loop_bridge is not None:
        loop_bridge.start()

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
        #
        # Under SIGTERM the operator wants a fast exit. Capture is also
        # shutting down so END will never arrive on imucam.sample / frame.depth
        # -- waiting 120 s on `odom.done` would let the launcher SIGKILL us
        # (10 s deadline), leaking every vio_rings slot. Cap the wait at 2 s
        # under SIGTERM, then Module.stop() forces the drain thread out at the
        # top of its next loop iteration. Natural END (finished.is_set()) keeps
        # the generous 120 s ceiling so a busy backend can finish.
        in_bridge.stop()
        # Stop the closed-loop feedback bridge too (no more corrections needed
        # once we're tearing down). Idempotent + safe if its connect never
        # succeeded (the thread already exited).
        if loop_bridge is not None:
            loop_bridge.stop()
        drain_timeout = 2.0 if stop[0] else 120.0
        odom.done.wait(timeout=drain_timeout)
        odom.stop()
        backend.done.wait(timeout=drain_timeout)
        backend.stop()
        # The modules already forward END on their declared downstream topics
        # via `_emit_end` (see `Module._handle_end`), but those go onto the
        # LOCAL bus -- the publisher bridge then mirrors them onto the IPC
        # server. So no explicit `server.publish_end` is needed here.
        time.sleep(0.3)
        pub_kf.stop()
        pub_pose.stop()
        server.close()
        cap_rings.close()
        vio_rings.unlink()
        vio_rings.close()
        LOG.info("vio: shutdown complete")
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
    ap.add_argument("--slam-endpoint", default=None,
                    help="SLAM IPC endpoint to subscribe loop.correction from for "
                         "the CLOSED-LOOP feedback (slam->vio). Only takes effect "
                         "with --tight; off when unset (open-loop, like before).")
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--no-gyro", action="store_true")
    ap.add_argument("--worker", action="store_true",
                    help="run windowed BA solve in a child process (release GIL)")
    ap.add_argument("--calib-timeout", type=float, default=30.0,
                    help="seconds to wait for the calib.bundle on boot")
    ap.add_argument("--backend-window", type=int, default=6)
    ap.add_argument("--backend-iters", type=int, default=5)
    ap.add_argument("--tight", action="store_true",
                    help="select the TIGHT-coupled VIO backend (joint visual + "
                         "IMU window optimiser, imu_info_weight=True) instead of "
                         "the default LOOSE windowed-BA backend. Opt-in; the "
                         "default (loose) path is byte-identical to before.")
    ap.add_argument("--stabilize-velocity", action="store_true",
                    help="tight only: enable Phase-4 velocity regularisation "
                         "(CV prior + gated ZUPT) to curb 54x42/shake velocity "
                         "divergence. Opt-in; ignored without --tight.")
    ap.add_argument("--depth-icp", action="store_true",
                    help="tight only: enable the Phase-4 dense-ICP relative-pose "
                         "factor (IMU-seeded point-to-plane ICP between keyframe "
                         "depth clouds) to anchor inter-keyframe translation at "
                         "54x42. Opt-in; ignored without --tight.")
    ap.add_argument("--ba-window", action="store_true",
                    help="loose only: publish ba.window solve snapshots (window "
                         "keyframe poses + 3D landmarks + observation rays + "
                         "reprojection error) for the UI's BA Window visualiser. "
                         "Opt-in; OFF by default (oracle byte-identical); ignored "
                         "with --tight.")
    args = ap.parse_args()

    return run_vio(
        capture_endpoint=args.capture_endpoint,
        endpoint=args.endpoint,
        slam_endpoint=args.slam_endpoint,
        kf_every=args.kf_every,
        use_gyro=not args.no_gyro,
        worker=args.worker,
        calib_timeout_s=args.calib_timeout,
        backend_window=args.backend_window,
        backend_iters=args.backend_iters,
        tight=args.tight,
        stabilize_velocity=args.stabilize_velocity,
        depth_icp=args.depth_icp,
        ba_window=args.ba_window,
    )


if __name__ == "__main__":
    # Use os._exit (not SystemExit / return-from-main) so a lingering non-daemon
    # thread -- IPCSubscriber's recv loop, the InProcessEngine worker, a numba
    # thread pool, etc -- cannot keep the process alive past
    # `vio: shutdown complete`. Without this the launcher waits its full 10 s
    # deadline and SIGKILLs us. Mirrors the same pattern in `imu_camera.main`.
    import os as _os
    _rc = main()
    LOG.info("vio: main returned, calling os._exit(%d)", int(_rc))
    logging.shutdown()
    _os.sys.stdout.flush()
    _os.sys.stderr.flush()
    _os._exit(int(_rc))
