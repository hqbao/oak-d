"""The VIO pipeline -- odometry (RGB-D VO + gyro prior) and the windowed
bundle-adjustment backend, as PROCEDURAL Python.

This replaces the old reactive ``OdometryModule(Module)`` /
``BackendModule(Module)`` + ``Step`` chains. The per-message work is now plain
functions (:func:`process_imucam` / :func:`process_frame` / :func:`process_kf`)
that call the step functions in the EXACT same order the reactive chains ran -- so
every output stream (``pose.odom`` / ``keyframe`` / ``pose.refined`` / the opt-in
viz topics) stays byte-identical and BOTH gates hold:

* the LOOSE path IS the frozen gap=0 oracle (the offline replay runs through this
  odometry worker), and
* the TIGHT path's live regression (``propagate_imu`` nav-state + the closed-loop
  loop-correction blend) is unchanged.

The framework indirection (``Module.on`` / ``_routes`` / ``_run_chain``
short-circuit-on-None / ``ctx.state[...]`` lookups inside every step) is gone; the
data flow reads as straight-line code. What did NOT vanish, and is replicated
EXPLICITLY in the two worker threads:

* :class:`OdometryWorker` -- the **2-input multi-END join**. The reactive module
  registered TWO routes (``.on(IMUCAM_SAMPLE, ...)`` + ``.on(FRAME_DEPTH, ...)``)
  on ONE inbox keyed by topic, and forwarded END only after BOTH inputs ENDed
  (``expected_ends = 2``). Here the inbox carries ``(topic, msg)`` tuples, the
  worker loop routes each by topic to the right step chain, and an explicit END
  counter forwards END downstream + sets :attr:`done` only once both
  ``imucam.sample`` AND ``frame.depth`` have ENDed. This is the load-bearing
  concurrency the ``Module`` gave for free.
* :class:`BackendWorker` -- the single ``keyframe`` input, the ``worker=True``
  subprocess-engine boundary, the ``tight`` engine switch, and the END-forward to
  ``pose.refined`` (+ ``ba.window`` when the capture engine is built).

The step functions own no module state; the odometry worker holds a
:class:`~vio.comms.module.ModuleContext` (a plain ``(bus, name, state)`` holder --
NOT the reactive substrate) so the per-run state the steps thread through
(``vo`` / ``priors`` / ``imu_segs`` / the live ``live_nav`` / ``loop_inbox`` ...)
lives in ONE place, and the selftests that reach into ``odom.ctx.state`` keep
working byte-for-byte.

``OdometryModule`` / ``BackendModule`` are kept as public aliases for the
procedural workers (``vio.main`` + the vio/verification selftests import them).
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import replace
from typing import Any

import numpy as np

from vio.comms import LocalPubSub, topics
from vio.comms.messages import END
from vio.comms.module import ModuleContext

from sky.front.frontend import CaptureKLTFrontend, FrontendConfig, KLTFrontend
from sky.front.odometry import OdometryConfig, RGBDVisualOdometry
from sky.backend.bundle import BAConfig
from sky.backend.windowed import WindowedConfig
from sky.vio.window import WindowedVIOConfig
from vio.mathlib.engine import make_ba_engine, make_vi_engine
from .preintegrate_prior import preintegrate_prior
from .track_features import track_features
from .publish_tracks import publish_tracks
from .align_gravity import align_gravity
from .pull_prior import pull_prior
from .estimate_motion import estimate_motion
from .correct_tilt import correct_tilt
from .publish_inliers import publish_inliers
from .publish_gyrofuse import publish_gyrofuse
from .publish_frontend_viz import publish_frontend_viz
from .propagate_imu import propagate_imu
from .loop_inbox import LoopCorrectionInbox
from .publish_pose import publish_pose
from .publish_vo import publish_vo
from .emit_keyframe import emit_keyframe
from .run_ba import run_ba
from .publish_refined import publish_refined
from .publish_ba_window import publish_ba_window

LOG = logging.getLogger("vio.pipeline")

#: Inbox sentinel to unblock ``queue.get`` on ``stop()``. Mirrors ``Module._SENTINEL``.
_SENTINEL = object()
#: Inbox payload marker for the coalescing path: "the real message is the current
#: self._latest[topic]". Mirrors the old ``Module._LATEST`` token.
_LATEST = object()


# =========================================================================== #
# Odometry: the 2-input front-end join (imucam.sample + frame.depth)
# =========================================================================== #
def process_imucam(ctx: ModuleContext, msg) -> None:
    """Run the ``imucam.sample`` edge for one synced packet.

    The reactive module routed this edge to ``[PreintegratePrior()]`` -- a single
    step that buffers the frame's IMU prior in ``ctx.state["priors"]`` (and, on
    the tight path, its raw IMU segment in ``ctx.state["imu_segs"]``) and returns
    ``None`` (terminal). Here that is just the one call.
    """
    preintegrate_prior(ctx.state, msg)


def process_frame(ctx: ModuleContext, msg) -> None:
    """Run the full ``frame.depth`` step chain for one depth frame.

    Byte-identical order to the old reactive frame-chain (``OdometryModule``
    built ``[TrackFeatures, PublishTracks, (PublishFrontendViz,) AlignGravity,
    PullPrior, EstimateMotion, CorrectTilt, PublishInliers, PublishGyroFuse,
    PropagateImu, PublishPose, (PublishVo,) EmitKeyframe]`` and ran them in that
    sequence, each step's output feeding the next; a step returning ``None``
    short-circuited the chain). The order here is exactly the same call order --
    gap=0 depends on it. The optional steps (``frontend_viz`` / ``publish_vo``)
    are gated by the same flags the module gated their wiring on.
    """
    state = ctx.state
    bus = ctx.bus
    vo: RGBDVisualOdometry = state["vo"]

    # TrackFeatures (KLT, the only numba-parallel section) -> PublishTracks.
    tracked = track_features(vo, msg)
    tracked = publish_tracks(bus, tracked)
    # PublishFrontendViz (opt-in, --frontend-viz): runs RIGHT AFTER PublishTracks
    # so the capture frontend's snap is fresh; passes the carrier through unchanged.
    if state.get("frontend_viz"):
        tracked = publish_frontend_viz(vo, bus, tracked)
    # AlignGravity (one-shot startup bootstrap) -> PullPrior (the IMU<->vision
    # join, pops priors[seq]) -> EstimateMotion (RGB-D PnP + gyro fusion).
    tracked = align_gravity(vo, state, tracked)
    primed = pull_prior(state["priors"], tracked)
    step = estimate_motion(vo, primed)
    # CorrectTilt (LIVE-only at-rest leveling) -> PublishInliers / PublishGyroFuse
    # (post-solve diagnostic publishers; gyrofuse self-skips on non-fused frames).
    step = correct_tilt(vo, state.get("level_tilt", False), step)
    step = publish_inliers(ctx, step)
    step = publish_gyrofuse(ctx, step)
    # PropagateImu (TIGHT path only, gated on retain_imu): forward-propagates the
    # live nav-state + applies the closed-loop SLAM correction, replaces step.pose.
    # On the LOOSE path it is a pass-through no-op (byte-identical pose.odom). It
    # also owns the keyframe-cadence boolean EmitKeyframe consumes (tight path).
    step = propagate_imu(ctx, step)
    step = publish_pose(bus, step)
    # PublishVo (LIVE-only, publish_vo): the pure-vision pose.vo line, right after
    # PublishPose; reads vo.pose_vo (not step.pose) so CorrectTilt/PropagateImu
    # never affect it.
    if state.get("publish_vo"):
        step = publish_vo(vo, bus, step)
    # EmitKeyframe: every kf_every frames publish a keyframe (terminal).
    emit_keyframe(vo, state, bus, step)


class OdometryWorker(threading.Thread):
    """The VIO odometry front-end: a plain thread joining two input edges.

    A procedural replacement for the old reactive ``OdometryModule(Module)``. It
    joins the two edges of the unified acquisition front-end:

    * ``imucam.sample`` -> :func:`process_imucam` (IMU prior preintegration), and
    * ``frame.depth`` -> :func:`process_frame` (KLT track -> RGB-D PnP -> gyro
      fusion -> pose -> keyframe).

    Both inputs come from the SAME upstream (capture's imu_cam module publishes
    ``imucam.sample`` and ``frame.depth``); over IPC the VIO process's subscriber
    bridge mirrors them onto this local bus. The worker owns ONE inbox carrying
    ``(topic, msg)`` tuples; the loop routes each message to the right chain by
    topic. Because BOTH inputs carry END, the worker waits for BOTH ENDs before
    forwarding END downstream + setting :attr:`done` (the old
    ``expected_ends = 2`` join) -- so it never signals "done" early on a single
    input's END.

    ``R_imu_cam`` (IMU->camera rotation) drives the gyro prior; ``accel_align`` is
    the one-shot startup gravity reference (camera frame) capture measured, seeded
    here so the solve levels the initial attitude. Both may be ``None`` (pure
    vision / no usable IMU). All per-run state lives in :attr:`ctx` (a
    :class:`~vio.comms.module.ModuleContext`), the same place the selftests reach
    into.

    WARNING: ``latest_only=True`` makes BOTH inboxes coalesce, dropping frames
    when the chain falls behind -- this breaks VIO's gyro continuity
    (preintegrate_prior) and KLT continuity (track_features). ONLY pass it for a
    UI-only graph with no odometry downstream. For VIO / replay / the oracle, keep
    the default FIFO inbox and put backpressure at the IPC boundary.
    """

    def __init__(self, bus: LocalPubSub, K: np.ndarray,
                 R_imu_cam: np.ndarray | None = None,
                 accel_align: np.ndarray | None = None,
                 odom_cfg: OdometryConfig | None = None,
                 frontend_cfg: FrontendConfig | None = None,
                 kf_every: int = 5, use_gyro: bool = True,
                 latest_only: bool = False, level_tilt: bool = False,
                 publish_vo: bool = False, retain_imu: bool = False,
                 loop_correct: bool = False, frontend_viz: bool = False) -> None:
        super().__init__(name="odometry", daemon=True)
        self.bus = bus
        self.ctx = ModuleContext(bus, "odometry")
        state = self.ctx.state

        # ``frontend_cfg`` carries the resolution-scaled KLT + corner-detection
        # geometry. Defaulting to None keeps the historical full-quality
        # FrontendConfig() the offline byte-parity oracle relies on.
        #
        # ``frontend_viz`` (opt-in, --frontend-viz) builds a CaptureKLTFrontend
        # instead of the plain KLTFrontend so the frontend stashes a per-frame
        # FrontendVizSnap for the UI's "Frontend Internals" view. The capture
        # frontend returns BYTE-IDENTICAL tracks, so the motion estimate -- and
        # the oracle -- are UNAFFECTED. It is LIVE-only (the offline oracle never
        # sets it); the publish step (publish_frontend_viz) runs in the
        # frame-chain only when on.
        state["frontend_viz"] = bool(frontend_viz)
        if frontend_viz:
            # Force capture on; keep all the resolution-scaled geometry. (When no
            # frontend_cfg was supplied, capture the historical default config.)
            cap_cfg = replace(frontend_cfg or FrontendConfig(), capture=True)
            fe = CaptureKLTFrontend(cap_cfg)
        else:
            fe = KLTFrontend(frontend_cfg) if frontend_cfg is not None else None
        state["vo"] = RGBDVisualOdometry(
            K, odom_cfg or OdometryConfig(), frontend=fe)
        state["kf_every"] = int(kf_every)
        state["use_gyro"] = bool(use_gyro)
        # Continuous at-rest roll/pitch leveling (correct_tilt). LIVE-only so the
        # offline replay/scoring pose.odom stays byte-identical; the live builder
        # turns it on. Lets the live view self-level without a startup hold-still.
        state["level_tilt"] = bool(level_tilt)
        state["publish_vo"] = bool(publish_vo)
        state["priors"] = {}
        # TIGHT path only: retain the per-frame raw IMU samples (camera frame) so
        # emit_keyframe can hand the inter-keyframe block to the tight backend. The
        # default (False) keeps the LOOSE / oracle front-end byte-identical -- the
        # extra retention is a no-op (preintegrate_prior / emit_keyframe gate on it).
        state["retain_imu"] = bool(retain_imu)
        if retain_imu:
            state["imu_segs"] = {}
            state["last_kf_seq"] = -1
            # Fixed world gravity ACCELERATION vector (optical-world "down" = +y),
            # matching WindowedVIOConfig.g_world. Used by propagate_imu's per-frame
            # forward-integration + ZUPT. Kept identical to the tight backend so the
            # live dead-reckoning and the keyframe nav-state share one gravity model.
            state["g_world"] = (0.0, 9.81, 0.0)
        # Closed-loop SLAM correction (LIVE + --tight only). When on, propagate_imu
        # bleeds the SLAM pose-graph correction (loop.correction) back into the live
        # nav-state so accumulated drift is BOUNDED on revisits. The correction
        # arrives on a DIFFERENT thread (the slam-endpoint IPC subscriber that
        # vio.main wires onto this local bus), so a thread-safe inbox hands it to
        # the odometry thread; propagate_imu drains it per frame. Gated so the
        # offline / oracle / loose path is byte-identical (loop_correct stays
        # False there -> no inbox, no subscription, no blend). Requires retain_imu
        # (the --tight nav-state); ignored otherwise.
        if loop_correct and retain_imu:
            inbox = LoopCorrectionInbox()
            state["loop_correct"] = True
            state["loop_inbox"] = inbox
            # Feed corrections from the local bus (vio.main republishes the slam
            # endpoint's loop.correction here) into the inbox. END is ignored.
            bus.subscribe(
                topics.LOOP_CORRECTION,
                lambda m: inbox.push(m) if m is not None and m is not END
                else None)
        state["R_imu_cam"] = (
            None if R_imu_cam is None else np.asarray(R_imu_cam, dtype=np.float64))
        if accel_align is not None:
            state["accel_align"] = np.asarray(accel_align, dtype=np.float64)

        # Downstream topics END is forwarded to (was Module.forwards_to).
        self._downstream = [topics.POSE_ODOM, topics.KEYFRAME, topics.FRAME_TRACKS,
                            topics.FRAME_INLIERS, topics.FRAME_GYROFUSE]
        if publish_vo:
            self._downstream.append(topics.POSE_VO)
        if frontend_viz:
            self._downstream.append(topics.FRAME_FRONTEND)

        # The two input edges, each routed to its own chain function. This is the
        # 2-input join the old ``.on(topic, chain)`` pair set up.
        self._routes = {
            topics.IMUCAM_SAMPLE: process_imucam,
            topics.FRAME_DEPTH: process_frame,
        }
        #: see BOTH ENDs before forwarding END / draining (was expected_ends = 2).
        self.expected_ends = len(self._routes)

        self._latest_only = bool(latest_only)
        self._inbox: "queue.Queue" = queue.Queue()
        self._latest: dict[str, Any] = {}      # topic -> newest unprocessed msg
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self.done = threading.Event()          #: set after all expected ENDs drained
        self._ends_seen = 0
        self._emitted_end = False

        # Subscribe both inbox feeders in __init__ (matches the old Module.on
        # timing) so a message published between construction and start() is never
        # lost. Each subscription captures its topic so the inbox stays keyed.
        for topic in self._routes:
            if self._latest_only:
                self.bus.subscribe(
                    topic, lambda m, t=topic: self._coalesce(t, m))
            else:
                self.bus.subscribe(
                    topic, lambda m, t=topic: self._inbox.put((t, m)))

    # -- inbox feeders (run on the PUBLISHER's thread, kept cheap) ----------- #
    def _coalesce(self, topic: str, msg: Any) -> None:
        """Keep only the newest unprocessed ``msg`` per topic (latest-only mode).

        Byte-for-byte the old ``Module._coalesce``: the inbox carries just a topic
        token; the message lives in ``self._latest[topic]`` and is overwritten by
        each newer arrival, so a backlog never builds. A token is enqueued only
        when nothing was pending for the topic -- EXCEPT END, which always enqueues
        a token (losing the last frame is fine; dropping END is not).
        """
        with self._latest_lock:
            pending = topic in self._latest
            self._latest[topic] = msg
            enqueue = (not pending) or (msg is END)
        if enqueue:
            self._inbox.put((topic, _LATEST))

    # -- thread body -------------------------------------------------------- #
    def stop(self) -> None:
        self._stop.set()
        self._inbox.put((_SENTINEL, _SENTINEL))   # unblock the queue.get

    def run(self) -> None:
        while not self._stop.is_set():
            topic, msg = self._inbox.get()
            if msg is _SENTINEL:
                break
            if msg is _LATEST:
                # Coalescing inbox: the token names a topic; pull its current
                # newest message (already drained by an earlier token -> skip).
                with self._latest_lock:
                    msg = self._latest.pop(topic, _SENTINEL)
                if msg is _SENTINEL:
                    continue
            if msg is END:
                self._handle_end()
                continue
            self._routes[topic](self.ctx, msg)

    def _handle_end(self) -> None:
        # The 2-input multi-END join: forward our own END downstream + signal done
        # only once EVERY END-bearing input has drained (expected_ends == 2). A
        # single input's END must NOT signal done early. Byte-for-byte the old
        # Module._handle_end semantics, specialised to this worker.
        self._ends_seen += 1
        if self._ends_seen >= self.expected_ends and not self._emitted_end:
            self._emitted_end = True
            for t in self._downstream:
                self.bus.publish(t, END)
        if self._ends_seen >= self.expected_ends:
            self.done.set()


# =========================================================================== #
# Backend: the windowed bundle-adjustment sink over `keyframe`
# =========================================================================== #
def process_kf(engine, bus: LocalPubSub, tight: bool, capture_window: bool,
               kf) -> None:
    """Run the backend chain for one keyframe.

    Byte-identical order to the old reactive chain. ``run_ba`` submits the
    keyframe's snapshot (loose 5-tuple / tight 6-tuple) to the engine and returns
    the refined pose (or ``None`` -> chain short-circuit). With capture on,
    ``publish_ba_window`` runs between ``run_ba`` and ``publish_refined`` (it
    forwards the pose UNCHANGED, so ``pose.refined`` is byte-identical to the
    no-capture chain).
    """
    msg = run_ba(engine, tight, kf)
    if msg is None:                  # no tracks this kf / no refined pose yet
        return
    if capture_window:
        msg = publish_ba_window(engine, bus, msg)
    publish_refined(bus, msg)


class BackendWorker(threading.Thread):
    """The windowed back-end over the ``keyframe`` stream: a plain thread.

    A procedural replacement for the old reactive ``BackendModule(Module)``. Two
    selectable backends, picked by ``tight`` (a clean engine switch, NOT a
    pipeline fork):

    * ``tight=False`` (default, LOOSE) -- :func:`make_ba_engine` builds the
      vision-only ``WindowedBAMap`` (reproj + depth + optional VO/gravity priors).
      Byte-identical to the pre-tight build; the offline oracle relies on this.
    * ``tight=True`` (``--tight``, opt-in) -- :func:`make_vi_engine` builds the
      tight-coupled ``WindowedVIOMap`` (joint visual + IMU window optimiser). The
      IMU factor is weighted by the per-edge information square root
      (``imu_info_weight=True``); ``run_ba`` then submits the SUPERSET snapshot
      (keyframe ts + raw inter-keyframe IMU block).

    The heavy solve runs behind an :class:`~vio.mathlib.engine.base.Engine`:
    ``worker=False`` (default, offline) runs it synchronously in-thread --
    byte-identical to the old path; ``worker=True`` (live) runs it in a separate
    process so it cannot hold the read loop's GIL (the fast-push undershoot fix).
    The engine is closed on THIS thread when the loop exits, so a subprocess
    worker is reaped without a cross-thread race.

    Single ``keyframe`` input, so the first END is terminal (no join). END is
    forwarded to ``pose.refined`` (+ ``ba.window`` when the capture engine built).
    """

    def __init__(self, bus: LocalPubSub, K: np.ndarray,
                 window: int = 6, iters: int = 5,
                 latest_only: bool = False, worker: bool = False,
                 tight: bool = False, stabilize_velocity: bool = False,
                 depth_icp: bool = False, capture_window: bool = False) -> None:
        super().__init__(name="backend", daemon=True)
        self.bus = bus
        self.tight = bool(tight)
        # BA-window capture (opt-in, --ba-window) is a LOOSE-backend-only viz: the
        # capture-aware ``WindowedBAMap`` engine snapshots each solve for the UI's
        # "BA Window". Ignored on the tight path + OFF by default so the oracle
        # stays byte-identical.
        self.capture_window = bool(capture_window) and not tight
        if tight:
            # Tight backend: enable the covariance-correct IMU weight (Phase 1's
            # opt-in flag) on a copy of WindowedVIOConfig's validated defaults.
            # ``imu_info_weight`` is the only baseline override -- everything else
            # (window, lock_tilt, tight vel/pos sigmas, kf_every) keeps the values
            # the vio_ba_selftest / vio oracle entries were tuned against.
            vio_cfg = WindowedVIOConfig()
            vio_cfg.vio.imu_info_weight = True
            # Phase-4 velocity regularisation (opt-in, LIVE --tight only): the
            # single ``stabilize_velocity`` knob makes ``run_ba`` flip on BOTH
            # the CV smoothness prior and the excitation-gated ZUPT for every
            # solve, curbing the 54x42 / shake window-velocity divergence. Left
            # OFF by default so the tight-without-flag path -- and the oracle --
            # stay byte-identical; only the operator's --stabilize-velocity sets it.
            if stabilize_velocity:
                vio_cfg.stabilize_velocity = True
                LOG.info("vio: tight velocity-stabilize ON "
                         "(CV prior + gated ZUPT)")
            # Phase-4 dense-ICP relative-pose factor (opt-in, LIVE --tight only):
            # ``depth_icp`` makes ``run_ba`` add an IMU-seeded point-to-plane ICP
            # relative-pose factor between adjacent in-window keyframes, anchoring
            # the inter-keyframe TRANSLATION the feature-starved 54x42 frontend
            # leaves unobservable. OFF by default so the tight-without-flag path
            # and the oracle stay byte-identical; only --depth-icp sets it.
            if depth_icp:
                vio_cfg.depth_icp = True
                LOG.info("vio: tight dense-ICP relative-pose factor ON "
                         "(translation anchor for feature-starved frames)")
            self.engine = make_vi_engine(K, vio_cfg, worker=worker)
        else:
            cfg = WindowedConfig(window=window, ba=BAConfig(max_iters=iters))
            self.engine = make_ba_engine(K, cfg, worker=worker,
                                         capture_window=self.capture_window)
            if self.capture_window:
                LOG.info("vio: BA-window capture ON (--ba-window) -- publishing "
                         "ba.window solve snapshots for the UI visualiser")

        # END is forwarded to whatever this worker publishes (was forwards_to):
        # pose.refined always, ba.window when the capture engine is built.
        self._downstream = [topics.POSE_REFINED]
        if self.capture_window:
            self._downstream.append(topics.BA_WINDOW)

        self._latest_only = bool(latest_only)
        self._inbox: "queue.Queue" = queue.Queue()
        self._latest: Any = _SENTINEL          # single-slot newest unprocessed kf
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self.done = threading.Event()          #: set after END is handled
        self._emitted_end = False

        # Subscribe the inbox feeder to keyframe in __init__ (old Module.on timing).
        self.bus.subscribe(topics.KEYFRAME, self._on_keyframe)

    # -- inbox feeder (runs on the PUBLISHER's thread, kept cheap) ----------- #
    def _on_keyframe(self, msg: Any) -> None:
        """Bus handler for ``keyframe``: enqueue (coalescing or strict FIFO)."""
        if not self._latest_only:
            self._inbox.put(msg)
            return
        # Coalescing (LIVE visualiser-fed graphs): keep only the newest
        # unprocessed keyframe; enqueue a token only when nothing pending -- EXCEPT
        # END, which always enqueues. Byte-for-byte the old Module._coalesce,
        # specialised to this worker's single keyframe topic.
        with self._latest_lock:
            pending = self._latest is not _SENTINEL
            self._latest = msg
            enqueue = (not pending) or (msg is END)
        if enqueue:
            self._inbox.put(_LATEST)

    # -- thread body -------------------------------------------------------- #
    def stop(self) -> None:
        self._stop.set()
        self._inbox.put(_SENTINEL)             # unblock the queue.get

    def run(self) -> None:
        # Close the engine on THIS thread when the loop exits (stop sentinel or
        # _stop), so a subprocess worker is reaped without a cross-thread race.
        try:
            self._loop()
        finally:
            self.engine.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            item = self._inbox.get()
            if item is _SENTINEL:
                break
            if item is _LATEST:
                with self._latest_lock:
                    msg, self._latest = self._latest, _SENTINEL
                if msg is _SENTINEL:
                    continue
            else:
                msg = item                      # strict-FIFO payload
            if msg is END:
                self._handle_end()
                continue
            process_kf(self.engine, self.bus, self.tight,
                       self.capture_window, msg)

    def _handle_end(self) -> None:
        # Single-input sink (one keyframe topic), so the first END is terminal.
        if not self._emitted_end:
            self._emitted_end = True
            for t in self._downstream:
                self.bus.publish(t, END)
        self.done.set()


#: Public names kept for vio.main + the vio/verification selftests. They are now
#: the procedural workers, not reactive Modules.
OdometryModule = OdometryWorker
BackendModule = BackendWorker
