"""The VIO reactive modules -- odometry (RGB-D VO + gyro prior) and the windowed
bundle-adjustment backend.

Ported VERBATIM from ``ours.flows.odometry.odometry_flow`` (->
:class:`OdometryModule`) + ``ours.flows.backend.backend_flow`` (->
:class:`BackendModule`); only the import roots + class names changed (Flow ->
Module, Bus -> LocalPubSub), so the wiring + algorithm are unchanged.

:class:`OdometryModule`
-----------------------
Wires the odometry steps (one file each) into a reactive module that joins the
two edges of the unified acquisition front-end:

* ``imucam.sample`` -> [:class:`~vio.modules.preintegrate_prior.PreintegratePrior`]
* ``frame.depth`` -> [:class:`~vio.modules.track_features.TrackFeatures`,
  :class:`~vio.modules.publish_tracks.PublishTracks`,
  :class:`~vio.modules.align_gravity.AlignGravity`,
  :class:`~vio.modules.pull_prior.PullPrior`,
  :class:`~vio.modules.estimate_motion.EstimateMotion`,
  :class:`~vio.modules.correct_tilt.CorrectTilt`,
  :class:`~vio.modules.publish_inliers.PublishInliers`,
  :class:`~vio.modules.publish_gyrofuse.PublishGyroFuse`,
  :class:`~vio.modules.publish_pose.PublishPose`,
  (:class:`~vio.modules.publish_vo.PublishVo`),
  :class:`~vio.modules.emit_keyframe.EmitKeyframe`]

Both inputs come from the SAME upstream (capture's imu_cam module publishes
``imucam.sample`` and, with its depth step, ``frame.depth``); over IPC the VIO
process's subscriber bridge mirrors them onto this local bus. The module owns the
IMU->prior fusion itself (``PreintegratePrior``). The frame-chain splits the
visual odometry into small single-purpose steps. ``TrackFeatures`` (KLT) is the
only numba-parallel section and holds the parallel lock; everything after it is
pure NumPy and runs lock-free, so the heavy motion solve overlaps the next
frame's depth matcher instead of serialising against it. ``PublishTracks`` emits
the same KLT tracks on ``frame.tracks`` for the keypoint-depth visualiser.
``AlignGravity`` does the one-shot startup attitude bootstrap; ``PullPrior`` is
the IMU<->vision join that pops the preintegrated prior for the frame's ``seq``;
``EstimateMotion`` is then just the RGB-D PnP (+ gyro fusion) solve.
``PublishInliers`` then emits that solve's PnP inlier track ids on
``frame.inliers`` so the visualiser can mark the clean subset the motion estimate
actually trusted. ``PublishGyroFuse`` then emits that solve's gyro-fusion
diagnostic on ``frame.gyrofuse`` (vision-vs-gyro rotation, disagreement, gain,
translation-trust) for the "Gyro fusion" strip chart -- only on gyro-fused frames
(it self-skips when gyro is off / PnP failed). The :class:`~vio.modules.tracked.Tracked` carrier threads the
frame + tracks down to ``PullPrior``, which swaps it for the
:class:`~vio.modules.primed.Primed` carrier (tracks + joined prior); the
:class:`~vio.modules.step.Step` carrier then threads the result through the rest
of the chain.

Joining two END-bearing inputs (``imucam.sample`` + ``frame.depth``) means the
module must see BOTH ENDs before draining: ``expected_ends = 2``.

``R_imu_cam`` (IMU->camera rotation) drives the gyro prior; ``accel_align`` is the
one-shot startup gravity reference (camera frame) capture measured, seeded here so
``EstimateMotion`` levels the initial attitude. Both may be ``None`` (pure vision
/ no usable IMU).

:class:`BackendModule`
----------------------
Wires the two backend steps (one file each) into a reactive module over
``keyframe``:

1. :class:`~vio.modules.run_ba.RunBA` -- submit the keyframe's track snapshot to
   the BA engine; forward any refined pose it returns.
2. :class:`~vio.modules.publish_refined.PublishRefined` -- emit it on
   ``pose.refined``.

The heavy solve runs behind a :class:`~vio.mathlib.engine.base.Engine`:
``worker=False`` (default, offline) runs it synchronously in-thread --
byte-identical to the old path; ``worker=True`` (live) runs it in a separate
process so it cannot hold the read loop's GIL (the fast-push undershoot fix). The
keyframe pose ``T_world_cam`` is inverted to the ``T_cw`` the BA map expects
inside ``RunBA``.
"""
from __future__ import annotations

import logging

import numpy as np

from vio.comms import Module, LocalPubSub, topics
from vio.comms.messages import END
from sky.front.frontend import FrontendConfig, KLTFrontend
from sky.front.odometry import OdometryConfig, RGBDVisualOdometry
from sky.backend.bundle import BAConfig
from sky.backend.windowed import WindowedConfig
from vio.mathlib.backend.vio_window import WindowedVIOConfig
from vio.mathlib.engine import make_ba_engine, make_vi_engine
from .preintegrate_prior import PreintegratePrior
from .track_features import TrackFeatures
from .publish_tracks import PublishTracks
from .align_gravity import AlignGravity
from .pull_prior import PullPrior
from .estimate_motion import EstimateMotion
from .correct_tilt import CorrectTilt
from .publish_inliers import PublishInliers
from .publish_gyrofuse import PublishGyroFuse
from .propagate_imu import PropagateImu
from .loop_inbox import LoopCorrectionInbox
from .publish_pose import PublishPose
from .publish_vo import PublishVo
from .emit_keyframe import EmitKeyframe
from .run_ba import RunBA
from .publish_refined import PublishRefined

LOG = logging.getLogger("vio.pipeline")


class OdometryModule(Module):
    def __init__(self, bus: LocalPubSub, K: np.ndarray,
                 R_imu_cam: np.ndarray | None = None,
                 accel_align: np.ndarray | None = None,
                 odom_cfg: OdometryConfig | None = None,
                 frontend_cfg: FrontendConfig | None = None,
                 kf_every: int = 5, use_gyro: bool = True,
                 latest_only: bool = False, level_tilt: bool = False,
                 publish_vo: bool = False, retain_imu: bool = False,
                 loop_correct: bool = False) -> None:
        super().__init__("odometry", bus, latest_only=latest_only)
        # ``frontend_cfg`` carries the resolution-scaled KLT + corner-detection
        # geometry (window/pyramid/corner budget, and at low res the block_size=3
        # + bucketed coverage levers). Defaulting to None keeps the historical
        # full-quality FrontendConfig() the offline byte-parity oracle relies on.
        fe = KLTFrontend(frontend_cfg) if frontend_cfg is not None else None
        self.ctx.state["vo"] = RGBDVisualOdometry(
            K, odom_cfg or OdometryConfig(), frontend=fe)
        self.ctx.state["kf_every"] = int(kf_every)
        self.ctx.state["use_gyro"] = bool(use_gyro)
        # Continuous at-rest roll/pitch leveling (CorrectTilt). LIVE-only so the
        # offline replay/scoring pose.odom stays byte-identical; the live builder
        # turns it on. Lets the live view self-level without a startup hold-still.
        self.ctx.state["level_tilt"] = bool(level_tilt)
        self.ctx.state["priors"] = {}
        # TIGHT path only: retain the per-frame raw IMU samples (camera frame) so
        # EmitKeyframe can hand the inter-keyframe block to the tight backend. The
        # default (False) keeps the LOOSE / oracle front-end byte-identical -- the
        # extra retention is a no-op (PreintegratePrior / EmitKeyframe gate on it).
        self.ctx.state["retain_imu"] = bool(retain_imu)
        if retain_imu:
            self.ctx.state["imu_segs"] = {}
            self.ctx.state["last_kf_seq"] = -1
            # Fixed world gravity ACCELERATION vector (optical-world "down" = +y),
            # matching WindowedVIOConfig.g_world. Used by PropagateImu's per-frame
            # forward-integration + ZUPT. Kept identical to the tight backend so the
            # live dead-reckoning and the keyframe nav-state share one gravity model.
            self.ctx.state["g_world"] = (0.0, 9.81, 0.0)
        # Closed-loop SLAM correction (LIVE + --tight only). When on, PropagateImu
        # bleeds the SLAM pose-graph correction (loop.correction) back into the live
        # nav-state so accumulated drift is BOUNDED on revisits. The correction
        # arrives on a DIFFERENT thread (the slam-endpoint IPC subscriber that
        # vio.main wires onto this local bus), so a thread-safe inbox hands it to
        # the odometry thread; PropagateImu drains it per frame. Gated so the
        # offline / oracle / loose path is byte-identical (loop_correct stays
        # False there -> no inbox, no subscription, no blend). Requires retain_imu
        # (the --tight nav-state); ignored otherwise.
        if loop_correct and retain_imu:
            inbox = LoopCorrectionInbox()
            self.ctx.state["loop_correct"] = True
            self.ctx.state["loop_inbox"] = inbox
            # Feed corrections from the local bus (vio.main republishes the slam
            # endpoint's loop.correction here) into the inbox. END is ignored.
            bus.subscribe(
                topics.LOOP_CORRECTION,
                lambda m: inbox.push(m) if m is not None and m is not END
                else None)
        self.ctx.state["R_imu_cam"] = (
            None if R_imu_cam is None else np.asarray(R_imu_cam, dtype=np.float64))
        if accel_align is not None:
            self.ctx.state["accel_align"] = np.asarray(accel_align, dtype=np.float64)
        self.expected_ends = 2          # imucam.sample + frame.depth both end
        self.on(topics.IMUCAM_SAMPLE, [PreintegratePrior()])
        # The frame chain. ``PublishVo`` is LIVE-only (publish_vo): it emits the
        # pure-vision ``pose.vo`` after EstimateMotion has advanced ``pose_vo``.
        # Off by default so the offline deterministic path never publishes it and
        # pose.odom byte-parity holds (mirrors the level_tilt opt-in). It runs
        # right after PublishPose; CorrectTilt only touches self.pose (never
        # pose_vo), so placing it after CorrectTilt does not affect the VO line.
        # ``PublishGyroFuse`` runs right after ``PublishInliers`` (both are
        # post-EstimateMotion diagnostic publishers). It self-skips on frames
        # where the gyro fusion did not run (gyro off / PnP failed / bootstrap),
        # so it is safe to always wire in -- it never emits a garbage frame.
        # ``PropagateImu`` (TIGHT path only, gated on retain_imu) runs right
        # before PublishPose: it forward-propagates the live nav-state with the
        # per-frame IMU and replaces ``step.pose`` so the live ``pose.odom`` keeps
        # moving via the IMU when vision is absent/weak, and applies a ZUPT at
        # rest. On the LOOSE path it is a pass-through no-op (byte-identical
        # pose.odom). It sits after CorrectTilt (so it re-anchors to the final
        # vision pose on keyframes) and before PublishPose / PublishVo (PublishVo
        # reads vo.pose_vo, not step.pose, so the pure-vision line is unaffected).
        frame_chain = [TrackFeatures(), PublishTracks(), AlignGravity(),
                       PullPrior(), EstimateMotion(), CorrectTilt(),
                       PublishInliers(), PublishGyroFuse(),
                       PropagateImu(), PublishPose()]
        if publish_vo:
            frame_chain.append(PublishVo())
        frame_chain.append(EmitKeyframe())
        self.on(topics.FRAME_DEPTH, frame_chain)
        fwd = [topics.POSE_ODOM, topics.KEYFRAME, topics.FRAME_TRACKS,
               topics.FRAME_INLIERS, topics.FRAME_GYROFUSE]
        if publish_vo:
            fwd.append(topics.POSE_VO)
        self.forwards_to(*fwd)


class BackendModule(Module):
    """Windowed back-end over the ``keyframe`` stream.

    Two selectable backends, picked by ``tight`` (a clean engine switch, NOT a
    pipeline fork):

    * ``tight=False`` (default, LOOSE) -- literally today's code path:
      :func:`make_ba_engine` builds the vision-only ``WindowedBAMap`` (reproj +
      depth + optional VO/gravity priors). Byte-identical to the pre-tight build;
      the offline oracle relies on this.
    * ``tight=True`` (``--tight``, opt-in) -- :func:`make_vi_engine` builds the
      tight-coupled ``WindowedVIOMap`` (the joint visual + IMU window optimiser
      from :mod:`vio.mathlib.backend.vio_window`). The IMU factor is weighted by
      the per-edge information square root (``imu_info_weight=True``, the
      covariance-correct Phase-1 weight) -- the live tight path the PLAN
      prescribes. ``RunBA`` then submits the SUPERSET snapshot (keyframe ts + raw
      inter-keyframe IMU block) instead of the loose at-rest accel.

    ``window`` / ``iters`` size the loose ``WindowedConfig``; the tight backend
    uses ``WindowedVIOConfig``'s own (validated) window + iteration defaults.
    """

    def __init__(self, bus: LocalPubSub, K: np.ndarray,
                 window: int = 6, iters: int = 5,
                 latest_only: bool = False, worker: bool = False,
                 tight: bool = False, stabilize_velocity: bool = False) -> None:
        super().__init__("backend", bus, latest_only=latest_only)
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
            self.engine = make_vi_engine(K, vio_cfg, worker=worker)
        else:
            cfg = WindowedConfig(window=window, ba=BAConfig(max_iters=iters))
            self.engine = make_ba_engine(K, cfg, worker=worker)
        self.ctx.state["engine"] = self.engine
        self.ctx.state["tight"] = bool(tight)    # RunBA picks the snapshot shape
        self.on(topics.KEYFRAME, [RunBA(), PublishRefined()])
        self.forwards_to(topics.POSE_REFINED)

    def run(self) -> None:
        # Close the engine on THIS thread when the loop exits (stop sentinel or
        # _stop), so a subprocess worker is reaped without a cross-thread race.
        try:
            super().run()
        finally:
            self.engine.close()
