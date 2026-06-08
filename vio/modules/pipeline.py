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
actually trusted. The :class:`~vio.modules.tracked.Tracked` carrier threads the
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

import numpy as np

from vio.comms import Module, LocalPubSub, topics
from vio.mathlib.frontend.frontend import FrontendConfig, KLTFrontend
from vio.mathlib.odometry.odometry import OdometryConfig, RGBDVisualOdometry
from vio.mathlib.backend.bundle import BAConfig
from vio.mathlib.backend.windowed import WindowedConfig
from vio.mathlib.engine import make_ba_engine
from .preintegrate_prior import PreintegratePrior
from .track_features import TrackFeatures
from .publish_tracks import PublishTracks
from .align_gravity import AlignGravity
from .pull_prior import PullPrior
from .estimate_motion import EstimateMotion
from .correct_tilt import CorrectTilt
from .publish_inliers import PublishInliers
from .publish_pose import PublishPose
from .publish_vo import PublishVo
from .emit_keyframe import EmitKeyframe
from .run_ba import RunBA
from .publish_refined import PublishRefined


class OdometryModule(Module):
    def __init__(self, bus: LocalPubSub, K: np.ndarray,
                 R_imu_cam: np.ndarray | None = None,
                 accel_align: np.ndarray | None = None,
                 odom_cfg: OdometryConfig | None = None,
                 frontend_cfg: FrontendConfig | None = None,
                 kf_every: int = 5, use_gyro: bool = True,
                 latest_only: bool = False, level_tilt: bool = False,
                 publish_vo: bool = False) -> None:
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
        frame_chain = [TrackFeatures(), PublishTracks(), AlignGravity(),
                       PullPrior(), EstimateMotion(), CorrectTilt(),
                       PublishInliers(), PublishPose()]
        if publish_vo:
            frame_chain.append(PublishVo())
        frame_chain.append(EmitKeyframe())
        self.on(topics.FRAME_DEPTH, frame_chain)
        fwd = [topics.POSE_ODOM, topics.KEYFRAME, topics.FRAME_TRACKS,
               topics.FRAME_INLIERS]
        if publish_vo:
            fwd.append(topics.POSE_VO)
        self.forwards_to(*fwd)


class BackendModule(Module):
    def __init__(self, bus: LocalPubSub, K: np.ndarray,
                 window: int = 6, iters: int = 5,
                 latest_only: bool = False, worker: bool = False) -> None:
        super().__init__("backend", bus, latest_only=latest_only)
        cfg = WindowedConfig(window=window, ba=BAConfig(max_iters=iters))
        self.engine = make_ba_engine(K, cfg, worker=worker)
        self.ctx.state["engine"] = self.engine
        self.on(topics.KEYFRAME, [RunBA(), PublishRefined()])
        self.forwards_to(topics.POSE_REFINED)

    def run(self) -> None:
        # Close the engine on THIS thread when the loop exits (stop sentinel or
        # _stop), so a subprocess worker is reaped without a cross-thread race.
        try:
            super().run()
        finally:
            self.engine.close()
