"""odometry flow implementation.

IMU chain (topic ``imu.sample``), routed by message type:

* :class:`~ours.lib.flow.messages.ImuInit`  -- stash the startup gravity-align accel.
* :class:`~ours.lib.flow.messages.ImuPrior` -- stash this frame's gyro rotation prior
  (keyed by ``seq``) so the matching depth frame can pick it up.

Frame chain (per ``frame.depth``):

1. ``_ProcessVO``     -- gravity-align on the first frame, pull the stashed gyro
   rotation prior for this ``seq``, run RGB-D PnP odometry.
2. ``_PublishPose``   -- publish the resulting pose on ``pose.odom``.
3. ``_EmitKeyframe``  -- every ``kf_every`` frames, publish a ``keyframe``
   carrying the pose, image, depth, the current track snapshot and (when the
   camera was at rest) the gravity accel for the back-end.

The capture flow owns the IMU->prior fusion, so this flow is identical for replay
and live and stays bit-for-bit aligned with the offline ``vio_run`` f2f driver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...lib.flow import topics
from ...lib.flow import Flow
from ...lib.flow.messages import DepthFrame, ImuInit, ImuPrior, Keyframe, PoseMsg
from ...lib.odometry.odometry import OdometryConfig, RGBDVisualOdometry
from ...lib.flow.pubsub import Bus
from ...lib.flow.runtime import NUMBA_PARALLEL_LOCK
from ...lib.flow.task import Task


@dataclass
class _Step:
    """Internal carrier threading one frame's result through the task chain."""

    frame: DepthFrame
    pose: np.ndarray
    info: dict
    accel_cam: np.ndarray | None
    at_rest: bool


class _RouteImu(Task):
    name = "route_imu"

    def run(self, ctx, msg):
        if isinstance(msg, ImuInit):
            if msg.accel_align is not None:
                ctx.state["accel_align"] = msg.accel_align
        elif isinstance(msg, ImuPrior):
            ctx.state["priors"][msg.seq] = msg
        return None


class _ProcessVO(Task):
    name = "process_vo"

    def run(self, ctx, msg: DepthFrame):
        vo: RGBDVisualOdometry = ctx.state["vo"]
        if not ctx.state.get("aligned") and "accel_align" in ctx.state:
            vo.align_to_gravity(ctx.state["accel_align"])
            ctx.state["aligned"] = True
        prior: ImuPrior | None = ctx.state["priors"].pop(msg.seq, None)
        R_prior = prior.R_prior if prior is not None else None
        with NUMBA_PARALLEL_LOCK:        # KLT tracker uses numba parallel=True
            pose = vo.process(msg.gray_left, msg.depth_m, R_prior=R_prior)
        accel_cam = prior.accel_cam if prior is not None else None
        at_rest = prior.at_rest if prior is not None else False
        return _Step(msg, pose.copy(), dict(vo.last_info), accel_cam, at_rest)


class _PublishPose(Task):
    name = "publish_pose"

    def run(self, ctx, step: _Step):
        ctx.bus.publish(topics.POSE_ODOM,
                        PoseMsg(step.frame.seq, step.frame.ts_ns,
                                step.pose, step.info))
        return step


class _EmitKeyframe(Task):
    name = "emit_keyframe"

    def run(self, ctx, step: _Step):
        n = ctx.state.get("kf_count", 0) + 1
        if n < ctx.state["kf_every"]:
            ctx.state["kf_count"] = n
            return None
        ctx.state["kf_count"] = 0
        vo: RGBDVisualOdometry = ctx.state["vo"]
        tr = vo.frontend.tracks
        ids = tr.ids.copy() if tr is not None and tr.ids is not None else None
        px = tr.points.copy() if tr is not None and tr.points is not None else None
        # Only hand the back-end a gravity measurement when the camera is at rest
        # (a moving keyframe's lateral acceleration would bias the gravity dir).
        accel = step.accel_cam if step.at_rest else None
        ctx.bus.publish(topics.KEYFRAME,
                        Keyframe(step.frame.seq, step.pose,
                                 step.frame.gray_left, step.frame.depth_m,
                                 track_ids=ids, track_px=px, accel=accel))
        return None


class OdometryFlow(Flow):
    def __init__(self, bus: Bus, K: np.ndarray,
                 odom_cfg: OdometryConfig | None = None,
                 kf_every: int = 5, use_gyro: bool = True) -> None:
        super().__init__("odometry", bus)
        self.ctx.state["vo"] = RGBDVisualOdometry(K, odom_cfg or OdometryConfig())
        self.ctx.state["kf_every"] = int(kf_every)
        self.ctx.state["use_gyro"] = bool(use_gyro)
        self.ctx.state["priors"] = {}
        self.on(topics.IMU_SAMPLE, [_RouteImu()])
        self.on(topics.FRAME_DEPTH,
                [_ProcessVO(), _PublishPose(), _EmitKeyframe()])
        self.forwards_to(topics.POSE_ODOM, topics.KEYFRAME)
