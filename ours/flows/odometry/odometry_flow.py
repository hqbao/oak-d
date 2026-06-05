"""odometry flow: real-time RGB-D visual odometry (+ gyro prior).

Wires the odometry tasks (one file each) into a reactive flow that joins the two
edges of the unified acquisition front-end:

* ``imucam.sample`` ->
  [:class:`~ours.flows.odometry.preintegrate_prior.PreintegratePrior`]
* ``frame.depth`` -> [:class:`~ours.flows.odometry.process_vo.ProcessVO`,
  :class:`~ours.flows.odometry.publish_pose.PublishPose`,
  :class:`~ours.flows.odometry.emit_keyframe.EmitKeyframe`,
  :class:`~ours.flows.odometry.signal_done.SignalDone`]

``SignalDone`` is the tail: it publishes one ``frame.done`` per processed frame so
the imu-reader's realtime admission gate frees the frame's in-flight credit (the
backpressure loop). It runs after ``EmitKeyframe`` (which now returns the ``Step``
so the chain continues).

Both come from the SAME stream (the imu-reader publishes ``imucam.sample`` and the
depth flow turns it into ``frame.depth``), so this flow owns the IMU->prior fusion
itself (``PreintegratePrior``) instead of a separate capture flow. The
:class:`~ours.flows.odometry.step.Step` carrier threads one frame's result between
the frame-chain tasks.

Joining two END-bearing inputs (``imucam.sample`` direct + ``frame.depth`` via
depth) means the flow must see BOTH ENDs before draining: ``expected_ends = 2``.

``R_imu_cam`` (IMU->camera rotation) drives the gyro prior; ``accel_align`` is the
one-shot startup gravity reference (camera frame) the front-end measured, seeded
here so ``ProcessVO`` levels the initial attitude. Both may be ``None`` (pure
vision / no usable IMU).
"""
from __future__ import annotations

import numpy as np

from ...lib.flow import Flow, Bus, topics
from ...lib.odometry.odometry import OdometryConfig, RGBDVisualOdometry
from .preintegrate_prior import PreintegratePrior
from .process_vo import ProcessVO
from .publish_pose import PublishPose
from .emit_keyframe import EmitKeyframe
from .signal_done import SignalDone


class OdometryFlow(Flow):
    def __init__(self, bus: Bus, K: np.ndarray,
                 R_imu_cam: np.ndarray | None = None,
                 accel_align: np.ndarray | None = None,
                 odom_cfg: OdometryConfig | None = None,
                 kf_every: int = 5, use_gyro: bool = True) -> None:
        super().__init__("odometry", bus)
        self.ctx.state["vo"] = RGBDVisualOdometry(K, odom_cfg or OdometryConfig())
        self.ctx.state["kf_every"] = int(kf_every)
        self.ctx.state["use_gyro"] = bool(use_gyro)
        self.ctx.state["priors"] = {}
        self.ctx.state["R_imu_cam"] = (
            None if R_imu_cam is None else np.asarray(R_imu_cam, dtype=np.float64))
        if accel_align is not None:
            self.ctx.state["accel_align"] = np.asarray(accel_align, dtype=np.float64)
        self.expected_ends = 2          # imucam.sample + frame.depth both end
        self.on(topics.IMUCAM_SAMPLE, [PreintegratePrior()])
        self.on(topics.FRAME_DEPTH,
                [ProcessVO(), PublishPose(), EmitKeyframe(), SignalDone()])
        self.forwards_to(topics.POSE_ODOM, topics.KEYFRAME)
