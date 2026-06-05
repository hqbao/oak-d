"""odometry flow: real-time RGB-D visual odometry (+ gyro prior).

Wires the odometry tasks (one file each) into a reactive flow:

* ``imu.sample`` -> [:class:`~ours.flows.odometry.route_imu.RouteImu`]
* ``frame.depth`` -> [:class:`~ours.flows.odometry.process_vo.ProcessVO`,
  :class:`~ours.flows.odometry.publish_pose.PublishPose`,
  :class:`~ours.flows.odometry.emit_keyframe.EmitKeyframe`]

The :class:`~ours.flows.odometry.step.Step` carrier threads one frame's result
between the frame-chain tasks. The capture flow owns the IMU->prior fusion, so
this flow is identical for replay and live and stays bit-for-bit aligned with the
offline ``vio_run`` f2f driver.
"""
from __future__ import annotations

import numpy as np

from ...lib.flow import Flow, Bus, topics
from ...lib.odometry.odometry import OdometryConfig, RGBDVisualOdometry
from .route_imu import RouteImu
from .process_vo import ProcessVO
from .publish_pose import PublishPose
from .emit_keyframe import EmitKeyframe


class OdometryFlow(Flow):
    def __init__(self, bus: Bus, K: np.ndarray,
                 odom_cfg: OdometryConfig | None = None,
                 kf_every: int = 5, use_gyro: bool = True) -> None:
        super().__init__("odometry", bus)
        self.ctx.state["vo"] = RGBDVisualOdometry(K, odom_cfg or OdometryConfig())
        self.ctx.state["kf_every"] = int(kf_every)
        self.ctx.state["use_gyro"] = bool(use_gyro)
        self.ctx.state["priors"] = {}
        self.on(topics.IMU_SAMPLE, [RouteImu()])
        self.on(topics.FRAME_DEPTH,
                [ProcessVO(), PublishPose(), EmitKeyframe()])
        self.forwards_to(topics.POSE_ODOM, topics.KEYFRAME)
