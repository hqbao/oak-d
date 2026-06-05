"""Canonical pub/sub topic names for the live ``ours`` pipeline.

Each constant is the string key used on the :class:`~ours.lib.flow.pubsub.Bus`.
Keeping them in one place documents the data flow between flows:

    capture  --frame.raw-->  depth
    capture  --imu.sample--> odometry
    depth    --frame.depth-> odometry
    odometry --pose.odom---> ui, backend
    odometry --keyframe----> backend, slam
    backend  --pose.refined-> ui
    slam     --loop.correction-> ui, odometry
"""
from __future__ import annotations

FRAME_RAW = "frame.raw"
IMU_SAMPLE = "imu.sample"
FRAME_DEPTH = "frame.depth"
POSE_ODOM = "pose.odom"
KEYFRAME = "keyframe"
POSE_REFINED = "pose.refined"
LOOP_CORRECTION = "loop.correction"
