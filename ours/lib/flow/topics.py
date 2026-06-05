"""Canonical pub/sub topic names for the live ``ours`` pipeline.

Each constant is the string key used on the :class:`~ours.lib.flow.pubsub.Bus`.
Keeping them in one place documents the data flow between flows (edges below are
exactly the ``self.on(...)`` subscriptions wired in each ``*_flow.py``).

There is ONE acquisition front-end (camera-reader + imu-reader); both the depth
and odometry flows consume its synced ``imucam.sample``::

    cam-reader --cam.sync------> imu-reader      (trigger: stereo pair + ts)
    imu-reader --imu.raw-------> (visualiser)    (raw IMU for the interval)
    imu-reader --imucam.sample-> depth, odometry (frames + CALIBRATED IMU)
    depth      --frame.depth---> odometry
    odometry   --pose.odom-----> ui-collector, ui-render
    odometry   --keyframe------> backend, slam
    backend    --pose.refined--> ui-collector
    slam       --loop.correction-> ui-collector

``cam.sync`` carries the frames so the IMU flow packs them with the inertial
samples it drains up to the frame timestamp. For each trigger the IMU flow emits
the uncalibrated samples on ``imu.raw`` (honest, what the sensor reported) and
the frames bundled with the *calibrated* IMU on ``imucam.sample`` -- the
combined, time-synced unit consumers (depth, odometry, visualiser) subscribe to.
The odometry flow owns the IMU->prior fusion itself (no separate capture flow).

A backpressure CONTROL edge closes the loop on the live path only::

    odometry  --frame.done----> imu-reader   (one credit per admitted frame)

The imu-reader admits at most ``N`` frames in flight; ``frame.done`` (published
once per processed frame by the odometry tail) frees a credit. Over budget, the
imu-reader skips a camera frame at the source -- before the heavy packet is built
-- so the host backlog stays bounded and the device link never stalls. Replay
admits everything (deterministic). See ``flows/imu_reader/admission.py``.

Note: the back-end and SLAM flows both trigger off ``keyframe`` (NOT
``pose.odom``); the SLAM ``loop.correction`` is currently consumed only by the
UI collector -- it is not yet fed back into odometry (no closed loop on the live
pose path). Wire an odometry ``self.on(LOOP_CORRECTION, ...)`` when that
feedback is added.
"""
from __future__ import annotations

FRAME_DEPTH = "frame.depth"
POSE_ODOM = "pose.odom"
KEYFRAME = "keyframe"
POSE_REFINED = "pose.refined"
LOOP_CORRECTION = "loop.correction"

# Acquisition front-end (camera-reader <-> imu-reader).
CAM_SYNC = "cam.sync"
IMUCAM_SAMPLE = "imucam.sample"
# Raw IMU for each frame interval, published BEFORE calibration (honest, what
# the sensor reported). ``imucam.sample`` carries the CALIBRATED IMU.
IMU_RAW = "imu.raw"
# Backpressure CONTROL topic (not data): the odometry tail publishes a one-per-
# admitted-frame FrameDone here; the imu-reader's admission gate consumes it to
# free an in-flight credit. Live-only effect (replay admits everything); never
# END-forwarded, so it does not enter the graph's drain accounting.
FRAME_DONE = "frame.done"
