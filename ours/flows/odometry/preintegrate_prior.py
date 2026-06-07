"""``preintegrate_prior`` task: build this frame's IMU prior from the packet.

Replaces the old ``route_imu`` task. In the unified front-end the odometry flow
consumes the SAME synced :class:`~ours.lib.flow.messages.ImuCamPacket` the imu_cam
flow's depth task does (one acquisition stream, no separate capture monolith). The
IMU->prior fusion that used to live in the capture flow now lives HERE, per packet:

* ``R_prior`` -- the inter-frame camera-frame rotation integrated from the
  packet's gyro (already bias-corrected by the imu_cam flow's ApplyCalibration),
  conjugated into the camera frame by ``R_imu_cam``. ``None`` when gyro fusion is
  off or the packet carries fewer than two samples (e.g. the first frame).
* ``accel_cam`` / ``at_rest`` -- the camera-frame accelerometer this frame and a
  stillness flag, so a keyframe can carry a gravity prior into the back-end.

The result is stashed in ``priors[seq]`` so the matching depth frame
(``EstimateMotion``) picks it up by ``seq``.
"""
from __future__ import annotations

import numpy as np

from ...lib.flow.messages import ImuCamPacket, ImuPrior
from ...lib.flow.task import Task
from ...lib.imu.imu import integrate_gyro_camera

# Stillness gate for the keyframe gravity prior: low angular rate and accel close
# to 1 g. Mirrors the live capture flow's at-rest thresholds.
_STILL_GYRO = 0.15      # rad/s
_GRAVITY = 9.81         # m/s^2
_STILL_ACCEL_DEV = 0.6  # m/s^2 deviation of |accel| from gravity
# Loose "definitely moving" gate for the IMU-vetoed low-inlier freeze (see
# OdometryConfig.min_inliers_for_translation): higher than the still gates so the
# middle band stays "neither". 0.3 rad/s ~ 17 deg/s (any real hand turn); 0.5
# m/s^2 ~ 5 % of g (any real linear push).
_MOVING_GYRO = 0.3      # rad/s
_MOVING_ACCEL_DEV = 0.5  # m/s^2 deviation of |accel| from gravity


class PreintegratePrior(Task):
    name = "preintegrate_prior"

    def run(self, ctx, msg: ImuCamPacket):
        R_imu_cam = ctx.state.get("R_imu_cam")
        use_gyro = ctx.state.get("use_gyro", True)

        gyro = np.asarray(msg.gyro, dtype=np.float64)
        accel = np.asarray(msg.accel, dtype=np.float64)

        R_prior = None
        if use_gyro and R_imu_cam is not None and gyro.shape[0] >= 2:
            R_prior = integrate_gyro_camera(msg.imu_ts, gyro, R_imu_cam)

        accel_cam = None
        at_rest = False
        imu_moving = False
        if R_imu_cam is not None and accel.size:
            a_mean = accel.mean(axis=0)
            accel_cam = np.asarray(R_imu_cam, dtype=np.float64) @ a_mean
            gyro_mag = (0.0 if gyro.size == 0
                        else float(np.linalg.norm(gyro, axis=1).mean()))
            accel_dev = abs(float(np.linalg.norm(a_mean)) - _GRAVITY)
            gyro_still = (gyro.size == 0 or gyro_mag < _STILL_GYRO)
            accel_still = accel_dev < _STILL_ACCEL_DEV
            at_rest = bool(gyro_still and accel_still)
            # Loose "definitely moving" flag: vetoes the low-inlier translation
            # freeze when motion blur (not a textureless wall) is starving PnP.
            imu_moving = bool(gyro_mag > _MOVING_GYRO
                              or accel_dev > _MOVING_ACCEL_DEV)

        ctx.state["priors"][msg.seq] = ImuPrior(
            msg.seq, R_prior, accel_cam, at_rest, imu_moving)
        # Safety cap: in the normal full-fidelity path each prior is popped by the
        # matching depth frame, so the dict stays ~1 entry and this never fires.
        # Under a latest-only visualiser graph, frames whose depth was coalesced
        # away leave their prior un-popped; drop the oldest so the dict can't grow
        # without bound over a long live session.
        priors = ctx.state["priors"]
        if len(priors) > 256:
            for seq in sorted(priors)[:-256]:
                priors.pop(seq, None)
        return None
