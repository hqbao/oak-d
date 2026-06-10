"""``preintegrate_prior`` task: build this frame's IMU prior from the packet.

Replaces the old ``route_imu`` task. In the unified front-end the odometry flow
consumes the SAME synced :class:`~vio.comms.messages.ImuCamPacket` the imu_cam
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

from vio.comms.messages import ImuCamPacket, ImuPrior
from vio.comms import Step
from vio.mathlib.imu.imu import integrate_gyro_camera

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


class PreintegratePrior(Step):
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

        # TIGHT path only: retain this interval's RAW IMU samples (rotated into
        # the camera optical frame) keyed by frame seq, so EmitKeyframe can hand
        # the inter-keyframe IMU block to the tight backend's preintegrator. This
        # is a no-op for the LOOSE / oracle path (``retain_imu`` defaults False),
        # so it never allocates on or perturbs the byte-identical loose path. The
        # tight backend's body frame == camera optical frame, hence the per-sample
        # ``R_imu_cam @ v`` rotation here (the gyro prior above already uses the
        # same extrinsic via integrate_gyro_camera).
        if ctx.state.get("retain_imu") and R_imu_cam is not None:
            R_ic = np.asarray(R_imu_cam, dtype=np.float64)
            imu_segs = ctx.state["imu_segs"]
            if gyro.shape[0]:
                gyro_cam = gyro @ R_ic.T          # (M,3) IMU-frame -> camera frame
                accel_arr = accel @ R_ic.T
                imu_segs[msg.seq] = (
                    np.asarray(msg.imu_ts, dtype=np.int64).copy(),
                    gyro_cam.copy(), accel_arr.copy())
            else:
                imu_segs[msg.seq] = (np.zeros(0, np.int64),
                                     np.zeros((0, 3)), np.zeros((0, 3)))
            # Same bounded-growth cap as ``priors`` -- EmitKeyframe pops the
            # consumed segments, but a coalesced latest-only graph can leave gaps.
            if len(imu_segs) > 512:
                for seq in sorted(imu_segs)[:-512]:
                    imu_segs.pop(seq, None)
        return None
