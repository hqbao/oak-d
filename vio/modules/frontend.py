"""Sparse visual VO frontend: KLT track -> RGB-D PnP (+ gyro fusion).

The two pure-vision steps of the odometry frame-chain, in pipeline order:

* :func:`track_features` -- the KLT frontend on the frame's left image. The ONLY
  ``numba parallel=True`` section on the odometry thread, so the only place that
  takes :data:`~vio.comms.runtime.NUMBA_PARALLEL_LOCK` (serialising just against
  the depth matcher / SGM on the imu_cam thread).
* :func:`estimate_motion` -- the pure-NumPy motion solve (RGB-D PnP + optional
  gyro fusion + pose compose). Runs lock-free so it overlaps the next frame's SGM.

The one-shot gravity bootstrap + the IMU prior join live upstream in
:mod:`vio.modules.imu_prior`; this file is just the visual solve.
"""
from __future__ import annotations

from sky.front.odometry import RGBDVisualOdometry

from vio.comms.messages import DepthFrame
from vio.comms.runtime import NUMBA_PARALLEL_LOCK
from .carriers import Primed, Step, Tracked


def track_features(vo: RGBDVisualOdometry, frame: DepthFrame) -> Tracked:
    """KLT-track the frame's left image; return the :class:`Tracked` carrier.

    Was ``TrackFeatures(Step)``; the odometry instance is passed explicitly
    instead of read off ``ctx.state["vo"]``.
    """
    with NUMBA_PARALLEL_LOCK:            # KLT tracker uses numba parallel=True
        obs = vo.track(frame.gray_left)
    return Tracked(frame, obs)


def estimate_motion(vo: RGBDVisualOdometry, primed: Primed) -> Step:
    """RGB-D PnP (+ gyro fusion) -> the per-frame :class:`Step` carrier.

    Was ``EstimateMotion(StepBase)``; the odometry instance is passed explicitly
    instead of read off ``ctx.state["vo"]``.
    """
    prior = primed.prior
    R_prior = prior.R_prior if prior is not None else None
    # Pass the IMU "definitely moving" flag so the low-inlier translation
    # freeze (textureless wall) is vetoed during a real motion-blurred shake.
    # See OdometryConfig.min_inliers_for_translation + ImuPrior.imu_moving.
    imu_moving = prior.imu_moving if prior is not None else False
    pose = vo.estimate(primed.obs, primed.frame.depth_m, R_prior=R_prior,
                       imu_moving=imu_moving)
    accel_cam = prior.accel_cam if prior is not None else None
    at_rest = prior.at_rest if prior is not None else False
    return Step(primed.frame, pose.copy(), dict(vo.last_info),
                accel_cam, at_rest)
