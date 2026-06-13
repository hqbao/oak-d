"""``estimate_motion`` step: RGB-D PnP (+ gyro fusion) -> per-frame Step.

Fifth step of the odometry frame-chain. Consumes the :class:`Primed` carrier
(frame + KLT tracks + the IMU prior already joined by :func:`pull_prior`) and runs
the pure-NumPy motion estimate
(:meth:`~sky.front.odometry.RGBDVisualOdometry.estimate`) -- build
correspondences -> RGB-D PnP -> optional gyro fusion (when the prior carries an
``R_prior``) -> pose compose -- then packages the result, plus the prior's
``accel_cam`` / ``at_rest``, into a :class:`Step` for the downstream steps. The
one-shot gravity bootstrap (:func:`align_gravity`) and the prior join
(:func:`pull_prior`) now run upstream, so this step is just the solve. No numba
parallel region runs here, so unlike ``track_features`` it takes no parallel lock.
"""
from __future__ import annotations

from sky.front.odometry import RGBDVisualOdometry
from .primed import Primed
from .step import Step


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
