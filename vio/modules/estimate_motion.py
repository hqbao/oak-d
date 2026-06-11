"""``estimate_motion`` task: RGB-D PnP (+ gyro fusion) -> per-frame Step.

Fifth task of the odometry frame-chain. Consumes the :class:`Primed` carrier
(frame + KLT tracks + the IMU prior already joined by :class:`PullPrior`) and runs
the pure-NumPy motion estimate
(:meth:`~sky.front.odometry.RGBDVisualOdometry.estimate`) -- build
correspondences -> RGB-D PnP -> optional gyro fusion (when the prior carries an
``R_prior``) -> pose compose -- then packages the result, plus the prior's
``accel_cam`` / ``at_rest``, into a :class:`Step` for the downstream tasks. The
one-shot gravity bootstrap (:class:`AlignGravity`) and the prior join
(:class:`PullPrior`) now run upstream, so this task is just the solve. No numba
parallel region runs here, so unlike ``TrackFeatures`` it takes no parallel lock.
"""
from __future__ import annotations

from vio.comms import Step as StepBase
from sky.front.odometry import RGBDVisualOdometry
from .primed import Primed
from .step import Step


class EstimateMotion(StepBase):
    name = "estimate_motion"

    def run(self, ctx, primed: Primed):
        vo: RGBDVisualOdometry = ctx.state["vo"]
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
