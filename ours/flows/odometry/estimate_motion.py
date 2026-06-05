"""``estimate_motion`` task: gravity-align, pull the gyro prior, RGB-D PnP.

Second task of the odometry frame-chain. Consumes the
:class:`~ours.flows.odometry.tracked.Tracked` carrier produced by
:class:`~ours.flows.odometry.track_features.TrackFeatures` (the frame + its KLT
tracks) and runs the pure-NumPy motion estimate
(:meth:`~ours.lib.odometry.odometry.RGBDVisualOdometry.estimate`) -- build
correspondences -> RGB-D PnP -> optional gyro fusion -> pose compose. No numba
parallel region runs here, so unlike ``TrackFeatures`` it takes no parallel lock.
"""
from __future__ import annotations

from ...lib.flow.messages import ImuPrior
from ...lib.flow.task import Task
from ...lib.odometry.odometry import RGBDVisualOdometry
from .step import Step
from .tracked import Tracked


class EstimateMotion(Task):
    name = "estimate_motion"

    def run(self, ctx, tracked: Tracked):
        vo: RGBDVisualOdometry = ctx.state["vo"]
        frame = tracked.frame
        if not ctx.state.get("aligned") and "accel_align" in ctx.state:
            vo.align_to_gravity(ctx.state["accel_align"])
            ctx.state["aligned"] = True
        prior: ImuPrior | None = ctx.state["priors"].pop(frame.seq, None)
        R_prior = prior.R_prior if prior is not None else None
        pose = vo.estimate(tracked.obs, frame.depth_m, R_prior=R_prior)
        accel_cam = prior.accel_cam if prior is not None else None
        at_rest = prior.at_rest if prior is not None else False
        return Step(frame, pose.copy(), dict(vo.last_info), accel_cam, at_rest)
