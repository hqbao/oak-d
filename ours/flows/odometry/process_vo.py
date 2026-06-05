"""``process_vo`` task: gravity-align, pull the gyro prior, run RGB-D PnP."""
from __future__ import annotations

from ...lib.flow.messages import DepthFrame, ImuPrior
from ...lib.flow.runtime import NUMBA_PARALLEL_LOCK
from ...lib.flow.task import Task
from ...lib.odometry.odometry import RGBDVisualOdometry
from .step import Step


class ProcessVO(Task):
    name = "process_vo"

    def run(self, ctx, msg: DepthFrame):
        vo: RGBDVisualOdometry = ctx.state["vo"]
        if not ctx.state.get("aligned") and "accel_align" in ctx.state:
            vo.align_to_gravity(ctx.state["accel_align"])
            ctx.state["aligned"] = True
        prior: ImuPrior | None = ctx.state["priors"].pop(msg.seq, None)
        R_prior = prior.R_prior if prior is not None else None
        with NUMBA_PARALLEL_LOCK:        # KLT tracker uses numba parallel=True
            pose = vo.process(msg.gray_left, msg.depth_m, R_prior=R_prior)
        accel_cam = prior.accel_cam if prior is not None else None
        at_rest = prior.at_rest if prior is not None else False
        return Step(msg, pose.copy(), dict(vo.last_info), accel_cam, at_rest)
