"""``track_features`` task: run the KLT frontend on the frame's left image.

First task of the odometry frame-chain. It is the ONLY ``numba parallel=True``
section on the odometry thread (the KLT tracker), so it is the only place that
takes :data:`~ours.lib.flow.runtime.NUMBA_PARALLEL_LOCK` -- serialising just
against the depth matcher (SGM) which runs the same numba layer on the imu_cam
thread. The downstream :class:`~ours.flows.odometry.estimate_motion.EstimateMotion`
(PnP + fusion) is pure NumPy and runs lock-free, so it overlaps the next frame's
SGM instead of blocking it.
"""
from __future__ import annotations

from ...lib.flow.messages import DepthFrame
from ...lib.flow.runtime import NUMBA_PARALLEL_LOCK
from ...lib.flow.task import Task
from ...lib.odometry.odometry import RGBDVisualOdometry
from .tracked import Tracked


class TrackFeatures(Task):
    name = "track_features"

    def run(self, ctx, msg: DepthFrame):
        vo: RGBDVisualOdometry = ctx.state["vo"]
        with NUMBA_PARALLEL_LOCK:        # KLT tracker uses numba parallel=True
            obs = vo.track(msg.gray_left)
        return Tracked(msg, obs)
