"""``track_features`` task: run the KLT frontend on the frame's left image.

First task of the odometry frame-chain. It is the ONLY ``numba parallel=True``
section on the odometry thread (the KLT tracker), so it is the only place that
takes :data:`~vio.comms.runtime.NUMBA_PARALLEL_LOCK` -- serialising just
against the depth matcher (SGM) which runs the same numba layer on the imu_cam
thread. The downstream :class:`~vio.modules.estimate_motion.EstimateMotion`
(PnP + fusion) is pure NumPy and runs lock-free, so it overlaps the next frame's
SGM instead of blocking it.
"""
from __future__ import annotations

from vio.comms.messages import DepthFrame
from vio.comms.runtime import NUMBA_PARALLEL_LOCK
from vio.comms import Step
from sky.front.odometry import RGBDVisualOdometry
from .tracked import Tracked


class TrackFeatures(Step):
    name = "track_features"

    def run(self, ctx, msg: DepthFrame):
        vo: RGBDVisualOdometry = ctx.state["vo"]
        with NUMBA_PARALLEL_LOCK:        # KLT tracker uses numba parallel=True
            obs = vo.track(msg.gray_left)
        return Tracked(msg, obs)
