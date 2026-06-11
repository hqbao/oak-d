"""``align_gravity`` task: one-shot startup gravity alignment.

Third task of the odometry frame-chain (after :class:`TrackFeatures` /
:class:`PublishTracks`). It is a one-shot bootstrap, not a per-frame solve: the
first time the front-end's startup gravity reference (``accel_align``, camera
frame) is available it levels the initial attitude via
:meth:`~sky.front.odometry.RGBDVisualOdometry.align_to_gravity`, then
never fires again. Pulled out of :class:`EstimateMotion` so the per-frame motion
solve carries no init branch. Passes the :class:`Tracked` carrier through
unchanged. A no-op when there is no usable IMU (no ``accel_align``).
"""
from __future__ import annotations

from vio.comms import Step
from sky.front.odometry import RGBDVisualOdometry
from .tracked import Tracked


class AlignGravity(Step):
    name = "align_gravity"

    def run(self, ctx, tracked: Tracked):
        if not ctx.state.get("aligned") and "accel_align" in ctx.state:
            vo: RGBDVisualOdometry = ctx.state["vo"]
            vo.align_to_gravity(ctx.state["accel_align"])
            ctx.state["aligned"] = True
        return tracked
