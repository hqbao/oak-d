"""``route_imu`` task: stash the IMU chunk (gravity-align accel + gyro priors).

Routed by message type on ``imu.sample``:

* :class:`~ours.flows.core.messages.ImuInit`  -- the startup gravity-align accel.
* :class:`~ours.flows.core.messages.ImuPrior` -- this frame's gyro rotation prior,
  keyed by ``seq`` so the matching depth frame can pick it up.
"""
from __future__ import annotations

from ..core.messages import ImuInit, ImuPrior
from ..core.task import Task


class RouteImu(Task):
    name = "route_imu"

    def run(self, ctx, msg):
        if isinstance(msg, ImuInit):
            if msg.accel_align is not None:
                ctx.state["accel_align"] = msg.accel_align
        elif isinstance(msg, ImuPrior):
            ctx.state["priors"][msg.seq] = msg
        return None
