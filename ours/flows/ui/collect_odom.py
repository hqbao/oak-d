"""``collect_odom`` task: record each ``pose.odom`` position by sequence."""
from __future__ import annotations

from ..core.messages import PoseMsg
from ..core.task import Task


class CollectOdom(Task):
    name = "collect_odom"

    def run(self, ctx, msg: PoseMsg):
        ctx.state["odom"][msg.seq] = msg.T_world_cam[:3, 3].copy()
        return None
