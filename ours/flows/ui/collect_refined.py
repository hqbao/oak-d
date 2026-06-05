"""``collect_refined`` task: record each ``pose.refined`` position by sequence."""
from __future__ import annotations

from ...lib.flow.messages import PoseMsg
from ...lib.flow.task import Task


class CollectRefined(Task):
    name = "collect_refined"

    def run(self, ctx, msg: PoseMsg):
        ctx.state["refined"][msg.seq] = msg.T_world_cam[:3, 3].copy()
        return None
