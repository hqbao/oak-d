"""``render_pose`` task: hand each ``pose.odom`` to the viewer callback."""
from __future__ import annotations

from ...lib.flow.messages import PoseMsg
from ...lib.flow.task import Task


class RenderPose(Task):
    name = "render"

    def run(self, ctx, msg: PoseMsg):
        ctx.state["on_pose"](msg)
        return None
