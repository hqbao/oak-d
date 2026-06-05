"""``publish_depth`` task: emit the computed depth frame on ``frame.depth``."""
from __future__ import annotations

from ...lib.flow import topics
from ...lib.flow.messages import DepthFrame
from ...lib.flow.task import Task


class PublishDepth(Task):
    name = "publish_depth"

    def run(self, ctx, msg: DepthFrame):
        ctx.bus.publish(topics.FRAME_DEPTH, msg)
        return None
