"""``publish_depth`` task: emit the computed depth frame on ``frame.depth``."""
from __future__ import annotations

from ..core import topics
from ..core.messages import DepthFrame
from ..core.task import Task


class PublishDepth(Task):
    name = "publish_depth"

    def run(self, ctx, msg: DepthFrame):
        ctx.bus.publish(topics.FRAME_DEPTH, msg)
        return None
