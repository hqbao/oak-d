"""``publish_correction`` task: emit the loop correction on ``loop.correction``."""
from __future__ import annotations

from ..core import topics
from ..core.messages import LoopCorrection
from ..core.task import Task


class PublishCorrection(Task):
    name = "publish_correction"

    def run(self, ctx, msg: LoopCorrection):
        ctx.bus.publish(topics.LOOP_CORRECTION, msg)
        return None
