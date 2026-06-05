"""``publish_correction`` task: emit the loop correction on ``loop.correction``."""
from __future__ import annotations

from ...lib.flow import topics
from ...lib.flow.messages import LoopCorrection
from ...lib.flow.task import Task


class PublishCorrection(Task):
    name = "publish_correction"

    def run(self, ctx, msg: LoopCorrection):
        ctx.bus.publish(topics.LOOP_CORRECTION, msg)
        return None
