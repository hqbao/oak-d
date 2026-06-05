"""``collect_correction`` task: append each ``loop.correction`` to the buffer."""
from __future__ import annotations

from ...lib.flow.messages import LoopCorrection
from ...lib.flow.task import Task


class CollectCorrection(Task):
    name = "collect_correction"

    def run(self, ctx, msg: LoopCorrection):
        ctx.state["corrections"].append(msg)
        return None
