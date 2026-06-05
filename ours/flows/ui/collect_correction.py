"""``collect_correction`` task: append each ``loop.correction`` to the buffer."""
from __future__ import annotations

from ..core.messages import LoopCorrection
from ..core.task import Task


class CollectCorrection(Task):
    name = "collect_correction"

    def run(self, ctx, msg: LoopCorrection):
        ctx.state["corrections"].append(msg)
        return None
