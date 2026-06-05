"""slam flow: loop closure SLAM.

Wires the two slam tasks (one file each) into a reactive flow over ``keyframe``:

1. :class:`~ours.flows.slam.slam_step.SlamStep` -- add the keyframe; on a
   confirmed loop, optimise the pose graph and forward the rewritten poses.
2. :class:`~ours.flows.slam.publish_correction.PublishCorrection` -- emit it on
   ``loop.correction``.
"""
from __future__ import annotations

from ..core import Flow, Bus, topics
from ...lib.loop.slam import SlamConfig, SlamMap
from .slam_step import SlamStep
from .publish_correction import PublishCorrection


class SlamFlow(Flow):
    def __init__(self, bus: Bus, K, cfg: SlamConfig | None = None) -> None:
        super().__init__("slam", bus)
        self.ctx.state["slam"] = SlamMap(K, cfg or SlamConfig())
        self.on(topics.KEYFRAME, [SlamStep(), PublishCorrection()])
        self.forwards_to(topics.LOOP_CORRECTION)

    @property
    def slam(self) -> SlamMap:
        return self.ctx.state["slam"]
