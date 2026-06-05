"""ui collector flow: record the streamed trajectory for offline scoring.

Wires three tiny tasks (one file each), one per subscribed topic, each stashing
its message into the flow's public buffers:

* :class:`~ours.flows.ui.collect_odom.CollectOdom`             -- ``pose.odom``
* :class:`~ours.flows.ui.collect_refined.CollectRefined`       -- ``pose.refined``
* :class:`~ours.flows.ui.collect_correction.CollectCorrection` -- ``loop.correction``
"""
from __future__ import annotations

from ...lib.flow import Flow, Bus, topics
from ...lib.flow.messages import LoopCorrection
from .collect_odom import CollectOdom
from .collect_refined import CollectRefined
from .collect_correction import CollectCorrection


class UiCollectorFlow(Flow):
    """Sink flow that records the streamed trajectory for offline scoring."""

    def __init__(self, bus: Bus) -> None:
        super().__init__("ui", bus)
        self.odom: dict[int, "object"] = {}
        self.refined: dict[int, "object"] = {}
        self.corrections: list[LoopCorrection] = []
        self.ctx.state["odom"] = self.odom
        self.ctx.state["refined"] = self.refined
        self.ctx.state["corrections"] = self.corrections
        self.expected_ends = 3       # pose.odom + pose.refined + loop.correction
        self.on(topics.POSE_ODOM, [CollectOdom()])
        self.on(topics.POSE_REFINED, [CollectRefined()])
        self.on(topics.LOOP_CORRECTION, [CollectCorrection()])
