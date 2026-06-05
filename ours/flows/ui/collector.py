"""ui collector flow implementation.

Records the trajectory streamed by the pipeline. Three tiny tasks, one per
subscribed topic, each stash their message into the flow's public buffers:

* ``odom``        -- ``{seq: position}`` from ``pose.odom``
* ``refined``     -- ``{seq: position}`` from ``pose.refined``
* ``corrections`` -- list of :class:`~ours.lib.flow.messages.LoopCorrection`
"""
from __future__ import annotations

from ...lib.flow import topics
from ...lib.flow import Flow
from ...lib.flow.messages import LoopCorrection, PoseMsg
from ...lib.flow.pubsub import Bus
from ...lib.flow.task import Task


class _CollectOdom(Task):
    name = "collect_odom"

    def run(self, ctx, msg: PoseMsg):
        ctx.state["odom"][msg.seq] = msg.T_world_cam[:3, 3].copy()
        return None


class _CollectRefined(Task):
    name = "collect_refined"

    def run(self, ctx, msg: PoseMsg):
        ctx.state["refined"][msg.seq] = msg.T_world_cam[:3, 3].copy()
        return None


class _CollectCorrection(Task):
    name = "collect_correction"

    def run(self, ctx, msg: LoopCorrection):
        ctx.state["corrections"].append(msg)
        return None


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
        self.on(topics.POSE_ODOM, [_CollectOdom()])
        self.on(topics.POSE_REFINED, [_CollectRefined()])
        self.on(topics.LOOP_CORRECTION, [_CollectCorrection()])
