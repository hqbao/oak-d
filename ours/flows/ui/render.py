"""ui render flow: forward streamed poses to a callback (live viewer).

Where :class:`~ours.flows.ui.collector.UiCollectorFlow` records poses for offline
scoring, this sink hands each ``pose.odom`` message to an ``on_pose`` callback --
the bridge that drives the Qt 3D viewer
(:class:`~ours.ui.live_source.FlowPoseSource`).
"""
from __future__ import annotations

from typing import Callable

from ...lib import topics
from ...lib.flow import Flow
from ...lib.messages import PoseMsg
from ...lib.pubsub import Bus
from ...lib.task import Task


class _Render(Task):
    name = "render"

    def run(self, ctx, msg: PoseMsg):
        ctx.state["on_pose"](msg)
        return None


class UiRenderFlow(Flow):
    """Sink flow that forwards each ``pose.odom`` to ``on_pose``."""

    def __init__(self, bus: Bus, on_pose: Callable[[PoseMsg], None]) -> None:
        super().__init__("ui", bus)
        self.ctx.state["on_pose"] = on_pose
        self.expected_ends = 1                       # only pose.odom carries END
        self.on(topics.POSE_ODOM, [_Render()])
