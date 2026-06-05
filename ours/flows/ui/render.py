"""ui render flow: forward streamed poses to a callback (live viewer).

Where :class:`~ours.flows.ui.collector.UiCollectorFlow` records poses for offline
scoring, this sink hands each ``pose.odom`` message to an ``on_pose`` callback --
the bridge that drives the Qt 3D viewer
(:class:`~ours.ui.live_source.FlowPoseSource`). The single task lives in
:mod:`ours.flows.ui.render_pose`.
"""
from __future__ import annotations

from typing import Callable

from ..core import Flow, Bus, topics
from ..core.messages import PoseMsg
from .render_pose import RenderPose


class UiRenderFlow(Flow):
    """Sink flow that forwards each ``pose.odom`` to ``on_pose``."""

    def __init__(self, bus: Bus, on_pose: Callable[[PoseMsg], None]) -> None:
        super().__init__("ui", bus)
        self.ctx.state["on_pose"] = on_pose
        self.expected_ends = 1                       # only pose.odom carries END
        self.on(topics.POSE_ODOM, [RenderPose()])
