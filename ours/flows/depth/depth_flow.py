"""depth flow: SGM dense depth.

Wires the two depth tasks (one file each) into a reactive flow:

1. :class:`~ours.flows.depth.compute_depth.ComputeDepth` -- SGM on the stereo pair.
2. :class:`~ours.flows.depth.publish_depth.PublishDepth`  -- emit ``frame.depth``.
"""
from __future__ import annotations

from ...lib.flow import Flow, Bus, topics
from ...lib.stereo.stereo import SGMStereoMatcher
from .compute_depth import ComputeDepth
from .publish_depth import PublishDepth


class DepthFlow(Flow):
    def __init__(self, bus: Bus, matcher: SGMStereoMatcher) -> None:
        super().__init__("depth", bus)
        self.ctx.state["matcher"] = matcher
        self.on(topics.FRAME_RAW, [ComputeDepth(), PublishDepth()])
        self.forwards_to(topics.FRAME_DEPTH)
