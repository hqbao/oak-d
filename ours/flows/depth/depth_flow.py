"""depth flow implementation: SGM dense depth.

Tasks (run sequentially per ``frame.raw``):

1. ``_ComputeDepth`` -- run the SGM matcher on (rectified left, raw right).
2. ``_PublishDepth``  -- publish the :class:`~ours.lib.flow.messages.DepthFrame`.
"""
from __future__ import annotations

from ...lib.flow import topics
from ...lib.flow import Flow
from ...lib.flow.messages import DepthFrame, RawFrame
from ...lib.flow.pubsub import Bus
from ...lib.flow.runtime import NUMBA_PARALLEL_LOCK
from ...lib.stereo.stereo import SGMStereoMatcher
from ...lib.flow.task import Task


class _ComputeDepth(Task):
    name = "compute_depth"

    def run(self, ctx, msg: RawFrame):
        matcher: SGMStereoMatcher = ctx.state["matcher"]
        with NUMBA_PARALLEL_LOCK:        # SGM uses numba parallel=True
            # Returns the tracking-grid left + metric depth on the SAME grid.
            # For a replay matcher (rectify_left off) the left passes through
            # unchanged; for the live matcher (rectify_left on, raw cameras) the
            # left is rectified here so depth + tracking share one grid.
            gray_track, depth = matcher.dense_depth_rectified_left(
                msg.gray_left, msg.gray_right)
        return DepthFrame(msg.seq, msg.ts_ns, gray_track, depth)


class _PublishDepth(Task):
    name = "publish_depth"

    def run(self, ctx, msg: DepthFrame):
        ctx.bus.publish(topics.FRAME_DEPTH, msg)
        return None


class DepthFlow(Flow):
    def __init__(self, bus: Bus, matcher: SGMStereoMatcher) -> None:
        super().__init__("depth", bus)
        self.ctx.state["matcher"] = matcher
        self.on(topics.FRAME_RAW, [_ComputeDepth(), _PublishDepth()])
        self.forwards_to(topics.FRAME_DEPTH)
