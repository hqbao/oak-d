"""``publish_capture`` task: route a produced capture message to its topic.

Shared by both capture sources (:class:`~ours.flows.capture.replay.ReplayCaptureFlow`
and :class:`~ours.flows.capture.live.LiveCaptureFlow`) so they emit the exact same
topics and one odometry flow serves both.
"""
from __future__ import annotations

from ...lib.flow import topics
from ...lib.flow.messages import ImuInit, ImuPrior, RawFrame
from ...lib.flow.task import Task


class PublishCapture(Task):
    name = "publish_capture"

    def run(self, ctx, msg):
        if isinstance(msg, (ImuInit, ImuPrior)):
            ctx.bus.publish(topics.IMU_SAMPLE, msg)
        elif isinstance(msg, RawFrame):
            ctx.bus.publish(topics.FRAME_RAW, msg)
        return None
