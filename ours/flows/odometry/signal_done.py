"""``signal_done`` task: publish a backpressure credit for this frame.

The tail of the odometry frame chain. It publishes one
:class:`~ours.lib.flow.messages.FrameDone` per processed frame on
``topics.FRAME_DONE`` so the imu-reader's admission gate can free the frame's
in-flight credit. It fires for EVERY frame that reaches here -- including
tracking failures -- because a credit that never returns would deadlock the live
gate at full budget. On the replay path the gate ignores it (admits everything).
"""
from __future__ import annotations

from ...lib.flow import topics
from ...lib.flow.messages import FrameDone
from ...lib.flow.task import Task
from .step import Step


class SignalDone(Task):
    name = "signal_done"

    def run(self, ctx, step: Step):
        ctx.bus.publish(topics.FRAME_DONE, FrameDone(step.frame.seq))
        return None
