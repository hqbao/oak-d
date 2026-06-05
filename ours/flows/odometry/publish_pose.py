"""``publish_pose`` task: emit the per-frame pose on ``pose.odom``."""
from __future__ import annotations

from ...lib.flow import topics
from ...lib.flow.messages import PoseMsg
from ...lib.flow.task import Task
from .step import Step


class PublishPose(Task):
    name = "publish_pose"

    def run(self, ctx, step: Step):
        ctx.bus.publish(topics.POSE_ODOM,
                        PoseMsg(step.frame.seq, step.frame.ts_ns,
                                step.pose, step.info))
        return step
