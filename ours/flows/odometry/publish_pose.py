"""``publish_pose`` task: emit the per-frame pose on ``pose.odom``."""
from __future__ import annotations

from ..core import topics
from ..core.messages import PoseMsg
from ..core.task import Task
from .step import Step


class PublishPose(Task):
    name = "publish_pose"

    def run(self, ctx, step: Step):
        ctx.bus.publish(topics.POSE_ODOM,
                        PoseMsg(step.frame.seq, step.frame.ts_ns,
                                step.pose, step.info))
        return step
