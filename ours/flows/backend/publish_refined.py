"""``publish_refined`` task: emit the BA-refined pose on ``pose.refined``."""
from __future__ import annotations

from ...lib.flow import topics
from ...lib.flow.messages import PoseMsg
from ...lib.flow.task import Task


class PublishRefined(Task):
    name = "publish_refined"

    def run(self, ctx, msg: PoseMsg):
        ctx.bus.publish(topics.POSE_REFINED, msg)
        return None
