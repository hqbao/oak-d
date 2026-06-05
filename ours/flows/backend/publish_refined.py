"""``publish_refined`` task: emit the BA-refined pose on ``pose.refined``."""
from __future__ import annotations

from ..core import topics
from ..core.messages import PoseMsg
from ..core.task import Task


class PublishRefined(Task):
    name = "publish_refined"

    def run(self, ctx, msg: PoseMsg):
        ctx.bus.publish(topics.POSE_REFINED, msg)
        return None
