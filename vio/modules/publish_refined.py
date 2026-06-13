"""``publish_refined`` step: emit the BA-refined pose on ``pose.refined``."""
from __future__ import annotations

from vio.comms import LocalPubSub, topics
from vio.comms.messages import PoseMsg


def publish_refined(bus: LocalPubSub, msg: PoseMsg) -> None:
    """Publish the BA-refined pose on ``pose.refined`` (terminal step).

    Was ``PublishRefined(Step)``; identical publish, the bus passed explicitly.
    """
    bus.publish(topics.POSE_REFINED, msg)
    return None
