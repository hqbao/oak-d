"""``publish_pose`` step: emit the per-frame pose on ``pose.odom``."""
from __future__ import annotations

from vio.comms import LocalPubSub, topics
from vio.comms.messages import PoseMsg
from .step import Step


def publish_pose(bus: LocalPubSub, step: Step) -> Step:
    """Publish the per-frame VIO pose on ``pose.odom``; pass the carrier on.

    Was ``PublishPose(Step)``; identical publish, the bus passed explicitly.
    """
    bus.publish(topics.POSE_ODOM,
                PoseMsg(step.frame.seq, step.frame.ts_ns,
                        step.pose, step.info))
    return step
