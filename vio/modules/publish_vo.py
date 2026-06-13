"""``publish_vo`` step: emit the pure-vision frame-to-frame pose on ``pose.vo``.

Mirrors :func:`~vio.modules.publish_pose.publish_pose`, but publishes the
PURE-VISION (no-IMU, no-BA) accumulated pose
(:attr:`~sky.front.odometry.RGBDVisualOdometry.pose_vo`) instead of the
gyro-fused VIO pose. The UI draws this as its "VO" line to compare against the
VIO ``pose.odom``.

LIVE-only: wired into the odometry worker (:class:`~vio.modules.pipeline.OdometryWorker`)
only when ``publish_vo=True`` (the proc4 live builder sets it). The offline
deterministic path leaves it off, so it never publishes there and pose.odom byte
parity is unaffected.

``pose_vo`` is updated upstream by :func:`~vio.modules.estimate_motion.estimate_motion`,
so this step must run AFTER it in the frame chain. The per-frame ``seq`` /
``ts_ns`` come from the same :class:`~vio.modules.step.Step` carrier
``publish_pose`` uses; the pose is read live off the shared
:class:`~sky.front.odometry.RGBDVisualOdometry` instance ``vo``.
"""
from __future__ import annotations

from vio.comms import LocalPubSub, topics
from vio.comms.messages import PoseMsg
from sky.front.odometry import RGBDVisualOdometry
from .step import Step


def publish_vo(vo: RGBDVisualOdometry, bus: LocalPubSub, step: Step) -> Step:
    """Publish the pure-vision accumulated pose on ``pose.vo``; pass on the carrier.

    Was ``PublishVo(Step)``; identical publish, the odometry instance + bus passed
    explicitly instead of read off ``ctx.state``.
    """
    # Copy the accumulator: the same instance keeps mutating pose_vo on the
    # next frame, so the published message must own an independent snapshot
    # (the wire/bridge layer reads it asynchronously).
    bus.publish(topics.POSE_VO,
                PoseMsg(step.frame.seq, step.frame.ts_ns,
                        vo.pose_vo.copy(), step.info))
    return step
