"""slam flow implementation: loop closure SLAM.

Tasks (per ``keyframe``):

1. ``_SlamStep`` -- add the keyframe to the SLAM map; if it confirmed a loop,
   optimise the pose graph and forward the rewritten keyframe poses.
2. ``_PublishCorrection`` -- publish the correction on ``loop.correction``.
"""
from __future__ import annotations

from ...lib.flow import topics
from ...lib.flow import Flow
from ...lib.loop.slam import SlamConfig, SlamMap
from ...lib.flow.messages import Keyframe, LoopCorrection
from ...lib.flow.pubsub import Bus
from ...lib.flow.task import Task


class _SlamStep(Task):
    name = "slam_step"

    def run(self, ctx, kf: Keyframe):
        slam: SlamMap = ctx.state["slam"]
        events = slam.add_keyframe(kf.T_world_cam, kf.gray_left, kf.depth_m,
                                   seq=kf.seq)
        if not events:
            return None
        slam.optimize()
        kf_poses = {int(slam.kf_seq[i]): slam.kf_pose[i].copy()
                    for i in range(len(slam.kf_pose))}
        return LoopCorrection(kf.seq, kf_poses, len(slam.loop_events))


class _PublishCorrection(Task):
    name = "publish_correction"

    def run(self, ctx, msg: LoopCorrection):
        ctx.bus.publish(topics.LOOP_CORRECTION, msg)
        return None


class SlamFlow(Flow):
    def __init__(self, bus: Bus, K, cfg: SlamConfig | None = None) -> None:
        super().__init__("slam", bus)
        self.ctx.state["slam"] = SlamMap(K, cfg or SlamConfig())
        self.on(topics.KEYFRAME, [_SlamStep(), _PublishCorrection()])
        self.forwards_to(topics.LOOP_CORRECTION)

    @property
    def slam(self) -> SlamMap:
        return self.ctx.state["slam"]
