"""``slam_step`` task: add a keyframe to the SLAM map, optimise on loop close."""
from __future__ import annotations

from ..core.messages import Keyframe, LoopCorrection
from ..core.task import Task
from ...lib.loop.slam import SlamMap


class SlamStep(Task):
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
