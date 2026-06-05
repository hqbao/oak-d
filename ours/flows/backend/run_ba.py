"""``run_ba`` task: add a keyframe to the sliding window and run BA."""
from __future__ import annotations

import numpy as np

from ...lib.flow.messages import Keyframe, PoseMsg
from ...lib.flow.task import Task
from ...lib.backend.windowed import WindowedBAMap


class RunBA(Task):
    name = "run_ba"

    def run(self, ctx, kf: Keyframe):
        if kf.track_ids is None or kf.track_px is None:
            return None
        ba: WindowedBAMap = ctx.state["ba"]
        T_cw = np.linalg.inv(kf.T_world_cam)
        ba.add_keyframe(T_cw, kf.track_ids, kf.track_px, kf.depth_m,
                        accel_cam=kf.accel)
        post = ba.run_ba()                       # refined latest T_cw, or None
        if post is None:
            return None
        return PoseMsg(kf.seq, 0, np.linalg.inv(post), {"refined": True})
