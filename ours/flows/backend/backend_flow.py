"""backend flow implementation: windowed bundle adjustment.

Tasks (per ``keyframe``):

1. ``_RunBA`` -- add the keyframe's track snapshot to the sliding window and run
   BA; if it produced a refined pose, forward it.
2. ``_PublishRefined`` -- publish the refined pose on ``pose.refined``.

The keyframe pose ``T_world_cam`` is inverted to the ``T_cw`` the BA map expects
(it keeps the map in the raw f2f world frame, exactly like the live worker).
"""
from __future__ import annotations

import numpy as np

from ...lib.flow import topics
from ...lib.backend.bundle import BAConfig
from ...lib.backend.windowed import WindowedBAMap, WindowedConfig
from ...lib.flow import Flow
from ...lib.flow.messages import Keyframe, PoseMsg
from ...lib.flow.pubsub import Bus
from ...lib.flow.task import Task


class _RunBA(Task):
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


class _PublishRefined(Task):
    name = "publish_refined"

    def run(self, ctx, msg: PoseMsg):
        ctx.bus.publish(topics.POSE_REFINED, msg)
        return None


class BackendFlow(Flow):
    def __init__(self, bus: Bus, K: np.ndarray,
                 window: int = 6, kf_every: int = 1, iters: int = 5) -> None:
        super().__init__("backend", bus)
        cfg = WindowedConfig(window=window, kf_every=kf_every,
                             ba=BAConfig(max_iters=iters))
        self.ctx.state["ba"] = WindowedBAMap(K, cfg)
        self.on(topics.KEYFRAME, [_RunBA(), _PublishRefined()])
        self.forwards_to(topics.POSE_REFINED)
