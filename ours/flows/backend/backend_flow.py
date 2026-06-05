"""backend flow: windowed bundle adjustment.

Wires the two backend tasks (one file each) into a reactive flow over
``keyframe``:

1. :class:`~ours.flows.backend.run_ba.RunBA` -- add the keyframe's track
   snapshot to the sliding window and run BA; forward any refined pose.
2. :class:`~ours.flows.backend.publish_refined.PublishRefined` -- emit it on
   ``pose.refined``.

The keyframe pose ``T_world_cam`` is inverted to the ``T_cw`` the BA map expects
(it keeps the map in the raw f2f world frame, exactly like the live worker).
"""
from __future__ import annotations

import numpy as np

from ..core import Flow, Bus, topics
from ...lib.backend.bundle import BAConfig
from ...lib.backend.windowed import WindowedBAMap, WindowedConfig
from .run_ba import RunBA
from .publish_refined import PublishRefined


class BackendFlow(Flow):
    def __init__(self, bus: Bus, K: np.ndarray,
                 window: int = 6, kf_every: int = 1, iters: int = 5) -> None:
        super().__init__("backend", bus)
        cfg = WindowedConfig(window=window, kf_every=kf_every,
                             ba=BAConfig(max_iters=iters))
        self.ctx.state["ba"] = WindowedBAMap(K, cfg)
        self.on(topics.KEYFRAME, [RunBA(), PublishRefined()])
        self.forwards_to(topics.POSE_REFINED)
