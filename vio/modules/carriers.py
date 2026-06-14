"""Per-frame carrier records threaded through the odometry frame-chain.

These three small dataclasses are NOT tasks and never go on the bus -- they are
flow-internal messages that thread one frame's state from one step function to the
next inside the odometry worker (:mod:`vio.modules.pipeline`). Listed in the order
the frame-chain produces them:

* :class:`Tracked` -- frame + freshly KLT-tracked ``{id: pixel}`` features
  (output of :func:`~vio.modules.frontend.track_features`).
* :class:`Primed` -- the same, plus the IMU prior joined for that frame's ``seq``
  (output of :func:`~vio.modules.imu_prior.pull_prior`; the IMU<->vision join).
* :class:`Step` -- the solved per-frame result: pose + info + gravity accel + the
  at-rest flag (output of :func:`~vio.modules.frontend.estimate_motion`), consumed
  by the downstream publishers + :func:`~vio.modules.backend.emit_keyframe`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vio.comms.messages import DepthFrame, ImuPrior


@dataclass
class Tracked:
    frame: DepthFrame
    obs: dict[int, np.ndarray]


@dataclass
class Primed:
    frame: DepthFrame
    obs: dict[int, np.ndarray]
    prior: ImuPrior | None


@dataclass
class Step:
    frame: DepthFrame
    pose: np.ndarray
    info: dict
    accel_cam: np.ndarray | None
    at_rest: bool
