"""Internal carrier threading one frame's result through the odometry chain.

Not a task -- a small flow-internal message handed from :class:`EstimateMotion` to
the downstream :class:`PublishPose` / :class:`EmitKeyframe` tasks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...lib.flow.messages import DepthFrame


@dataclass
class Step:
    frame: DepthFrame
    pose: np.ndarray
    info: dict
    accel_cam: np.ndarray | None
    at_rest: bool
