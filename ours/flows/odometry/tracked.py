"""Internal carrier from :class:`TrackFeatures` to :class:`EstimateMotion`.

Not a task -- a small flow-internal message that threads one frame's depth input
plus its freshly tracked ``{id: pixel}`` features between the two halves of the
odometry chain. Stays inside the odometry flow (never goes on the Bus), the same
role :class:`~ours.flows.odometry.step.Step` plays for the downstream tasks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...lib.flow.messages import DepthFrame


@dataclass
class Tracked:
    frame: DepthFrame
    obs: dict[int, np.ndarray]
