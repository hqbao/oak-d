"""Message types passed between flows over the pub/sub bus.

These are plain immutable carriers -- one per topic in ``ours.flows.core.topics``. Keeping
them here documents exactly what each flow consumes and produces, and keeps the
flows themselves free of ad-hoc dicts.

The :data:`END` sentinel is published on a topic when its upstream flow has no
more data (e.g. the recorded session ran out). Reactive flows forward it to their
own downstream topics so the whole graph drains cleanly; the UI flow uses it to
know the run is finished.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Published on a topic to signal "no more messages will follow on this topic".
END = object()


@dataclass(frozen=True)
class ImuInit:
    """One-shot IMU startup info, published by capture before any frame.

    ``accel_align`` is the near-static startup accelerometer mean in the camera
    optical frame (m/s^2), used once by the odometry flow to gravity-align the
    initial attitude. ``None`` when the device has no usable IMU.
    """

    accel_align: np.ndarray | None


@dataclass(frozen=True)
class ImuPrior:
    """Per-frame IMU prior, published by capture alongside each raw frame.

    The capture flow owns the IMU->prior fusion so the same odometry flow drives
    both replay and live:

    * ``R_prior`` -- inter-frame camera-frame rotation ``R_cam(prev->cur)`` from
      the gyro (the rotation prior handed to PnP), or ``None`` on the first frame.
    * ``accel_cam`` / ``at_rest`` -- the camera-frame accelerometer this frame and
      whether the camera was still; supplied so a keyframe can carry a gravity
      prior into the back-end. ``at_rest`` is ``False`` (accel ``None``) when no
      usable gravity measurement is available (e.g. replay scoring).
    """

    seq: int
    R_prior: np.ndarray | None
    accel_cam: np.ndarray | None = None
    at_rest: bool = False


@dataclass(frozen=True)
class RawFrame:
    """A captured stereo pair (rectified left + raw right) with its timestamp."""

    seq: int
    ts_ns: int
    gray_left: np.ndarray
    gray_right: np.ndarray | None


@dataclass(frozen=True)
class DepthFrame:
    """A left image with a metric depth map aligned to it."""

    seq: int
    ts_ns: int
    gray_left: np.ndarray
    depth_m: np.ndarray


@dataclass(frozen=True)
class PoseMsg:
    """An estimated camera pose (4x4 ``T_world_cam``) for one frame."""

    seq: int
    ts_ns: int
    T_world_cam: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Keyframe:
    """A keyframe handed to the back-end / SLAM flows.

    Carries both the high-level snapshot the SLAM map needs (pose + image +
    depth) and the low-level track snapshot the sliding-window BA needs
    (``track_ids`` + ``track_px`` from the odometry front-end, plus an optional
    at-rest ``accel`` for the gravity prior).
    """

    seq: int
    T_world_cam: np.ndarray
    gray_left: np.ndarray
    depth_m: np.ndarray
    track_ids: np.ndarray | None = None
    track_px: np.ndarray | None = None
    accel: np.ndarray | None = None


@dataclass(frozen=True)
class LoopCorrection:
    """A pose-graph correction: rewritten keyframe poses after loop closure."""

    seq: int
    kf_poses: dict[int, np.ndarray]
    n_loops: int
