"""ui tracks flow: forward streamed KLT tracks to a callback (keypoints view).

The sink that drives the keypoint-depth tracker window
(:class:`~ours.ui.keypoints_window.KeypointTrackWindow`). It subscribes
``frame.tracks`` -- the REAL frontend tracks the odometry flow publishes -- and
``frame.depth`` (the matching rectified-left image + metric depth published by
the ``imu_cam`` flow), joins them by ``seq``, and hands a
:class:`TracksWithFrame` bundle (the same attributes the old all-in-one
``FrameTracks`` exposed) to an ``on_tracks`` callback. The join lives here so the
producer side (odometry's :class:`~ours.flows.odometry.publish_tracks.PublishTracks`)
ships ONLY the per-frame tracks dataclass -- the gray + depth are written once
by capture into its shared-memory rings (single writer), instead of VIO racing
capture to overwrite the same slots from a separate process.

When an ``on_inliers`` callback is given it ALSO subscribes ``frame.inliers`` --
the RGB-D PnP inlier track ids the odometry solve emits (a separate REAL output)
-- and forwards each :class:`~ours.lib.flow.messages.FrameInliers` to it (task in
:mod:`ours.flows.ui.render_inliers`), so the window can mark the clean subset the
motion estimate trusted. ``expected_ends`` accounts for every END-bearing input
the flow subscribes to.

The window keeps the overlay rendering (per-id trails + depth-coloured dots)
UI-side -- that is honest buffering of the subscribed tracks, not a parallel
detector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ...lib.flow import Bus, Flow, topics
from ...lib.flow.messages import FrameInliers
from .render_inliers import RenderInliers
from .render_tracks import RenderTracks, StashFrameDepth


@dataclass(frozen=True)
class TracksWithFrame:
    """One frame's KLT tracks bundled with its rectified-left image + depth.

    Built inside :class:`UiTracksFlow` by joining
    :class:`~ours.lib.flow.messages.FrameTracks` (ids + pixels, from odometry)
    with :class:`~ours.lib.flow.messages.DepthFrame` (gray + depth, from imu_cam)
    by ``seq``. The attribute names match the old monolithic ``FrameTracks`` so
    consumer code (the keypoints window) reads ``msg.gray_left`` / ``msg.depth_m``
    exactly as before -- only the producer-side wiring changed.

    * ``ids`` -- ``(N,)`` int64 persistent track ids.
    * ``points`` -- ``(N, 2)`` float32 pixel coordinates (same order as ``ids``).
    * ``gray_left`` -- ``(H, W)`` uint8, the rectified-left image for this frame.
    * ``depth_m`` -- ``(H, W)`` float32, metric depth aligned to ``gray_left``.
    """

    seq: int
    ts_ns: int
    ids: np.ndarray
    points: np.ndarray
    gray_left: np.ndarray
    depth_m: np.ndarray


class UiTracksFlow(Flow):
    """Sink flow: join ``frame.tracks`` + ``frame.depth`` by seq, forward to ``on_tracks``."""

    def __init__(self, bus: Bus, on_tracks: Callable[[TracksWithFrame], None], *,
                 on_inliers: Callable[[FrameInliers], None] | None = None,
                 latest_only: bool = False) -> None:
        super().__init__("ui", bus, latest_only=latest_only)
        self.ctx.state["on_tracks"] = on_tracks
        # Per-seq stash of (gray_left, depth_m) buffered from frame.depth so the
        # matching frame.tracks for that seq can join immediately. Bounded -- see
        # StashFrameDepth.
        self.ctx.state["frame_buf"] = {}
        # Two END-bearing inputs: frame.tracks (odometry) + frame.depth (imu_cam).
        self.expected_ends = 2
        self.on(topics.FRAME_DEPTH, [StashFrameDepth()])
        self.on(topics.FRAME_TRACKS, [RenderTracks()])
        if on_inliers is not None:
            self.ctx.state["on_inliers"] = on_inliers
            self.expected_ends = 3                   # + frame.inliers
            self.on(topics.FRAME_INLIERS, [RenderInliers()])
