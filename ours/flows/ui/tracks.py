"""ui tracks flow: forward streamed KLT tracks to a callback (keypoints view).

The sink that drives the keypoint-depth tracker window
(:class:`~ours.ui.keypoints_window.KeypointTrackWindow`). It subscribes
``frame.tracks`` -- the REAL frontend tracks the odometry flow publishes -- and
hands each :class:`~ours.lib.flow.messages.FrameTracks` to an ``on_tracks``
callback, exactly as :class:`~ours.flows.ui.render.UiRenderFlow` does for
``pose.odom``. The single task lives in :mod:`ours.flows.ui.render_tracks`.

The window keeps the overlay rendering (per-id trails + depth-coloured dots)
UI-side -- that is honest buffering of the subscribed tracks, not a parallel
detector.
"""
from __future__ import annotations

from typing import Callable

from ...lib.flow import Flow, Bus, topics
from ...lib.flow.messages import FrameTracks
from .render_tracks import RenderTracks


class UiTracksFlow(Flow):
    """Sink flow that forwards each ``frame.tracks`` to ``on_tracks``."""

    def __init__(self, bus: Bus, on_tracks: Callable[[FrameTracks], None], *,
                 latest_only: bool = False) -> None:
        super().__init__("ui", bus, latest_only=latest_only)
        self.ctx.state["on_tracks"] = on_tracks
        self.expected_ends = 1                       # only frame.tracks carries END
        self.on(topics.FRAME_TRACKS, [RenderTracks()])
