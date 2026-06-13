"""In-process engine: run the heavy solve synchronously on the calling thread.

This is the OFFLINE / deterministic engine. ``submit`` runs the *whole* step
(``add_keyframe`` + solve) right now and stashes its single result; ``poll`` hands
that one result back and clears it. Because the back-end flow's FIFO inbox calls
``submit`` then ``poll`` inside the *same* task invocation per keyframe, the
behaviour is identical to the old in-thread ``RunBA`` / ``RunVIO`` -- zero
latency, byte-identical replay output. See :mod:`vio.engine.base` for why
``poll`` must be one-shot (not latest-wins).
"""
from __future__ import annotations

from typing import Any, Callable

from .base import Engine


class InProcessEngine(Engine):
    def __init__(self, make_map: Callable[[], Any], step,
                 overlay=None) -> None:
        self._make_map = make_map
        self._map = make_map()
        self._step = step
        self._overlay = overlay
        self._pending: Any = None
        self._has = False

    def submit(self, snapshot: Any) -> None:
        # Run the full solve now; stash the result (including None for a warmup
        # keyframe, so the matching poll returns None rather than a stale result).
        self._pending = self._step(self._map, snapshot)
        self._has = True

    def poll(self) -> Any:
        if not self._has:
            return None
        res, self._pending, self._has = self._pending, None, False
        return res

    def poll_overlay(self) -> Any:
        if self._overlay is None:
            return None
        return self._overlay(self._map)

    def reset(self) -> None:
        self._map = self._make_map()
        self._pending, self._has = None, False

    def close(self) -> None:
        pass

    @property
    def map(self) -> Any:
        """The live map object (read-only access for overlays / tests)."""
        return self._map
