"""``slam.engine`` -- swappable runners for the heavy keyframe optimisers.

A flow picks how its optimiser runs with one ``worker`` flag:

* ``worker=False`` (default, OFFLINE) -> :class:`InProcessEngine` -- synchronous,
  deterministic, byte-identical replay output.
* ``worker=True`` (LIVE) -> :class:`~slam.engine.subprocess.SubprocessEngine`
  -- the solve runs in a separate process so it never holds the camera read loop's
  GIL (the fast-push undershoot fix).

The engine wraps the shared loop-closure SLAM library (``sky.slam.slam``)
and knows nothing about flows or the bus -- it is pure machinery (``lib``),
called by the module steps.
"""
from __future__ import annotations

import numpy as np

from .base import Engine, SlamResult
from .inprocess import InProcessEngine
from .steps import slam_step, slam_overlay
from .subprocess import SubprocessEngine, _slam_worker_main

__all__ = ["Engine", "SlamResult", "InProcessEngine", "SubprocessEngine",
           "make_slam_engine"]


def make_slam_engine(K: np.ndarray, cfg, *, worker: bool = False,
                     capture_loops: bool = False) -> Engine:
    """Build a loop-closure SLAM engine (in-process unless ``worker``).

    ``capture_loops`` (LIVE-only) makes the engine capture each verified
    candidate's match funnel so the SLAM module can publish ``slam.loop`` for the
    UI's loop-closure view. It is wired ON only on the live publish-map path; the
    OFFLINE / oracle path leaves it False, so the map runs the byte-frozen
    ``verify`` (no funnel work) and the deterministic ``loop.correction`` scoring
    stays bit-identical. The subprocess engine is LIVE-only and always captures
    (its worker builds the map with ``capture_loops=True``), so the flag here only
    governs the in-process engine.
    """
    if worker:
        return SubprocessEngine(_slam_worker_main, K, cfg)
    from sky.slam.slam import SlamMap
    return InProcessEngine(lambda: SlamMap(K, cfg, capture_loops=capture_loops),
                           slam_step, slam_overlay)
