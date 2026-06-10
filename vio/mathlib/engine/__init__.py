"""``vio.mathlib.engine`` -- swappable runners for the heavy keyframe optimisers.

A flow picks how its optimiser runs with one ``worker`` flag:

* ``worker=False`` (default, OFFLINE) -> :class:`InProcessEngine` -- synchronous,
  deterministic, byte-identical replay output.
* ``worker=True`` (LIVE) -> :class:`~vio.mathlib.engine.subprocess.SubprocessEngine`
  -- the solve runs in a separate process so it never holds the camera read loop's
  GIL (the fast-push undershoot fix).

The engines wrap the existing algorithm libraries (``lib.backend.windowed`` /
``lib.loop.slam``) and know nothing about flows or the bus -- they are pure
machinery (``lib``), called by the flow tasks.
"""
from __future__ import annotations

import numpy as np

from .base import Engine, SlamResult
from .inprocess import InProcessEngine
from .steps import (ba_step, slam_step, vio_step,
                    ba_overlay, slam_overlay, vio_overlay)
from .subprocess import (SubprocessEngine, _ba_worker_main, _slam_worker_main,
                         _vio_worker_main)

__all__ = ["Engine", "SlamResult", "InProcessEngine", "SubprocessEngine",
           "make_ba_engine", "make_vi_engine", "make_slam_engine"]


def make_ba_engine(K: np.ndarray, cfg, *, worker: bool = False) -> Engine:
    """Build a windowed-BA engine (in-process unless ``worker``)."""
    if worker:
        return SubprocessEngine(_ba_worker_main, K, cfg)
    from ..backend.windowed import WindowedBAMap
    return InProcessEngine(lambda: WindowedBAMap(K, cfg), ba_step, ba_overlay)


def make_vi_engine(K: np.ndarray, cfg, *, worker: bool = False) -> Engine:
    """Build a tight-coupled VIO engine (in-process unless ``worker``).

    Symmetric with :func:`make_ba_engine` but wraps
    :class:`vio.mathlib.backend.vio_window.WindowedVIOMap` (the joint visual +
    IMU window optimiser) instead of the visual-only ``WindowedBAMap``. The live
    path feeds each keyframe's raw IMU segment via the snapshot, so the map is
    built with no stored IMU stream (``cfg=cfg`` only).
    """
    if worker:
        return SubprocessEngine(_vio_worker_main, K, cfg)
    from ..backend.vio_window import WindowedVIOMap
    return InProcessEngine(lambda: WindowedVIOMap(K, cfg=cfg),
                           vio_step, vio_overlay)


def make_slam_engine(K: np.ndarray, cfg, *, worker: bool = False) -> Engine:
    """Build a loop-closure SLAM engine (in-process unless ``worker``)."""
    if worker:
        return SubprocessEngine(_slam_worker_main, K, cfg)
    from ..loop.slam import SlamMap
    return InProcessEngine(lambda: SlamMap(K, cfg), slam_step, slam_overlay)
