"""``vio.engine`` -- swappable runners for the heavy keyframe optimisers.

A flow picks how its optimiser runs with one ``worker`` flag:

* ``worker=False`` (default, OFFLINE) -> :class:`InProcessEngine` -- synchronous,
  deterministic, byte-identical replay output.
* ``worker=True`` (LIVE) -> :class:`~vio.engine.subprocess.SubprocessEngine`
  -- the solve runs in a separate process so it never holds the camera read loop's
  GIL (the fast-push undershoot fix).

The engines wrap the shared algorithm libraries (``sky.backend.windowed``
/ ``sky.vio.window``) and know nothing about flows or the bus --
they are pure machinery (``lib``), called by the flow tasks.
"""
from __future__ import annotations

import numpy as np

from .base import Engine
from .inprocess import InProcessEngine
from .steps import ba_step, vio_step, ba_overlay, vio_overlay
from .ba_capture import ba_step_capture, ba_window_overlay
from .subprocess import (SubprocessEngine, _ba_worker_main,
                         _ba_capture_worker_main, _vio_worker_main)

__all__ = ["Engine", "InProcessEngine", "SubprocessEngine",
           "make_ba_engine", "make_vi_engine"]


def make_ba_engine(K: np.ndarray, cfg, *, worker: bool = False,
                   capture_window: bool = False) -> Engine:
    """Build a windowed-BA engine (in-process unless ``worker``).

    ``capture_window`` (opt-in, ``--ba-window``) selects the RICHER capture step +
    overlay (:func:`~vio.engine.ba_capture.ba_step_capture` /
    :func:`~vio.engine.ba_capture.ba_window_overlay`): the SAME frozen
    ``run_ba`` solve plus a read-only PRE/POST snapshot for the UI's "BA Window"
    visualiser. Default OFF -> the historical ``ba_step`` / ``ba_overlay`` path,
    byte-identical to before (the oracle relies on this).
    """
    step = ba_step_capture if capture_window else ba_step
    overlay = ba_window_overlay if capture_window else ba_overlay
    if worker:
        worker_main = _ba_capture_worker_main if capture_window else _ba_worker_main
        return SubprocessEngine(worker_main, K, cfg)
    from sky.backend.windowed import WindowedBAMap
    return InProcessEngine(lambda: WindowedBAMap(K, cfg), step, overlay)


def make_vi_engine(K: np.ndarray, cfg, *, worker: bool = False) -> Engine:
    """Build a tight-coupled VIO engine (in-process unless ``worker``).

    Symmetric with :func:`make_ba_engine` but wraps
    :class:`sky.vio.window.WindowedVIOMap` (the joint visual +
    IMU window optimiser) instead of the visual-only ``WindowedBAMap``. The live
    path feeds each keyframe's raw IMU segment via the snapshot, so the map is
    built with no stored IMU stream (``cfg=cfg`` only).
    """
    if worker:
        return SubprocessEngine(_vio_worker_main, K, cfg)
    from sky.vio.window import WindowedVIOMap
    return InProcessEngine(lambda: WindowedVIOMap(K, cfg=cfg),
                           vio_step, vio_overlay)
