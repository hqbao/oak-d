"""Engine contract: a swappable runner for the heavy keyframe optimisers.

An *engine* owns one heavy map optimiser (windowed BA or loop-closure SLAM) and
exposes a tiny, uniform interface so the flow tasks (``RunBA`` / ``SlamStep``)
never care *where* the solve runs:

* :class:`InProcessEngine` -- runs the solve synchronously on the calling thread.
  Used by the OFFLINE replay/scoring path, where determinism matters and there is
  no real-time constraint: ``submit`` does the whole solve, ``poll`` returns its
  one result. This keeps the offline numbers byte-identical to the old in-thread
  flow.
* :class:`~ours.lib.engine.subprocess.SubprocessEngine` -- ships each keyframe to
  a separate process and reads the result back asynchronously. Used by the LIVE
  path so the mostly-pure-Python solve never holds the GIL of the camera read
  loop (the cause of the fast-push stall / undershoot -- see
  ``ours/lib/engine/subprocess.py``).

Both implement the same four methods, so a flow picks one with a single ``worker``
flag and nothing else changes.

CONTRACT (critical for offline byte-parity)
-------------------------------------------
``poll()`` is **one-shot** for the in-process engine: it returns at most the one
result stashed by the matching ``submit`` and then clears it. It is NOT
latest-wins. A warmup keyframe whose solve returns ``None`` must make ``poll``
return ``None`` -- never re-surface a previous keyframe's result -- otherwise the
back-end would emit an extra ``pose.refined`` and break the replay self-test count
(``len(refined) <= ceil(frames/kf_every)``). The subprocess engine, used only
live, is free to be latest-wins (it drops backlog on purpose).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SlamResult:
    """A SLAM step's output: the rewritten keyframe poses + loop count.

    Returned by :func:`ours.lib.engine.steps.slam_step` only on a keyframe that
    confirmed a loop (so the pose graph was optimised). ``SlamStep`` frames it
    into a :class:`~ours.lib.flow.messages.LoopCorrection` for the bus.

    * ``kf_poses`` -- ``{keyframe seq: T_world_cam}`` after pose-graph optimise.
    * ``n_loops`` -- total confirmed loop closures so far.
    """

    kf_poses: dict[int, np.ndarray]
    n_loops: int


class Engine(ABC):
    """Runs one heavy optimiser; results are produced by ``submit`` + ``poll``."""

    @abstractmethod
    def submit(self, snapshot: Any) -> None:
        """Hand the optimiser one keyframe snapshot (non-blocking)."""

    @abstractmethod
    def poll(self) -> Any:
        """Return a ready result (refined ``T_cw`` for BA, :class:`SlamResult`
        for SLAM) or ``None`` if nothing is ready."""

    @abstractmethod
    def poll_overlay(self) -> Any:
        """Return the latest MAP overlay snapshot (for the live 3D view) or
        ``None``. Separate channel from :meth:`poll` so the UI can read the map
        without stealing the correction the flow task consumes. Offline never
        calls this (no live viewer)."""

    @abstractmethod
    def reset(self) -> None:
        """Forget the whole map and start fresh (UI "clear keyframes")."""

    @abstractmethod
    def close(self) -> None:
        """Tear down; idempotent."""
