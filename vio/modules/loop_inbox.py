"""Thread-safe inbox for SLAM ``loop.correction`` messages (LIVE + --tight only).

The closed-loop feedback ``slam -> vio`` crosses a thread boundary: the
``loop.correction`` arrives on the VIO process's slam-endpoint IPC subscriber
thread, but it must be applied to the live nav-state on the ODOMETRY module's
thread (the single owner of ``live_nav``). This tiny holder is the safe handoff:

* the subscriber side calls :meth:`push` (under a lock),
* :class:`~vio.modules.propagate_imu.PropagateImu` calls :meth:`drain` once per
  frame on the odometry thread and applies the corrections there.

It coalesces nothing -- every queued correction is returned in arrival order so a
burst of loop closures (e.g. a long revisit) all fold into the pending delta. The
queue is bounded so a wedged consumer cannot grow it without limit; the oldest
correction is dropped on overflow (a stale correction is the safe one to lose --
the freshest pose-graph rewrite supersedes it anyway).

This module is imported ONLY by the live ``--tight`` path (``OdometryModule`` when
``loop_correct=True``, wired by ``vio.main``). The offline / oracle / loose path
never constructs it, so the closed-loop feedback is purely additive there.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

#: Max queued corrections before the oldest is dropped. Loop closures are rare
#: events (one per revisit), so this only ever fills if the odometry thread wedges
#: -- in which case the freshest correction is the one worth keeping.
_MAX_QUEUED = 16


class LoopCorrectionInbox:
    """A small lock-guarded queue of ``LoopCorrection`` messages."""

    __slots__ = ("_lock", "_q")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._q: "deque[Any]" = deque(maxlen=_MAX_QUEUED)

    def push(self, correction: Any) -> None:
        """Enqueue a correction from the IPC subscriber thread (drops oldest on
        overflow via the bounded deque)."""
        with self._lock:
            self._q.append(correction)

    def drain(self) -> list[Any]:
        """Return + clear all queued corrections (called on the odometry thread).

        Returns them in arrival order so the pending-delta composition stacks the
        oldest unfinished correction first.
        """
        with self._lock:
            if not self._q:
                return []
            items = list(self._q)
            self._q.clear()
        return items
