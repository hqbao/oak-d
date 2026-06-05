"""Realtime admission control for the acquisition front-end.

On the live path the camera produces frames at a fixed rate (e.g. 20 fps) but the
downstream VIO (SGM depth + PnP + windowed BA) only sustains a lower rate. With
unbounded per-flow inboxes the surplus frames -- each a ~0.5 MB stereo packet --
pile up on the host until memory pressure starves the depthai XLink thread and
the device firmware watchdog crashes the camera.

The fix is a closed-loop *credit* budget enforced at the imu-reader, the single
funnel BEFORE the depth/odometry fan-out (so both branches always see the exact
same surviving subset -- the diamond never desyncs). At most ``budget`` frames
are admitted "in flight"; each admitted frame frees its credit when the odometry
tail publishes a :class:`~ours.lib.flow.messages.FrameDone`. Over budget, the
imu-reader skips the camera frame at the source -- before the heavy packet is
built and before the IMU is drained -- so the skipped interval's inertial samples
simply fold into the next admitted frame (gyro preintegration stays gap-free).

Two strategies, selected by run mode:

* :class:`AdmitAll`  -- replay / offline: admit every frame, never count. The
  graph stays byte-for-byte deterministic (one packet per input frame).
* :class:`BudgetAdmission` -- live: the ``N``-credit gate above.

Both are touched only on the imu-reader's own thread (``try_admit`` from the
``cam.sync`` handler, ``complete`` from the ``frame.done`` handler -- same inbox,
same thread), so no locking is required for correctness; a lock is kept anyway as
cheap insurance against future re-wiring.
"""
from __future__ import annotations

import threading


class Admission:
    """Strategy interface: gate frame admission and account for completions."""

    def try_admit(self, seq: int) -> bool:
        """Return True if this frame may enter the pipeline (and reserve it)."""
        raise NotImplementedError

    def complete(self, seq: int) -> None:
        """Account for a finished frame, freeing its reservation."""


class AdmitAll(Admission):
    """Admit every frame, count nothing (replay / offline determinism)."""

    def try_admit(self, seq: int) -> bool:
        return True

    def complete(self, seq: int) -> None:
        pass


class BudgetAdmission(Admission):
    """Allow at most ``budget`` frames in flight; a completion frees a credit.

    ``budget`` of 2 keeps depth pipelined (one computing, one queued); 3 adds
    slack for jitter. Larger values just reintroduce latency and backlog. The
    in-flight count never goes negative (a stray completion is ignored).
    """

    def __init__(self, budget: int = 2) -> None:
        self.budget = max(1, int(budget))
        self._in_flight = 0
        self._lock = threading.Lock()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def try_admit(self, seq: int) -> bool:
        with self._lock:
            if self._in_flight >= self.budget:
                return False
            self._in_flight += 1
            return True

    def complete(self, seq: int) -> None:
        with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1
