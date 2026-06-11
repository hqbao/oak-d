"""Thread-safe, time-indexed ring buffer for raw IMU samples.

The split-acquisition front-end reads the IMU on its own thread and the cameras
on another. The IMU thread pushes every raw sample ``(t_ns, gyro, accel)`` into
this buffer as it arrives; the camera thread, after grabbing a stereo pair with
device timestamp ``t``, asks the buffer for *every IMU sample up to ``t``*. That
is exactly the set of inertial measurements that belong to the interval ending at
that frame -- the honest, timestamp-based way to bind IMU data to a camera frame
(the IMU has no frame serial number, only a device clock; see the capture flow).

The buffer therefore offers two operations the two threads need:

* :meth:`append` -- called by the IMU reader for each incoming sample.
* :meth:`drain_until` -- called by the camera-synced consumer: returns and
  removes all buffered samples with ``sample_t <= t_ns``. Because each drain
  removes everything up to its cut, successive drains yield disjoint, contiguous
  intervals ``(prev_cut, t_ns]`` with no sample dropped or double-counted.

:meth:`wait_until` lets the consumer block until the buffer has actually received
a sample at/after the requested time (coverage), so a frame is never packed with
a half-filled interval merely because the IMU thread had not been scheduled yet.
A bounded timeout (and :meth:`close`) guarantees it never blocks forever -- the
final frame, whose timestamp may lie past the last IMU sample, drains whatever is
available instead of hanging.

Pure standard-library + numpy; no device or framework imports, so it is fully
unit-testable offline.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np


class TimedImuBuffer:
    """A thread-safe FIFO of timestamped IMU samples, drained by timestamp.

    Samples are expected in non-decreasing ``t_ns`` order (a single sensor on one
    monotonic device clock). Out-of-order late arrivals are still stored; the
    drain pops from the front while the front timestamp is ``<= t_ns``, so a rare
    swap only delays a straggler by one frame rather than corrupting the stream.
    """

    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._cap = int(capacity)
        self._buf: deque[tuple[int, np.ndarray, np.ndarray]] = deque()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._newest_t: int | None = None
        self._closed = False
        self._dropped = 0  #: samples evicted by capacity before being drained

    # ------------------------------------------------------------------ #
    def append(self, t_ns: int, gyro, accel) -> None:
        """Store one IMU sample. Called by the IMU reader thread.

        ``gyro`` / ``accel`` are copied into contiguous float64 ``(3,)`` arrays so
        the caller may reuse its own buffers. Oldest samples are evicted once
        ``capacity`` is exceeded (a slow/absent consumer must not grow memory
        unbounded); the eviction count is tracked for diagnostics.
        """
        g = np.asarray(gyro, dtype=np.float64).reshape(3).copy()
        a = np.asarray(accel, dtype=np.float64).reshape(3).copy()
        t = int(t_ns)
        with self._cv:
            self._buf.append((t, g, a))
            while len(self._buf) > self._cap:
                self._buf.popleft()
                self._dropped += 1
            if self._newest_t is None or t > self._newest_t:
                self._newest_t = t
            self._cv.notify_all()

    def close(self) -> None:
        """Mark the source exhausted so :meth:`wait_until` stops blocking."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    # ------------------------------------------------------------------ #
    def wait_until(self, t_ns: int, timeout: float = 0.5) -> bool:
        """Block until a sample at/after ``t_ns`` has arrived (coverage).

        Returns ``True`` if the buffer now covers ``t_ns`` (newest sample
        ``>= t_ns``), ``False`` if it timed out or the source closed first. A
        ``False`` return is normal for the last frame (its timestamp can lie past
        the final IMU sample) -- the caller then drains whatever is present.
        """
        cut = int(t_ns)
        deadline = time.monotonic() + float(timeout)
        with self._cv:
            while True:
                if self._newest_t is not None and self._newest_t >= cut:
                    return True
                if self._closed:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._cv.wait(remaining)

    def drain_until(self, t_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove and return every buffered sample with ``sample_t <= t_ns``.

        Returns ``(ts, gyro, accel)`` with shapes ``(M,)``, ``(M, 3)``,
        ``(M, 3)`` in time order (``M`` may be 0). Successive calls return
        disjoint, contiguous intervals because each call also discards what it
        returned.
        """
        cut = int(t_ns)
        out: list[tuple[int, np.ndarray, np.ndarray]] = []
        with self._cv:
            while self._buf and self._buf[0][0] <= cut:
                out.append(self._buf.popleft())
        if not out:
            return (np.empty((0,), dtype=np.int64),
                    np.empty((0, 3), dtype=np.float64),
                    np.empty((0, 3), dtype=np.float64))
        ts = np.fromiter((s[0] for s in out), dtype=np.int64, count=len(out))
        gyro = np.stack([s[1] for s in out])
        accel = np.stack([s[2] for s in out])
        return ts, gyro, accel

    # ------------------------------------------------------------------ #
    @property
    def newest_t(self) -> int | None:
        """Device timestamp (ns) of the most recent appended sample, or ``None``."""
        with self._lock:
            return self._newest_t

    @property
    def dropped(self) -> int:
        """Number of samples evicted by the capacity cap before being drained."""
        with self._lock:
            return self._dropped

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
