"""Pose-source contract for the 3D viewer + a device-free fake source.

The viewer ([ours.ui.mainwindow.MainWindow]) consumes 6-DoF poses from anything
implementing :class:`PoseSource`: it ``start(callback)``s the source, which runs
a background thread and pushes :class:`~ours.lib.pose.Pose` samples (NED) to the
callback. This is the UI's input contract -- it deliberately knows nothing about
how the poses are produced.

Two implementations live here:

* :class:`PoseSource`     -- the abstract base (thread + callback + error state).
* :class:`FakePoseSource` -- a procedural figure-8 trajectory for UI bring-up
  without a camera.

The real live source is :class:`ours.ui.live_source.FlowPoseSource`, which runs
the flow pipeline and bridges its poses to this contract.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from ..lib.frames import rpy_to_quat
from ..lib.pose import Pose

PoseCallback = Callable[[Pose], None]


class PoseSource(ABC):
    """Pushes :class:`Pose` samples to a callback in a background thread."""

    def __init__(self) -> None:
        self._cb: PoseCallback | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.fps: float = 0.0
        # Set by ``_fail`` when the source aborts (e.g. bad startup attitude).
        # The UI polls this to surface the reason and reset its Start button.
        self.error: str | None = None

    def start(self, callback: PoseCallback) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("source already running")
        self._cb = callback
        self.error = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_wrapper, name=type(self).__name__, daemon=True
        )
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run_wrapper(self) -> None:
        try:
            self._run()
        except Exception as e:                                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[{type(self).__name__}] stopped: {e}")

    @abstractmethod
    def _run(self) -> None: ...

    def _emit(self, pose: Pose) -> None:
        if self._cb is not None:
            self._cb(pose)

    def _fail(self, msg: str) -> None:
        """Abort the source with a user-facing reason (polled by the UI)."""
        self.error = msg
        print(f"[{type(self).__name__}] {msg}")


class FakePoseSource(PoseSource):
    """Smooth figure-8 trajectory in the horizontal plane, with altitude bob.

    All positions are expressed in NED metres relative to start.
    """

    def __init__(self, rate_hz: float = 100.0, radius_m: float = 3.0,
                 period_s: float = 12.0, alt_amp_m: float = 0.5) -> None:
        super().__init__()
        self.rate_hz = float(rate_hz)
        self.radius = float(radius_m)
        self.period = float(period_s)
        self.alt_amp = float(alt_amp_m)

    def _run(self) -> None:
        dt = 1.0 / self.rate_hz
        t0 = time.monotonic()
        prev_pos = np.zeros(3)
        prev_t = t0

        last_fps_t = t0
        frames = 0

        while not self._stop.is_set():
            now = time.monotonic()
            t = now - t0

            # figure-8 in horizontal plane (Lissajous 1:2)
            w = 2.0 * np.pi / self.period
            n_pos = self.radius * np.sin(w * t)
            e_pos = self.radius * np.sin(2.0 * w * t) * 0.5
            d_pos = -1.0 + self.alt_amp * np.sin(0.7 * w * t)  # ~1 m up, gentle bob
            pos = np.array([n_pos, e_pos, d_pos], dtype=np.float64)

            # tangent yaw: heading along velocity in NE plane
            dn_dt = self.radius * w * np.cos(w * t)
            de_dt = self.radius * w * np.cos(2.0 * w * t)
            yaw = float(np.arctan2(de_dt, dn_dt))
            roll = 0.15 * np.sin(2.0 * w * t)
            pitch = 0.08 * np.sin(w * t)
            q = rpy_to_quat(roll, pitch, yaw)

            # velocity by finite difference
            dt_real = now - prev_t
            vel = (pos - prev_pos) / dt_real if dt_real > 1e-6 else np.zeros(3)
            prev_pos, prev_t = pos, now

            self._emit(Pose(t=t, pos_ned=pos, vel_ned=vel, quat_wxyz=q,
                            tracking_ok=True))

            frames += 1
            if now - last_fps_t >= 0.5:
                self.fps = frames / (now - last_fps_t)
                frames = 0
                last_fps_t = now

            # sleep to maintain rate (best-effort)
            remain = dt - (time.monotonic() - now)
            if remain > 0:
                time.sleep(remain)
