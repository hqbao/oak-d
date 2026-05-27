"""Procedural pose generator — figure-8 + altitude bob, for UI bring-up."""
from __future__ import annotations

import time

import numpy as np

from ..frames import rpy_to_quat
from ..pose import Pose
from .base import PoseSource


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
            d_pos = -1.0 + self.alt_amp * np.sin(0.7 * w * t)  # ~1 m above start, gentle bob
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

            self._emit(Pose(t=t, pos_ned=pos, vel_ned=vel, quat_wxyz=q, tracking_ok=True))

            frames += 1
            if now - last_fps_t >= 0.5:
                self.fps = frames / (now - last_fps_t)
                frames = 0
                last_fps_t = now

            # sleep to maintain rate (best-effort)
            remain = dt - (time.monotonic() - now)
            if remain > 0:
                time.sleep(remain)
