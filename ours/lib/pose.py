"""Pose dataclass and fixed-size ring buffer for the live trajectory."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
import time

import numpy as np

from .frames import quat_to_rot, quat_to_rpy


@dataclass
class Pose:
    """A single 6-DoF pose sample in the world (NED) frame."""
    t: float                                                     # seconds, monotonic
    pos_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))   # metres
    vel_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))   # m/s
    quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    tracking_ok: bool = True
    # True when this sample's position step was produced by a loop-closure
    # correction slewing the pose (a "teleport" while the camera barely moved),
    # not by real camera motion. The viewer colours these path segments
    # differently so the discontinuity reads as a map correction, not odometry.
    teleport: bool = False
    # Optional accelerometer-only attitude (gravity-leveled, yaw=0) in the same
    # NED body frame, for live comparison against the fused ``quat_wxyz``. ``None``
    # when the source has no IMU.
    accel_quat_wxyz: np.ndarray | None = None

    @property
    def R(self) -> np.ndarray:
        return quat_to_rot(self.quat_wxyz)

    @property
    def rpy_rad(self) -> tuple[float, float, float]:
        return quat_to_rpy(self.quat_wxyz)

    @property
    def rpy_deg(self) -> tuple[float, float, float]:
        r, p, y = self.rpy_rad
        return float(np.degrees(r)), float(np.degrees(p)), float(np.degrees(y))

    @property
    def accel_rpy_deg(self) -> tuple[float, float, float] | None:
        if self.accel_quat_wxyz is None:
            return None
        r, p, y = quat_to_rpy(self.accel_quat_wxyz)
        return float(np.degrees(r)), float(np.degrees(p)), float(np.degrees(y))


class PoseHistory:
    """Thread-safe append + snapshot ring buffer of recent positions."""

    def __init__(self, capacity: int = 4096):
        self._cap = int(capacity)
        self._buf = np.zeros((self._cap, 3), dtype=np.float32)
        self._flags = np.zeros(self._cap, dtype=bool)
        self._n = 0
        self._head = 0
        self._lock = Lock()
        self._latest: Pose | None = None
        self._t_start = time.monotonic()

    def push(self, pose: Pose) -> None:
        with self._lock:
            self._buf[self._head] = pose.pos_ned.astype(np.float32)
            self._flags[self._head] = bool(pose.teleport)
            self._head = (self._head + 1) % self._cap
            if self._n < self._cap:
                self._n += 1
            self._latest = pose

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, Pose | None]:
        """Return (positions Nx3 chronological, teleport-flags N, latest pose)."""
        with self._lock:
            n = self._n
            if n == 0:
                return (np.empty((0, 3), dtype=np.float32),
                        np.empty((0,), dtype=bool), None)
            if n < self._cap:
                arr = self._buf[:n].copy()
                flags = self._flags[:n].copy()
            else:
                arr = np.concatenate(
                    (self._buf[self._head:], self._buf[:self._head]), axis=0
                )
                flags = np.concatenate(
                    (self._flags[self._head:], self._flags[:self._head]), axis=0
                )
            return arr, flags, self._latest

    def clear(self) -> None:
        with self._lock:
            self._n = 0
            self._head = 0
            self._latest = None
            self._t_start = time.monotonic()

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self._t_start
