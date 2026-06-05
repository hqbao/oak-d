"""Stillness detection + capture state machines for the IMU calibration wizards.

These are pure, hardware-agnostic state machines: the UI feeds them raw IMU
samples one at a time and polls a status snapshot to drive the on-screen wizard.
Keeping the logic here (not in the Qt dialogs or the device loop) means it is
unit-tested offline -- a calibration that must be trustworthy on a military
product cannot have its capture logic live only in an untested UI callback.

Two collectors, sharing one stillness primitive:

* :class:`StaticCollector` -- accumulates a still window: it tracks the running
  mean of the current motionless streak and reports ``ready`` once the device has
  held still (low gyro rate AND steady accel) for ``window_s``. Any motion clears
  the streak and restarts it. This is the same gate the live startup uses,
  extracted so both paths share one tested implementation.
* :class:`SixFaceCollector` -- drives the six-position accel routine: it watches
  for a still window, identifies which face is up from the gravity direction,
  captures one mean per distinct face, requires the operator to move to a NEW
  face between captures, and once all six are in hand solves the calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .accel_calib import (
    G_STANDARD,
    AccelCalibration,
    solve_accel_calibration,
)

_FACE_NAMES = ("+X up", "-X up", "+Y up", "-Y up", "+Z up", "-Z up")
# Face k's expected specific-force direction (what a level, perfect sensor reads
# at rest). Index matches accel_calib.SIX_FACES.
_FACE_DIRS = np.array([
    [+1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, +1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, +1.0], [0.0, 0.0, -1.0],
])


def face_name(index: int) -> str:
    """Human label for face index 0..5 (``+X up`` ... ``-Z up``)."""
    return _FACE_NAMES[index]


@dataclass
class StaticCollectorConfig:
    """Stillness gate thresholds (tighter than live startup for cal quality)."""

    gyro_thresh: float = 0.05      # rad/s; max |gyro| considered still
    accel_dev_thresh: float = 0.4  # m/s^2; max accel deviation from streak mean
    window_s: float = 0.6          # how long it must hold still
    min_samples: int = 30          # and at least this many samples


class StaticCollector:
    """Accumulate the mean of a motionless streak; reset on any motion."""

    def __init__(self, cfg: StaticCollectorConfig | None = None) -> None:
        self.cfg = cfg or StaticCollectorConfig()
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._gsum = np.zeros(3)
        self._asum = np.zeros(3)
        self._t_start: float | None = None
        self._t_last: float | None = None

    def feed(self, gyro, accel, t_s: float) -> bool:
        """Add one sample. Returns ``True`` if the device is currently moving."""
        g = np.asarray(gyro, dtype=np.float64)
        a = np.asarray(accel, dtype=np.float64)
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(a))):
            return True
        moving = float(np.linalg.norm(g)) > self.cfg.gyro_thresh
        if self._n > 0:
            mean_a = self._asum / self._n
            if float(np.linalg.norm(a - mean_a)) > self.cfg.accel_dev_thresh:
                moving = True
        if moving:
            self.reset()
            return True
        self._n += 1
        self._gsum += g
        self._asum += a
        if self._t_start is None:
            self._t_start = t_s
        self._t_last = t_s
        return False

    @property
    def progress(self) -> float:
        """Fraction of the still window elapsed (0..1)."""
        if self._t_start is None or self._t_last is None:
            return 0.0
        return min(1.0, (self._t_last - self._t_start) / self.cfg.window_s)

    @property
    def ready(self) -> bool:
        """True once a clean still window of ``window_s`` is in hand."""
        return (self._n >= self.cfg.min_samples
                and self._t_start is not None
                and (self._t_last - self._t_start) >= self.cfg.window_s)

    @property
    def n(self) -> int:
        return self._n

    @property
    def gyro_mean(self) -> np.ndarray:
        return self._gsum / max(self._n, 1)

    @property
    def accel_mean(self) -> np.ndarray:
        return self._asum / max(self._n, 1)


@dataclass
class SixFaceStatus:
    """Snapshot the UI polls each frame to render the six-face wizard."""

    moving: bool
    progress: float                       # current still-window progress 0..1
    captured: tuple                       # face indices already captured
    just_captured: int | None             # face captured on THIS feed, else None
    complete: bool
    message: str
    calibration: AccelCalibration | None = None   # set once complete


@dataclass
class SixFaceConfig:
    static: StaticCollectorConfig = field(default_factory=StaticCollectorConfig)
    # A still mean only counts as a face if its dominant axis carries at least
    # this fraction of g (i.e. the device is held close enough to a true face).
    axis_min_frac: float = 0.92
    g: float = G_STANDARD


class SixFaceCollector:
    """Six-position accel capture: one mean per distinct face, then solve."""

    def __init__(self, cfg: SixFaceConfig | None = None) -> None:
        self.cfg = cfg or SixFaceConfig()
        self._coll = StaticCollector(self.cfg.static)
        self._caps: dict[int, np.ndarray] = {}
        self._need_move = False            # must leave a face before the next
        self._cal: AccelCalibration | None = None

    # -- introspection ----------------------------------------------------- #
    @property
    def captured_faces(self) -> tuple:
        return tuple(sorted(self._caps))

    def remaining_faces(self) -> tuple:
        return tuple(i for i in range(6) if i not in self._caps)

    @property
    def complete(self) -> bool:
        return len(self._caps) == 6

    def reset(self) -> None:
        self._coll.reset()
        self._caps.clear()
        self._need_move = False
        self._cal = None

    def _identify_face(self, accel_mean: np.ndarray) -> int | None:
        """Which face (0..5) the mean accel implies, or None if ambiguous."""
        ax = int(np.argmax(np.abs(accel_mean)))
        val = accel_mean[ax]
        if abs(val) < self.cfg.axis_min_frac * self.cfg.g:
            return None
        return ax * 2 + (0 if val > 0 else 1)

    # -- drive ------------------------------------------------------------- #
    def feed(self, gyro, accel, t_s: float) -> SixFaceStatus:
        if self.complete:
            return self._status(False, None, "All six faces captured.")

        moving = self._coll.feed(gyro, accel, t_s)
        if moving:
            # Leaving a face clears the "must move first" latch.
            self._need_move = False
            return self._status(True, None, self._prompt())

        if self._need_move:
            # Still, but we just captured a face here -- wait for the operator
            # to rotate to a new face (motion) before accepting another.
            return self._status(False, None,
                                "Captured. Rotate to the next face.")

        if not self._coll.ready:
            return self._status(False, None,
                                "Hold still on a face... " + self._prompt())

        face = self._identify_face(self._coll.accel_mean)
        if face is None:
            return self._status(False, None,
                                "Not square to a face -- align a face up/down.")
        if face in self._caps:
            return self._status(False, None,
                                f"{face_name(face)} already done. " + self._prompt())

        # Accept this face.
        self._caps[face] = self._coll.accel_mean.copy()
        self._need_move = True
        self._coll.reset()
        if self.complete:
            self._solve()
            return self._status(False, face, "Calibration complete.")
        return self._status(False, face, f"Captured {face_name(face)}.")

    def _solve(self) -> None:
        idx = sorted(self._caps)
        caps = [self._caps[i] for i in idx]
        dirs = [_FACE_DIRS[i] for i in idx]
        self._cal = solve_accel_calibration(caps, directions=dirs, g=self.cfg.g)

    def _prompt(self) -> str:
        rem = self.remaining_faces()
        if not rem:
            return ""
        return "Remaining: " + ", ".join(face_name(i) for i in rem)

    def _status(self, moving: bool, just: int | None,
                message: str) -> SixFaceStatus:
        return SixFaceStatus(
            moving=moving,
            progress=self._coll.progress,
            captured=self.captured_faces,
            just_captured=just,
            complete=self.complete,
            message=message,
            calibration=self._cal,
        )
