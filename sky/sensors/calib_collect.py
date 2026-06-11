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


# -- quality gates --------------------------------------------------------- #
# A military product must REFUSE a bad calibration, not silently store it. These
# are the acceptance bounds for a hand-held bench calibration; the wizard keeps
# SAVE disabled until the captured data clears them.
GYRO_MAX_STD = 0.02          # rad/s; max per-axis gyro noise in the still window
GYRO_MIN_SAMPLES = 80        # need a long enough window for a trustworthy mean
ACCEL_MAX_RESIDUAL_G = 0.5   # m/s^2; max RMS |a_cal|-g over the six faces


@dataclass(frozen=True)
class CalibVerdict:
    """Accept/reject decision for a finished calibration.

    ``ok`` gates the wizard's SAVE button; ``message`` is shown to the operator;
    ``metric`` is the figure that was tested (gyro noise std or accel residual).
    """

    ok: bool
    message: str
    metric: float = 0.0


def gyro_bias_verdict(std_max: float, n: int, *,
                      max_std: float = GYRO_MAX_STD,
                      min_samples: int = GYRO_MIN_SAMPLES) -> CalibVerdict:
    """Accept a gyro-bias window only if it is long AND genuinely steady.

    The stillness gate already rejects per-sample motion, but a creeping or
    vibrating rest can still pass it while leaving the window noisy -- which
    poisons the mean. We additionally require the per-axis gyro std over the
    window to stay under ``max_std``.
    """
    if n < min_samples:
        return CalibVerdict(
            False, f"Too few still samples ({n} < {min_samples}). "
            "Hold still longer.", float(std_max))
    if not np.isfinite(std_max) or std_max > max_std:
        return CalibVerdict(
            False, f"Surface not steady enough (gyro noise {std_max:.4f} > "
            f"{max_std:.4f} rad/s). Use a firmer rest and retry.",
            float(std_max))
    return CalibVerdict(
        True, f"Accepted (gyro noise {std_max:.4f} rad/s).", float(std_max))


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
        self._gsq = np.zeros(3)
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
        self._gsq += g * g
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
    def gyro_std(self) -> np.ndarray:
        """Per-axis std of gyro over the still streak (window steadiness)."""
        if self._n < 2:
            return np.zeros(3)
        mean = self._gsum / self._n
        var = self._gsq / self._n - mean * mean
        return np.sqrt(np.clip(var, 0.0, None))

    @property
    def gyro_std_max(self) -> float:
        """Worst-axis gyro std over the streak (the figure the gate tests)."""
        return float(np.max(self.gyro_std))

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
    # A still mean counts as a face only if the pose is SQUARE: the dominant
    # axis must carry at least this fraction of the vector's LENGTH (i.e. the
    # off-axis tilt is small). This is judged RELATIVE to the measured vector,
    # not against an absolute g, so an axis whose UNCALIBRATED gain/bias is off
    # still classifies. Gating on absolute magnitude (>= 0.92 g) would make a
    # mis-scaled axis impossible to ever capture -- the exact axis the user is
    # trying to calibrate -- a chicken-and-egg trap. 0.95 ~= within 18 deg of
    # square (a 45 deg tilt gives only 0.71 and is rejected).
    axis_dom_frac: float = 0.95
    # Loose sanity band on |a| (as multiples of g) to reject free-fall / heavy
    # vibration; deliberately wide so per-axis scale errors still pass.
    mag_lo_frac: float = 0.6
    mag_hi_frac: float = 1.4
    # Reject the solved calibration if its sphere residual exceeds this (the
    # faces were held too crooked / unsteady to trust the fit).
    max_residual_g: float = ACCEL_MAX_RESIDUAL_G
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

    @property
    def calibration(self) -> AccelCalibration | None:
        """The solved calibration once all six faces are captured, else None."""
        return self._cal

    def verdict(self) -> CalibVerdict:
        """Accept the solved calibration only if all six faces fit the sphere.

        The capture loop already enforces six distinct, square faces; this gates
        the FIT quality: a high RMS residual means the poses were too crooked or
        unsteady to trust, so SAVE stays disabled until the operator re-runs.
        """
        if not self.complete or self._cal is None:
            return CalibVerdict(
                False, f"Incomplete ({len(self._caps)}/6 faces captured).", 0.0)
        r = self._cal.residual_g
        if not np.isfinite(r) or r > self.cfg.max_residual_g:
            return CalibVerdict(
                False, f"Residual too high ({r:.4f} > "
                f"{self.cfg.max_residual_g:.4f} m/s²). Re-run holding each face "
                "squarer and stiller.", float(r))
        return CalibVerdict(
            True, f"Accepted (residual {r:.4f} m/s²).", float(r))

    def reset(self) -> None:
        self._coll.reset()
        self._caps.clear()
        self._need_move = False
        self._cal = None

    def _identify_face(self, accel_mean: np.ndarray) -> int | None:
        """Which face (0..5) the mean accel implies, or None if ambiguous.

        Judges the pose by SQUARENESS (the dominant axis dominates the vector's
        length), not by hitting an absolute g -- so an axis with a real scale or
        bias error is still classifiable and therefore calibratable.
        """
        norm = float(np.linalg.norm(accel_mean))
        g = self.cfg.g
        if norm < self.cfg.mag_lo_frac * g or norm > self.cfg.mag_hi_frac * g:
            return None
        ax = int(np.argmax(np.abs(accel_mean)))
        val = float(accel_mean[ax])
        if abs(val) < self.cfg.axis_dom_frac * norm:
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
