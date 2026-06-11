"""Combined IMU correction: gyro bias + six-position accel calibration.

The acquisition front-end publishes the *raw* IMU (exactly what the sensor
reported, before any correction) alongside the *calibrated* IMU. This module is
the single place that turns the persisted per-device calibration into a function
that maps a batch of raw samples to corrected ones, so the live flow and any
offline test apply the identical maths:

* **gyro**  -- subtract the zero-rate bias: ``w_cal = w_raw - bias``.
* **accel** -- the affine six-position correction ``a_cal = T (a_raw - b)``
  (see :class:`~sky.sensors.accel_calib.AccelCalibration`).

A missing piece is a pass-through: with no cached gyro bias the gyro is returned
unchanged, with no accel calibration the accel is returned unchanged. An
all-missing calibration is :meth:`is_identity` and the flow can skip the copy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .accel_calib import AccelCalibration
from .calib_store import load_accel_calib, load_gyro_bias


@dataclass(frozen=True)
class ImuCalibration:
    """Per-device IMU correction (either part may be absent -> pass-through)."""

    gyro_bias: np.ndarray | None = None         # (3,) rad/s, subtracted
    accel: AccelCalibration | None = None       # affine a_cal = T (a_raw - b)

    @property
    def is_identity(self) -> bool:
        """True when neither correction is present (raw == calibrated)."""
        return self.gyro_bias is None and self.accel is None

    def apply(self, gyro: np.ndarray, accel: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
        """Correct a batch of raw samples.

        ``gyro`` / ``accel`` are ``(M, 3)`` (or empty ``(0, 3)``); returns the
        corrected ``(gyro_cal, accel_cal)`` as new float64 arrays. The maths is
        linear, so applying it per-sample equals applying it to a mean -- callers
        may correct either.
        """
        g = np.asarray(gyro, dtype=np.float64)
        a = np.asarray(accel, dtype=np.float64)
        if self.gyro_bias is not None and g.size:
            g = g - self.gyro_bias
        else:
            g = g.copy()
        if self.accel is not None and a.size:
            a = self.accel.apply(a)
        else:
            a = a.copy()
        return g, a

    @classmethod
    def load(cls, device_id: str, path: Path | None = None) -> "ImuCalibration":
        """Load whatever calibration is cached for ``device_id`` (may be empty)."""
        return cls(
            gyro_bias=load_gyro_bias(device_id, path),
            accel=load_accel_calib(device_id, path),
        )
