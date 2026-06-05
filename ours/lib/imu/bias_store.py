"""Backward-compatibility shim -- the gyro bias store moved into ``calib_store``.

The IMU calibration cache grew from gyro-bias-only to a unified store (gyro bias
+ six-position accel calibration). The full API now lives in
:mod:`ours.lib.imu.calib_store`; this module simply re-exports the gyro helpers
so older imports keep working. Prefer importing from ``calib_store`` directly.
"""
from __future__ import annotations

from .calib_store import (  # noqa: F401
    default_path,
    load_gyro_bias,
    save_gyro_bias,
)

__all__ = ["default_path", "load_gyro_bias", "save_gyro_bias"]
