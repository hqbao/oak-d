"""``imu_camera.device`` -- live OAK-D acquisition + boot-time calibration.

The only hardware-touching corner of the project: opening the single shared
device, reading the boot references the live pipeline needs, and decoding the
device's raw IMU packets. ``depthai`` is imported lazily inside these modules, so
importing this package never pulls the device library (keeps the offline / replay
path depthai-free).

* :class:`~imu_camera.device.oak_live.SharedLiveDevice` -- one
  reference-counted pipeline (stereo + IMU) shared by the cam / imu reader
  modules.
* :func:`~imu_camera.device.live_calib.read_live_calibration` --
  intrinsics + IMU->camera extrinsics + the startup gravity-align / cached gyro
  bias.
* :mod:`~imu_camera.device.imu_decode` -- decode the OAK device's raw IMU
  packets into ``(ts, gyro, accel)`` samples (live only).
* :mod:`~imu_camera.device.calib_status` /
  :mod:`~imu_camera.device.camera_calib_store` -- the per-device user-calibration
  status check + on-disk store.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .oak_live import SharedLiveDevice

__all__ = ["SharedLiveDevice", "LiveFrontEndCalib", "read_live_calibration"]

if TYPE_CHECKING:                       # pragma: no cover -- type-checkers only
    from .live_calib import LiveFrontEndCalib, read_live_calibration


def __getattr__(name: str):
    """Lazily re-export the live-calibration API.

    ``live_calib`` reads boot-time references via ``ResolutionProfile`` (from the
    vendored comms config) and is HARDWARE-only; deferring its import keeps the
    package importable on the offline / replay path -- the live front-end builder
    imports it explicitly only when ``--live`` is used.
    """
    if name in ("LiveFrontEndCalib", "read_live_calibration"):
        from . import live_calib
        return getattr(live_calib, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
