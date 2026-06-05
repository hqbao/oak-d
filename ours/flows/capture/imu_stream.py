"""Minimal IMU-only device stream for the calibration wizards.

The gyro/accel calibration wizards need the raw IMU at full rate but NOT the
cameras or the VIO graph -- opening the heavy stereo pipeline just to read the
accelerometer would be wasteful and would couple calibration to the depth code.
``ImuStream`` therefore opens a tiny depthai pipeline with only the IMU node and
pushes each sample to a callback on a background thread.

It mirrors the :class:`ours.ui.source.PoseSource` lifecycle (``start`` /
``stop`` / ``error`` / ``device_id``) so the Qt dialogs can drive it the same
way they drive a pose source. depthai is imported lazily inside ``start`` so
importing this module never pulls the device library on the offline/replay path.

This is hardware-facing and cannot run in the offline test harness; the logic it
feeds (the stillness gate + six-face collector) is what carries the unit tests.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

# (gyro rad/s, accel m/s^2, t_seconds) -> None
ImuCallback = Callable[[np.ndarray, np.ndarray, float], None]


class ImuStream:
    """Background reader of the OAK-D IMU (accelerometer + gyroscope)."""

    def __init__(self, rate_hz: int = 200) -> None:
        self.rate_hz = int(rate_hz)
        self.error: str | None = None
        self.device_id: str = "default"
        self._cb: ImuCallback | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pipeline = None

    # ------------------------------------------------------------------ #
    def start(self, callback: ImuCallback) -> None:
        """Open the device and begin streaming IMU samples to ``callback``."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._cb = callback
        self.error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 1.5) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        try:
            import depthai as dai
        except Exception as e:                                    # noqa: BLE001
            self._fail(f"depthai not available: {e}")
            return
        p = None
        try:
            p = dai.Pipeline()
            imu = p.create(dai.node.IMU)
            imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW,
                                 dai.IMUSensor.GYROSCOPE_RAW], self.rate_hz)
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(20)
            q = imu.out.createOutputQueue(maxSize=100, blocking=False)
            p.start()
            self._pipeline = p
            self.device_id = self._read_device_id(p)

            while not self._stop.is_set() and p.isRunning():
                msg = q.tryGet()
                if msg is None:
                    time.sleep(0.002)
                    continue
                for pkt in msg.packets:
                    a = pkt.acceleroMeter
                    g = pkt.gyroscope
                    accel = np.array([a.x, a.y, a.z], dtype=np.float64)
                    gyro = np.array([g.x, g.y, g.z], dtype=np.float64)
                    try:
                        t_s = g.getTimestampDevice().total_seconds()
                    except Exception:
                        t_s = time.monotonic()
                    if self._cb is not None:
                        self._cb(gyro, accel, t_s)
        except Exception as e:                                    # noqa: BLE001
            self._fail(f"IMU stream failed: {e}")
        finally:
            if p is not None:
                try:
                    p.stop()
                except Exception:
                    pass

    @staticmethod
    def _read_device_id(pipeline) -> str:
        try:
            dev = pipeline.getDefaultDevice()
        except Exception:
            return "default"
        for attr in ("getDeviceId", "getMxId", "getDeviceName"):
            fn = getattr(dev, attr, None)
            if callable(fn):
                try:
                    val = fn()
                    if val:
                        return str(val)
                except Exception:
                    pass
        return "default"

    def _fail(self, msg: str) -> None:
        self.error = msg
        import sys
        print(f"[imu-stream] {msg}", file=sys.stderr)
