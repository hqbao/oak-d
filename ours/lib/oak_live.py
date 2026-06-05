"""A single OAK-D device shared by the split camera and IMU reader flows.

The OAK-D is **single-client**: only one ``depthai`` pipeline may be connected to
a given device at a time. The split front-end has two independent sources
(:class:`~ours.flows.cam_reader.sources.LiveCamSource` and
:class:`~ours.flows.imu_reader.sources.LiveImuSource`); if each opened its own
pipeline the second one would fail with ``X_LINK_DEVICE_NOT_FOUND``.

:class:`SharedLiveDevice` is the fix: it owns ONE pipeline carrying both mono
cameras and the IMU node, and hands out the per-stream output queues. Both
sources :meth:`acquire` it (reference-counted, so whoever connects first opens
the device and the other just attaches) and :meth:`release` it on stop; the
pipeline is closed once the last user releases it.

``depthai`` is imported lazily inside :meth:`acquire`, so importing this module on
the offline path never pulls the device library. Hardware-only: validated on the
bench, not in the offline test harness (the reference-counting lifecycle is
covered offline with a fake opener).
"""
from __future__ import annotations

import threading
from typing import Callable


class SharedLiveDevice:
    """One OAK-D pipeline (stereo + IMU) shared, reference-counted, thread-safe.

    ``opener`` is an injection seam for testing: it returns
    ``(handle, q_left, q_right, q_imu)`` and defaults to the real depthai opener.
    ``is_running_fn`` reports whether the handle is still streaming.
    """

    def __init__(self, *, width: int = 640, height: int = 400, fps: int = 20,
                 imu_rate_hz: int = 200,
                 opener: Callable[["SharedLiveDevice"], tuple] | None = None,
                 ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.imu_rate_hz = int(imu_rate_hz)
        self._opener = opener or _open_oak_pipeline
        self._lock = threading.Lock()
        self._refs = 0
        self._handle = None
        self.q_left = None
        self.q_right = None
        self.q_imu = None

    def acquire(self) -> None:
        """Connect (the first caller opens; later callers just attach).

        Raises whatever the opener raises (e.g. ``X_LINK_DEVICE_NOT_FOUND``)
        WITHOUT incrementing the reference count, so a failed open leaves the
        device closed and a later :meth:`release` is a safe no-op.
        """
        with self._lock:
            if self._handle is None:
                handle, q_left, q_right, q_imu = self._opener(self)
                self._handle = handle
                self.q_left = q_left
                self.q_right = q_right
                self.q_imu = q_imu
            self._refs += 1

    def release(self) -> None:
        """Detach; close the pipeline once the last user releases it."""
        with self._lock:
            if self._refs > 0:
                self._refs -= 1
            if self._refs == 0 and self._handle is not None:
                try:
                    self._handle.stop()
                except Exception:
                    pass
                self._handle = None
                self.q_left = self.q_right = self.q_imu = None

    def is_running(self) -> bool:
        handle = self._handle
        if handle is None:
            return False
        try:
            return bool(handle.isRunning())
        except Exception:
            return False


def _open_oak_pipeline(dev: SharedLiveDevice) -> tuple:
    """Open one depthai pipeline with both mono cameras + the IMU (lazy import)."""
    import depthai as dai

    p = dai.Pipeline()
    left = p.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B, sensorFps=dev.fps)
    right = p.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C, sensorFps=dev.fps)
    imu = p.create(dai.node.IMU)
    imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW,
                         dai.IMUSensor.GYROSCOPE_RAW], dev.imu_rate_hz)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(20)

    left_out = left.requestOutput((dev.width, dev.height))
    right_out = right.requestOutput((dev.width, dev.height))
    q_left = left_out.createOutputQueue(maxSize=4, blocking=False)
    q_right = right_out.createOutputQueue(maxSize=4, blocking=False)
    q_imu = imu.out.createOutputQueue(maxSize=100, blocking=False)

    p.start()
    return p, q_left, q_right, q_imu
