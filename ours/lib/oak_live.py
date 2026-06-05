"""A single OAK-D device shared by the split camera and IMU reader flows.

The OAK-D is **single-client**: only one ``depthai`` pipeline may be connected to
a given device at a time. The split front-end has two independent sources
(:class:`~ours.flows.cam.sources.LiveCamSource` and
:class:`~ours.flows.imu_cam.sources.LiveImuSource`); if each opened its own
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
        self.device_id = "default"      # set once the pipeline opens (cal key)
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
                self.device_id = _read_device_id(handle)
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
        with self._lock:
            if self._handle is None:
                return False
            try:
                return bool(self._handle.isRunning())
            except Exception:
                return False

    def poll(self, which: str):
        """Thread-safe non-blocking read of one output queue.

        ``which`` is ``"left"`` / ``"right"`` / ``"imu"``; returns the next
        depthai message, or ``None`` if nothing is queued, the device is closed,
        or the queue raised. The read shares ``self._lock`` with
        :meth:`release`'s pipeline teardown, so a reader thread can NEVER call
        ``tryGet`` on a queue whose pipeline another thread is destroying -- the
        lifetime race that aborted the host with ``mutex lock failed: Invalid
        argument`` (which then starved the XLink and tripped the device
        watchdog). It also serialises the camera and IMU reads, so the two never
        enter depthai's per-device link concurrently.
        """
        with self._lock:
            if self._handle is None:
                return None
            q = {"left": self.q_left, "right": self.q_right,
                 "imu": self.q_imu}.get(which)
            if q is None:
                return None
            try:
                return q.tryGet()
            except Exception:
                return None

    def read_calibration(self):
        """Return the depthai ``CalibrationHandler`` of the open device.

        The shared device must already be :meth:`acquire`-d. Used once by the
        live front-end builder to read intrinsics + IMU->camera extrinsics, the
        same handle the monolithic capture flow read.
        """
        with self._lock:
            if self._handle is None:
                raise RuntimeError("device not acquired")
            return self._handle.getDefaultDevice().readCalibration()


def _read_device_id(handle) -> str:
    """Best-effort unique id of the open device (the per-device cache key).

    ``handle`` is the depthai pipeline; mirrors the id lookup the monolithic
    capture flow uses so both paths key the IMU calibration cache the same way.
    A fake handle (offline tests) without these methods just yields "default".
    """
    try:
        dev = handle.getDefaultDevice()
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
