"""IMU sample sources for the split-acquisition :class:`ImuReaderFlow`.

The IMU-reader flow owns a :class:`~ours.lib.imu.timed_buffer.TimedImuBuffer` but
not the *origin* of the samples -- that is injected as an ``ImuSource`` so the
exact same flow runs offline (deterministic replay of a recorded session) and on
the bench (the OAK-D IMU). A source is a tiny lifecycle object:

* :meth:`ImuSource.start` -- begin pushing ``(t_ns, gyro, accel)`` to a callback
  on a background thread (the IMU never blocks the camera/consumer threads).
* :meth:`ImuSource.stop`  -- stop and join that thread.

``on_exhausted`` (replay only) lets the flow know the recorded stream ended so it
can close the buffer; the live source never exhausts until stopped.

Only :class:`LiveImuSource` touches depthai, and it imports it lazily inside
``start`` -- importing this module on the offline path never pulls the device
library.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from ...lib.io.reader import SessionReader

# (t_ns, gyro(3,), accel(3,)) -> None
SampleCallback = Callable[[int, np.ndarray, np.ndarray], None]


class ImuSource:
    """Lifecycle base for an IMU sample producer."""

    def start(self, on_sample: SampleCallback,
              on_exhausted: Callable[[], None] | None = None) -> None:
        raise NotImplementedError

    def stop(self, timeout: float = 2.0) -> None:
        raise NotImplementedError

    def is_running(self) -> bool:
        raise NotImplementedError


class ReplayImuSource(ImuSource):
    """Replays a recorded session's IMU on a background thread.

    Pushes the recorded samples in timestamp order. By default it streams them as
    fast as possible (deterministic offline tests don't want wall-clock pacing);
    set ``realtime=True`` to pace by the recorded inter-sample interval (scaled by
    ``speed``) for a faithful live-like visualisation.
    """

    def __init__(self, reader: SessionReader, *, realtime: bool = False,
                 speed: float = 1.0) -> None:
        self._imu = reader.load_imu()
        self._realtime = bool(realtime)
        self._speed = max(1e-6, float(speed))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self, on_sample: SampleCallback,
              on_exhausted: Callable[[], None] | None = None) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(on_sample, on_exhausted), daemon=True)
        self._thread.start()

    def _run(self, on_sample: SampleCallback,
             on_exhausted: Callable[[], None] | None) -> None:
        ts = self._imu["ts_ns"]
        gyro = self._imu["gyro"]
        accel = self._imu["accel"]
        prev_t = None
        for i in range(ts.shape[0]):
            if self._stop.is_set():
                break
            if self._realtime and prev_t is not None:
                dt = (int(ts[i]) - prev_t) * 1e-9 / self._speed
                if dt > 0:
                    time.sleep(min(dt, 0.2))
            prev_t = int(ts[i])
            on_sample(int(ts[i]), gyro[i], accel[i])
        if on_exhausted is not None:
            on_exhausted()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class LiveImuSource(ImuSource):
    """Streams the OAK-D IMU (accelerometer + gyroscope) at ``rate_hz``.

    Opens a minimal depthai pipeline with only the IMU node and pushes every
    decoded sample to the callback, tagging it with the gyro device timestamp
    (the clock shared with the camera frames). depthai is imported lazily.

    Hardware-facing: this path is validated on the bench, not in the offline test
    harness. The OAK-D is single-client, so this must not run while a
    :class:`~ours.flows.capture.live.LiveCaptureFlow` (which owns its own IMU) is
    open on the same device.
    """

    def __init__(self, rate_hz: int = 200) -> None:
        self.rate_hz = int(rate_hz)
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pipeline = None

    def start(self, on_sample: SampleCallback,
              on_exhausted: Callable[[], None] | None = None) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.error = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(on_sample,), daemon=True)
        self._thread.start()

    def _run(self, on_sample: SampleCallback) -> None:
        try:
            import depthai as dai
            from ...lib.imu.decode import decode_imu_packets
        except Exception as e:                                    # noqa: BLE001
            self.error = f"depthai not available: {e}"
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
            while not self._stop.is_set() and p.isRunning():
                msg = q.tryGet()
                if msg is None:
                    time.sleep(0.002)
                    continue
                for gyro, accel, t_s in decode_imu_packets(msg):
                    t_ns = int(t_s * 1e9) if t_s is not None else time.monotonic_ns()
                    on_sample(t_ns, gyro, accel)
        except Exception as e:                                    # noqa: BLE001
            self.error = f"IMU stream failed: {e}"
        finally:
            if p is not None:
                try:
                    p.stop()
                except Exception:
                    pass

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
