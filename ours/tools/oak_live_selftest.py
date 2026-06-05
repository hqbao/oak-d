#!/usr/bin/env python3
"""Offline self-test for the single-client shared OAK-D device.

The OAK-D allows only one pipeline per device. The split front-end has two live
sources (camera + IMU); before this they each opened their own pipeline, so the
second failed with ``X_LINK_DEVICE_NOT_FOUND``.
:class:`~ours.lib.oak_live.SharedLiveDevice` fixes that by sharing one pipeline,
reference-counted. This test covers that lifecycle with a FAKE opener (no
device): the depthai pipeline itself is bench-only, but the single-client
reference-counting -- the part that was broken -- is verified here.

Run::

    python -m ours.tools.oak_live_selftest
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.flows.cam.sources import LiveCamSource                    # noqa: E402
from ours.flows.imu_cam.sources import LiveImuSource                # noqa: E402
from ours.lib.oak_live import SharedLiveDevice                      # noqa: E402


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


class _FakeHandle:
    def __init__(self) -> None:
        self._running = True
        self.stopped = False

    def isRunning(self) -> bool:
        return self._running

    def stop(self) -> None:
        self.stopped = True
        self._running = False


class _NullQueue:
    def tryGet(self):
        return None


class _FrameQueue:
    """Yields the given frames once, then None."""

    def __init__(self, frames) -> None:
        self._frames = list(frames)

    def tryGet(self):
        return self._frames.pop(0) if self._frames else None


class _FakeFrame:
    def __init__(self, seq: int) -> None:
        self._seq = seq

    def getSequenceNum(self) -> int:
        return self._seq

    def getCvFrame(self) -> np.ndarray:
        return np.full((4, 4), self._seq, dtype=np.uint8)

    def getTimestampDevice(self):
        raise RuntimeError("no device clock in the fake")   # falls back to wall


def _counting_opener(calls, queues=None):
    def _open(dev):
        calls.append(1)
        ql, qr, qi = queues or (_NullQueue(), _NullQueue(), _NullQueue())
        return _FakeHandle(), ql, qr, qi
    return _open


def test_refcount_single_open() -> None:
    print(" reference-counted single open/close")
    calls: list[int] = []
    dev = SharedLiveDevice(opener=_counting_opener(calls))
    dev.acquire()
    dev.acquire()
    _check(len(calls) == 1, "two acquires open the pipeline exactly once")
    _check(dev.is_running(), "device reports running after acquire")
    handle = dev._handle
    dev.release()
    _check(dev.is_running() and not handle.stopped,
           "still open while one user holds it")
    dev.release()
    _check(handle.stopped and not dev.is_running(),
           "closed once the last user releases it")


def test_failed_open_is_safe() -> None:
    print(" failed open leaves no dangling reference")
    def _bad(dev):
        raise RuntimeError("Failed to find device: X_LINK_DEVICE_NOT_FOUND")
    dev = SharedLiveDevice(opener=_bad)
    raised = False
    try:
        dev.acquire()
    except RuntimeError:
        raised = True
    _check(raised, "acquire propagates the open error")
    _check(dev._refs == 0, "failed open did not increment the ref count")
    dev.release()                                    # must be a safe no-op
    _check(not dev.is_running(), "release after failed open is a no-op")


def test_single_client_shared() -> None:
    print(" camera + IMU share ONE device (single-client)")
    calls: list[int] = []
    dev = SharedLiveDevice(opener=_counting_opener(calls))
    cam = LiveCamSource(dev)
    imu = LiveImuSource(dev)

    cam.open()                                       # acquire #1 -> opens
    imu.start(lambda t, g, a: None)                  # acquire #2 -> attaches
    deadline = time.time() + 3.0
    while time.time() < deadline and imu.error is None and dev._refs < 2:
        time.sleep(0.01)
    _check(len(calls) == 1, "the camera and IMU opened the device only once")
    _check(imu.error is None, f"IMU attached without error ({imu.error})")

    imu.stop()
    cam.close()
    _check(not dev.is_running(), "device closed after both released it")


def test_cam_read_pairs_by_seq() -> None:
    print(" camera reads a seq-matched stereo pair off the shared queues")
    calls: list[int] = []
    queues = (_FrameQueue([_FakeFrame(0)]), _FrameQueue([_FakeFrame(0)]),
              _NullQueue())
    dev = SharedLiveDevice(opener=_counting_opener(calls, queues))
    cam = LiveCamSource(dev)
    cam.open()
    item = cam.read()
    _check(item is not None, "read returns a pair")
    seq, ts_ns, left, right = item
    _check(seq == 0, "paired on the shared sequence number")
    _check(left.shape == (4, 4) and right.shape == (4, 4), "grayscale frames")
    _check(isinstance(ts_ns, int) and ts_ns > 0, "timestamp falls back to wall")
    cam.close()
    _check(not dev.is_running(), "device closed on cam.close()")


class _MutexQueue:
    """Fake output queue that raises if read AFTER its pipeline was stopped.

    Emulates depthai's ``mutex lock failed: Invalid argument`` -- ``tryGet`` on a
    queue whose pipeline (and its C++ queue mutex) was destroyed by a concurrent
    ``release``. A correct :meth:`SharedLiveDevice.poll` (read + teardown under
    one lock) must NEVER call ``tryGet`` once the handle is stopped.
    """

    def __init__(self, handle) -> None:
        self._handle = handle

    def tryGet(self):
        if self._handle.stopped:
            raise RuntimeError("mutex lock failed: Invalid argument")
        return None


def test_poll_is_teardown_race_safe() -> None:
    print(" queue reads never touch a destroyed pipeline (teardown race)")

    box: dict[str, object] = {}

    def _opener(dev):
        h = _FakeHandle()
        box["h"] = h
        q = _MutexQueue(h)
        return h, q, q, q

    dev = SharedLiveDevice(opener=_opener)
    dev.acquire()                                    # cam ref
    dev.acquire()                                    # imu ref

    errors: list[Exception] = []
    stop = threading.Event()

    def hammer() -> None:
        try:
            while not stop.is_set():
                dev.poll("imu")
                dev.poll("left")
                dev.poll("right")
        except Exception as e:                       # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=hammer) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.05)                                 # let the readers spin
    dev.release()                                    # refs 2 -> 1 (no destroy)
    dev.release()                                    # refs 1 -> 0 -> destroy
    time.sleep(0.05)                                 # readers race the teardown
    stop.set()
    for t in threads:
        t.join(timeout=2.0)

    _check(box["h"].stopped, "pipeline was destroyed by the last release")
    _check(not errors,
           f"no reader hit a destroyed queue mutex ({errors[:1]})")
    _check(dev.poll("imu") is None, "poll returns None once the device is closed")


def main() -> int:
    print("oak_live_selftest")
    test_refcount_single_open()
    test_failed_open_is_safe()
    test_single_client_shared()
    test_cam_read_pairs_by_seq()
    test_poll_is_teardown_race_safe()
    _check("depthai" not in sys.modules,
           "fake-opener path never imported depthai (stays lazy)")
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
