#!/usr/bin/env python3
"""Offline self-test for :class:`ours.lib.imu.timed_buffer.TimedImuBuffer`.

Verifies the timestamp-drain contract the split camera/IMU front-end relies on,
including the multi-threaded producer/consumer path -- all without any device.

Run::

    python -m ours.tools.timed_buffer_selftest
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.imu.timed_buffer import TimedImuBuffer  # noqa: E402


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def test_drain_intervals() -> None:
    """Successive drains return disjoint, contiguous, ordered intervals."""
    buf = TimedImuBuffer()
    for i in range(10):
        buf.append(i * 100, [i, 0, 0], [0, 0, 9.8])  # t = 0,100,...,900

    ts1, g1, a1 = buf.drain_until(250)          # t <= 250 -> 0,100,200
    ts2, g2, a2 = buf.drain_until(550)          # 300,400,500
    ts3, g3, a3 = buf.drain_until(10_000)       # the rest

    _check(ts1.tolist() == [0, 100, 200], "first drain = (.., 250]")
    _check(ts2.tolist() == [300, 400, 500], "second drain = (250, 550]")
    _check(ts3.tolist() == [600, 700, 800, 900], "third drain = remainder")
    _check(g1.shape == (3, 3) and a1.shape == (3, 3), "shapes (M,3)")
    _check(float(g2[0, 0]) == 3.0, "payload carried through (gyro x)")
    total = len(ts1) + len(ts2) + len(ts3)
    _check(total == 10, "no sample lost or duplicated across drains")


def test_empty_and_dtypes() -> None:
    buf = TimedImuBuffer()
    ts, g, a = buf.drain_until(123)
    _check(ts.shape == (0,) and g.shape == (0, 3) and a.shape == (0, 3),
           "empty drain returns correctly-shaped empties")
    buf.append(5, [1, 2, 3], [4, 5, 6])
    ts, g, a = buf.drain_until(4)               # cut before the only sample
    _check(len(ts) == 0, "sample after the cut is NOT drained")
    ts, g, a = buf.drain_until(5)               # inclusive bound
    _check(len(ts) == 1 and ts.dtype == np.int64, "inclusive cut; int64 ts")
    _check(g.dtype == np.float64, "float64 gyro")


def test_capacity_eviction() -> None:
    buf = TimedImuBuffer(capacity=4)
    for i in range(10):
        buf.append(i, [i, 0, 0], [0, 0, 0])
    _check(len(buf) == 4, "capacity caps live size")
    _check(buf.dropped == 6, "evicted-sample count tracked")
    ts, _, _ = buf.drain_until(100)
    _check(ts.tolist() == [6, 7, 8, 9], "only newest survive eviction")


def test_wait_until_timeout() -> None:
    buf = TimedImuBuffer()
    t0 = time.monotonic()
    covered = buf.wait_until(1000, timeout=0.05)
    dt = time.monotonic() - t0
    _check(covered is False, "wait_until times out when uncovered")
    _check(dt >= 0.05, "wait_until actually blocked for the timeout")
    buf.append(1000, [0, 0, 0], [0, 0, 0])
    _check(buf.wait_until(1000, timeout=0.05) is True, "covered after append")


def test_wait_until_close() -> None:
    buf = TimedImuBuffer()

    def closer():
        time.sleep(0.02)
        buf.close()

    threading.Thread(target=closer, daemon=True).start()
    covered = buf.wait_until(1_000_000, timeout=5.0)
    _check(covered is False, "close() unblocks wait_until promptly")


def test_threaded_producer_consumer() -> None:
    """A background producer + timestamp-driven consumer never lose a sample."""
    buf = TimedImuBuffer(capacity=10_000)
    n = 2000

    def producer():
        for i in range(n):
            buf.append(i, [i, -i, 0], [0, 0, 9.8])
            if i % 200 == 0:
                time.sleep(0.001)
        buf.close()

    collected: list[int] = []
    th = threading.Thread(target=producer, daemon=True)
    th.start()

    cut = 0
    while True:
        covered = buf.wait_until(cut, timeout=0.2)
        ts, _, _ = buf.drain_until(cut)
        collected.extend(int(x) for x in ts)
        if not covered and buf.newest_t is not None and cut > buf.newest_t:
            break
        cut += 50
        if cut > n + 100:
            ts, _, _ = buf.drain_until(cut)     # final sweep
            collected.extend(int(x) for x in ts)
            break
    th.join(timeout=2.0)

    _check(collected == sorted(collected), "consumer saw samples in time order")
    _check(collected == list(range(n)),
           f"every produced sample drained exactly once ({len(collected)}/{n})")


def main() -> int:
    print("timed_buffer_selftest")
    test_drain_intervals()
    test_empty_and_dtypes()
    test_capacity_eviction()
    test_wait_until_timeout()
    test_wait_until_close()
    test_threaded_producer_consumer()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
