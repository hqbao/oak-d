#!/usr/bin/env python3
"""Headless self-test for the :mod:`ours.tools.imucam_view` renderers.

Exercises the pure rendering functions (no window, no device) on a synthetic
:class:`~ours.lib.flow.messages.ImuCamPacket` so the visualiser's drawing path is
covered by the offline sweep. The interactive ``cv2.imshow`` loop is bench/eyeball
only and is not exercised here.

Run::

    python -m ours.tools.imucam_view_selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.flow.messages import ImuCamPacket                     # noqa: E402
from ours.tools.imucam_view import (                                # noqa: E402
    GyroChart, compose, render_accel3d, render_cameras,
)


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def _packet(seq: int, m: int) -> ImuCamPacket:
    rng = np.random.default_rng(seq)
    left = rng.integers(0, 255, size=(200, 320), dtype=np.uint8)
    right = rng.integers(0, 255, size=(200, 320), dtype=np.uint8)
    ts = np.arange(m, dtype=np.int64) * 5_000_000 + seq * 50_000_000
    gyro = rng.normal(0, 0.3, size=(m, 3))
    accel = np.tile([0.1, 9.7, 0.2], (m, 1)) + rng.normal(0, 0.05, (m, 3))
    return ImuCamPacket(seq=seq, ts_ns=int(ts[-1]) if m else seq,
                        gray_left=left, gray_right=right,
                        imu_ts=ts, gyro=gyro, accel=accel)


def test_cameras() -> None:
    img = render_cameras(np.zeros((200, 320), np.uint8),
                         np.zeros((200, 320), np.uint8))
    _check(img.ndim == 3 and img.shape[2] == 3, "camera panel is BGR")
    _check(img.shape[0] == 360, "camera panel fitted to panel height")
    mono = render_cameras(np.zeros((200, 320), np.uint8), None)
    _check(mono.shape[1] < img.shape[1], "right=None -> left only (narrower)")


def test_accel3d() -> None:
    a = render_accel3d(np.tile([0, 9.8, 0], (5, 1)))
    _check(a.shape == (360, 360, 3), "accel panel shape")
    empty = render_accel3d(np.empty((0, 3)))
    _check(empty.shape == (360, 360, 3), "accel panel handles empty samples")


def test_chart_and_compose() -> None:
    chart = GyroChart()
    chart.add(np.zeros((0, 3)))                         # empty is a no-op
    _check(chart.render().shape == (360, 360, 3), "empty chart renders")
    row = None
    for seq in range(5):
        row = compose(_packet(seq, m=9), chart)
    _check(row is not None and row.ndim == 3, "compose returns an image")
    # cameras(320*2 fitted to 360 -> 576 each ~1152) + gyro 360 + accel 360.
    _check(row.shape[0] == 360 and row.shape[1] > 360 * 3,
           "composed row spans cameras + gyro + accel")
    _check(row.dtype == np.uint8, "composed row is uint8")

    chart.clear()
    _check(len(chart._hist) == 0, "chart clear empties history")

    # A packet with no IMU samples must still compose (degenerate interval).
    row0 = compose(_packet(99, m=0), chart)
    _check(row0.ndim == 3, "packet with 0 IMU samples still composes")


def main() -> int:
    print("imucam_view_selftest")
    test_cameras()
    test_accel3d()
    test_chart_and_compose()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
