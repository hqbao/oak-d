"""Shared decode of OAK-D (depthai) IMU report packets into plain samples.

The live capture flow and the IMU-only calibration stream both read the same
depthai ``IMU`` node. This module is the ONE place that knows the depthai packet
layout (``msg.packets[i].acceleroMeter`` / ``.gyroscope`` with a per-sample
device timestamp), so the two readers don't each re-implement -- and slowly
drift apart on -- the field extraction.

It is duck-typed over the depthai message and never imports depthai, so the
layout is regression-locked offline with a tiny fake packet object
(``imu_decode_selftest``); only the actual device read stays hardware-only.
"""
from __future__ import annotations

import numpy as np


def _vec3(o) -> np.ndarray:
    """Pull an ``(x, y, z)`` depthai vector into a float64 array."""
    return np.array([o.x, o.y, o.z], dtype=np.float64)


def decode_imu_packets(msg) -> list[tuple[np.ndarray, np.ndarray, float | None]]:
    """Decode one depthai IMU batch into ``[(gyro, accel, t_s), ...]``.

    ``gyro`` (rad/s) and ``accel`` (m/s^2) are float64 3-vectors in the raw
    sensor frame; ``t_s`` is the gyroscope device timestamp in seconds, or
    ``None`` when the packet carries no usable timestamp.

    Finite-checking and the choice of fallback clock are deliberately left to the
    caller: the live VIO loop, the startup leveler and the calibration stream
    each want a slightly different policy (drop the sample vs. substitute a host
    clock), so this function owns only the packet-field layout, nothing else.
    """
    out: list[tuple[np.ndarray, np.ndarray, float | None]] = []
    for pkt in msg.packets:
        gyro = _vec3(pkt.gyroscope)
        accel = _vec3(pkt.acceleroMeter)
        try:
            t_s = pkt.gyroscope.getTimestampDevice().total_seconds()
        except Exception:
            t_s = None
        out.append((gyro, accel, t_s))
    return out
