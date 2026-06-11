"""The per-keyframe solve, factored out so in-process and subprocess engines run
*exactly the same code* (the whole offline byte-parity argument depends on this).

Each ``*_step`` takes a live map object + one keyframe snapshot and returns the
solve result (or ``None`` when there is nothing to publish for that keyframe).
These are pure functions of (map, snapshot): no threads, no queues, no flow/bus
knowledge -- they receive the map instance so the same function drives both the
synchronous :class:`~vio.mathlib.engine.inprocess.InProcessEngine` and the child of
:class:`~vio.mathlib.engine.subprocess.SubprocessEngine`.

The logic is lifted verbatim from the old in-thread ``RunBA`` task so the offline
path stays identical.
"""
from __future__ import annotations

from typing import Any


def ba_step(ba_map, snap: Any):
    """One windowed-BA keyframe: add the track snapshot, run BA.

    ``snap`` = ``(T_cw, ids, pts, depth_m, accel)`` in the raw f2f world frame
    (the flow inverts ``T_world_cam`` -> ``T_cw`` before submitting). Returns the
    refined latest ``T_cw`` (``4x4``) or ``None`` when the window has not yet
    enough structure to optimise.
    """
    T_cw, ids, pts, depth_m, accel = snap
    if ids is None or pts is None:
        return None
    ba_map.add_keyframe(T_cw, ids, pts, depth_m, accel_cam=accel)
    return ba_map.run_ba()                    # refined latest T_cw, or None


def vio_step(vio_map, snap: Any):
    """One tight-coupled VIO keyframe: add the track snapshot + IMU block, solve.

    Mirrors :func:`ba_step` but for the tight backend
    (:class:`vio.mathlib.backend.vio_window.WindowedVIOMap`): the snapshot is a
    SUPERSET of the loose one carrying the keyframe timestamp + the raw
    inter-keyframe IMU segment the joint optimiser preintegrates --

        ``snap`` = ``(T_cw, ids, pts, depth_m, ts_ns, imu_seg)``

    where ``imu_seg`` is ``(ts_ns, gyro_cam, accel_cam)`` in the camera optical
    frame (or ``None`` -> the map slices its stored stream, empty live). Returns
    the refined latest ``T_cw`` (``4x4``) or ``None`` when the window has not yet
    enough structure / IMU to optimise.
    """
    T_cw, ids, pts, depth_m, ts_ns, imu_seg = snap
    if ids is None or pts is None:
        return None
    vio_map.add_keyframe(T_cw, ids, pts, depth_m, ts_ns, imu_seg=imu_seg)
    return vio_map.run_ba()                    # refined latest T_cw, or None


# --------------------------------------------------------------------------- #
# Overlay extractors: a cheap, picklable snapshot of the live MAP for the 3D
# viewer (the visible "refined map behind the responsive marker"). All positions
# are camera world-frame (optical); the UI applies the single optical->NED display
# transform. These read REAL map outputs (refined keyframe poses / corrected
# SLAM poses + loop events) -- never a parallel/derived pipeline.
# --------------------------------------------------------------------------- #

def ba_overlay(ba_map):
    """BA window snapshot: ``{kf_id: refined camera-world position}``.

    Keyed by the map's monotonic keyframe id so the UI can accumulate a full
    refined trajectory across the sliding window (ids that leave the window keep
    their last-refined position). ``inv(T_cw)`` maps each keyframe pose to its
    camera-in-world position.
    """
    import numpy as np
    out = {}
    for kf in ba_map.keyframes:
        T_cw = kf["T_cw"]
        out[int(kf["id"])] = (np.linalg.inv(T_cw)[:3, 3]).copy()
    return out


def vio_overlay(vio_map):
    """Tight-VIO window snapshot: ``{kf_index: refined camera-world position}``.

    Mirrors :func:`ba_overlay` but the tight map's keyframes are a plain list
    (no monotonic ``id`` field), so the snapshot is keyed by the keyframe's index
    within the current window. ``inv(T_cw)`` maps each keyframe pose to its
    camera-in-world position (camera-optical frame; the UI applies the single
    optical->NED display transform). Read-only over real refined map outputs.
    """
    import numpy as np
    out = {}
    for i, kf in enumerate(vio_map.keyframes):
        T_cw = kf["T_cw"]
        out[int(i)] = (np.linalg.inv(T_cw)[:3, 3]).copy()
    return out
