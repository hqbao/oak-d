"""``emit_keyframe`` step: every ``kf_every`` frames, publish a ``keyframe``.

The keyframe carries the pose, image, depth, the current track snapshot and --
only when the camera was at rest -- the gravity accel for the back-end (a moving
keyframe's lateral acceleration would bias the gravity direction).
"""
from __future__ import annotations

import numpy as np

from vio.comms import LocalPubSub, topics
from vio.comms.messages import Keyframe
from sky.front.odometry import RGBDVisualOdometry
from .step import Step


def emit_keyframe(vo: RGBDVisualOdometry, state: dict, bus: LocalPubSub,
                  step: Step) -> Step:
    """Publish a ``keyframe`` on the cadence; pass the carrier through.

    Was ``EmitKeyframe(StepBase)``; the odometry instance, the worker's shared
    state dict (``ctx.state``), and the bus are passed explicitly.
    """
    # Keyframe cadence. LOOSE path (default): emit_keyframe owns the counter
    # exactly as before -- byte-identical. TIGHT path (retain_imu): propagate_imu
    # runs first and is the SINGLE source of truth for the cadence (so the live
    # nav-state re-anchors on the same frame a keyframe is emitted); here we
    # just consume the boolean it stamped.
    if state.get("retain_imu"):
        if not state.get("is_kf_frame"):
            return step
    else:
        n = state.get("kf_count", 0) + 1
        if n < state["kf_every"]:
            state["kf_count"] = n
            return step
        state["kf_count"] = 0
    tr = vo.frontend.tracks
    ids = tr.ids.copy() if tr is not None and tr.ids is not None else None
    px = tr.points.copy() if tr is not None and tr.points is not None else None
    accel = step.accel_cam if step.at_rest else None
    inl = step.info.get("inlier_ids")        # PnP inliers this frame (clean subset)
    inl = None if inl is None else np.asarray(inl).copy()
    # TIGHT path only: attach the keyframe timestamp + the raw inter-keyframe
    # IMU block (camera optical frame) the tight backend preintegrates. The
    # block is the concatenation of every retained per-frame IMU segment
    # (preintegrate_prior) since the previous emitted keyframe, in seq order,
    # so the preintegrated ``dt`` spans the FULL keyframe interval -- not just
    # the last frame's. Loose path: ``retain_imu`` is False, so ``ts_ns``
    # stays 0 and ``imu_seg`` stays None (byte-identical Keyframe payload).
    ts_ns = 0
    imu_seg = _collect_imu_seg(state, step.frame.seq)
    if state.get("retain_imu"):
        ts_ns = int(step.frame.ts_ns)
    bus.publish(topics.KEYFRAME,
                Keyframe(step.frame.seq, step.pose,
                         step.frame.gray_left, step.frame.depth_m,
                         track_ids=ids, track_px=px, accel=accel,
                         inlier_ids=inl, ts_ns=ts_ns, imu_seg=imu_seg))
    return step


def _collect_imu_seg(state: dict, kf_seq: int):
    """Concatenate retained per-frame IMU segments for the interval since the
    previous emitted keyframe; return ``(ts, gyro_cam, accel_cam)`` or None.

    Returns None on the loose path (no retained segments) so the keyframe
    carries no IMU block. Consumed segments are popped so the dict stays
    bounded over a long live session.
    """
    if not state.get("retain_imu"):
        return None
    imu_segs = state["imu_segs"]
    if not imu_segs:
        return None
    prev = state.get("last_kf_seq", -1)
    # all frame seqs in (prev_kf_seq, kf_seq], in order -- the frames whose
    # IMU intervals together span this keyframe interval.
    seqs = sorted(s for s in imu_segs if prev < s <= kf_seq)
    state["last_kf_seq"] = int(kf_seq)
    ts_list, g_list, a_list = [], [], []
    for s in seqs:
        t, g, a = imu_segs.pop(s)
        if t.size:
            ts_list.append(t)
            g_list.append(g)
            a_list.append(a)
    if len(ts_list) < 1:
        return None
    ts = np.concatenate(ts_list)
    gyro = np.concatenate(g_list)
    accel = np.concatenate(a_list)
    # Guard against duplicate / out-of-order device timestamps at the segment
    # joins (the preintegrator needs strictly increasing ts to form dt > 0).
    order = np.argsort(ts, kind="stable")
    ts, gyro, accel = ts[order], gyro[order], accel[order]
    keep = np.concatenate(([True], np.diff(ts) > 0))
    ts, gyro, accel = ts[keep], gyro[keep], accel[keep]
    if ts.size < 2:
        return None
    return (ts, gyro, accel)
