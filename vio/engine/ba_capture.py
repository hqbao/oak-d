"""BA-window VISUALIZER capture (opt-in ``--ba-window``).

A RICHER sibling of the core per-keyframe solve (:mod:`vio.engine.steps`'s
``ba_step`` / ``ba_overlay``) that snapshots the FULL windowed-BA solve state
(window keyframe poses + shared 3D landmarks + observation rays + per-observation
reprojection error + the PRE-solve state for a before/after toggle).

Split out of ``steps.py`` so the core solve file stays minimal. The capture runs
the SAME frozen ``ba_map.run_ba()`` -- the only addition is a read-only PRE
snapshot before the solve and a read-only snapshot build after it; the frozen
``ba_step`` / ``run_ba`` / ``optimize`` are never edited and stay byte-identical,
so this path is selected ONLY by the opt-in capture engine and the oracle is
unaffected (gap=0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


#: Hard cap on landmarks shipped on the wire (the M with the MOST keyframe
#: observations are kept). Lives ONLY here in the capture/overlay path, never in
#: the frozen solve, so it can never change a refined pose. The window keyframe
#: count N is already bounded by ``WindowedConfig.window`` (<= 8).
_BA_WINDOW_MAX_LM = 100


def _mat_to_quat(R):
    """Rotation matrix (3x3) -> unit quaternion ``(qw, qx, qy, qz)``.

    Numerically-stable Shepperd branch (pick the largest diagonal term to avoid a
    near-zero divisor). Pure read-only helper for the snapshot; the sign is
    canonicalised so ``qw >= 0`` (a quaternion and its negation are the same
    rotation -- fixing the sign keeps the wire bytes deterministic).
    """
    import numpy as np
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    if q[0] < 0.0:                         # canonical sign (qw >= 0)
        q = -q
    n = np.linalg.norm(q)
    return q / n if n > 0.0 else np.array([1.0, 0.0, 0.0, 0.0])


@dataclass
class BaWindowSnap:
    """A picklable, fully-built windowed-BA snapshot the publisher ships verbatim.

    Built by :func:`_build_ba_window` from the live map's PUBLIC state (keyframes /
    landmarks / last_info) AFTER the frozen solve. Carries exactly the columns of
    :class:`comms.messages.BaWindow` so the publish step is a thin 1:1 copy. All
    positions are camera-optical world frame; quaternions are ``(qw,qx,qy,qz)`` of
    the camera-in-world rotation. ``*_pre`` are the PRE-solve poses/landmarks.

    Picklable (plain numpy fields, module-level class) so it crosses the
    subprocess-engine ``ov_q`` under the ``spawn`` start method, exactly like the
    ``{kf_id: pos}`` dict ``ba_overlay`` returns.
    """

    seq: int
    ts_ns: int
    kf_ids: Any
    kf_quat: Any
    kf_pos: Any
    lm_ids: Any
    lm_xyz: Any
    obs_kf: Any
    obs_lm: Any
    obs_uv: Any
    obs_reproj_px: Any
    ba_reproj_px: float
    kf_quat_pre: Any
    kf_pos_pre: Any
    lm_xyz_pre: Any
    n_kf: int
    n_lm: int


def _snapshot_pre(ba_map):
    """Shallow-copy the PRE-solve keyframe poses + landmarks (read-only).

    Returns ``({kf_id: T_cw_copy}, {tid: xyz_copy})`` captured BEFORE
    ``run_ba`` mutates the map, so the snapshot can show the before/after toggle.
    Copies are independent allocations so the subsequent solve cannot alias them.
    """
    import numpy as np
    pre_poses = {int(kf["id"]): np.asarray(kf["T_cw"], np.float64).copy()
                 for kf in ba_map.keyframes}
    pre_lms = {int(t): np.asarray(p, np.float64).copy()
               for t, p in ba_map.landmarks.items()}
    return pre_poses, pre_lms


def _build_ba_window(ba_map, pre, seq: int, ts_ns: int) -> BaWindowSnap:
    """Build a :class:`BaWindowSnap` from the live map AFTER ``run_ba``.

    Reads only PUBLIC map state (``keyframes`` / ``landmarks`` / ``last_info``),
    so it never touches the frozen solve. The landmark set is capped to the
    :data:`_BA_WINDOW_MAX_LM` with the most keyframe observations (the cap lives
    ONLY here, never in the solve). ``pre`` is the ``_snapshot_pre`` result; a
    landmark/keyframe absent there (e.g. born this keyframe) falls back to its
    post value so the ghost is simply coincident, never missing.
    """
    import numpy as np
    pre_poses, pre_lms = pre
    kfs = list(ba_map.keyframes)

    # Keyframe poses (camera-in-world = inv(T_cw)); window order (oldest first).
    kf_ids, kf_quat, kf_pos = [], [], []
    kf_quat_pre, kf_pos_pre = [], []
    kf_id_to_idx = {}
    for kf in kfs:
        kid = int(kf["id"])
        kf_id_to_idx[kid] = len(kf_ids)
        Twc = np.linalg.inv(np.asarray(kf["T_cw"], np.float64))
        kf_ids.append(kid)
        kf_quat.append(_mat_to_quat(Twc[:3, :3]))
        kf_pos.append(Twc[:3, 3].copy())
        # PRE pose (fall back to POST when this keyframe was inserted this solve).
        Tcw_pre = pre_poses.get(kid, np.asarray(kf["T_cw"], np.float64))
        Twc_pre = np.linalg.inv(np.asarray(Tcw_pre, np.float64))
        kf_quat_pre.append(_mat_to_quat(Twc_pre[:3, :3]))
        kf_pos_pre.append(Twc_pre[:3, 3].copy())

    # Landmarks observed by >= 2 in-window keyframes (the ones BA constrains),
    # ranked by observation count, then capped to the wire bound.
    from collections import Counter
    obs_count: Counter = Counter()
    for kf in kfs:
        for tid in kf["obs"]:
            if tid in ba_map.landmarks:
                obs_count[int(tid)] += 1
    ranked = [t for t, c in obs_count.most_common() if c >= 2]
    if not ranked:                              # degenerate: keep any landmarks
        ranked = [int(t) for t in ba_map.landmarks]
    ranked = ranked[:_BA_WINDOW_MAX_LM]
    lm_id_to_idx = {t: i for i, t in enumerate(ranked)}

    lm_ids, lm_xyz, lm_xyz_pre = [], [], []
    for t in ranked:
        xyz = np.asarray(ba_map.landmarks[t], np.float64)
        lm_ids.append(int(t))
        lm_xyz.append(xyz.copy())
        lm_xyz_pre.append(np.asarray(pre_lms.get(t, xyz), np.float64).copy())

    # Observation rays: every (in-window keyframe, kept landmark) pixel obs, with
    # its post-solve reprojection error in pixels (pinhole(K, T_cw @ X)).
    K = np.asarray(ba_map.K, np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    obs_kf, obs_lm, obs_uv, obs_reproj = [], [], [], []
    for kf in kfs:
        ci = kf_id_to_idx[int(kf["id"])]
        Tcw = np.asarray(kf["T_cw"], np.float64)
        Rcw, tcw = Tcw[:3, :3], Tcw[:3, 3]
        for tid, uvz in kf["obs"].items():
            j = lm_id_to_idx.get(int(tid))
            if j is None:
                continue
            u, v = float(uvz[0]), float(uvz[1])
            Xc = Rcw @ np.asarray(ba_map.landmarks[int(tid)], np.float64) + tcw
            z = Xc[2]
            if z > 1e-6:                        # in front of the camera
                up = fx * Xc[0] / z + cx
                vp = fy * Xc[1] / z + cy
                err = float(np.hypot(up - u, vp - v))
            else:                               # behind/degenerate -> mark large
                err = float("inf")
            obs_kf.append(ci)
            obs_lm.append(j)
            obs_uv.append((u, v))
            obs_reproj.append(err if np.isfinite(err) else 1e3)

    n = len(kf_ids)
    m = len(lm_ids)
    return BaWindowSnap(
        seq=int(seq), ts_ns=int(ts_ns),
        kf_ids=np.asarray(kf_ids, np.int64).reshape(-1),
        kf_quat=np.asarray(kf_quat, np.float64).reshape(n, 4) if n else
        np.zeros((0, 4), np.float64),
        kf_pos=np.asarray(kf_pos, np.float64).reshape(n, 3) if n else
        np.zeros((0, 3), np.float64),
        lm_ids=np.asarray(lm_ids, np.int64).reshape(-1),
        lm_xyz=np.asarray(lm_xyz, np.float64).reshape(m, 3) if m else
        np.zeros((0, 3), np.float64),
        obs_kf=np.asarray(obs_kf, np.int32).reshape(-1),
        obs_lm=np.asarray(obs_lm, np.int32).reshape(-1),
        obs_uv=np.asarray(obs_uv, np.float32).reshape(-1, 2) if obs_uv else
        np.zeros((0, 2), np.float32),
        obs_reproj_px=np.asarray(obs_reproj, np.float32).reshape(-1),
        ba_reproj_px=float(ba_map.last_info.get("ba_reproj_px", 0.0)),
        kf_quat_pre=np.asarray(kf_quat_pre, np.float64).reshape(n, 4) if n else
        np.zeros((0, 4), np.float64),
        kf_pos_pre=np.asarray(kf_pos_pre, np.float64).reshape(n, 3) if n else
        np.zeros((0, 3), np.float64),
        lm_xyz_pre=np.asarray(lm_xyz_pre, np.float64).reshape(m, 3) if m else
        np.zeros((0, 3), np.float64),
        n_kf=int(n), n_lm=int(m))


def ba_step_capture(ba_map, snap: Any):
    """Capture-aware windowed-BA keyframe: identical solve + a snapshot side-car.

    Mirrors :func:`ba_step` (same ``add_keyframe`` + same frozen ``run_ba``) but
    PRE-snapshots the poses/landmarks before the solve and stashes a built
    :class:`BaWindowSnap` on the map (``ba_map._ba_window_snap``) for the overlay
    to hand to the publisher. The RETURN value is byte-identical to ``ba_step``'s
    (the refined latest ``T_cw`` or ``None``) so the responsive/refined pose path
    is unchanged -- only the overlay channel carries the richer snapshot.
    """
    T_cw, ids, pts, depth_m, accel = snap
    if ids is None or pts is None:
        ba_map._ba_window_snap = None
        return None
    pre = _snapshot_pre(ba_map)                # PRE state (before run_ba mutates)
    ba_map.add_keyframe(T_cw, ids, pts, depth_m, accel_cam=accel)
    post = ba_map.run_ba()                      # the SAME frozen solve
    # Build the snapshot only when the solve actually ran (else there is no
    # refined window to show -- the warmup keyframe returns None, like ba_step).
    if post is None:
        ba_map._ba_window_snap = None
    else:
        ba_map._ba_window_snap = _build_ba_window(
            ba_map, pre, seq=ba_map._kf_counter - 1, ts_ns=0)
    return post


def ba_window_overlay(ba_map):
    """Overlay for the capture engine: return the last-built BA-window snapshot.

    Returns the :class:`BaWindowSnap` ``ba_step_capture`` stashed for the most
    recent keyframe (or ``None`` on a warmup keyframe). Rides the SAME ``ov_q``
    overlay channel ``ba_overlay`` uses; the publisher (``publish_ba_window``)
    polls ``engine.poll_overlay()`` and ships it on ``ba.window``.
    """
    return getattr(ba_map, "_ba_window_snap", None)
