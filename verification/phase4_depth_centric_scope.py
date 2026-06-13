#!/usr/bin/env python3
"""SCOPE EXPERIMENT (read-only): is DENSE-DEPTH ICP a better 54x42 relative
pose than the current SPARSE feature path?  --- decide before building.

THE HYPOTHESIS
--------------
At 54x42 the sparse frontend tracks only ~1-3 features (the confirmed Phase-4
failure: docs/.../oakd-phase4-tight-stabilize.md), so inter-frame
TRANSLATION/velocity is poorly observed.  But the `--vl53l9cx` ToF sim gives an
ACCURATE dense depth map (~99% valid, computed at source res then
block-median-downsampled).  We currently use depth ONLY at the ~1-3 tracked
feature pixels and discard the other ~2265 depth pixels.

  H: dense frame-to-frame point-to-plane ICP on the FULL 54x42 depth map gives a
     markedly better / more robust RELATIVE TRANSLATION than the sparse path at
     54x42, and the regime flips at full-res (sparse catches up / wins).

WHAT THIS SCRIPT DOES (per gold session, at 54x42 AND full-res)
---------------------------------------------------------------
For CONSECUTIVE frame pairs (i-1, i):
  1. SPARSE relative pose  T_rel_sparse  -- mirrors `sky.front.odometry`
     EXACTLY: KLT track i-1 -> i, back-project prev pixels via prev depth,
     `sky.front.pnp.solve_pnp_ransac` (cur<-prev).  FAILS if < min_pnp tracks.
  2. DENSE relative pose   T_rel_icp     -- back-project BOTH frames' full depth
     to clouds (K), run a compact point-to-plane ICP (cur<-prev).  FAILS if it
     does not converge (too few valid points / ill-conditioned normal eqn).
  3. GROUND TRUTH          T_rel_gt      -- from the Basalt reference full poses
     at the two seqs:  T_rel_gt = inv(T_w_b(i-1)) @ T_w_b(i)  (body frame).

COMPARISON (standard RPE)
-------------------------
GT lives in the Basalt BODY frame; our estimates live in the camera OPTICAL
frame.  They are related by a single fixed rotation R_align (optical->body).  We
recover R_align ONCE per (session, resolution) from a global Umeyama fit of the
sparse trajectory onto the GT trajectory (rigid, no scale -- depth is metric).
Then per pair:
    t_err  = | R_align @ t_est  -  t_gt |                       (metres)
    rot_err = geodesic angle( R_align R_est R_align^T , R_gt )  (degrees)
We report MEDIAN t_err / rot_err and the FAILURE RATE for each method.  Median
(not mean) so a few blow-ups don't hide the typical behaviour.

This isolates RELATIVE pose quality -- it does NOT chain poses, so it is not an
ATE; it measures exactly the per-step translation the sparse path can't see.

HARD SCOPE / SAFETY
-------------------
READ-ONLY scratch.  Imports only side-effect-free helpers + the same math
classes the live projects use; modifies NOTHING the byte-parity oracle depends
on.  Reuses `loose_vs_tight_bench`'s producer-side 54x42 reduction verbatim
(`_tof_reduce`, `_scale_K_to_tof`) so the ToF profile is identical to live.
Confirm `oracle_replay_selftest.py` still gap=0 after running.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- reusable, side-effect-free helpers (NOT modified) --------------------- #
from verification.oracle_replay import load_basalt_positions, umeyama  # noqa: E402
from verification.loose_vs_tight_bench import (  # noqa: E402
    _tof_reduce,
    _scale_K_to_tof,
    basalt_ref_is_broken,
)
from imu_camera.io.reader import SessionReader  # noqa: E402
from imu_camera.modules.pipeline import TOF_W, TOF_H  # noqa: E402
from sky.depth.stereo import SGMConfig, SGMStereoMatcher  # noqa: E402
from sky.front.frontend import KLTFrontend  # noqa: E402
from sky.front.pnp import solve_pnp_ransac  # noqa: E402
from sky.front.odometry import OdometryConfig  # noqa: E402
# Production config builders -> the sparse path uses the SAME frontend/odometry
# config the live 54x42 ToF pipeline runs (faithful comparison, not ad-hoc).
from vio.comms.lib.config.resolution import ResolutionProfile  # noqa: E402
from vio.resolution_build import frontend_config, odometry_config  # noqa: E402


GOLD_DIR = Path("sessions/gold")
# Same Basalt-ref sanity threshold as the bench (drop diverged references).
# A recorded Basalt trajectory is only a valid reference if it didn't blow up.


# --------------------------------------------------------------------------- #
# Ground truth: Basalt FULL pose (pos + quat) -> 4x4 body-in-world per seq.
# load_basalt_positions only returns positions; we need rotation for rel-pose.
# --------------------------------------------------------------------------- #
def load_basalt_poses(session_dir: Path) -> dict[int, np.ndarray]:
    """seq -> 4x4 T_world_body from basalt/vio_pose.jsonl (quat is wxyz)."""
    import json
    out: dict[int, np.ndarray] = {}
    path = session_dir / "basalt" / "vio_pose.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        T = np.eye(4)
        T[:3, :3] = _quat_wxyz_to_R(d["quat_wxyz"])
        T[:3, 3] = np.asarray(d["pos"], dtype=np.float64)
        out[int(d["seq"])] = T
    return out


def _quat_wxyz_to_R(q) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _rot_angle_deg(R: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix in degrees."""
    c = (np.trace(R) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# SPARSE relative pose -- mirrors sky.front.odometry.estimate's correspondence
# build + PnP (cur<-prev) for ONE frame pair.  Returns (R, t, n_pnp) or
# (None, None, n_pnp) on failure (too few usable correspondences / PnP reject).
# --------------------------------------------------------------------------- #
def _backproject_px(u, v, z, K) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def sparse_rel_pose(prev_obs: dict[int, np.ndarray], cur_obs: dict[int, np.ndarray],
                    prev_depth: np.ndarray, K: np.ndarray, odom_cfg: OdometryConfig):
    """T_rel(prev->cur) in the optical frame as (R, t): X_cur = R @ X_prev + t.

    Faithful to odometry.estimate: shared track ids, prev-depth back-projection
    with the same [min_depth_m, max_depth_m] gate and min_pnp_points threshold,
    then solve_pnp_ransac with the SAME ransac params (no gyro seed here -- we
    want the raw vision relative pose).
    """
    h, w = prev_depth.shape
    obj_pts, img_pts = [], []
    for tid, cur_px in cur_obs.items():
        prev_px = prev_obs.get(tid)
        if prev_px is None:
            continue
        pu, pv = int(round(prev_px[0])), int(round(prev_px[1]))
        if not (0 <= pu < w and 0 <= pv < h):
            continue
        z = float(prev_depth[pv, pu])
        if not (odom_cfg.min_depth_m <= z <= odom_cfg.max_depth_m):
            continue
        obj_pts.append(_backproject_px(prev_px[0], prev_px[1], z, K))
        img_pts.append(cur_px)

    n_pnp = len(obj_pts)
    if n_pnp < odom_cfg.min_pnp_points:
        return None, None, n_pnp
    obj = np.asarray(obj_pts, dtype=np.float64)
    img = np.asarray(img_pts, dtype=np.float64)
    ok, R, t, inliers = solve_pnp_ransac(
        obj, img, K,
        reproj_px=odom_cfg.ransac_reproj_px, iters=odom_cfg.ransac_iters,
        conf=odom_cfg.ransac_conf, min_points=odom_cfg.min_pnp_points)
    if not ok:
        return None, None, n_pnp
    return R, t, n_pnp


# --------------------------------------------------------------------------- #
# DENSE point-to-plane ICP (cur<-prev) on the full depth map.  Compact, correct,
# pure NumPy (no open3d/scipy in this venv).  Frame-to-frame, projective data
# association via the pinhole model (standard for organised depth).
# --------------------------------------------------------------------------- #
def backproject_depth(depth: np.ndarray, K: np.ndarray):
    """Organised depth -> (pts (H,W,3) float64, valid (H,W) bool). 0/NaN = invalid."""
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)
    valid = np.isfinite(z) & (z > 1e-6)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)
    return pts, valid


def estimate_normals(pts: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-pixel surface normals from the organised cloud via local image gradients.

    n = normalize( dP/du x dP/dv ), central differences on the 3D points; the
    standard cheap normal for organised (depth-image) clouds.  Invalid / edge
    pixels get a zero normal (treated as invalid downstream).
    """
    h, w = valid.shape
    n = np.zeros_like(pts)
    # central differences (interior pixels only)
    du = np.zeros_like(pts)
    dv = np.zeros_like(pts)
    du[:, 1:-1, :] = pts[:, 2:, :] - pts[:, :-2, :]
    dv[1:-1, :, :] = pts[2:, :, :] - pts[:-2, :, :]
    cr = np.cross(du, dv)
    norm = np.linalg.norm(cr, axis=-1)
    ok = norm > 1e-9
    n[ok] = cr[ok] / norm[ok][:, None]
    # only keep normals where the pixel and its 4 diff neighbours are valid
    nb = np.zeros_like(valid)
    nb[1:-1, 1:-1] = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
                      & valid[2:, 1:-1] & valid[:-2, 1:-1])
    n[~nb] = 0.0
    return n


def _project(P: np.ndarray, K: np.ndarray):
    """3D points (N,3) -> pixel (u,v) (N,2) + z (N,). Pinhole, no distortion."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = P[:, 2]
    u = fx * P[:, 0] / z + cx
    v = fy * P[:, 1] / z + cy
    return np.stack([u, v], axis=-1), z


def icp_point_to_plane(prev_depth: np.ndarray, cur_depth: np.ndarray, K: np.ndarray,
                       max_iters: int = 20, dist_thresh_m: float = 0.10,
                       min_corr: int = 40, tol: float = 1e-6):
    """Point-to-plane ICP aligning the PREV cloud onto the CUR cloud.

    Returns (R, t, info) with X_cur ~= R @ X_prev + t (cur<-prev, the same
    convention solve_pnp_ransac returns), or (None, None, info) on failure.

    Projective association: transform prev points by the current estimate,
    project into the CUR image, look up the cur point+normal at that pixel,
    accept if within dist_thresh_m.  Linearised point-to-plane normal equations
    (Low 2004) solved each iteration; small-angle increment composed on the left.
    info carries n_corr (final), iters, converged.
    """
    Pp, vp = backproject_depth(prev_depth, K)
    Pc, vc = backproject_depth(cur_depth, K)
    nc = estimate_normals(Pc, vc)
    h, w = cur_depth.shape

    # source = valid prev points (flattened); target lookups index the cur image.
    src = Pp[vp]                              # (Ns, 3)
    if src.shape[0] < min_corr:
        return None, None, {"reason": "src_sparse", "n_corr": 0, "iters": 0}

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R = np.eye(3)
    t = np.zeros(3)
    last_err = None
    n_corr = 0
    it = 0
    for it in range(1, max_iters + 1):
        Ps = src @ R.T + t                    # prev points in cur estimate
        u = fx * Ps[:, 0] / Ps[:, 2] + cx
        v = fy * Ps[:, 1] / Ps[:, 2] + cy
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        inb = (Ps[:, 2] > 1e-6) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if inb.sum() < min_corr:
            return None, None, {"reason": "no_overlap", "n_corr": int(inb.sum()),
                                 "iters": it}
        ui, vi = ui[inb], vi[inb]
        q = Pc[vi, ui]                        # corresponding cur points
        nq = nc[vi, ui]                       # cur normals
        ps = Ps[inb]
        # reject: invalid target / no normal / too far
        d = q - ps
        dist = np.linalg.norm(d, axis=1)
        nn = np.linalg.norm(nq, axis=1)
        keep = (nn > 1e-6) & (dist < dist_thresh_m) & np.isfinite(dist)
        if keep.sum() < min_corr:
            return None, None, {"reason": "few_corr", "n_corr": int(keep.sum()),
                                 "iters": it}
        ps, q, nq, d = ps[keep], q[keep], nq[keep], d[keep]
        n_corr = ps.shape[0]

        # Point-to-plane linear system (Low 2004): unknown x = [a(3), b(3)] with
        # increment R_inc = I + skew(a), t_inc = b applied on the LEFT of (R,t).
        # residual r = ((q - ps) . n) ; jacobian rows = [ (ps x n) , n ].
        b_res = np.einsum("ij,ij->i", d, nq)          # (M,)
        cxn = np.cross(ps, nq)                         # (M,3)
        A = np.hstack([cxn, nq])                       # (M,6)
        AtA = A.T @ A
        Atb = A.T @ b_res
        # solve (regularised lstsq for conditioning at low res / planar scenes)
        try:
            x = np.linalg.solve(AtA + 1e-9 * np.eye(6), Atb)
        except np.linalg.LinAlgError:
            return None, None, {"reason": "singular", "n_corr": n_corr, "iters": it}
        a = x[:3]
        b = x[3:]
        ang = float(np.linalg.norm(a))
        if ang < 1e-12:
            R_inc = np.eye(3)
        else:
            ax = a / ang
            Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R_inc = np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * (Kx @ Kx)
        # compose increment on the left: X_cur = R_inc (R X + t) + b
        R = R_inc @ R
        t = R_inc @ t + b

        err = float(np.sqrt(np.mean(b_res ** 2)))
        if last_err is not None and abs(last_err - err) < tol:
            break
        last_err = err

    converged = n_corr >= min_corr
    return (R, t, {"reason": "ok", "n_corr": n_corr, "iters": it,
                   "rms_m": last_err}) if converged else \
           (None, None, {"reason": "not_converged", "n_corr": n_corr, "iters": it})


# --------------------------------------------------------------------------- #
# Per-session run: produce sparse + ICP + GT relative poses for every pair,
# recover R_align (optical->body) from a global rigid Umeyama on the sparse
# trajectory, then score each method's relative pose (RPE) against GT.
# --------------------------------------------------------------------------- #
def run_session(session_dir: Path, *, resolution: str, max_frames: int = 0):
    reader = SessionReader(session_dir)
    n = len(reader) if max_frames <= 0 else min(max_frames, len(reader))

    basalt_pos = load_basalt_positions(reader.dir)
    if not basalt_pos or basalt_ref_is_broken(basalt_pos):
        return None
    basalt_T = load_basalt_poses(reader.dir)

    f0 = reader.load_frame(0, load_right=False)
    src_h, src_w = f0.gray_left.shape[:2]

    tof = (resolution == "tof54")
    if tof:
        K_solve = _scale_K_to_tof(reader.K, src_w, src_h)
        matcher = SGMStereoMatcher.from_calib(reader.calib, SGMConfig())
        # Faithful production sparse path: the SAME frontend + odometry config the
        # live 54x42 ToF pipeline builds (bucketed Shi-Tomasi, scaled reproj gate),
        # numba=True for the full corner budget so we give sparse its BEST shot.
        prof = ResolutionProfile.for_resolution(TOF_W, TOF_H)
    else:
        K_solve = reader.K
        matcher = None
        prof = ResolutionProfile.for_resolution(src_w, src_h)
    front_cfg = frontend_config(prof, numba=True)
    odom_cfg = odometry_config(prof)
    frontend = KLTFrontend(front_cfg)

    # --- per-frame: gray+depth, KLT tracks, store rel poses for consecutive pairs
    prev_obs = None
    prev_depth = None
    prev_seq = None
    rel = []                 # list of dicts per pair
    n_warmup_dropped = 0     # pairs skipped because depth was (near-)empty

    # Depth-coverage gate: the SGM matcher returns no valid depth for the first
    # few warm-up frames (organised cloud is empty), so NEITHER method can
    # produce a relative pose there -- that is an SGM-init artifact, not a
    # relative-pose-method failure, and counting it would corrupt both methods'
    # failure rate identically. We drop a pair only when prev OR cur depth has
    # essentially no valid pixels (< 5% valid); genuine method failures on
    # well-populated depth are kept and counted.
    _MIN_DEPTH_COVER = 0.05

    for i in range(n):
        f = reader.load_frame(i, load_right=tof)
        if tof:
            gray_src, depth_src = matcher.dense_depth_rectified_left(
                f.gray_left, f.gray_right)
            if gray_src.dtype != np.uint8:
                gray_src = np.clip(gray_src, 0.0, 255.0).astype(np.uint8)
            gray, depth = _tof_reduce(gray_src, depth_src)
        else:
            gray, depth = f.gray_left, f.depth_m

        state = frontend.process(gray)
        cur_obs = {int(tid): p for tid, p in zip(state.ids, state.points)}

        if prev_obs is not None and prev_seq in basalt_T and f.seq in basalt_T:
            prev_cover = float((prev_depth > 1e-6).mean())
            cur_cover = float((depth > 1e-6).mean())
            if prev_cover < _MIN_DEPTH_COVER or cur_cover < _MIN_DEPTH_COVER:
                n_warmup_dropped += 1
                prev_obs = cur_obs
                prev_depth = depth
                prev_seq = f.seq
                continue

            # GT relative pose (body frame): T_prev->cur = inv(T_w_prev) @ T_w_cur
            Tg = np.linalg.inv(basalt_T[prev_seq]) @ basalt_T[f.seq]
            R_gt, t_gt = Tg[:3, :3], Tg[:3, 3]

            R_sp, t_sp, n_pnp = sparse_rel_pose(prev_obs, cur_obs, prev_depth,
                                                K_solve, odom_cfg)
            R_ic, t_ic, info = icp_point_to_plane(prev_depth, depth, K_solve)

            rel.append({
                "seq": f.seq,
                "R_gt": R_gt, "t_gt": t_gt,
                "R_sp": R_sp, "t_sp": t_sp, "n_pnp": n_pnp,
                "R_ic": R_ic, "t_ic": t_ic, "icp": info,
            })

        prev_obs = cur_obs
        prev_depth = depth
        prev_seq = f.seq

    if len(rel) < 5:
        return None

    # --- recover R_align (optical->body) from a GLOBAL rigid Umeyama fit ------ #
    # Chain the SPARSE relative poses into a trajectory (identity where sparse
    # failed -> hold position), align onto the Basalt positions at matching seqs.
    # R_align is the rotation that best rotates our optical-frame motion into the
    # Basalt body/world frame; it is the SAME for both methods (shared cameras).
    R_align = _recover_alignment(rel, basalt_pos)

    # --- score each method's relative pose against GT ------------------------ #
    def score(method: str):
        t_errs, r_errs = [], []
        n_fail = 0
        for p in rel:
            R_e = p[f"R_{method}"]
            t_e = p[f"t_{method}"]
            if R_e is None:
                n_fail += 1
                continue
            t_e_body = R_align @ t_e
            t_err = float(np.linalg.norm(t_e_body - p["t_gt"]))
            R_e_body = R_align @ R_e @ R_align.T
            r_err = _rot_angle_deg(R_e_body @ p["R_gt"].T)
            t_errs.append(t_err)
            r_errs.append(r_err)
        n_tot = len(rel)
        return {
            "n_pairs": n_tot,
            "n_fail": n_fail,
            "fail_rate": n_fail / n_tot if n_tot else 1.0,
            "t_err_med_cm": float(np.median(t_errs)) * 100 if t_errs else None,
            "t_err_mean_cm": float(np.mean(t_errs)) * 100 if t_errs else None,
            "r_err_med_deg": float(np.median(r_errs)) if r_errs else None,
        }

    # diagnostics: typical GT inter-frame translation (the thing to recover) and
    # typical sparse track count (the starvation we are testing).
    gt_steps = np.array([np.linalg.norm(p["t_gt"]) for p in rel])
    n_pnp_arr = np.array([p["n_pnp"] for p in rel])
    icp_corr = np.array([p["icp"].get("n_corr", 0) for p in rel])

    return {
        "sparse": score("sp"),
        "icp": score("ic"),
        "gt_step_med_cm": float(np.median(gt_steps)) * 100,
        "gt_step_max_cm": float(np.max(gt_steps)) * 100,
        "n_pnp_med": float(np.median(n_pnp_arr)),
        "icp_corr_med": float(np.median(icp_corr)),
        "n_warmup_dropped": n_warmup_dropped,
    }


def _recover_alignment(rel, basalt_pos):
    """R_align (optical->body) from a global rigid Umeyama of the sparse chain.

    Chain sparse relative poses (cur<-prev) into world poses T_w_cur =
    T_w_prev @ inv(T_rel); identity where sparse failed.  Then Umeyama (rigid)
    the resulting positions onto Basalt's at matching seqs.  Robust even if a
    handful of pairs failed -- it only needs the gross trajectory direction to
    fix the optical->body rotation.
    """
    # Anchor the chain at the smallest Basalt seq (origin), then chain the
    # sparse relative poses forward; rel[i]["seq"] is the CUR seq of pair i.
    T = np.eye(4)
    poses = {}
    anchor_seq = min(basalt_pos)
    poses[anchor_seq] = T[:3, 3].copy()
    for p in rel:
        R_e, t_e = p["R_sp"], p["t_sp"]
        if R_e is not None:
            Trel = np.eye(4)
            Trel[:3, :3] = R_e
            Trel[:3, 3] = t_e
            T = T @ np.linalg.inv(Trel)        # T_w_cur = T_w_prev @ inv(rel)
        poses[p["seq"]] = T[:3, 3].copy()

    common = sorted(set(poses) & set(basalt_pos))
    if len(common) < 5:
        return np.eye(3)
    src = np.array([poses[s] for s in common])
    dst = np.array([basalt_pos[s] for s in common])
    if np.linalg.norm(src.std(axis=0)) < 1e-6:
        return np.eye(3)                       # degenerate (no motion) -> identity
    R_align, _, _ = umeyama(src, dst, with_scale=False)
    return R_align


# --------------------------------------------------------------------------- #
# Table + verdict.
# --------------------------------------------------------------------------- #
def _row(label, s):
    if s is None:
        return f"  {label:8s}  {'--':>9s} {'--':>9s} {'--':>9s} {'--':>9s}"
    tm = f"{s['t_err_med_cm']:.1f}" if s["t_err_med_cm"] is not None else "--"
    tmn = f"{s['t_err_mean_cm']:.1f}" if s["t_err_mean_cm"] is not None else "--"
    rm = f"{s['r_err_med_deg']:.2f}" if s["r_err_med_deg"] is not None else "--"
    fr = f"{100*s['fail_rate']:.0f}%"
    return f"  {label:8s}  {tm:>9s} {tmn:>9s} {rm:>9s} {fr:>9s}"


def run(sessions, max_frames):
    names = sorted(d for d in GOLD_DIR.iterdir()
                   if (d / "basalt" / "vio_pose.jsonl").exists())
    if sessions:
        names = [d for d in names if d.name in sessions]

    all_res = {}
    for d in names:
        all_res[d.name] = {}
        t0 = time.perf_counter()
        for res in ("tof54", "full"):
            all_res[d.name][res] = run_session(d, resolution=res, max_frames=max_frames)
        print(f"  scored {d.name}  ({time.perf_counter()-t0:.0f}s)")

    for res, label in (("tof54", "54x42 ToF (VL53L9CX target) -- THE REGIME UNDER TEST"),
                       ("full", "FULL-RES (chip depth) -- contrast")):
        print()
        print("#" * 92)
        print(f"#  {label}")
        print("#  per-pair RELATIVE pose error vs Basalt GT (RPE). median t-err is the headline.")
        print("#" * 92)
        print(f"{'session':22s}  method      tErr_med  tErr_mean  rErr_med   failR")
        print(f"{'':22s}  {'(units)':10s}      (cm)      (cm)     (deg)")
        print("-" * 78)
        for d in names:
            r = all_res[d.name][res]
            print(f"{d.name:22s}")
            if r is None:
                print(f"  {'(no scoreable Basalt overlap)':40s}")
                continue
            print(_row("SPARSE", r["sparse"]))
            print(_row("ICP", r["icp"]))
            print(f"            GT step med={r['gt_step_med_cm']:.1f}cm "
                  f"max={r['gt_step_max_cm']:.1f}cm | "
                  f"sparse n_pnp med={r['n_pnp_med']:.0f} | "
                  f"icp corr med={r['icp_corr_med']:.0f} | "
                  f"warmup-dropped={r['n_warmup_dropped']}")
            print(f"            verdict: {_pair_verdict(r)}")
        print()

    print("#" * 92)
    print("#  AGGREGATE")
    print("#" * 92)
    _aggregate(all_res, names)


def _pair_verdict(r) -> str:
    sp, ic = r["sparse"], r["icp"]
    spv = sp["t_err_med_cm"]
    icv = ic["t_err_med_cm"]
    if spv is None and icv is None:
        return "both unusable"
    if spv is None:
        return f"SPARSE all-fail; ICP t_med={icv:.1f}cm (fail {100*ic['fail_rate']:.0f}%)"
    if icv is None:
        return f"ICP all-fail; SPARSE t_med={spv:.1f}cm (fail {100*sp['fail_rate']:.0f}%)"
    if icv < spv:
        return (f"ICP better trans ({spv:.1f}->{icv:.1f}cm, "
                f"-{100*(spv-icv)/spv:.0f}%) | fail sp {100*sp['fail_rate']:.0f}% "
                f"ic {100*ic['fail_rate']:.0f}%")
    return (f"SPARSE better trans ({icv:.1f}->{spv:.1f}cm, "
            f"-{100*(icv-spv)/icv:.0f}%) | fail sp {100*sp['fail_rate']:.0f}% "
            f"ic {100*ic['fail_rate']:.0f}%")


def _aggregate(all_res, names):
    for res, label in (("tof54", "54x42 ToF"), ("full", "FULL-RES")):
        icp_wins = sparse_wins = scored = 0
        sp_t, ic_t = [], []
        sp_fr, ic_fr = [], []
        for d in names:
            r = all_res[d.name][res]
            if r is None:
                continue
            spv = r["sparse"]["t_err_med_cm"]
            icv = r["icp"]["t_err_med_cm"]
            sp_fr.append(r["sparse"]["fail_rate"])
            ic_fr.append(r["icp"]["fail_rate"])
            if spv is None and icv is None:
                continue
            scored += 1
            if spv is not None:
                sp_t.append(spv)
            if icv is not None:
                ic_t.append(icv)
            if icv is not None and (spv is None or icv < spv):
                icp_wins += 1
            elif spv is not None and (icv is None or spv < icv):
                sparse_wins += 1
        print(f"  [{label}]  ICP better trans on {icp_wins}/{scored}, "
              f"sparse better on {sparse_wins}/{scored}")
        if sp_t:
            print(f"      median-of-medians t-err: "
                  f"SPARSE {np.median(sp_t):.1f}cm  ICP {np.median(ic_t):.1f}cm")
        print(f"      mean failure rate: "
              f"SPARSE {100*np.mean(sp_fr):.0f}%  ICP {100*np.mean(ic_fr):.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = all frames (default); >0 quick smoke")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these session names")
    args = ap.parse_args()
    run(args.only, args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
