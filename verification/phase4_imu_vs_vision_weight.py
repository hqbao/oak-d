#!/usr/bin/env python3
"""Phase-4 WEIGHT-RATIO DIAGNOSTIC: by how much does the IMU factor out-weight
the vision constraint on the latest keyframe's VELOCITY when vision is
feature-starved (54x42 ToF) vs full-res?

WHY
---
``phase4_velocity_bisect.py`` proved the failure mode: at 54x42 the tight solve
leaves ``|v_opt|`` ~6x ground-truth (vision can't pull it down), while at
full-res the solve drives ``|v_opt|`` back to ~0.1x (vision dominates). The seed
is identically wrong in BOTH cases, so the divergence is NOT a seed bug -- it is
that at 54x42 the tightly-weighted IMU vel/pos factor OUT-WEIGHS the weak vision
constraint, so the optimiser leaves velocity sitting on the (drifting) IMU seed.

This script QUANTIFIES that weight imbalance so the fix knob can be chosen
(down-weight IMU adaptively, by how much / add a velocity regulariser). It is the
information-theoretic companion to the bisect: the bisect shows the SYMPTOM
(v_opt stays inflated), this shows the CAUSE (the constraint information ratio).

WHAT IS MEASURED, per IMU edge ``(i = ci-1, j = ci)`` in the live window, with
``j`` the LATEST keyframe (the one whose velocity diverges)
------------------------------------------------------------------------------
The tight solve whitens the 9-vector IMU residual ``[rRot; rVel; rPos]`` by the
per-edge information sqrt ``Omega_I = pre.sqrt_info`` (``Omega_I.T @ Omega_I ==
inv(cov)``), block order [dphi 0:3 | dvel 3:6 | dpos 6:9] -- exactly the live
tight path (``imu_info_weight=True``).

1. IMU VEL-BLOCK weight on ``v_j``.  ``rVel = R_i^T (v_j - v_i - g dt) - dv``, so
   ``d rVel / d v_j = R_i^T``.  The whitened-residual Jacobian wrt ``v_j`` is
   ``J_v = Omega_I[:, 3:6] @ R_i^T`` (9x3), and the IMU INFORMATION on ``v_j`` is
   ``Info_v = J_v^T J_v`` (3x3).  Its spectral norm (largest eigenvalue) is the
   stiffest 1/sigma^2 the IMU puts on velocity; ``1/sqrt(lambda_min)`` is the
   loosest implied 1-sigma in m/s.  (``R_i`` is orthonormal, so the singular
   values of ``J_v`` equal those of ``Omega_I[:, 3:6]`` -- the velocity-block
   whitening magnitude, reported as ``||Omega_vel||``.)

2. IMU POS-BLOCK weight on ``p_j``.  ``rPos`` carries the ``d p_j`` term:
   ``d rPos / d p_j = R_i^T`` and ``d rVel / d p_j = 0``, so the position
   information is built from ``Omega_I[:, 6:9]`` the same way: ``Info_p = J_p^T
   J_p`` with ``J_p = Omega_I[:, 6:9] @ R_i^T``.  Reported as ``||Omega_pos||``
   and an implied 1-sigma in metres.

   The IMU also constrains ``v_j`` THROUGH the position factor of the NEXT edge
   (``p`` integrates ``v``), but ``j`` is the latest KF (no next edge yet), so the
   honest "vision must fight the IMU on v_j" weight is the vel-block info above,
   and the pos factor's grip on the latest TRANSLATION is ``Info_p`` -- the two
   IMU numbers vision has to overcome.

3. VISION weight on ``p_j``.  Each observation on KF j gives a whitened
   reprojection row pair (``1/sigma_px`` each) + a depth row (``1/(coeff z^2)``).
   The vision INFORMATION on KF j's translation is ``Info_vis_p = sum_obs J_o^T
   J_o`` over the 3 TRANSLATION columns of that observation's pose Jacobian --
   computed with the IDENTICAL finite-difference assembly ``optimize_vio`` uses
   (so it is the real number the solver sees, robust IRLS weights included).
   Reported alongside #tracks / #obs-with-z on KF j and the per-obs reproj weight
   ``1/sigma_px`` and depth weight ``1/(coeff z^2)``.

4. THE RATIO.  Vision constrains VELOCITY only INDIRECTLY (it pins translation;
   velocity is tied to translation through the IMU position factor over dt). The
   honest comparison the fix needs:
     * ``ratio_v`` = IMU velocity information / vision-implied velocity info,
       where vision-implied velocity info = ``Info_vis_p * dt^2`` (a position
       constraint of stiffness ``I_p`` over an interval ``dt`` constrains the
       velocity producing that displacement with stiffness ``I_p * dt^2``).
     * ``ratio_p`` = IMU position-block info / vision translation info (the raw
       translation tug-of-war).
   Reported as the eigenvalue ratio (spectral, i.e. stiffest-vs-stiffest) AND the
   trace ratio (total), per KF and as a window median.

CONSTRAINTS
-----------
READ-ONLY / ADDITIVE.  It DRIVES the exact live tight path (the same
``WindowedVIORGBDOdometry`` / ``imu_info_weight=True`` the benchmark uses) and
only OBSERVES: ``run_ba`` is wrapped so that, immediately BEFORE the unmodified
``optimize_vio`` call, the already-assembled (st, obs_*, imu_factors) are copied
out and re-run through a standalone re-implementation of the SAME whitening +
finite-difference Jacobian assembly to read the information matrices.  It does
NOT change solve behaviour, the loose path, or any baseline.  Confirm
``verification/oracle_replay_selftest.py`` stays gap=0 after running.

Output: per-(session,res) per-KF table + the headline ratio numbers.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from imu_camera.io.reader import SessionReader  # noqa: E402
from sky.depth.stereo import SGMConfig, SGMStereoMatcher  # noqa: E402
from imu_camera.modules.tof_downsample import _block_median_valid  # noqa: E402
from imu_camera.modules.pipeline import TOF_W, TOF_H  # noqa: E402

from vio.mathlib.imu.imu import GyroPreintegrator  # noqa: E402
from sky.front.odometry import OdometryConfig  # noqa: E402
from sky.math import so3_exp_unit as so3_exp  # noqa: E402
from vio.mathlib.backend.vio_window import (  # noqa: E402
    WindowedVIOConfig,
    WindowedVIORGBDOdometry,
    T_cw_to_body_world,
    _pose_perturb,
    _imu_residual,
    _bias_rw_residual,
)

GOLD_DIR = Path("sessions/gold")


# --------------------------------------------------------------------------- #
# 54x42 ToF reduction -- VERBATIM from the benchmark / existing diags.
# --------------------------------------------------------------------------- #
def _scale_K_to_tof(K: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    sx = TOF_W / float(src_w)
    sy = TOF_H / float(src_h)
    Kt = np.asarray(K, dtype=np.float64).copy()
    Kt[0, 0] *= sx
    Kt[0, 2] *= sx
    Kt[1, 1] *= sy
    Kt[1, 2] *= sy
    return Kt


def _tof_reduce(gray_src: np.ndarray, depth_src: np.ndarray):
    gray_tof = cv2.resize(gray_src, (TOF_W, TOF_H), interpolation=cv2.INTER_AREA)
    if gray_tof.dtype != np.uint8:
        gray_tof = gray_tof.astype(np.uint8)
    depth_tof = _block_median_valid(depth_src.astype(np.float32), TOF_H, TOF_W)
    return gray_tof, depth_tof


# --------------------------------------------------------------------------- #
# IMU information on the latest keyframe's velocity / position.
#
# The live solve whitens r = [rRot; rVel; rPos] by Omega_I = pre.sqrt_info.
# Jacobian of the WHITENED residual wrt v_j is Omega_I[:,3:6] @ (d rVel/d v_j),
# and d rVel/d v_j = R_i^T (orthonormal). Wrt p_j it is Omega_I[:,6:9] @ R_i^T.
# Info_v = J_v^T J_v ; Info_p = J_p^T J_p (3x3 SPD information matrices).
# --------------------------------------------------------------------------- #
def _imu_vel_pos_info(pre, R_i: np.ndarray):
    Omega = pre.sqrt_info                       # (9,9) sqrt-information
    Ri_T = R_i.T
    J_v = Omega[:, 3:6] @ Ri_T                  # d(whitened r)/d v_j
    J_p = Omega[:, 6:9] @ Ri_T                  # d(whitened r)/d p_j
    Info_v = J_v.T @ J_v
    Info_p = J_p.T @ J_p
    return Info_v, Info_p


# --------------------------------------------------------------------------- #
# Vision information on a keyframe's TRANSLATION, assembled EXACTLY as
# optimize_vio's build_system does for that keyframe's observations (same
# finite-difference, same per-obs reproj/depth whitening, same IRLS robust
# sqrt-weights from the current residual). We only keep the 3 TRANSLATION
# columns of each obs's pose Jacobian and accumulate J_t^T J_t over the obs on
# that keyframe -> the vision information bearing on its position.
# --------------------------------------------------------------------------- #
def _vision_translation_info(K, st, obs_cam, obs_lm, obs_uv, obs_depth, cfg,
                             kf_index):
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    sigma_px = cfg.sigma_px
    huber_px = cfg.huber_px
    depth_huber = cfg.depth_huber
    eps = cfg.fd_eps
    fd_tiny = 1e-300

    sel = np.asarray(obs_cam) == kf_index
    n_obs = int(sel.sum())
    if n_obs == 0:
        return np.zeros((3, 3)), 0, 0
    R_kf = st.R[kf_index]
    p_kf = st.p[kf_index]
    lm = st.landmarks
    obs_lm_k = np.asarray(obs_lm)[sel]
    uv_k = np.asarray(obs_uv, np.float64)[sel]
    z_k = np.asarray(obs_depth, np.float64)[sel]
    lm_k = lm[obs_lm_k]                          # (n,3) world landmarks

    R_obs = np.repeat(R_kf[None], n_obs, axis=0)
    p_obs = np.repeat(p_kf[None], n_obs, axis=0)

    use_depth = bool(cfg.use_depth)
    depth_mask = (z_k > 0) if use_depth else np.zeros(n_obs, dtype=bool)
    sz = cfg.depth_sigma_coeff * np.where(depth_mask, z_k, 1.0) ** 2

    def rows(Rb, pb, lmb):
        d = lmb - pb
        Xc = np.einsum('nki,nk->ni', Rb, d)      # R^T (Xw - p)
        Z = Xc[:, 2]
        Zc = np.where(Z > cfg.min_view_z, Z, cfg.min_view_z)
        u = fx * Xc[:, 0] / Zc + cx
        v = fy * Xc[:, 1] / Zc + cy
        r = np.empty((n_obs, 3))
        r[:, 0] = (u - uv_k[:, 0]) / sigma_px
        r[:, 1] = (v - uv_k[:, 1]) / sigma_px
        r[:, 2] = np.where(depth_mask, (Z - z_k) / sz, 0.0)
        return r

    r0 = rows(R_obs, p_obs, lm_k)
    # 3 translation FD columns (p <- p + eps R[:,d]); EXACTLY optimize_vio.
    Jt = np.empty((n_obs, 3, 3))                 # (obs, residual-row, t-DoF)
    for dax in range(3):
        pp = p_obs + eps * R_obs[:, :, dax]
        Jt[:, :, dax] = (rows(R_obs, pp, lm_k) - r0) / eps

    # IRLS robust sqrt-weights from the current residual (same as build_system).
    e_px = np.hypot(r0[:, 0], r0[:, 1]) * sigma_px
    sw = np.where(e_px <= huber_px, 1.0,
                  np.sqrt(huber_px / np.maximum(e_px, fd_tiny)))
    az = np.abs(r0[:, 2])
    thr = depth_huber / sz
    dw = np.where(az <= thr, 1.0, np.sqrt(thr / np.maximum(az, 1e-12)))
    dw = np.where(depth_mask, dw, 1.0)
    Jt[:, 0, :] *= sw[:, None]
    Jt[:, 1, :] *= sw[:, None]
    Jt[:, 2, :] *= dw[:, None]

    Info = np.einsum('nri,nrj->ij', Jt, Jt)      # sum_obs J_t^T J_t  (3x3)
    n_z = int(np.count_nonzero(depth_mask))
    return Info, n_obs, n_z


# --------------------------------------------------------------------------- #
# run_ba wrapper: capture the assembled solve inputs immediately before the
# (unmodified) optimize_vio runs, and measure the weights for the LATEST KF.
# --------------------------------------------------------------------------- #
class _WeightRecorder:
    def __init__(self, vmap):
        self.vmap = vmap
        self._orig_ba = vmap.run_ba
        vmap.run_ba = self._run_ba  # type: ignore[assignment]
        self.last = None             # dict of measured weights for latest KF

    def _run_ba(self):
        # Snapshot the live window BEFORE the solve mutates it. We re-derive the
        # same (st, obs_*, imu_factors) the solve will build, read-only.
        snap = self._snapshot_solve_inputs()
        out = self._orig_ba()        # the REAL solve runs untouched
        if snap is not None:
            self.last = self._measure(snap)
        return out

    # Mirror WindowedVIOMap.run_ba's assembly EXACTLY (read-only copy) so the
    # measured matrices are the ones the live solve uses. We do NOT call the
    # private method; we reconstruct the same lists from the live keyframes.
    def _snapshot_solve_inputs(self):
        from collections import Counter
        m = self.vmap
        kfs = m.keyframes
        if len(kfs) < 2:
            return None
        cnt = Counter()
        for kf in kfs:
            for tid in kf["obs"]:
                if tid in m.landmarks:
                    cnt[tid] += 1
        ba_tids = [t for t, c in cnt.items() if c >= m.cfg.min_ba_views]
        if len(ba_tids) < 6:
            return None
        lm_index = {t: j for j, t in enumerate(ba_tids)}

        from vio.mathlib.backend.vio_window import VioState
        st = VioState(
            R=[], p=[], v=[], bg=[], ba=[],
            landmarks=np.array([m.landmarks[t] for t in ba_tids]))
        for kf in kfs:
            R_wb, p_wb = T_cw_to_body_world(kf["T_cw"])
            st.R.append(R_wb)
            st.p.append(p_wb)
            st.v.append(kf["v"].copy())
            st.bg.append(kf["bg"].copy())
            st.ba.append(kf["ba"].copy())

        obs_cam, obs_lm, obs_uv, obs_depth = [], [], [], []
        for ci, kf in enumerate(kfs):
            for tid, uvz in kf["obs"].items():
                j = lm_index.get(tid)
                if j is None:
                    continue
                obs_cam.append(ci)
                obs_lm.append(j)
                obs_uv.append(uvz[:2])
                obs_depth.append(uvz[2])
        if len(obs_cam) < 12:
            return None

        # IMU edges, read from the per-edge cache (same as run_ba). The tight
        # path relinearises before reading edge.pre, so do the same to read the
        # ACTUAL sqrt_info the solve uses this call.
        relinearize = bool(m.cfg.vio.imu_info_weight)
        imu_factors = []
        for ci in range(1, len(kfs)):
            edge = kfs[ci]["edge"]
            if edge is None:
                continue
            if relinearize:
                edge.maybe_relinearize(kfs[ci - 1]["bg"], kfs[ci - 1]["ba"])
            imu_factors.append((ci - 1, ci, edge.pre))
        return {
            "st": st, "obs_cam": np.array(obs_cam), "obs_lm": np.array(obs_lm),
            "obs_uv": np.array(obs_uv), "obs_depth": np.array(obs_depth),
            "imu_factors": imu_factors, "n_kf": len(kfs),
            "g_world": m.g_world.copy(), "cfg": m.cfg.vio,
            "n_total_obs": len(obs_cam), "n_lms": len(ba_tids),
        }

    def _measure(self, s):
        st = s["st"]
        cfg = s["cfg"]
        j = s["n_kf"] - 1                        # latest keyframe index
        K = self.vmap.K

        # ---- vision information on the latest KF's translation ----------------
        Info_vis_p, n_obs_j, n_z_j = _vision_translation_info(
            K, st, s["obs_cam"], s["obs_lm"], s["obs_uv"], s["obs_depth"],
            cfg, j)
        n_tracks_j = int((np.asarray(s["obs_cam"]) == j).sum())

        # ---- IMU information on the latest KF's velocity + position -----------
        # The inbound edge to the latest KF is (j-1, j, pre).
        edge_in = next((f for f in s["imu_factors"] if f[1] == j), None)
        Info_imu_v = np.zeros((3, 3))
        Info_imu_p = np.zeros((3, 3))
        dt = float("nan")
        if edge_in is not None and edge_in[2].sqrt_info is not None:
            i, _, pre = edge_in
            R_i = st.R[i]
            Info_imu_v, Info_imu_p = _imu_vel_pos_info(pre, R_i)
            dt = float(pre.dt)

        # per-obs vision whitening weights (single-obs scalars; sz from median z)
        z_on_j = np.asarray(s["obs_depth"], np.float64)[
            np.asarray(s["obs_cam"]) == j]
        z_valid = z_on_j[z_on_j > 0]
        med_z = float(np.median(z_valid)) if z_valid.size else float("nan")
        reproj_w = 1.0 / cfg.sigma_px            # per-pixel-row whitening 1/sigma
        depth_w = (1.0 / (cfg.depth_sigma_coeff * med_z ** 2)
                   if med_z == med_z else float("nan"))

        def _spec(M):                            # largest eigenvalue (stiffest)
            w = np.linalg.eigvalsh(0.5 * (M + M.T))
            return float(np.max(w)) if w.size else 0.0

        def _trace(M):
            return float(np.trace(M))

        # implied 1-sigma from the IMU blocks (stiffest direction).
        spec_iv = _spec(Info_imu_v)
        spec_ip = _spec(Info_imu_p)
        sigma_v = (1.0 / np.sqrt(spec_iv)) if spec_iv > 0 else float("nan")
        sigma_p = (1.0 / np.sqrt(spec_ip)) if spec_ip > 0 else float("nan")

        spec_vis = _spec(Info_vis_p)             # vision info on translation
        # vision-implied VELOCITY info = vision pos info * dt^2 (a pos constraint
        # of stiffness I over dt constrains the velocity producing it by I*dt^2).
        dt2 = dt * dt if dt == dt else float("nan")
        spec_vis_v = spec_vis * dt2
        trace_vis_v = _trace(Info_vis_p) * dt2

        ratio_v_spec = (spec_iv / spec_vis_v) if spec_vis_v > 0 else float("inf")
        ratio_v_trace = ((_trace(Info_imu_v) / trace_vis_v)
                         if trace_vis_v > 0 else float("inf"))
        ratio_p_spec = (spec_ip / spec_vis) if spec_vis > 0 else float("inf")
        ratio_p_trace = ((_trace(Info_imu_p) / _trace(Info_vis_p))
                         if _trace(Info_vis_p) > 0 else float("inf"))

        return {
            "n_kf": s["n_kf"], "n_lms": s["n_lms"], "n_total_obs": s["n_total_obs"],
            "n_tracks_j": n_tracks_j, "n_obs_j": n_obs_j, "n_z_j": n_z_j,
            "dt": dt,
            # IMU block weights (spectral whitening magnitude ||Omega_block||)
            "omega_vel": np.sqrt(spec_iv), "omega_pos": np.sqrt(spec_ip),
            "imu_sigma_v": sigma_v, "imu_sigma_p": sigma_p,
            # vision per-obs whitening scalars + total info
            "reproj_w": reproj_w, "depth_w": depth_w, "med_z": med_z,
            "vis_info_p_spec": spec_vis, "vis_info_p_trace": _trace(Info_vis_p),
            # ratios (the headline)
            "ratio_v_spec": ratio_v_spec, "ratio_v_trace": ratio_v_trace,
            "ratio_p_spec": ratio_p_spec, "ratio_p_trace": ratio_p_trace,
        }


# --------------------------------------------------------------------------- #
# Drive the live tight path; capture the per-KF weights.
# --------------------------------------------------------------------------- #
def measure_session(session_dir: Path, *, resolution: str,
                    min_ba_views: int | None, max_kf: int) -> dict | None:
    reader = SessionReader(session_dir)
    n = len(reader)
    tof = (resolution == "tof54")

    if tof:
        f0 = reader.load_frame(0, load_right=False)
        sh, sw = f0.gray_left.shape[:2]
        K_solve = _scale_K_to_tof(reader.K, sw, sh)
    else:
        K_solve = reader.K

    matcher = SGMStereoMatcher.from_calib(reader.calib, SGMConfig()) if tof else None
    odom_cfg = OdometryConfig(gyro_fuse=True,
                              use_own_pnp=os.environ.get("OAKD_OWN_PNP", "1") != "0")

    if not reader.calib.has_imu_extrinsics:
        return None
    imu = reader.load_imu()
    if imu["ts_ns"].size <= 1:
        return None
    R_imu_cam = reader.calib.T_imu_left[:3, :3]
    gyro_cam = (R_imu_cam @ imu["gyro"].T).T
    accel_cam = (R_imu_cam @ imu["accel"].T).T
    t0 = imu["ts_ns"][0]
    win = imu["ts_ns"] <= t0 + int(0.3 * 1e9)
    bg0 = gyro_cam[win].mean(axis=0) if win.any() else np.zeros(3)
    ba0 = np.zeros(3)

    wcfg = WindowedVIOConfig()
    wcfg.vio = replace(wcfg.vio, imu_info_weight=True)   # the LIVE tight path
    if min_ba_views is not None:
        wcfg.min_ba_views = int(min_ba_views)

    vo = WindowedVIORGBDOdometry(
        K_solve, imu["ts_ns"], gyro_cam, accel_cam,
        bg0=bg0, ba0=ba0, cfg=wcfg, odom_cfg=odom_cfg)
    rec = _WeightRecorder(vo.map)

    pre = GyroPreintegrator(imu["ts_ns"], imu["gyro"], reader.calib.T_imu_left)
    accel_imu = imu["accel"][win].mean(axis=0)
    vo.align_to_gravity(R_imu_cam @ accel_imu)

    prev_ts = None
    rows = []
    kf_idx = -1
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

        R_prior = pre.delta_rotation(prev_ts, f.ts_ns) if prev_ts is not None else None
        vo.process(gray, depth, f.ts_ns, R_prior=R_prior)
        prev_ts = f.ts_ns

        info = vo.last_info
        if not info.get("is_kf"):
            continue
        kf_idx += 1
        ran_ba = any(k in info for k in ("vio_imu", "vio_reproj_px"))
        if ran_ba and rec.last is not None:
            m = dict(rec.last)
            m["kf"] = kf_idx
            m["frame"] = i
            m["seq"] = int(f.seq)
            rows.append(m)
        if kf_idx >= max_kf:
            break

    return {"name": session_dir.name, "resolution": resolution, "rows": rows,
            "min_ba_views": wcfg.min_ba_views, "window": wcfg.window}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_table(d: dict) -> None:
    rows = d["rows"]
    print("=" * 132)
    print(f"SESSION {d['name']}  res={d['resolution']}  "
          f"min_ba_views={d['min_ba_views']}  window={d['window']}   "
          f"(IMU vel/pos info on LATEST KF vs vision constraint)")
    print("=" * 132)
    hdr = (f"{'kf':>3} {'frm':>4} {'dt':>5} "
           f"{'#trk':>5} {'#obsZ':>6} "
           f"{'reprojW':>8} {'depthW':>8} {'med_z':>6} "
           f"{'OmgVel':>8} {'OmgPos':>8} {'imu_sv':>8} {'imu_sp':>8} "
           f"{'visInfoP':>9} "
           f"{'rV_spec':>8} {'rV_tr':>8} {'rP_spec':>8} {'rP_tr':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['kf']:>3} {r['frame']:>4} {r['dt']:>5.2f} "
              f"{r['n_tracks_j']:>5} {r['n_z_j']:>6} "
              f"{r['reproj_w']:>8.2f} {r['depth_w']:>8.2f} {r['med_z']:>6.2f} "
              f"{r['omega_vel']:>8.1f} {r['omega_pos']:>8.1f} "
              f"{r['imu_sigma_v']:>8.4f} {r['imu_sigma_p']:>8.4f} "
              f"{r['vis_info_p_spec']:>9.1f} "
              f"{r['ratio_v_spec']:>8.1f} {r['ratio_v_trace']:>8.1f} "
              f"{r['ratio_p_spec']:>8.1f} {r['ratio_p_trace']:>8.1f}")
    print("  Legend: OmgVel/OmgPos = sqrt(spectral IMU info on v_j/p_j) "
          "[= stiffest 1/sigma]; imu_sv/imu_sp = implied 1-sigma (m/s, m);")
    print("          visInfoP = spectral vision info on p_j; "
          "rV_* = IMU-vel-info / vision-implied-vel-info (spectral, trace);")
    print("          rP_* = IMU-pos-info / vision-pos-info (spectral, trace). "
          "reprojW=1/sigma_px, depthW=1/(coeff*z^2).")


def _headline(label: str, d: dict, kf_floor: int) -> tuple[float, float]:
    rows = [r for r in d["rows"] if r["kf"] >= kf_floor
            and np.isfinite(r["ratio_v_spec"])]
    if not rows:
        return float("nan"), float("nan")
    med_v = float(np.median([r["ratio_v_spec"] for r in rows]))
    med_p = float(np.median([r["ratio_p_spec"] for r in rows]))
    med_v_tr = float(np.median([r["ratio_v_trace"] for r in rows]))
    med_p_tr = float(np.median([r["ratio_p_trace"] for r in rows]))
    print(f"  HEADLINE [{label}] (KF>={kf_floor}, n={len(rows)}):")
    print(f"    velocity  IMU/vision  spectral={med_v:8.1f}x   trace={med_v_tr:8.1f}x")
    print(f"    position  IMU/vision  spectral={med_p:8.1f}x   trace={med_p_tr:8.1f}x")
    return med_v, med_p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="name:res pairs ; res in {full,tof54}.")
    ap.add_argument("--max-kf", type=int, default=12,
                    help="last keyframe index to capture.")
    ap.add_argument("--kf-floor", type=int, default=5,
                    help="first KF index counted in the headline median "
                         "(KF5+ = where tof54 diverges).")
    ap.add_argument("--tof-min-ba-views", type=int, default=1)
    args = ap.parse_args()

    if args.sessions:
        targets = []
        for s in args.sessions:
            name, _, res = s.partition(":")
            targets.append((name, res or "full"))
    else:
        targets = [
            ("push_straight_fast_15s", "full"),
            ("push_straight_fast_15s", "tof54"),
        ]

    summary = {}
    for name, res in targets:
        sd = GOLD_DIR / name
        if not sd.exists():
            print(f"!! missing session {name}")
            continue
        mbv = args.tof_min_ba_views if res == "tof54" else None
        d = measure_session(sd, resolution=res, min_ba_views=mbv,
                            max_kf=args.max_kf)
        if d is None:
            print(f"!! {name} ({res}) could not run (no IMU extrinsics?)")
            continue
        _print_table(d)
        mv, mp = _headline(f"{name}:{res}", d, args.kf_floor)
        summary[(name, res)] = (mv, mp)
        print()

    # one-line cross-resolution headline if both res ran for the same session.
    print("=" * 132)
    print("CROSS-RESOLUTION HEADLINE (median IMU-vs-vision velocity weight ratio,"
          f" KF>={args.kf_floor}):")
    by_name: dict[str, dict] = {}
    for (name, res), (mv, mp) in summary.items():
        by_name.setdefault(name, {})[res] = (mv, mp)
    for name, byres in by_name.items():
        tof = byres.get("tof54", (float("nan"), float("nan")))
        full = byres.get("full", (float("nan"), float("nan")))
        print(f"  {name}:")
        print(f"    at 54x42 the IMU vel/pos factor out-weights the vision "
              f"velocity constraint by ~{tof[0]:.0f}x (pos ~{tof[1]:.0f}x);")
        print(f"    at full-res ~{full[0]:.1f}x (pos ~{full[1]:.1f}x).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
