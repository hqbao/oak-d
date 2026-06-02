"""Sliding-window bundle adjustment (BA) core — pure NumPy.

This is the optimisation *engine* only: given a set of keyframe poses, 3D
landmarks and 2D observations, it refines them by minimising the robust
reprojection error. It knows nothing about cameras, KLT or sessions — that
glue lives in :mod:`oakd.vio.windowed`. Keeping the solver standalone means we
can unit-test it on a synthetic scene with a known answer (see
``tools/ba_selftest.py``) before trusting it on real data.

Conventions
-----------
- A keyframe pose is the **world->camera** transform ``T_cw`` (4x4). A world
  point ``Xw`` maps to the camera frame as ``Xc = R_cw @ Xw + t_cw`` and then
  projects with the pinhole intrinsics ``K``.
- Pose increments use a **left** SE3 perturbation ``T <- Exp(xi) @ T`` with
  ``xi = [rho(3) ; phi(3)]`` (translation part first, rotation part second).
  The 3D point Jacobian is then ``d(Xc)/d(xi) = [I | -skew(Xc)]``.
- One keyframe (by default the oldest in the window) is held **fixed** to
  remove the 6-DoF gauge freedom; everything else is solved relative to it.

Solver: Levenberg-Marquardt with the Schur complement. The landmark block of
the Hessian is block-diagonal (3x3 per point), so we eliminate landmarks first
and solve the small reduced camera system, then back-substitute. This is the
standard structure that makes BA scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Lie-group helpers (SE3 / SO3)
# --------------------------------------------------------------------------- #
def skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map so3 -> SO3 (Rodrigues)."""
    theta = float(np.linalg.norm(phi))
    if theta < 1e-12:
        return np.eye(3) + skew(phi)
    k = phi / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map se3 -> SE3. xi = [rho(3); phi(3)] -> 4x4."""
    rho = xi[:3]
    phi = xi[3:]
    theta = float(np.linalg.norm(phi))
    R = so3_exp(phi)
    if theta < 1e-12:
        V = np.eye(3) + 0.5 * skew(phi)
    else:
        K = skew(phi / theta)
        V = (np.eye(3)
             + (1.0 - np.cos(theta)) / theta * K
             + (theta - np.sin(theta)) / theta * (K @ K))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


# --------------------------------------------------------------------------- #
# Configuration / result
# --------------------------------------------------------------------------- #
@dataclass
class BAConfig:
    max_iters: int = 15
    huber_px: float = 2.0      # robust kernel threshold (pixels)
    init_lambda: float = 1e-3  # LM damping
    min_lambda: float = 1e-7
    max_lambda: float = 1e7
    rel_tol: float = 1e-4      # stop when cost drops by less than this fraction
    min_view_z: float = 1e-3   # skip observations with Z <= this (behind cam)
    # --- metric depth anchoring ---------------------------------------------
    # Pure reprojection BA is scale-free: fixing one camera removes 6 DoF but
    # NOT global scale. With an RGB-D sensor the measured depth pins the metric
    # scale, so we add a residual ``(Z_pred - z_meas)`` per observation that has
    # a valid depth, weighted by a stereo-like quadratic noise model
    # ``sigma_z = depth_sigma_coeff * z_meas^2`` (so far, noisy points count
    # less). Set ``use_depth=False`` to recover plain reprojection BA.
    use_depth: bool = True
    depth_sigma_coeff: float = 0.02   # sigma_z = coeff * z^2  (metres)
    depth_huber: float = 0.10         # robust threshold on depth residual (m)


@dataclass
class BAResult:
    poses: list[np.ndarray]     # refined T_cw per keyframe
    landmarks: np.ndarray       # refined (M,3) world points
    iters: int
    cost0: float                # initial robust cost
    cost1: float                # final robust cost
    mean_reproj_px: float       # final mean reprojection error (inliers)


# --------------------------------------------------------------------------- #
# Core optimiser
# --------------------------------------------------------------------------- #
def _huber_weight(e: float, delta: float) -> float:
    """sqrt-weight so that (w*r)^2 equals the Huber loss of residual norm e."""
    if e <= delta:
        return 1.0
    return float(np.sqrt(delta / e))


def optimize(
    K: np.ndarray,
    poses: list[np.ndarray],
    fixed: list[bool],
    landmarks: np.ndarray,
    obs_cam: np.ndarray,
    obs_lm: np.ndarray,
    obs_uv: np.ndarray,
    obs_depth: np.ndarray | None = None,
    cfg: BAConfig | None = None,
) -> BAResult:
    """Refine ``poses`` (world->cam) and ``landmarks`` by reprojection BA.

    Parameters
    ----------
    K        : (3,3) pinhole intrinsics, shared by all keyframes.
    poses    : list of (4,4) ``T_cw`` world->camera transforms.
    fixed    : per-pose bool; True keyframes are held constant (gauge anchor).
    landmarks: (M,3) world points.
    obs_cam  : (N,) int, keyframe index of each observation.
    obs_lm   : (N,) int, landmark index of each observation.
    obs_uv   : (N,2) float, measured pixel of each observation.
    obs_depth: (N,) float or None. Measured metric depth (m) for each
               observation; values <= 0 or NaN are treated as "no depth". When
               provided (and ``cfg.use_depth``) these anchor the metric scale.
    """
    cfg = cfg or BAConfig()
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    poses = [p.copy() for p in poses]
    landmarks = landmarks.astype(np.float64).copy()
    nC = len(poses)
    M = landmarks.shape[0]

    obs_cam = np.asarray(obs_cam, np.int64)
    obs_lm = np.asarray(obs_lm, np.int64)
    obs_uv = np.asarray(obs_uv, np.float64)
    use_depth = bool(cfg.use_depth and obs_depth is not None)
    if use_depth:
        obs_depth = np.asarray(obs_depth, np.float64)
    else:
        obs_depth = np.zeros(obs_cam.shape[0])

    # Free-camera column indexing (fixed cameras get no columns).
    free_col = [-1] * nC
    nf = 0
    for i in range(nC):
        if not fixed[i]:
            free_col[i] = nf
            nf += 1
    free_col = np.asarray(free_col, np.int64)
    fc_obs = free_col[obs_cam]            # (N,) free column per obs, -1 if fixed
    N = obs_cam.shape[0]
    uv0 = obs_uv[:, 0]
    uv1 = obs_uv[:, 1]
    has_depth = use_depth & (obs_depth > 0)

    def _stack(poses_):
        Rs = np.stack([p[:3, :3] for p in poses_])    # (nC,3,3)
        ts = np.stack([p[:3, 3] for p in poses_])     # (nC,3)
        return Rs, ts

    def eval_cost(poses_, lms_):
        """Robust cost + mean inlier reprojection error (vectorised)."""
        Rs, ts = _stack(poses_)
        Xc = np.einsum('nij,nj->ni', Rs[obs_cam], lms_[obs_lm]) + ts[obs_cam]
        Z = Xc[:, 2]
        ok = Z > cfg.min_view_z
        Zc = np.where(ok, Z, 1.0)
        u = fx * Xc[:, 0] / Zc + cx
        v = fy * Xc[:, 1] / Zc + cy
        e = np.hypot(u - uv0, v - uv1)
        small = e <= cfg.huber_px
        rep = np.where(small, 0.5 * e * e,
                       cfg.huber_px * (e - 0.5 * cfg.huber_px))
        cost = float(np.where(ok, rep, cfg.huber_px ** 2).sum())
        mean_e = float(e[ok].mean()) if ok.any() else 0.0
        if use_depth:
            dm = has_depth & ok
            sig = np.where(dm, cfg.depth_sigma_coeff * obs_depth ** 2, 1.0)
            rz = np.where(dm, (Z - obs_depth) / sig, 0.0)
            thr = cfg.depth_huber / sig
            dsmall = np.abs(rz) <= thr
            dcost = np.where(dsmall, 0.5 * rz * rz,
                             thr * (np.abs(rz) - 0.5 * thr))
            cost += float(np.where(dm, dcost, 0.0).sum())
        return cost, mean_e

    # Pre-group free-camera observations by landmark (for the Schur S block).
    free_mask = fc_obs >= 0
    free_idx = np.nonzero(free_mask)[0]
    order = free_idx[np.argsort(obs_lm[free_idx], kind="stable")]
    seg_lm = obs_lm[order]
    bounds = np.searchsorted(seg_lm, np.arange(M + 1))

    cost0, _ = eval_cost(poses, landmarks)
    lam = cfg.init_lambda
    cost_prev = cost0
    it = 0

    for it in range(cfg.max_iters):
        Rs, ts = _stack(poses)
        Rc = Rs[obs_cam]                                  # (N,3,3)
        Xc = np.einsum('nij,nj->ni', Rc, landmarks[obs_lm]) + ts[obs_cam]
        Z = Xc[:, 2]
        ok = Z > cfg.min_view_z
        Zc = np.where(ok, Z, 1.0)
        invZ = 1.0 / Zc

        # --- visual reprojection rows (2 per obs) ---------------------------
        u = fx * Xc[:, 0] * invZ + cx
        v = fy * Xc[:, 1] * invZ + cy
        r = np.stack([u - uv0, v - uv1], axis=1)          # (N,2)
        e = np.linalg.norm(r, axis=1)
        w = np.where(e <= cfg.huber_px, 1.0,
                     np.sqrt(cfg.huber_px / np.maximum(e, 1e-12)))
        w = np.where(ok, w, 0.0)                          # drop behind-cam obs

        # d(proj)/d(Xc): (N,2,3)
        Jp = np.zeros((N, 2, 3))
        Jp[:, 0, 0] = fx * invZ
        Jp[:, 0, 2] = -fx * Xc[:, 0] * invZ * invZ
        Jp[:, 1, 1] = fy * invZ
        Jp[:, 1, 2] = -fy * Xc[:, 1] * invZ * invZ

        Jl = np.einsum('nij,njk->nik', Jp, Rc)            # (N,2,3) landmark
        # [I | -skew(Xc)] : (N,3,6)
        A = np.zeros((N, 3, 6))
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 2, 2] = 1.0
        A[:, 0, 4] = Xc[:, 2]
        A[:, 0, 5] = -Xc[:, 1]
        A[:, 1, 3] = -Xc[:, 2]
        A[:, 1, 5] = Xc[:, 0]
        A[:, 2, 3] = Xc[:, 1]
        A[:, 2, 4] = -Xc[:, 0]
        Jc = np.einsum('nij,njk->nik', Jp, A)             # (N,2,6) pose

        Jl_sw = w[:, None, None] * Jl
        Jc_sw = w[:, None, None] * Jc
        r_sw = w[:, None] * r

        Hpp = np.zeros((M, 3, 3))
        bp = np.zeros((M, 3))
        np.add.at(Hpp, obs_lm, np.einsum('nai,naj->nij', Jl_sw, Jl_sw))
        np.add.at(bp, obs_lm, np.einsum('nai,na->ni', Jl_sw, r_sw))

        Hcc_blk = np.zeros((max(nf, 1), 6, 6))
        bc_blk = np.zeros((max(nf, 1), 6))
        Eob = np.einsum('nai,naj->nij', Jc_sw, Jl_sw)     # (N,6,3) coupling
        if nf > 0 and free_mask.any():
            fm = free_mask
            np.add.at(Hcc_blk, fc_obs[fm],
                      np.einsum('nai,naj->nij', Jc_sw[fm], Jc_sw[fm]))
            np.add.at(bc_blk, fc_obs[fm],
                      np.einsum('nai,na->ni', Jc_sw[fm], r_sw[fm]))

        # --- metric depth rows (1 per obs with valid depth) ----------------
        if use_depth:
            dm = has_depth & ok
            sig = np.where(dm, cfg.depth_sigma_coeff * obs_depth ** 2, 1.0)
            rz = np.where(dm, (Z - obs_depth) / sig, 0.0)
            thr = cfg.depth_huber / sig
            wz = np.where(np.abs(rz) <= thr, 1.0,
                          np.sqrt(thr / np.maximum(np.abs(rz), 1e-12)))
            wz = np.where(dm, wz, 0.0)
            Jlz = Rc[:, 2, :] / sig[:, None]              # (N,3)
            Jlz_sw = wz[:, None] * Jlz
            np.add.at(Hpp, obs_lm, np.einsum('ni,nj->nij', Jlz_sw, Jlz_sw))
            np.add.at(bp, obs_lm, Jlz_sw * (wz * rz)[:, None])
            Jcz = np.zeros((N, 6))
            Jcz[:, 2] = 1.0 / sig
            Jcz[:, 3] = Xc[:, 1] / sig
            Jcz[:, 4] = -Xc[:, 0] / sig
            Jcz_sw = wz[:, None] * Jcz
            Eob = Eob + np.einsum('ni,nj->nij', Jcz_sw, Jlz_sw)
            if nf > 0 and free_mask.any():
                fm = free_mask
                np.add.at(Hcc_blk, fc_obs[fm],
                          np.einsum('ni,nj->nij', Jcz_sw[fm], Jcz_sw[fm]))
                np.add.at(bc_blk, fc_obs[fm], Jcz_sw[fm] * (wz * rz)[fm, None])

        # --- LM damping ----------------------------------------------------
        di = np.arange(3)
        Hpp_d = Hpp.copy()
        Hpp_d[:, di, di] += lam * np.clip(Hpp[:, di, di], 1e-9, None) + 1e-12
        Hpp_inv = np.linalg.inv(Hpp_d)                    # (M,3,3)

        # Assemble big camera Hessian (block-diagonal) + rhs.
        Hcc = np.zeros((6 * nf, 6 * nf))
        bc = np.zeros(6 * nf)
        for fcc in range(nf):
            s = slice(6 * fcc, 6 * fcc + 6)
            Hcc[s, s] = Hcc_blk[fcc]
            bc[s] = bc_blk[fcc]
        if nf > 0:
            dd = np.diag(Hcc)
            Hcc[np.diag_indices_from(Hcc)] += lam * np.clip(dd, 1e-9, None)

        # Schur complement: rhs = -bc + sum E Hpp^-1 bp ; S = Hcc - E Hpp^-1 E^T
        S = Hcc.copy()
        rhs = -bc.copy()
        if nf > 0:
            Y = np.einsum('nij,njk->nik', Eob, Hpp_inv[obs_lm])   # (N,6,3)
            t1 = np.einsum('nij,nj->ni', Y, bp[obs_lm])           # (N,6)
            np.add.at(rhs.reshape(nf, 6), fc_obs[free_mask], t1[free_mask])
            # Pairwise S blocks, grouped by landmark (few obs per landmark).
            for l in range(M):
                seg = order[bounds[l]:bounds[l + 1]]
                if seg.size == 0:
                    continue
                fcs = fc_obs[seg]
                blocks = np.einsum('aij,bkj->abik', Y[seg], Eob[seg])  # (k,k,6,6)
                for ai in range(seg.size):
                    ra = slice(6 * fcs[ai], 6 * fcs[ai] + 6)
                    for bi in range(seg.size):
                        cb = slice(6 * fcs[bi], 6 * fcs[bi] + 6)
                        S[ra, cb] -= blocks[ai, bi]

        # Solve reduced camera system.
        if nf > 0:
            try:
                dc = np.linalg.solve(S, rhs)
            except np.linalg.LinAlgError:
                dc = np.linalg.lstsq(S, rhs, rcond=None)[0]
        else:
            dc = np.zeros(0)

        # Back-substitute landmarks (vectorised).
        dc_obs = np.zeros((N, 6))
        if nf > 0:
            dc_blocks = dc.reshape(nf, 6)
            dc_obs[free_mask] = dc_blocks[fc_obs[free_mask]]
        sum_term = np.zeros((M, 3))
        np.add.at(sum_term, obs_lm, np.einsum('nij,ni->nj', Eob, dc_obs))
        dp = np.einsum('mij,mj->mi', Hpp_inv, -bp - sum_term)

        # Trial update.
        trial_poses = []
        for i in range(nC):
            fcc = free_col[i]
            if fcc >= 0:
                trial_poses.append(se3_exp(dc[6 * fcc:6 * fcc + 6]) @ poses[i])
            else:
                trial_poses.append(poses[i])
        trial_lms = landmarks + dp

        cost_new, mean_px = eval_cost(trial_poses, trial_lms)
        if cost_new < cost_prev:
            poses = trial_poses
            landmarks = trial_lms
            lam = max(cfg.min_lambda, lam * 0.5)
            improved = (cost_prev - cost_new) / max(cost_prev, 1e-12)
            cost_prev = cost_new
            if improved < cfg.rel_tol:
                break
        else:
            lam = min(cfg.max_lambda, lam * 4.0)

    final_cost, mean_px = eval_cost(poses, landmarks)
    return BAResult(
        poses=poses,
        landmarks=landmarks,
        iters=it + 1,
        cost0=cost0,
        cost1=final_cost,
        mean_reproj_px=mean_px,
    )
