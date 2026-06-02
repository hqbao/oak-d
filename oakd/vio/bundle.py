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

    def eval_cost(poses_):
        """Robust cost + mean inlier reprojection error over all observations."""
        cost = 0.0
        se = 0.0
        cnt = 0
        for n in range(obs_cam.shape[0]):
            T = poses_[obs_cam[n]]
            Xc = T[:3, :3] @ landmarks[obs_lm[n]] + T[:3, 3]
            Z = Xc[2]
            if Z <= cfg.min_view_z:
                cost += cfg.huber_px ** 2  # penalise but stay finite
                continue
            u = fx * Xc[0] / Z + cx
            v = fy * Xc[1] / Z + cy
            e = float(np.hypot(u - obs_uv[n, 0], v - obs_uv[n, 1]))
            if e <= cfg.huber_px:
                cost += 0.5 * e * e
            else:
                cost += cfg.huber_px * (e - 0.5 * cfg.huber_px)
            se += e
            cnt += 1
            # metric depth term
            zm = obs_depth[n]
            if use_depth and zm > 0:
                sig = cfg.depth_sigma_coeff * zm * zm
                rz = (Z - zm) / sig
                if abs(rz) <= cfg.depth_huber / sig:
                    cost += 0.5 * rz * rz
                else:
                    d = cfg.depth_huber / sig
                    cost += d * (abs(rz) - 0.5 * d)
        return cost, (se / cnt if cnt else 0.0)

    cost0, _ = eval_cost(poses)
    lam = cfg.init_lambda
    cost_prev = cost0

    for it in range(cfg.max_iters):
        # Accumulators for the normal equations.
        Hcc = np.zeros((6 * nf, 6 * nf))
        bc = np.zeros(6 * nf)
        Hpp = np.zeros((M, 3, 3))
        bp = np.zeros((M, 3))
        # E[l] = list of (free_col, 6x3 block) coupling landmark l to cameras.
        E: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(M)]

        for n in range(obs_cam.shape[0]):
            c = int(obs_cam[n])
            l = int(obs_lm[n])
            T = poses[c]
            R = T[:3, :3]
            Xc = R @ landmarks[l] + T[:3, 3]
            Z = Xc[2]
            if Z <= cfg.min_view_z:
                continue
            invZ = 1.0 / Z
            u = fx * Xc[0] * invZ + cx
            v = fy * Xc[1] * invZ + cy
            r = np.array([u - obs_uv[n, 0], v - obs_uv[n, 1]])
            e = float(np.linalg.norm(r))
            w = _huber_weight(e, cfg.huber_px)

            # d(proj)/d(Xc): 2x3
            Jp = np.array([
                [fx * invZ, 0.0, -fx * Xc[0] * invZ * invZ],
                [0.0, fy * invZ, -fy * Xc[1] * invZ * invZ],
            ])
            # Landmark Jacobian: 2x3
            Jl = Jp @ R
            # Pose Jacobian: 2x6 (only if camera is free)
            Jl_sw = w * Jl
            r_sw = w * r

            Hpp[l] += Jl_sw.T @ Jl_sw
            bp[l] += Jl_sw.T @ r_sw

            fc = free_col[c]
            Jc = None
            if fc >= 0:
                Jc = Jp @ np.hstack([np.eye(3), -skew(Xc)])  # 2x6
                Jc_sw = w * Jc
                s = slice(6 * fc, 6 * fc + 6)
                Hcc[s, s] += Jc_sw.T @ Jc_sw
                bc[s] += Jc_sw.T @ r_sw
                E[l].append((fc, Jc_sw.T @ Jl_sw))  # 6x3

            # --- metric depth residual: (Z - z_meas) anchors global scale ----
            zm = obs_depth[n]
            if use_depth and zm > 0:
                sig = cfg.depth_sigma_coeff * zm * zm
                rz = (Z - zm) / sig
                wz = _huber_weight(abs(rz), cfg.depth_huber / sig)
                # dZ/dXw = R[2,:] ; dZ/dxi = [0,0,1, Y, -X, 0]
                Jlz = (R[2, :] / sig)              # (3,)
                Jlz_sw = wz * Jlz
                Hpp[l] += np.outer(Jlz_sw, Jlz_sw)
                bp[l] += Jlz_sw * (wz * rz)
                if fc >= 0:
                    Jcz = np.array([0.0, 0.0, 1.0, Xc[1], -Xc[0], 0.0]) / sig
                    Jcz_sw = wz * Jcz
                    s = slice(6 * fc, 6 * fc + 6)
                    Hcc[s, s] += np.outer(Jcz_sw, Jcz_sw)
                    bc[s] += Jcz_sw * (wz * rz)
                    E[l].append((fc, np.outer(Jcz_sw, Jlz_sw)))  # 6x3

        # LM damping on camera block.
        if nf > 0:
            Hcc[np.diag_indices_from(Hcc)] += lam * np.diag(Hcc).clip(min=1e-9)

        # Schur complement: eliminate landmarks.
        S = Hcc.copy()
        rhs = -bc.copy()
        Hpp_inv = np.zeros_like(Hpp)
        for l in range(M):
            H = Hpp[l].copy()
            H[np.diag_indices_from(H)] += lam * np.clip(np.diag(H), 1e-9, None)
            try:
                Hi = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                Hi = np.linalg.pinv(H)
            Hpp_inv[l] = Hi
            if not E[l]:
                continue
            Hi_bp = Hi @ bp[l]
            for (fc, Ecl) in E[l]:
                s = slice(6 * fc, 6 * fc + 6)
                rhs[s] += Ecl @ Hi_bp                       # +E Hpp^-1 bp
                for (fc2, Ec2l) in E[l]:
                    s2 = slice(6 * fc2, 6 * fc2 + 6)
                    S[s, s2] -= Ecl @ Hi @ Ec2l.T           # -E Hpp^-1 E^T

        # Solve reduced camera system.
        if nf > 0:
            try:
                dc = np.linalg.solve(S, rhs)
            except np.linalg.LinAlgError:
                dc = np.linalg.lstsq(S, rhs, rcond=None)[0]
        else:
            dc = np.zeros(0)

        # Back-substitute landmarks: dp_l = Hpp^-1 (-bp - sum_c E^T dc_c)
        dp = np.zeros((M, 3))
        for l in range(M):
            rhs_l = -bp[l].copy()
            for (fc, Ecl) in E[l]:
                rhs_l -= Ecl.T @ dc[6 * fc:6 * fc + 6]
            dp[l] = Hpp_inv[l] @ rhs_l

        # Trial update.
        trial_poses = []
        for i in range(nC):
            fc = free_col[i]
            if fc >= 0:
                trial_poses.append(se3_exp(dc[6 * fc:6 * fc + 6]) @ poses[i])
            else:
                trial_poses.append(poses[i])
        trial_lms = landmarks + dp

        # Evaluate; accept/reject (LM).
        lm_bak = landmarks
        landmarks = trial_lms
        cost_new, mean_px = eval_cost(trial_poses)
        if cost_new < cost_prev:
            poses = trial_poses
            lam = max(cfg.min_lambda, lam * 0.5)
            improved = (cost_prev - cost_new) / max(cost_prev, 1e-12)
            cost_prev = cost_new
            if improved < cfg.rel_tol:
                break
        else:
            landmarks = lm_bak  # reject
            lam = min(cfg.max_lambda, lam * 4.0)

    final_cost, mean_px = eval_cost(poses)
    return BAResult(
        poses=poses,
        landmarks=landmarks,
        iters=it + 1,
        cost0=cost0,
        cost1=final_cost,
        mean_reproj_px=mean_px,
    )
