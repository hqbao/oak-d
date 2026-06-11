"""Sliding-window bundle adjustment (BA) core — pure NumPy.

This is the optimisation *engine* only: given a set of keyframe poses, 3D
landmarks and 2D observations, it refines them by minimising the robust
reprojection error. It knows nothing about cameras, KLT or sessions — that
glue lives in :mod:`vio.mathlib.backend.windowed`. Keeping the solver standalone means we
can unit-test it on a synthetic scene with a known answer (see
``vio/tests/vio_ba_selftest.py``) before trusting it on real data.

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

# Lie-group helpers (SE3 / SO3) -- sourced from the shared ``sky.math`` kernel.
# Bundle adjustment uses the BA-convention exponential (first-order ``I + skew``
# at zero) and the ``solve``-based ``se3_log``; ``se3_log`` / ``skew`` are also
# re-exported for :mod:`~vio.mathlib.backend.marginalize`. Numerics are
# byte-identical to the former local copies.
from sky.math import se3_exp, se3_log, skew


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
    #
    # ``depth_sigma_coeff`` IS the stereo noise constant: from triangulation
    # ``z = f*B/d`` the depth error is ``sigma_z = z^2 * sigma_disp / (f*B)``, so
    # ``coeff = sigma_disp / (f*B)``. For this OAK-D ``f*B = 282.2 px * 0.0751 m
    # = 21.2 px*m``, so a standard ~1 px SGM disparity noise gives ``coeff ~=
    # 0.05``. The earlier ``0.02`` implied a sub-pixel disparity precision of only
    # 0.42 px -- far too optimistic for block-matched SGM. Measured on the LIVE
    # SGM depth with the live window (kf_every=5, window=6, iters=5), the Sim3
    # scale vs Basalt is NON-MONOTONIC in this coefficient with a sharp optimum:
    # on push_straight_fast 0.02 -> 0.40 (badly collapsed, the "moves a bit then
    # stalls" symptom), 0.05 -> 0.91; push_fwdback 0.81 -> 0.85 (f2f ~0.87). The
    # plateau is broad (0.04-0.06 all land ~0.90), so 0.05 is both the physically
    # honest value AND the empirical optimum -- not a fragile tuned constant.
    # NB: this is sensitive to the depth SOURCE (recorded chip depth behaves
    # differently) and to the window/kf cadence -- validate any retune on the
    # LIVE SGM depth with the live window (window=6, kf_every=5, iters=5).
    use_depth: bool = True
    depth_sigma_coeff: float = 0.05   # sigma_z = coeff * z^2  (m); = sigma_disp/(f*B)
    depth_huber: float = 0.10         # robust threshold on depth residual (m)
    # --- gravity prior (accelerometer leveling inside BA) -------------------
    # Pure reprojection (+depth) BA optimises all 6-DoF of every keyframe but has
    # NO gravity awareness, so absolute roll/pitch slowly tilt-drifts with the
    # map. When a keyframe has a trustworthy at-rest accelerometer reading we add
    # a soft constraint pulling its world-down direction (rotated into the
    # camera) onto the measured gravity. This pins roll/pitch (NOT yaw -- rotating
    # about gravity leaves the constraint unchanged), so the map stays level
    # without a display-side hack. Off by default to keep the offline path
    # byte-identical; the live source opts in. ``sigma`` is the 1-sigma on the
    # unit-vector direction residual (~radians of tilt), ``huber`` the robust
    # threshold (rad), ``band`` the fractional |accel| window around the gravity
    # magnitude outside which the sample is treated as accelerating (rejected).
    use_gravity: bool = False
    gravity_sigma_rad: float = 0.05   # ~2.9 deg; smaller => trust accel more
    gravity_huber: float = 0.2        # robust threshold on tilt residual (rad)
    gravity_band: float = 0.15        # accept |a| within +/- this frac of g_ref
    # --- front-end relative-translation prior (the scale anchor) ------------
    # WHY: pure reprojection BA is scale-free along a low-parallax axis (a
    # straight forward push), so the metric-depth residual is the ONLY thing
    # holding metric scale there -- and it is too weak (1 row/obs, huberized) to
    # stop the window's inter-keyframe baseline from shrinking. Measured (gold,
    # --depth ours): frame-to-frame PnP keeps Sim3 scale 0.90-0.98 on a forward
    # push while windowed BA collapses to 0.30-0.39 on the SAME depth + SAME
    # frontend; no depth-weight retune fixes it (host-anchor/huber sweeps all
    # plateaued ~0.5). The frame-to-frame PnP translation IS metric (depth-
    # anchored 3D-2D), so we feed it back as a soft prior on the relative
    # translation between consecutive keyframes -- exactly the role IMU
    # preintegration plays in a tight-coupled VIO, but from our own VO. BA is
    # then still free to refine rotation, landmarks and loop drift, but can no
    # longer shrink the baseline below the front-end's correctly-scaled value.
    # ``sigma`` is the 1-sigma (m) on that relative translation; smaller => trust
    # the front-end more. ``huber`` robustifies against an occasional bad VO step
    # (tracking loss) so one wrong reference cannot warp the window. 0 = off
    # (byte-identical to plain BA); the caller passes the per-pair references.
    use_vo_trans_prior: bool = False
    vo_trans_sigma_m: float = 0.01    # 1-sigma on inter-KF relative translation
    vo_trans_huber_m: float = 0.10    # robust threshold on the residual (m)


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
    grav_meas: np.ndarray | None = None,
    grav_world: np.ndarray | None = None,
    grav_gref: float | None = None,
    prior_cams: list[int] | None = None,
    prior_H: np.ndarray | None = None,
    prior_b0: np.ndarray | None = None,
    prior_lin: np.ndarray | None = None,
    vo_rel_a: np.ndarray | None = None,
    vo_rel_b: np.ndarray | None = None,
    vo_rel_t: np.ndarray | None = None,
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
    grav_meas: (nC,3) float or None. Per-keyframe accelerometer vector in the
               camera optical frame (m/s^2). Rows that are NaN/inf mean "no
               gravity measurement for this keyframe". Used (with
               ``cfg.use_gravity``) to add the gravity-leveling prior.
    grav_world: (3,) unit world-down direction in the map frame; defaults to the
               optical-world down ``[0, 1, 0]``.
    grav_gref: reference gravity magnitude for the |accel| band gate; defaults
               to the median of the valid ``grav_meas`` norms.
    prior_cams: list of ``len(prior_cams)=k`` keyframe indices (into ``poses``)
               that the marginalization prior constrains, in the prior's block
               order. ``None`` disables the prior. All referenced cameras must be
               free (not in ``fixed``).
    prior_H  : (6k,6k) symmetric PSD prior information matrix (the Schur-condensed
               Hessian of the marginalized states), in the same ``[rho;phi]``
               left-perturbation ordering as the solver.
    prior_b0 : (6k,) prior gradient at the linearization point (same sign as the
               solver's ``b = J^T r``).
    prior_lin: (k,4,4) linearization-point ``T_cw`` per prior block. First-Estimate
               Jacobians: the prior Jacobian is held at identity, so the prior
               contributes ``H_p`` to the camera Hessian and ``b0 + H_p @ delta``
               to the gradient, where ``delta_a = se3_log(T_cur @ T_lin^-1)``.
    vo_rel_a : (P,) int or None. Older-keyframe pose index of each front-end
               relative-translation prior pair.
    vo_rel_b : (P,) int or None. Newer-keyframe pose index of each pair.
    vo_rel_t : (P,3) float or None. The front-end (f2f PnP) relative translation
               ``t_rel`` for each pair; the prior (with ``cfg.use_vo_trans_prior``)
               softly pulls the optimised relative translation toward it,
               anchoring metric scale. See ``BAConfig.use_vo_trans_prior``.
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

    # --- gravity prior precompute ------------------------------------------
    # For each keyframe with a trustworthy at-rest accel reading we constrain
    # ``R_cw @ g_world`` (world-down in the camera) to match the measured
    # down ``-accel/|accel|``. This is a 3-vector direction residual that only
    # bites on roll/pitch (yaw rotates about g_world, leaving it invariant). It
    # couples to no landmark, so it adds only to the per-camera Hessian blocks.
    use_grav = bool(cfg.use_gravity and grav_meas is not None)
    if use_grav:
        GA = np.asarray(grav_meas, np.float64)
        gw_world = (np.asarray(grav_world, np.float64)
                    if grav_world is not None else np.array([0.0, 1.0, 0.0]))
        gw_world = gw_world / max(float(np.linalg.norm(gw_world)), 1e-12)
        g_nrm = np.linalg.norm(GA, axis=1)
        g_finite = np.all(np.isfinite(GA), axis=1) & (g_nrm > 1e-6)
        if grav_gref is not None:
            g_ref = float(grav_gref)
        elif g_finite.any():
            g_ref = float(np.median(g_nrm[g_finite]))
        else:
            g_ref = 0.0
        g_band_ok = np.abs(g_nrm - g_ref) <= cfg.gravity_band * max(g_ref, 1e-9)
        fixed_arr = np.asarray(fixed, dtype=bool)
        grav_active = g_finite & g_band_ok & (~fixed_arr)   # free cameras only
        down_meas = np.zeros((nC, 3))
        if g_finite.any():
            down_meas[g_finite] = -GA[g_finite] / g_nrm[g_finite, None]
        g_sig = max(cfg.gravity_sigma_rad, 1e-6)
        g_thr = cfg.gravity_huber / g_sig
        grav_rows = np.nonzero(grav_active)[0]
    else:
        grav_rows = np.array([], dtype=np.int64)
        gw_world = np.array([0.0, 1.0, 0.0])
        down_meas = np.zeros((nC, 3))
        g_sig = 1.0
        g_thr = 0.0

    # --- marginalization prior precompute ----------------------------------
    # A linear-Gaussian prior over k keyframe poses, condensed from previously
    # marginalized keyframes/landmarks (see vio.mathlib.backend.marginalize). It is
    # camera-only (couples poses, no landmarks) and uses First-Estimate
    # Jacobians: the prior Jacobian is frozen at the linearization point, so it
    # contributes ``H_p`` to the camera Hessian and ``b0 + H_p @ delta`` to the
    # gradient, with ``delta_a = se3_log(T_cur @ T_lin^-1)`` per block. No robust
    # kernel (the whitening is already baked into H_p/b0).
    use_prior = (prior_cams is not None and prior_H is not None
                 and len(prior_cams) > 0)
    if use_prior:
        pri_cams = np.asarray(prior_cams, np.int64)
        pri_H = np.asarray(prior_H, np.float64)
        pri_b0 = np.asarray(prior_b0, np.float64)
        pri_lin_inv = np.stack([np.linalg.inv(np.asarray(prior_lin[a], np.float64))
                                for a in range(len(prior_cams))])
        k_pri = len(prior_cams)

        def _prior_delta(poses_):
            d = np.empty(6 * k_pri)
            for a in range(k_pri):
                d[6 * a:6 * a + 6] = se3_log(poses_[pri_cams[a]] @ pri_lin_inv[a])
            return d

    # --- front-end relative-translation prior precompute -------------------
    # Soft constraint pulling each consecutive-keyframe relative translation
    # ``t_rel = t_b - R_b R_a^T t_a`` toward the front-end (f2f PnP) value
    # ``vo_rel_t``. Camera-only (couples the two cameras of each pair, no
    # landmarks). Jacobian (left SE3 perturbation, [rho|phi] columns):
    #   d t_rel / d xi_b = [ I            , -skew(t_rel) ]
    #   d t_rel / d xi_a = [ -R_b R_a^T   ,  0           ]
    # At least one of the pair must be a free camera for the prior to bite; a
    # fixed camera contributes no columns (it acts as the anchor).
    use_vt = bool(cfg.use_vo_trans_prior and vo_rel_a is not None
                  and vo_rel_b is not None and vo_rel_t is not None
                  and len(vo_rel_a) > 0)
    if use_vt:
        vt_a = np.asarray(vo_rel_a, np.int64)
        vt_b = np.asarray(vo_rel_b, np.int64)
        vt_t = np.asarray(vo_rel_t, np.float64)
        vt_sig = max(cfg.vo_trans_sigma_m, 1e-6)
        vt_thr = cfg.vo_trans_huber_m / vt_sig
    else:
        vt_a = np.array([], dtype=np.int64)
        vt_b = np.array([], dtype=np.int64)
        vt_t = np.zeros((0, 3))
        vt_sig = 1.0
        vt_thr = 0.0

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
        if use_grav and grav_rows.size:
            gc = 0.0
            for i in grav_rows:
                gwc = poses_[i][:3, :3] @ gw_world
                rg = gwc - down_meas[i]
                e_rho = float(np.linalg.norm(rg)) / g_sig
                if e_rho <= g_thr:
                    gc += 0.5 * e_rho * e_rho
                else:
                    gc += g_thr * (e_rho - 0.5 * g_thr)
            cost += gc
        if use_prior:
            d = _prior_delta(poses_)
            cost += float(0.5 * d @ (pri_H @ d) + pri_b0 @ d)
        if use_vt and vt_a.size:
            Rs_, ts_ = _stack(poses_)
            for p in range(vt_a.size):
                a, b = vt_a[p], vt_b[p]
                t_rel = ts_[b] - Rs_[b] @ Rs_[a].T @ ts_[a]
                r = t_rel - vt_t[p]
                e_t = float(np.linalg.norm(r)) / vt_sig
                if e_t <= vt_thr:
                    cost += 0.5 * e_t * e_t
                else:
                    cost += vt_thr * (e_t - 0.5 * vt_thr)
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

        # --- gravity prior rows (camera-only, one 3-vector per keyframe) ----
        # Residual r_g = R_cw @ g_world - down_meas. With the left SE3
        # perturbation Exp(xi) @ T_cw the rotation updates as Exp(phi) @ R_cw, so
        # d r_g / d phi = -[R_cw @ g_world]_x and d r_g / d rho = 0 (no landmark
        # coupling). Whitened by 1/sigma with a Huber sqrt-weight, this adds only
        # to the rotation block (cols/rows 3:6) of the free camera's Hessian.
        if use_grav and grav_rows.size and nf > 0:
            for i in grav_rows:
                fcc = free_col[i]
                if fcc < 0:
                    continue
                gwc = Rs[i] @ gw_world
                rg = gwc - down_meas[i]
                e_rho = float(np.linalg.norm(rg)) / g_sig
                wh = 1.0 if e_rho <= g_thr else np.sqrt(
                    g_thr / max(e_rho, 1e-12))
                c = (wh / g_sig) ** 2
                Sg = skew(gwc)
                Hcc_blk[fcc][3:, 3:] += c * (Sg.T @ Sg)
                bc_blk[fcc][3:] += c * (Sg @ rg)

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
        # --- marginalization prior (camera-only, FEJ identity Jacobian) -----
        # Inject H_p into the (possibly off-diagonal) camera blocks and the
        # gradient b0 + H_p @ delta into bc, mapping each prior block to its free
        # camera column. Done before LM damping so the prior is damped with the
        # rest of the camera system.
        if use_prior and nf > 0:
            d = _prior_delta(poses)
            g_pri = pri_H @ d + pri_b0                     # (6k,) gradient
            fcols = free_col[pri_cams]                     # (k,) free col per block
            for a in range(k_pri):
                fa = fcols[a]
                if fa < 0:
                    continue
                ra = slice(6 * fa, 6 * fa + 6)
                bc[ra] += g_pri[6 * a:6 * a + 6]
                for b_ in range(k_pri):
                    fb = fcols[b_]
                    if fb < 0:
                        continue
                    cb = slice(6 * fb, 6 * fb + 6)
                    Hcc[ra, cb] += pri_H[6 * a:6 * a + 6, 6 * b_:6 * b_ + 6]
        # --- front-end relative-translation prior (camera-only, 2-cam block) -
        # For each pair add the Gauss-Newton contribution of the 3-vector
        # residual r = t_rel - vo_rel_t, with the analytic Jacobians above,
        # whitened by 1/sigma and Huber-robustified. Fixed cameras drop their
        # columns (no free_col), so a pair anchored on the fixed gauge still
        # constrains the free camera's baseline.
        if use_vt and vt_a.size and nf > 0:
            for p in range(vt_a.size):
                a, b = int(vt_a[p]), int(vt_b[p])
                Ra, Rb = Rs[a], Rs[b]
                ta, tb = ts[a], ts[b]
                t_rel = tb - Rb @ Ra.T @ ta
                r = t_rel - vt_t[p]
                e_t = float(np.linalg.norm(r)) / vt_sig
                wh = 1.0 if e_t <= vt_thr else np.sqrt(
                    vt_thr / max(e_t, 1e-12))
                c = (wh / vt_sig) ** 2
                Jb = np.zeros((3, 6))
                Jb[:, :3] = np.eye(3)
                Jb[:, 3:] = -skew(t_rel)
                Ja = np.zeros((3, 6))
                Ja[:, :3] = -(Rb @ Ra.T)
                fa, fb = free_col[a], free_col[b]
                if fb >= 0:
                    sb = slice(6 * fb, 6 * fb + 6)
                    Hcc[sb, sb] += c * (Jb.T @ Jb)
                    bc[sb] += c * (Jb.T @ r)
                if fa >= 0:
                    sa = slice(6 * fa, 6 * fa + 6)
                    Hcc[sa, sa] += c * (Ja.T @ Ja)
                    bc[sa] += c * (Ja.T @ r)
                if fa >= 0 and fb >= 0:
                    sa = slice(6 * fa, 6 * fa + 6)
                    sb = slice(6 * fb, 6 * fb + 6)
                    Hab = c * (Ja.T @ Jb)
                    Hcc[sa, sb] += Hab
                    Hcc[sb, sa] += Hab.T
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
