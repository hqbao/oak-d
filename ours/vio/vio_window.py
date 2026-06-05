"""Tight-coupled visual-inertial window optimizer (pure NumPy).

This is the Basalt-style core the loosely-coupled gyro fusion could not be: it
puts the raw visual measurements (reprojection + metric depth) **and** the IMU
preintegration factors (rotation, velocity, position increments) into ONE
non-linear least-squares problem, solving jointly for every keyframe's pose,
**velocity** and **gyro/accel bias**, plus the landmarks. Because the
accelerometer ties consecutive keyframes through ``v`` and ``p``, a pure in-place
rotation -- where the true linear acceleration is ~0 -- can no longer be
explained as a translation by slipped visual tracks: the IMU says "no
acceleration => no net translation", killing the phantom yaw-drift that the
vision-only / loosely-coupled paths leave behind.

Design choices (deliberate, to be correct and verifiable before fast):
  * **Body frame == camera frame** in this core. The IMU<->camera extrinsic is
    handled by the caller, which rotates the raw IMU samples into the camera
    optical frame before preintegrating (see :mod:`oakd.vio.imu`). The small
    IMU/camera lever arm is treated as modelling noise (the OAK-D IMU sits ~cm
    from the left camera); a future refinement can add it explicitly.
  * Poses are parametrised as **body->world** ``(R, p)`` with the perturbation
    ``R <- R Exp(dphi)``, ``p <- p + R dp`` (GTSAM Pose3 convention). Velocity
    and biases are plain additive vector states.
  * **Per-factor finite-difference Jacobians.** Each factor differentiates only
    its own local variables (a projection sees 1 pose + 1 landmark; an IMU
    factor sees the 2 adjacent nav states), so FD is cheap AND immune to the
    hand-derivation sign errors that plague analytic VIO Jacobians. The IMU
    residual formulas themselves are the ones validated in
    ``tools/imu_preint_selftest.py``.
  * **Dense Levenberg-Marquardt** over the whole window (no Schur complement).
    Correctness first; the window is small, so a dense solve is fine. Schur is a
    speed optimisation left for later if the live path needs it.

One keyframe (``anchor``, default 0) has its pose held fixed to pin the global
position+yaw gauge (gravity already fixes roll/pitch through the IMU factors).
Its velocity and biases stay free.

Validated end-to-end by ``tools/vio_ba_selftest.py``: a synthetic multi-segment
trajectory (fast yaw + translation under gravity) with consistent IMU + image
measurements is perturbed and recovered to sub-mm / sub-mdeg.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .frontend import FrontendConfig, KLTFrontend
from .imu import ImuPreintegration, preintegrate_imu, so3_exp, so3_log
from .odometry import OdometryConfig, RGBDVisualOdometry


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


# --------------------------------------------------------------------------- #
# State + configuration
# --------------------------------------------------------------------------- #
@dataclass
class VioState:
    """Mutable VIO window state. Lists are per-keyframe, indexed alike.

    R, p : body->world rotation (3x3) and position (3,) per keyframe.
    v    : world-frame velocity (3,) per keyframe.
    bg, ba: gyro / accel bias (3,) per keyframe.
    landmarks: (M,3) world points.
    """
    R: list = field(default_factory=list)
    p: list = field(default_factory=list)
    v: list = field(default_factory=list)
    bg: list = field(default_factory=list)
    ba: list = field(default_factory=list)
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def copy(self) -> "VioState":
        return VioState(
            R=[r.copy() for r in self.R],
            p=[x.copy() for x in self.p],
            v=[x.copy() for x in self.v],
            bg=[x.copy() for x in self.bg],
            ba=[x.copy() for x in self.ba],
            landmarks=self.landmarks.copy(),
        )


@dataclass
class VioConfig:
    # measurement sigmas (whiten each residual to be dimensionless)
    sigma_px: float = 1.0           # pixel reprojection sigma
    depth_sigma_coeff: float = 0.02  # sigma_z = coeff * z^2  (metres)
    depth_huber: float = 0.10       # robust threshold on depth residual (m)
    sigma_rot: float = 0.01         # rad, IMU rotation increment
    sigma_vel: float = 0.05         # m/s, IMU velocity increment
    sigma_pos: float = 0.05         # m, IMU position increment
    sigma_bg_rw: float = 1e-3       # gyro-bias random walk
    sigma_ba_rw: float = 1e-2       # accel-bias random walk
    huber_px: float = 2.0           # robust threshold on pixel residual
    use_depth: bool = True
    min_view_z: float = 1e-3
    # When True the camera ROLL/PITCH (tilt relative to gravity) is HELD FIXED
    # during optimisation: each pose only has 4 free DoF -- 3 translation + 1 YAW
    # about the world-vertical axis -- instead of the full 6. The accelerometer
    # (via the IMU factor's gravity term, and the f2f gravity-levelling that
    # seeds the input pose) already owns tilt absolutely; locking it stops the
    # reprojection factors from drifting roll/pitch and stops gravity leaking
    # into a horizontal translation. Vision/depth then refine only what the IMU
    # cannot observe: yaw and position. Default False keeps the full 6-DoF
    # solve (and the byte-identical self-test recovery of a perturbed tilt).
    lock_tilt: bool = False
    # LM
    max_iters: int = 30
    init_lambda: float = 1e-3
    min_lambda: float = 1e-9
    max_lambda: float = 1e9
    rel_tol: float = 1e-7
    fd_eps: float = 1e-6


@dataclass
class VioResult:
    state: VioState
    iters: int
    cost0: float
    cost1: float
    mean_reproj_px: float


# --------------------------------------------------------------------------- #
# Residual primitives (raw, unwhitened) -- shared by cost eval and FD assembly
# --------------------------------------------------------------------------- #
def _project(R, p, Xw, fx, fy, cx, cy, min_z):
    """Camera-frame point + pixel of a world landmark for a body->world pose."""
    Xc = R.T @ (Xw - p)
    Z = Xc[2]
    Zc = Z if Z > min_z else min_z
    u = fx * Xc[0] / Zc + cx
    v = fy * Xc[1] / Zc + cy
    return Xc, Z, u, v


def _imu_residual(R_i, p_i, v_i, bg_i, ba_i,
                  R_j, p_j, v_j,
                  pre: ImuPreintegration, g_world, cfg) -> np.ndarray:
    """15-vector whitened IMU + bias-random-walk residual between KF i and j.

    Order: [rRot(3), rVel(3), rPos(3), rBiasGyro(3), rBiasAccel(3)].
    Matches the increment convention validated in imu_preint_selftest:
        R_j = R_i dR ; v_j = v_i + g dt + R_i dv ;
        p_j = p_i + v_i dt + 0.5 g dt^2 + R_i dp
    The bias rows tie i and j (random walk); note the *corrected* increments use
    the CURRENT bias estimate at i, which is the Forster first-order relinearise.
    """
    dt = pre.dt
    dR, dv, dp = pre.corrected(bg_i, ba_i)
    Ri_T = R_i.T
    rR = so3_log(dR.T @ Ri_T @ R_j)
    rv = Ri_T @ (v_j - v_i - g_world * dt) - dv
    rp = Ri_T @ (p_j - p_i - v_i * dt - 0.5 * g_world * dt * dt) - dp
    # bias random walk uses bias_j - bias_i; bias_j lives on KF j's state, but
    # the preintegration only carries i's bias, so j's bias enters only here.
    return np.concatenate([
        rR / cfg.sigma_rot,
        rv / cfg.sigma_vel,
        rp / cfg.sigma_pos,
    ])


def _bias_rw_residual(bg_i, ba_i, bg_j, ba_j, cfg) -> np.ndarray:
    return np.concatenate([
        (bg_j - bg_i) / cfg.sigma_bg_rw,
        (ba_j - ba_i) / cfg.sigma_ba_rw,
    ])


# --------------------------------------------------------------------------- #
# Pose perturbation helper.
#   Full 6-DoF (up_axis None):  d = [dp(3), dphi(3)], R <- R Exp(dphi).
#   Tilt-locked 4-DoF (up_axis): d = [dp(3), dyaw(1)], R <- Exp(up_axis*dyaw) R.
# In the tilt-locked case the yaw increment is a rotation about the WORLD
# vertical axis applied on the LEFT, so the gravity direction expressed in the
# body frame (R^T @ down_world) is unchanged -- i.e. roll/pitch stay fixed and
# only the heading (yaw) moves. Translation stays a body-frame perturbation in
# both cases.
# --------------------------------------------------------------------------- #
def _pose_perturb(R, p, d, up_axis=None):
    dp = d[:3]
    if up_axis is None:
        return R @ so3_exp(d[3:6]), p + R @ dp
    return so3_exp(up_axis * d[3]) @ R, p + R @ dp


# --------------------------------------------------------------------------- #
# Main optimiser
# --------------------------------------------------------------------------- #
def optimize_vio(
    K: np.ndarray,
    state: VioState,
    obs_cam: np.ndarray,
    obs_lm: np.ndarray,
    obs_uv: np.ndarray,
    obs_depth: np.ndarray | None,
    imu_factors: list[tuple[int, int, ImuPreintegration]],
    g_world: np.ndarray,
    cfg: VioConfig | None = None,
    anchor: int = 0,
) -> VioResult:
    """Jointly refine poses, velocities, biases and landmarks.

    obs_cam/obs_lm : (N,) int keyframe / landmark index per observation.
    obs_uv         : (N,2) measured pixels.
    obs_depth      : (N,) metric depth (m), <=0 means none, or None to disable.
    imu_factors    : consecutive-keyframe preintegration factors (i, j, pre).
    g_world        : (3,) gravity ACCELERATION vector in the world frame
                     (e.g. optical-down [0, +9.81, 0]).
    """
    cfg = cfg or VioConfig()
    st = state.copy()
    fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]),
                      float(K[0, 2]), float(K[1, 2]))
    nC = len(st.R)
    M = st.landmarks.shape[0]
    g_world = np.asarray(g_world, np.float64)
    obs_cam = np.asarray(obs_cam, np.int64)
    obs_lm = np.asarray(obs_lm, np.int64)
    obs_uv = np.asarray(obs_uv, np.float64)
    use_depth = bool(cfg.use_depth and obs_depth is not None)
    obs_depth = (np.asarray(obs_depth, np.float64) if use_depth
                 else np.zeros(obs_cam.shape[0]))

    # --- column layout -----------------------------------------------------
    # tilt-lock: pose has 4 free DoF (3 translation + 1 yaw) instead of 6, with
    # the yaw perturbation about the world-vertical (gravity) axis.
    lock_tilt = bool(cfg.lock_tilt)
    pose_dof = 4 if lock_tilt else 6
    up_axis = None
    if lock_tilt:
        gn = float(np.linalg.norm(g_world))
        up_axis = (g_world / gn) if gn > 1e-9 else np.array([0.0, 1.0, 0.0])

    pose_col = np.full(nC, -1, np.int64)
    vel_col = np.zeros(nC, np.int64)
    bg_col = np.zeros(nC, np.int64)
    ba_col = np.zeros(nC, np.int64)
    n = 0
    for i in range(nC):
        if i != anchor:
            pose_col[i] = n
            n += pose_dof
    for i in range(nC):
        vel_col[i] = n; n += 3
        bg_col[i] = n; n += 3
        ba_col[i] = n; n += 3
    lm_col = np.zeros(M, np.int64)
    for m in range(M):
        lm_col[m] = n; n += 3
    ndim = n
    eps = cfg.fd_eps
    N = obs_cam.shape[0]
    sigma_px = cfg.sigma_px
    huber_px = cfg.huber_px
    depth_huber = cfg.depth_huber

    # --- precomputed constants for the vectorised factor assembly ----------
    # The projection + depth factors used to be a scalar Python loop over every
    # observation, each finite-differencing one column at a time. Run from the
    # background VIO worker that loop held the GIL for the whole ~100ms solve and
    # starved the realtime camera loop (frame drops -> feature loss on fast
    # motion). Here the identical finite-difference math is computed with batched
    # numpy ops (which release the GIL), and the per-observation J^T J / J^T r
    # blocks are scattered with np.add.at. Mirrors the BA/PGO vectorisations.
    depth_mask = ((obs_depth > 0) if use_depth
                  else np.zeros(N, dtype=bool))
    # safe per-observation depth sigma (1.0 where no depth; that row is masked)
    sz_all = cfg.depth_sigma_coeff * np.where(depth_mask, obs_depth, 1.0) ** 2
    lm_base = lm_col[obs_lm]                 # (N,) landmark column base
    pose_base_all = pose_col[obs_cam]        # (N,) pose column base, -1 anchor
    free_obs = pose_base_all >= 0
    pose_base_free = pose_base_all[free_obs]
    lm_base_free = lm_base[free_obs]
    ar_pose = np.arange(pose_dof, dtype=np.int64)
    ar3 = np.arange(3, dtype=np.int64)
    # Constant finite-difference rotation increments: the same pose DoF
    # perturbation is applied to every observation, so build the Exp(eps) 3x3
    # rotation(s) once (identical to the scalar _pose_perturb step).
    if lock_tilt:
        dR_yaw = so3_exp(up_axis * eps)
    else:
        dR_rot = np.stack([so3_exp(np.eye(3)[a] * eps) for a in range(3)])
    fd_tiny = 1e-300

    # --- batched residual evaluators ---------------------------------------
    def _proj_uvz(Rb, pb, lmb):
        """Vectorised _project over a batch of (R, p, landmark)."""
        d = lmb - pb
        Xc = np.einsum('nki,nk->ni', Rb, d)          # R^T @ (Xw - p)
        Z = Xc[:, 2]
        Zc = np.where(Z > cfg.min_view_z, Z, cfg.min_view_z)
        u = fx * Xc[:, 0] / Zc + cx
        v = fy * Xc[:, 1] / Zc + cy
        return u, v, Z

    def _rows(Rb, pb, lmb):
        """Raw (unweighted) residual rows (N,3); depth row 0 where absent."""
        u, v, Z = _proj_uvz(Rb, pb, lmb)
        r = np.empty((N, 3))
        r[:, 0] = (u - obs_uv[:, 0]) / sigma_px
        r[:, 1] = (v - obs_uv[:, 1]) / sigma_px
        r[:, 2] = np.where(depth_mask, (Z - obs_depth) / sz_all, 0.0)
        return r

    def _gather(stt):
        R_arr = np.array(stt.R) if nC else np.zeros((0, 3, 3))
        p_arr = np.array(stt.p) if nC else np.zeros((0, 3))
        return R_arr[obs_cam], p_arr[obs_cam], stt.landmarks[obs_lm]

    def total_cost(stt) -> tuple[float, float]:
        cost = 0.0
        mean_e = 0.0
        if N:
            R_obs, p_obs, lm_obs = _gather(stt)
            r = _rows(R_obs, p_obs, lm_obs)
            e_px = np.hypot(r[:, 0], r[:, 1]) * sigma_px
            w = np.where(e_px <= huber_px, 1.0,
                         huber_px / np.maximum(e_px, fd_tiny))
            cost += 0.5 * float(np.sum(w * (r[:, 0] ** 2 + r[:, 1] ** 2)))
            az = np.abs(r[:, 2])
            thr = depth_huber / sz_all
            dcost = np.where(az <= thr, 0.5 * r[:, 2] ** 2,
                             thr * (az - 0.5 * thr))
            cost += float(np.sum(np.where(depth_mask, dcost, 0.0)))
            mean_e = float(np.mean(e_px))
        for (i, j, pre) in imu_factors:
            ri = _imu_residual(stt.R[i], stt.p[i], stt.v[i], stt.bg[i], stt.ba[i],
                               stt.R[j], stt.p[j], stt.v[j], pre, g_world, cfg)
            rb = _bias_rw_residual(stt.bg[i], stt.ba[i], stt.bg[j], stt.ba[j], cfg)
            cost += 0.5 * float(ri @ ri + rb @ rb)
        return cost, mean_e

    # --- one Gauss-Newton/LM linear system ---------------------------------
    def build_system(stt):
        H = np.zeros((ndim, ndim))
        b = np.zeros(ndim)

        # projection + depth factors (vectorised over all observations)
        if N:
            R_obs, p_obs, lm_obs = _gather(stt)
            ncol = pose_dof + 3
            r0 = _rows(R_obs, p_obs, lm_obs)
            J = np.empty((N, 3, ncol))
            col = 0
            # pose translation columns: p <- p + R @ (eps e_d) = p + eps R[:,d]
            for dax in range(3):
                pp = p_obs + eps * R_obs[:, :, dax]
                J[:, :, col] = (_rows(R_obs, pp, lm_obs) - r0) / eps
                col += 1
            # pose rotation column(s): R <- R Exp(eps e_a) (or yaw-only locked)
            if lock_tilt:
                Rp = np.einsum('ij,njk->nik', dR_yaw, R_obs)
                J[:, :, col] = (_rows(Rp, p_obs, lm_obs) - r0) / eps
                col += 1
            else:
                for a in range(3):
                    Rp = np.einsum('nij,jk->nik', R_obs, dR_rot[a])
                    J[:, :, col] = (_rows(Rp, p_obs, lm_obs) - r0) / eps
                    col += 1
            # landmark columns
            for dax in range(3):
                lp = lm_obs.copy()
                lp[:, dax] += eps
                J[:, :, col] = (_rows(R_obs, p_obs, lp) - r0) / eps
                col += 1

            # IRLS robust sqrt-weights from the current residual (held fixed
            # across this linearisation), identical to the scalar version.
            r = r0.copy()
            e_px = np.hypot(r0[:, 0], r0[:, 1]) * sigma_px
            sw = np.where(e_px <= huber_px, 1.0,
                          np.sqrt(huber_px / np.maximum(e_px, fd_tiny)))
            r[:, 0] *= sw
            r[:, 1] *= sw
            J[:, 0, :] *= sw[:, None]
            J[:, 1, :] *= sw[:, None]
            az = np.abs(r0[:, 2])
            thr = depth_huber / sz_all
            dw = np.where(az <= thr, 1.0,
                          np.sqrt(thr / np.maximum(az, 1e-12)))
            dw = np.where(depth_mask, dw, 1.0)
            r[:, 2] *= dw
            J[:, 2, :] *= dw[:, None]

            Jp = J[:, :, :pose_dof]
            Jl = J[:, :, pose_dof:]

            # landmark-landmark block + landmark rhs (every observation)
            Hll = np.einsum('nri,nrj->nij', Jl, Jl)
            bl = np.einsum('nri,nr->ni', Jl, r)
            lm_rows = (lm_base[:, None, None] + ar3[None, :, None]
                       + np.zeros((1, 1, 3), np.int64))
            lm_cols = (lm_base[:, None, None] + ar3[None, None, :]
                       + np.zeros((1, 3, 1), np.int64))
            np.add.at(H, (lm_rows.ravel(), lm_cols.ravel()), Hll.ravel())
            np.add.at(b, (lm_base[:, None] + ar3[None, :]).ravel(), bl.ravel())

            # pose blocks (only free, non-anchor poses)
            if pose_base_free.size:
                Jpf = Jp[free_obs]
                Jlf = Jl[free_obs]
                rf = r[free_obs]
                Hpp = np.einsum('nri,nrj->nij', Jpf, Jpf)
                Hpl = np.einsum('nri,nrj->nij', Jpf, Jlf)
                bp = np.einsum('nri,nr->ni', Jpf, rf)
                pp_rows = (pose_base_free[:, None, None]
                           + ar_pose[None, :, None]
                           + np.zeros((1, 1, pose_dof), np.int64))
                pp_cols = (pose_base_free[:, None, None]
                           + ar_pose[None, None, :]
                           + np.zeros((1, pose_dof, 1), np.int64))
                np.add.at(H, (pp_rows.ravel(), pp_cols.ravel()), Hpp.ravel())
                np.add.at(b, (pose_base_free[:, None]
                              + ar_pose[None, :]).ravel(), bp.ravel())
                # pose-landmark cross blocks (+ symmetric transpose)
                pl_rows = (pose_base_free[:, None, None]
                           + ar_pose[None, :, None]
                           + np.zeros((1, 1, 3), np.int64))
                pl_cols = (lm_base_free[:, None, None]
                           + ar3[None, None, :]
                           + np.zeros((1, pose_dof, 1), np.int64))
                # H[pose+a, lm+b] += Hpl[n,a,b]; the symmetric H[lm+b, pose+a]
                # gets the SAME value (Jl^T Jp)[b,a] == Hpl[n,a,b], so reuse the
                # ravel with the row/col index arrays swapped (do NOT transpose
                # the value array -- its [n,a,b] order must match the indices).
                np.add.at(H, (pl_rows.ravel(), pl_cols.ravel()), Hpl.ravel())
                np.add.at(H, (pl_cols.ravel(), pl_rows.ravel()), Hpl.ravel())

        # IMU + bias-rw factors (few factors -> scalar FD, but evaluated on the
        # 10 nav vectors of the two adjacent keyframes directly: no VioState /
        # landmark copies per perturbation column, so the per-column GIL-held
        # Python work is minimal).
        def _imu_eval(pre, Ri, pi, vi, bgi, bai, Rj, pj, vj, bgj, baj):
            ri = _imu_residual(Ri, pi, vi, bgi, bai, Rj, pj, vj,
                               pre, g_world, cfg)
            rb = _bias_rw_residual(bgi, bai, bgj, baj, cfg)
            return np.concatenate([ri, rb])

        for (i, j, pre) in imu_factors:
            base_vals = [stt.R[i], stt.p[i], stt.v[i], stt.bg[i], stt.ba[i],
                         stt.R[j], stt.p[j], stt.v[j], stt.bg[j], stt.ba[j]]
            r0i = _imu_eval(pre, *base_vals)
            rows = r0i.shape[0]
            # local variable blocks: (kind, keyframe, col base, size)
            blocks = []
            if pose_col[i] >= 0:
                blocks.append(("pose", i, pose_col[i], pose_dof))
            blocks.append(("vel", i, vel_col[i], 3))
            blocks.append(("bg", i, bg_col[i], 3))
            blocks.append(("ba", i, ba_col[i], 3))
            if pose_col[j] >= 0:
                blocks.append(("pose", j, pose_col[j], pose_dof))
            blocks.append(("vel", j, vel_col[j], 3))
            blocks.append(("bg", j, bg_col[j], 3))
            blocks.append(("ba", j, ba_col[j], 3))

            idx = []
            for _, _, base, size in blocks:
                idx.extend(range(base, base + size))
            idx = np.asarray(idx, np.int64)
            Ji = np.zeros((rows, idx.shape[0]))

            # slot of each variable in base_vals: i-side 0..4, j-side 5..9
            # (pose occupies the R,p pair at slots 0/1 and 5/6).
            base_slot = {i: 0, j: 5}
            kind_off = {"pose": 0, "vel": 2, "bg": 3, "ba": 4}
            col = 0
            for kind, vi, base, size in blocks:
                s0 = base_slot[vi]
                for d in range(size):
                    vals = list(base_vals)
                    if kind == "pose":
                        dd = np.zeros(pose_dof); dd[d] = eps
                        vals[s0], vals[s0 + 1] = _pose_perturb(
                            stt.R[vi], stt.p[vi], dd, up_axis)
                    else:
                        si = s0 + kind_off[kind]
                        v = vals[si].copy(); v[d] += eps; vals[si] = v
                    Ji[:, col] = (_imu_eval(pre, *vals) - r0i) / eps
                    col += 1

            H[np.ix_(idx, idx)] += Ji.T @ Ji
            b[idx] += Ji.T @ r0i

        return H, b

    def retract(stt, delta):
        out = stt.copy()
        for i in range(nC):
            if pose_col[i] >= 0:
                dd = delta[pose_col[i]:pose_col[i] + pose_dof]
                out.R[i], out.p[i] = _pose_perturb(stt.R[i], stt.p[i], dd,
                                                   up_axis)
            out.v[i] = stt.v[i] + delta[vel_col[i]:vel_col[i] + 3]
            out.bg[i] = stt.bg[i] + delta[bg_col[i]:bg_col[i] + 3]
            out.ba[i] = stt.ba[i] + delta[ba_col[i]:ba_col[i] + 3]
        for m in range(M):
            out.landmarks[m] = stt.landmarks[m] + delta[lm_col[m]:lm_col[m] + 3]
        return out

    # --- LM loop -----------------------------------------------------------
    cost0, _ = total_cost(st)
    cost_prev = cost0
    lam = cfg.init_lambda
    it = 0
    for it in range(cfg.max_iters):
        H, b = build_system(st)
        diag = np.clip(np.diag(H).copy(), 1e-12, None)
        solved = False
        for _ in range(12):                      # inner LM damping retries
            A = H + lam * np.diag(diag)
            try:
                delta = np.linalg.solve(A, -b)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(A, -b, rcond=None)[0]
            trial = retract(st, delta)
            cost_new, _ = total_cost(trial)
            if cost_new < cost_prev:
                st = trial
                lam = max(cfg.min_lambda, lam * 0.5)
                improved = (cost_prev - cost_new) / max(cost_prev, 1e-15)
                cost_prev = cost_new
                solved = True
                break
            lam = min(cfg.max_lambda, lam * 4.0)
        if not solved:
            break
        if improved < cfg.rel_tol:
            break

    final_cost, mean_px = total_cost(st)
    return VioResult(state=st, iters=it + 1, cost0=cost0, cost1=final_cost,
                     mean_reproj_px=mean_px)


# --------------------------------------------------------------------------- #
# Frame conversions: pipeline T_cw (world->cam) <-> VioState body->world (R,p)
# --------------------------------------------------------------------------- #
def T_cw_to_body_world(T_cw: np.ndarray):
    """World->camera 4x4 -> body->world (R_wb, p_wb), with body == camera."""
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    R_wb = R_cw.T
    p_wb = -R_cw.T @ t_cw
    return R_wb, p_wb


def body_world_to_T_cw(R_wb: np.ndarray, p_wb: np.ndarray) -> np.ndarray:
    """Inverse of :func:`T_cw_to_body_world`."""
    T = np.eye(4)
    R_cw = R_wb.T
    T[:3, :3] = R_cw
    T[:3, 3] = -R_cw @ p_wb
    return T


def _imu_segment(ts_ns: np.ndarray, gyro: np.ndarray, accel: np.ndarray,
                 t0: int, t1: int):
    """Clamped IMU slice for the open interval ``(t0, t1]`` with the endpoints
    linearly interpolated, so the preintegrated ``dt`` matches the real frame
    interval exactly (no sub-sample truncation at the keyframe boundaries).

    Returns ``(ts_seg, gyro_seg, accel_seg)`` or ``None`` if the interval has no
    usable samples.
    """
    t0 = int(t0); t1 = int(t1)
    if t1 <= t0 or ts_ns.size < 2:
        return None

    def interp(t):
        j = int(np.searchsorted(ts_ns, t))
        if j <= 0:
            return gyro[0], accel[0]
        if j >= ts_ns.size:
            return gyro[-1], accel[-1]
        ta, tb = int(ts_ns[j - 1]), int(ts_ns[j])
        if tb == ta:
            return gyro[j], accel[j]
        a = (t - ta) / (tb - ta)
        return (gyro[j - 1] * (1 - a) + gyro[j] * a,
                accel[j - 1] * (1 - a) + accel[j] * a)

    # interior samples strictly inside (t0, t1)
    lo = int(np.searchsorted(ts_ns, t0, side="right"))
    hi = int(np.searchsorted(ts_ns, t1, side="left"))
    g0, a0 = interp(t0)
    g1, a1 = interp(t1)
    ts_list = [t0]
    g_list = [g0]
    a_list = [a0]
    for k in range(lo, hi):
        ts_list.append(int(ts_ns[k]))
        g_list.append(gyro[k])
        a_list.append(accel[k])
    ts_list.append(t1)
    g_list.append(g1)
    a_list.append(a1)
    if len(ts_list) < 2:
        return None
    return (np.asarray(ts_list, np.int64),
            np.asarray(g_list, np.float64),
            np.asarray(a_list, np.float64))


# --------------------------------------------------------------------------- #
# Windowed tight-coupled VIO map (Basalt-style sliding window)
# --------------------------------------------------------------------------- #
@dataclass
class WindowedVIOConfig:
    kf_every: int = 4            # insert a keyframe every N frames
    window: int = 8             # keyframes kept in the VIO window
    min_depth_m: float = 0.2
    max_depth_m: float = 8.0
    min_ba_views: int = 2       # landmark needs >= this many KF views
    # gravity ACCELERATION vector in the optical world frame: "down" is +y, and
    # at rest the accelerometer reads +g upward, so g_world points +y.
    g_world: tuple = (0.0, 9.81, 0.0)
    use_imu: bool = True         # set False to A/B the IMU factors (diagnostic)
    # The windowed VIO solves each pose with roll/pitch LOCKED (lock_tilt): the
    # accelerometer owns tilt absolutely (gravity is an absolute reference), so
    # the joint solve only refines what the IMU cannot observe -- yaw + position
    # + velocity + scale. Locking tilt removes the 2 DoF where gravity used to
    # leak into a horizontal translation (corridor scale drift) and where the
    # reprojection factors could fight the IMU. The IMU vel/pos sigmas are TIGHT
    # so the accelerometer (not the blur-biased RGB-D depth) anchors metric scale.
    vio: VioConfig = field(default_factory=lambda: VioConfig(
        max_iters=12, sigma_rot=0.02, sigma_vel=0.03, sigma_pos=0.03,
        lock_tilt=True))


class WindowedVIOMap:
    """Sliding-window tight-coupled VIO map (visual + IMU), tracker-agnostic.

    Mirrors :class:`oakd.vio.windowed.WindowedBAMap` but feeds the raw visual
    measurements **and** IMU preintegration factors into the joint optimiser
    :func:`optimize_vio`, solving for each keyframe's pose, velocity and
    gyro/accel bias together with the landmarks. The accelerometer ties the
    keyframes through velocity/position, so an in-place rotation (true linear
    acceleration ~0) can no longer be explained away as a translation by slipped
    visual tracks -- the phantom yaw-drift that the vision-only / loosely-coupled
    paths leave behind.

    The caller supplies the full IMU stream **already rotated into the camera
    optical frame** (``gyro_cam``/``accel_cam``) at construction. Each keyframe
    carries its device-clock timestamp; the map preintegrates the IMU between
    consecutive keyframe timestamps internally.
    """

    def __init__(self, K: np.ndarray, ts_ns: np.ndarray | None = None,
                 gyro_cam: np.ndarray | None = None,
                 accel_cam: np.ndarray | None = None,
                 bg0: np.ndarray | None = None, ba0: np.ndarray | None = None,
                 cfg: WindowedVIOConfig | None = None):
        self.K = np.asarray(K, dtype=np.float64)
        self.cfg = cfg or WindowedVIOConfig()
        # Offline: the full IMU stream is supplied up front and sliced between
        # keyframe timestamps. Live: no stream exists yet, so the caller hands
        # each keyframe's raw IMU segment to ``add_keyframe(imu_seg=...)`` and we
        # leave the stored stream empty.
        if ts_ns is None or len(ts_ns) == 0:
            self.imu_ts = np.zeros(0, np.int64)
            self.imu_gyro = np.zeros((0, 3), np.float64)
            self.imu_accel = np.zeros((0, 3), np.float64)
        else:
            order = np.argsort(ts_ns)
            self.imu_ts = np.asarray(ts_ns, np.int64)[order]
            self.imu_gyro = np.asarray(gyro_cam, np.float64)[order]
            self.imu_accel = np.asarray(accel_cam, np.float64)[order]
        self.bg0 = (np.zeros(3) if bg0 is None
                    else np.asarray(bg0, np.float64).copy())
        self.ba0 = (np.zeros(3) if ba0 is None
                    else np.asarray(ba0, np.float64).copy())
        self.g_world = np.asarray(self.cfg.g_world, np.float64)
        self.landmarks: dict[int, np.ndarray] = {}
        self.keyframes: list[dict] = []
        self.last_info: dict = {}

    def _backproject_world(self, T_cw: np.ndarray, u: float, v: float,
                           z: float) -> np.ndarray:
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        Xc = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
        R, t = T_cw[:3, :3], T_cw[:3, 3]
        return R.T @ (Xc - t)

    def add_keyframe(self, T_cw: np.ndarray, ids: np.ndarray,
                     pts: np.ndarray, depth_m: np.ndarray, ts_ns: int,
                     imu_seg: tuple[np.ndarray, np.ndarray, np.ndarray] | None
                     = None) -> None:
        """Register a keyframe (pose + track snapshot + depth + timestamp).

        ``imu_seg`` (live path) is the raw ``(ts_ns, gyro_cam, accel_cam)`` block
        of IMU samples spanning the interval since the previous keyframe, already
        rotated into the camera frame. When ``None`` (offline path) the segment
        is sliced from the stored full stream by timestamp.
        """
        h, w = depth_m.shape
        obs: dict[int, np.ndarray] = {}
        for tid, px in zip(ids, pts):
            tid = int(tid)
            u, v = float(px[0]), float(px[1])
            pu, pv = int(round(u)), int(round(v))
            if not (0 <= pu < w and 0 <= pv < h):
                continue
            z = float(depth_m[pv, pu])
            z_ok = self.cfg.min_depth_m <= z <= self.cfg.max_depth_m
            if tid not in self.landmarks:
                if not z_ok:
                    continue
                self.landmarks[tid] = self._backproject_world(T_cw, u, v, z)
            obs[tid] = np.array([u, v, z if z_ok else 0.0])

        T_cw = np.asarray(T_cw, float).copy()
        ts_ns = int(ts_ns)
        if not self.keyframes:
            kf = {"T_cw": T_cw, "obs": obs, "ts_ns": ts_ns, "pre": None,
                  "v": np.zeros(3), "bg": self.bg0.copy(), "ba": self.ba0.copy()}
        else:
            prev = self.keyframes[-1]
            bg_i, ba_i = prev["bg"], prev["ba"]
            if imu_seg is not None:
                ts_seg = np.asarray(imu_seg[0], np.int64)
                seg = None if ts_seg.size < 2 else (
                    ts_seg, np.asarray(imu_seg[1], np.float64),
                    np.asarray(imu_seg[2], np.float64))
            else:
                seg = _imu_segment(self.imu_ts, self.imu_gyro, self.imu_accel,
                                   prev["ts_ns"], ts_ns)
            pre = None
            v_j = prev["v"].copy()
            if seg is not None:
                pre = preintegrate_imu(seg[0], seg[1], seg[2], bg_i, ba_i)
                R_i, _ = T_cw_to_body_world(prev["T_cw"])
                dR, dv, dp = pre.corrected(bg_i, ba_i)
                # predict velocity from the IMU increment (position/rotation are
                # seeded from the visual pose instead, which is metric already).
                v_j = prev["v"] + self.g_world * pre.dt + R_i @ dv
            kf = {"T_cw": T_cw, "obs": obs, "ts_ns": ts_ns, "pre": pre,
                  "v": v_j, "bg": bg_i.copy(), "ba": ba_i.copy()}
        self.keyframes.append(kf)
        self._marginalize()

    def _marginalize(self) -> None:
        while len(self.keyframes) > self.cfg.window:
            self.keyframes.pop(0)
        live = set()
        for kf in self.keyframes:
            live.update(kf["obs"].keys())
        for tid in list(self.landmarks.keys()):
            if tid not in live:
                del self.landmarks[tid]

    def run_ba(self) -> np.ndarray | None:
        """Optimise the window; return the refined latest ``T_cw`` (or None)."""
        kfs = self.keyframes
        if len(kfs) < 2:
            return None
        cnt = Counter()
        for kf in kfs:
            for tid in kf["obs"]:
                if tid in self.landmarks:
                    cnt[tid] += 1
        ba_tids = [t for t, c in cnt.items() if c >= self.cfg.min_ba_views]
        if len(ba_tids) < 6:
            return None
        lm_index = {t: j for j, t in enumerate(ba_tids)}

        st = VioState(
            R=[], p=[], v=[], bg=[], ba=[],
            landmarks=np.array([self.landmarks[t] for t in ba_tids]),
        )
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

        # IMU factors between consecutive in-window keyframes. kf[ci]["pre"]
        # links kf[ci-1] -> kf[ci]; the window's first keyframe's own "pre"
        # (which linked to a now-dropped keyframe) is simply never referenced.
        imu_factors = []
        for ci in range(1, len(kfs)):
            pre = kfs[ci]["pre"]
            if pre is not None:
                imu_factors.append((ci - 1, ci, pre))
        if not self.cfg.use_imu:
            imu_factors = []

        res = optimize_vio(
            self.K, st,
            np.array(obs_cam), np.array(obs_lm), np.array(obs_uv),
            np.array(obs_depth), imu_factors, self.g_world,
            cfg=self.cfg.vio, anchor=0,
        )
        out = res.state
        for ci, kf in enumerate(kfs):
            kf["T_cw"] = body_world_to_T_cw(out.R[ci], out.p[ci])
            kf["v"] = out.v[ci].copy()
            kf["bg"] = out.bg[ci].copy()
            kf["ba"] = out.ba[ci].copy()
        for t, j in lm_index.items():
            self.landmarks[t] = out.landmarks[j]

        self.last_info = {
            "vio_kfs": len(kfs), "vio_lms": len(ba_tids),
            "vio_obs": len(obs_cam), "vio_imu": len(imu_factors),
            "vio_iters": res.iters, "vio_reproj_px": res.mean_reproj_px,
        }
        return kfs[-1]["T_cw"].copy()


class WindowedVIORGBDOdometry:
    """Frame-to-frame tracking with a tight-coupled sliding-window VIO backend.

    Drop-in sibling of :class:`oakd.vio.windowed.WindowedRGBDOdometry`: the same
    KLT/PnP frontend produces a smooth per-frame pose, but every keyframe is
    refined by :class:`WindowedVIOMap` (visual + IMU joint optimisation) instead
    of vision-only bundle adjustment. The caller passes the full IMU stream in
    the camera optical frame at construction; :meth:`process` takes the frame
    timestamp so the map can preintegrate the IMU between keyframes.
    """

    def __init__(self, K: np.ndarray, ts_ns: np.ndarray,
                 gyro_cam: np.ndarray, accel_cam: np.ndarray,
                 bg0: np.ndarray | None = None, ba0: np.ndarray | None = None,
                 cfg: WindowedVIOConfig | None = None,
                 frontend: KLTFrontend | None = None,
                 odom_cfg: OdometryConfig | None = None):
        self.K = np.asarray(K, dtype=np.float64)
        self.cfg = cfg or WindowedVIOConfig()
        fe = frontend or KLTFrontend(FrontendConfig())
        self.vo = RGBDVisualOdometry(self.K, odom_cfg or OdometryConfig(),
                                     frontend=fe)
        self.frontend = self.vo.frontend
        self.map = WindowedVIOMap(self.K, ts_ns, gyro_cam, accel_cam,
                                  bg0, ba0, self.cfg)
        self._frames_since_kf = 0
        self._frame_idx = -1
        self.pose = np.eye(4)
        self.last_info: dict = {}

    def align_to_gravity(self, accel_cam: np.ndarray) -> None:
        self.vo.align_to_gravity(accel_cam)
        self.pose = self.vo.pose.copy()

    @property
    def landmarks(self) -> dict[int, np.ndarray]:
        return self.map.landmarks

    @property
    def keyframes(self) -> list[dict]:
        return self.map.keyframes

    def process(self, gray: np.ndarray, depth_m: np.ndarray, ts_ns: int,
                R_prior: np.ndarray | None = None) -> np.ndarray:
        """Advance one frame; return the current 4x4 world pose (camera->world)."""
        self._frame_idx += 1
        self.pose = self.vo.process(gray, depth_m, R_prior=R_prior).copy()
        self.last_info = dict(self.vo.last_info)
        self._frames_since_kf += 1

        is_kf = (not self.keyframes) or (self._frames_since_kf >= self.cfg.kf_every)
        if is_kf:
            self._frames_since_kf = 0
            state = self.frontend.tracks
            self.map.add_keyframe(np.linalg.inv(self.pose),
                                  state.ids, state.points, depth_m, ts_ns)
            post = self.map.run_ba()
            if post is not None:
                self.pose = np.linalg.inv(post)
                self.vo.pose = self.pose.copy()
                self.last_info.update(self.map.last_info)
            self.last_info["is_kf"] = True
        else:
            self.last_info["is_kf"] = False
        return self.pose
