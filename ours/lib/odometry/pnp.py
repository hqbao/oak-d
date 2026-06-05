"""Library-free PnP (3D-2D) with RANSAC + Gauss-Newton refinement.

Drop-in replacement for the subset of ``cv2.solvePnPRansac`` the from-scratch
VIO needs: given 3D points in the previous camera frame (``obj``), their pixel
observations in the current frame (``img``) and the intrinsics ``K``, recover
the rigid transform ``(R, t)`` (current <- previous) that reprojects the points
onto the pixels, rejecting outliers.

Pipeline (pure NumPy):
  1. RANSAC over minimal 6-point DLT hypotheses (plus an optional hypothesis
     seeded from a rotation prior, e.g. the gyro), scored by reprojection
     inliers within ``reproj_px``.
  2. A robust (Huber) Gauss-Newton refinement on the inlier set, optimising the
     full 6-DoF pose (rotation on the left, translation in the camera frame).
  3. One inlier re-selection + refine pass for stability.

Returned contract mirrors what ``odometry.py`` consumed from cv2:
``(ok: bool, R: 3x3, t: (3,), inliers: (M,1) int32)``.
"""
from __future__ import annotations

import numpy as np

from ..imu.imu import so3_exp


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def _reproj_err(R: np.ndarray, t: np.ndarray, obj: np.ndarray,
                img: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Per-point reprojection error (px). Points behind the camera -> +inf."""
    P = obj @ R.T + t                       # (N,3) in current camera frame
    z = P[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * P[:, 0] / z + cx
        v = fy * P[:, 1] / z + cy
    du = u - img[:, 0]
    dv = v - img[:, 1]
    err = np.sqrt(du * du + dv * dv)
    err[z <= 1e-6] = np.inf
    return err


def _dlt(obj: np.ndarray, img: np.ndarray, K: np.ndarray):
    """Direct Linear Transform pose from >=6 correspondences. Returns R,t or None.

    Solves for the 3x4 projection in normalised coordinates, then projects the
    left 3x3 block onto SO(3) via SVD to recover a valid rotation + translation.
    """
    n = obj.shape[0]
    if n < 6:
        return None
    Kinv = np.linalg.inv(K)
    # normalised image coordinates (x = Kinv [u v 1])
    uv1 = np.hstack([img, np.ones((n, 1))])
    xy = (uv1 @ Kinv.T)[:, :2]
    A = np.zeros((2 * n, 12))
    X = np.hstack([obj, np.ones((n, 1))])   # homogeneous world points
    A[0::2, 0:4] = X
    A[0::2, 8:12] = -xy[:, 0:1] * X
    A[1::2, 4:8] = X
    A[1::2, 8:12] = -xy[:, 1:2] * X
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    P = Vt[-1].reshape(3, 4)
    R0 = P[:, :3]
    # scale so that R0 projects onto a proper rotation; sign from det
    U, S, Vt2 = np.linalg.svd(R0)
    if S[2] < 1e-9:
        return None
    R = U @ Vt2
    if np.linalg.det(R) < 0:
        R = -R
        P = -P
    scale = 1.0 / ((S[0] + S[1] + S[2]) / 3.0)
    t = P[:, 3] * scale
    # enforce points in front of camera (cheirality) — flip if majority behind
    zc = (obj @ R.T + t)[:, 2]
    if np.mean(zc > 0) < 0.5:
        return None
    return R, t


def _build_jac(R: np.ndarray, t: np.ndarray, obj: np.ndarray,
               img: np.ndarray, K: np.ndarray):
    """Reprojection residual + 6-DoF Jacobian (rotation on the left, then t).

    Returns ``(J, r, e, ok)`` where ``J`` is (2m,6), ``r`` is (2m,) stacked
    [du,dv] residuals, ``e`` is the per-point error magnitude (m,), and ``ok``
    is the boolean mask of points in front of the camera (length N).
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = obj @ R.T + t
    z = P[:, 2]
    good = z > 1e-6
    if good.sum() < 3:
        return None, None, None, good
    Pg = P[good]
    zg = Pg[:, 2]
    u = fx * Pg[:, 0] / zg + cx
    v = fy * Pg[:, 1] / zg + cy
    ru = u - img[good, 0]
    rv = v - img[good, 1]
    inv_z = 1.0 / zg
    j00 = fx * inv_z
    j02 = -fx * Pg[:, 0] * inv_z * inv_z
    j11 = fy * inv_z
    j12 = -fy * Pg[:, 1] * inv_z * inv_z
    m = Pg.shape[0]
    J = np.zeros((2 * m, 6))
    px, py, pz = Pg[:, 0], Pg[:, 1], Pg[:, 2]
    # -[Pc]_x = [[0, pz, -py],[-pz,0,px],[py,-px,0]]
    J[0::2, 0] = j02 * py
    J[0::2, 1] = j00 * pz + j02 * (-px)
    J[0::2, 2] = j00 * (-py)
    J[0::2, 3] = j00
    J[0::2, 5] = j02
    J[1::2, 0] = j11 * (-pz) + j12 * py
    J[1::2, 1] = j12 * (-px)
    J[1::2, 2] = j11 * px
    J[1::2, 4] = j11
    J[1::2, 5] = j12
    r = np.empty(2 * m)
    r[0::2] = ru
    r[1::2] = rv
    e = np.sqrt(ru * ru + rv * rv)
    return J, r, e, good


def _refine_lm(R: np.ndarray, t: np.ndarray, obj: np.ndarray,
               img: np.ndarray, K: np.ndarray, huber_px: float = np.inf,
               iters: int = 20):
    """Levenberg-Marquardt refinement of (R,t) minimising reprojection error.

    Mirrors ``cv2``'s ``refineLM`` final polish: damped Gauss-Newton with an
    adaptive ``lambda`` (accept the step + shrink lambda when the cost drops,
    reject + grow lambda otherwise). ``huber_px = inf`` gives a plain
    least-squares fit on the (already RANSAC-selected) inliers, exactly like
    cv2; a finite ``huber_px`` re-weights large residuals during intermediate
    fits where outliers may still be present.
    """
    R = R.copy()
    t = t.astype(np.float64).copy()
    lam = 1e-3

    def cost(Rc, tc):
        J, r, e, good = _build_jac(Rc, tc, obj, img, K)
        if J is None:
            return None, None, None, np.inf
        if np.isfinite(huber_px):
            w = np.ones_like(e)
            big = e > huber_px
            w[big] = huber_px / e[big]
            c = float(np.sum(np.minimum(e, huber_px) ** 2))
        else:
            w = None
            c = float(np.sum(e * e))
        return (J, r, w), good, e, c

    packed, _good, _e, c_cur = cost(R, t)
    if packed is None:
        return R, t
    for _ in range(iters):
        J, r, w = packed
        if w is not None:
            wpair = np.repeat(w, 2)
            Jw = J * wpair[:, None]
        else:
            Jw = J
        H = Jw.T @ J
        g = Jw.T @ r
        diag = np.diag(np.diag(H)).copy()
        improved = False
        for _try in range(8):
            try:
                delta = np.linalg.solve(H + lam * diag + 1e-12 * np.eye(6), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            R_new = so3_exp(delta[:3]) @ R
            t_new = t + delta[3:]
            packed_new, _gn, _en, c_new = cost(R_new, t_new)
            if packed_new is not None and c_new < c_cur:
                R, t = R_new, t_new
                packed, c_cur = packed_new, c_new
                lam = max(lam * 0.5, 1e-9)
                improved = True
                step = float(np.linalg.norm(delta))
                break
            lam *= 4.0
        if not improved or step < 1e-10:
            break
    return R, t


def solve_pnp_ransac(obj: np.ndarray, img: np.ndarray, K: np.ndarray,
                     R_init: np.ndarray | None = None,
                     t_init: np.ndarray | None = None,
                     reproj_px: float = 2.0, iters: int = 200,
                     conf: float = 0.999, min_points: int = 8,
                     rng: np.random.Generator | None = None):
    """RANSAC PnP. Returns ``(ok, R, t, inliers)`` (inliers as (M,1) int32).

    ``R_init``/``t_init`` (optional, e.g. a gyro rotation prior) seed an extra
    hypothesis so clean frames lock on immediately; RANSAC still runs so a bad
    prior cannot trap the solution.
    """
    obj = np.asarray(obj, np.float64)
    img = np.asarray(img, np.float64)
    n = obj.shape[0]
    if n < min_points:
        return False, np.eye(3), np.zeros(3), None
    if rng is None:
        rng = np.random.default_rng(0xC0FFEE)

    best_inl = None
    best_cnt = -1

    def consider(R, t):
        nonlocal best_inl, best_cnt
        if R is None:
            return
        err = _reproj_err(R, t, obj, img, K)
        inl = err < reproj_px
        c = int(inl.sum())
        if c > best_cnt:
            best_cnt = c
            best_inl = inl

    # seed hypothesis from the prior rotation + its linear-LS translation.
    R_seed_ref = None
    if R_init is not None:
        t_seed = _translation_given_rotation(obj, img, R_init, K)
        consider(R_init, t_seed)

    sample = 6
    max_iter = int(iters)
    it = 0
    while it < max_iter:
        it += 1
        idx = rng.choice(n, size=sample, replace=False)
        sol = _dlt(obj[idx], img[idx], K)
        if sol is not None:
            consider(sol[0], sol[1])
        # adaptive stop: if current best inlier ratio is high, fewer iters needed
        if best_cnt > 0:
            w = best_cnt / n
            denom = 1.0 - w ** sample
            if denom > 1e-12:
                need = np.log(max(1e-12, 1.0 - conf)) / np.log(max(1e-12, denom))
                if it >= need:
                    break

    # Rescue: on marginal frames (few correspondences) a raw 6-point DLT sample
    # is too noisy to bootstrap, so plain RANSAC under-counts inliers and would
    # fail -- exactly where cv2's ``useExtrinsicGuess`` LM still locks on.
    # Only when the prior is available AND plain RANSAC came up short do we add a
    # robust (Huber) LM polish of the prior over ALL points and re-score. This
    # is a pure fallback: healthy frames keep the plain result unchanged (so the
    # sessions that already matched/beat cv2 are untouched), and the low-inlier
    # frames cv2 solved (measured: push_fwdback seq 76/196/313/345) are
    # recovered instead of freezing real motion.
    if (R_init is not None
            and (best_inl is None or best_cnt < 2 * min_points)):
        t_seed = _translation_given_rotation(obj, img, R_init, K)
        R_seed_ref, t_seed_ref = _refine_lm(
            R_init, t_seed, obj, img, K, huber_px=2.0 * reproj_px, iters=10)
        err = _reproj_err(R_seed_ref, t_seed_ref, obj, img, K)
        inl = err < reproj_px
        if int(inl.sum()) > best_cnt:
            best_cnt = int(inl.sum())
            best_inl = inl

    if best_inl is None or best_cnt < min_points:
        return False, np.eye(3), np.zeros(3), None

    # refine on inliers, then re-select once for stability
    R = R_seed_ref.copy() if R_seed_ref is not None else (
        R_init.copy() if R_init is not None else None)
    if R is None:
        sol = _dlt(obj[best_inl], img[best_inl], K)
        if sol is None:
            return False, np.eye(3), np.zeros(3), None
        R, t = sol
    else:
        t = _translation_given_rotation(obj[best_inl], img[best_inl], R, K)
    # Plain least-squares LM polish on the inlier set (cv2's refineLM): the
    # inliers are already within ``reproj_px``, so a robust kernel would only
    # bias the fit by down-weighting the larger-but-valid residuals (depth
    # noise on fast pushes) -- exactly the regression that made the Huber
    # Gauss-Newton lose to cv2. One re-selection + re-fit for stability
    # (iterating further destabilised the fast-push fit; measured worse).
    R, t = _refine_lm(R, t, obj[best_inl], img[best_inl], K, huber_px=np.inf)
    err = _reproj_err(R, t, obj, img, K)
    inl = err < reproj_px
    if int(inl.sum()) >= best_cnt:
        best_inl = inl
        R, t = _refine_lm(R, t, obj[best_inl], img[best_inl], K,
                          huber_px=np.inf)

    if int(best_inl.sum()) < min_points:
        return False, np.eye(3), np.zeros(3), None
    inliers = np.nonzero(best_inl)[0].reshape(-1, 1).astype(np.int32)
    return True, R, t, inliers


def _translation_given_rotation(obj: np.ndarray, img: np.ndarray,
                                R: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Least-squares camera translation with the rotation held fixed.

    Mirror of the helper in ``odometry.py``: each correspondence constrains the
    rotated point to lie on the pixel's viewing ray; stacking gives ``A t = b``.
    """
    Kinv = np.linalg.inv(K)
    P = obj @ R.T
    ones = np.ones((img.shape[0], 1))
    rays = (np.hstack([img, ones]) @ Kinv.T)
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    A = np.zeros((2 * img.shape[0], 3))
    b = np.zeros(2 * img.shape[0])
    for i, (d, p) in enumerate(zip(rays, P)):
        # rows of [d]_x: two independent constraints d x (P + t) = 0
        dx = np.array([[0.0, -d[2], d[1]],
                       [d[2], 0.0, -d[0]],
                       [-d[1], d[0], 0.0]])
        A[2 * i] = dx[0]
        A[2 * i + 1] = dx[1]
        b[2 * i] = -(dx[0] @ p)
        b[2 * i + 1] = -(dx[1] @ p)
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return t
