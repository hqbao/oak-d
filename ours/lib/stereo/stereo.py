"""Sparse rectified-stereo depth: our own block matcher (library-free, portable).

This replaces the OAK-D ``StereoDepth`` node (the SGBM depth blob baked into the
chip) for the from-scratch ``ours`` VIO. The whole point is portability: when the
pipeline is ported to another platform there is no DepthAI ``StereoDepth`` to
lean on, so we compute metric depth ourselves from just the two **rectified**
grayscale frames + the stereo calibration.

Why *sparse* (and not a full dense disparity map):

    The ``ours`` VIO core only ever samples depth at a few hundred KLT feature
    pixels per frame (``odometry.py`` back-projects each tracked point). A dense
    640x400 x ~96-disparity SGBM is therefore mostly wasted work for VIO. We
    instead match only the query pixels along their (rectified) epipolar row.
    That is O(N_features) instead of O(W*H), so it stays real-time in pure
    Python and trivially portable.

Geometry (rectified pair, depth aligned to the **left** camera):

    A 3D point projects to the left image at ``(u, v)`` and to the right image at
    ``(u - d, v)`` with disparity ``d >= 0`` (right camera is the +x baseline
    neighbour, rows are aligned after rectification). Metric depth is then::

        Z = fx * baseline / d

    ``fx`` is the left rectified focal length (``K[0, 0]``) and ``baseline`` is
    ``|T_left_right translation|`` in metres (``StereoCalib.baseline_m``).

Matching is windowed **ZNCC** (zero-mean normalised cross-correlation), which is
robust to the small gain/offset difference between the two physical cameras (a
plain SAD would bias toward the brighter image). We reject a match when the peak
correlation is weak (textureless / occluded) or not unique (repetitive texture),
and parabola-fit the correlation peak for sub-pixel disparity. The chip depth
(``*_D.raw16``) is kept only as the validation oracle in ``stereo_selftest.py`` --
exactly how ``klt_selftest.py`` keeps cv2 as the optical-flow oracle.

Numba is an *optional* accelerator for the per-point search (same pattern as
``klt_numba.py``): we own the whole algorithm; Numba only compiles our own
explicit loops. Without numba the pure-NumPy path runs the identical math.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit, prange  # type: ignore
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only when numba is absent
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return wrap

    def prange(*args):  # type: ignore
        return range(*args)


@dataclass
class StereoConfig:
    """Tuning for the sparse block matcher (all measured, see stereo_selftest)."""

    patch_radius: int = 4        # window is (2r+1)^2; 9x9 at r=4
    min_disparity: int = 1       # px; Z_max ~ fx*B/min_disp
    max_disparity: int = 96      # px; Z_min ~ fx*B/max_disp
    v_radius: int = 1            # search +/- this many rows (residual rectif.)
    min_ncc: float = 0.55        # reject weak peaks (textureless / occluded)
    uniqueness: float = 0.03     # best - second_peak must exceed this
    subpixel: bool = True        # parabola fit on the correlation peak
    min_texture: float = 1e-3    # reject flat left patches (variance floor)
    lr_consistency: bool = True  # left->right->left round-trip agreement check
    lr_max_diff: float = 1.5     # px; reject if the round-trip disparity drifts


@njit(cache=True, fastmath=True)
def _match_points(left, right, us, vs, rad, dmin, dmax, vrad,
                  min_ncc, uniq, min_tex, subpixel, sign):
    """Per-point ZNCC disparity search along the epipolar row.

    Matches ``left`` patches into ``right``; the right column searched is
    ``u + sign * d`` (``sign = -1`` for the forward left->right pass, ``+1`` for
    the reverse right->left consistency pass). Returns a float disparity per
    query point; ``NaN`` where rejected. This is our own code -- numba only
    compiles the scalar loops to machine code.
    """
    H, W = left.shape
    N = us.shape[0]
    ndisp = dmax - dmin + 1
    out = np.full(N, np.nan)
    costs = np.empty(ndisp, dtype=np.float64)
    npx = (2 * rad + 1) * (2 * rad + 1)

    for n in range(N):
        u = us[n]
        v = vs[n]
        # The left window (plus the vertical search band) must be in-bounds.
        if (u - rad < 0 or u + rad >= W
                or v - rad - vrad < 0 or v + rad + vrad >= H):
            continue

        # Left patch mean + variance (computed once).
        lmean = 0.0
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                lmean += left[v + dy, u + dx]
        lmean /= npx
        lvar = 0.0
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                t = left[v + dy, u + dx] - lmean
                lvar += t * t
        if lvar < min_tex * npx:
            continue

        for k in range(ndisp):
            costs[k] = -2.0

        for k in range(ndisp):
            d = dmin + k
            ur = u + sign * d
            if ur - rad < 0 or ur + rad >= W:
                continue
            best_v = -2.0
            for dvv in range(-vrad, vrad + 1):
                vr = v + dvv
                rmean = 0.0
                for dy in range(-rad, rad + 1):
                    for dx in range(-rad, rad + 1):
                        rmean += right[vr + dy, ur + dx]
                rmean /= npx
                cross = 0.0
                rvar = 0.0
                for dy in range(-rad, rad + 1):
                    for dx in range(-rad, rad + 1):
                        lt = left[v + dy, u + dx] - lmean
                        rt = right[vr + dy, ur + dx] - rmean
                        cross += lt * rt
                        rvar += rt * rt
                if rvar < 1e-9:
                    continue
                ncc = cross / np.sqrt(lvar * rvar)
                if ncc > best_v:
                    best_v = ncc
            costs[k] = best_v

        # Peak + uniqueness (second-best peak excluding the +/-1 neighbours).
        best_k = -1
        best_c = -2.0
        for k in range(ndisp):
            if costs[k] > best_c:
                best_c = costs[k]
                best_k = k
        if best_k < 0 or best_c < min_ncc:
            continue
        second = -2.0
        for k in range(ndisp):
            if k < best_k - 1 or k > best_k + 1:
                if costs[k] > second:
                    second = costs[k]
        if best_c - second < uniq:
            continue

        d_best = float(dmin + best_k)
        if subpixel and 0 < best_k < ndisp - 1:
            cm = costs[best_k - 1]
            c0 = costs[best_k]
            cp = costs[best_k + 1]
            denom = (cm - 2.0 * c0 + cp)
            if denom < -1e-9:  # concave peak only
                d_best += 0.5 * (cm - cp) / denom
        out[n] = d_best

    return out


class StereoMatcher:
    """Sparse rectified-stereo depth at query pixels (depth aligned to left)."""

    def __init__(self, K: np.ndarray, baseline_m: float,
                 cfg: StereoConfig | None = None):
        self.K = np.asarray(K, dtype=np.float64)
        self.fx = float(self.K[0, 0])
        self.baseline_m = float(baseline_m)
        self.cfg = cfg or StereoConfig()

    # -- core -------------------------------------------------------------- #
    def disparity_at(self, left: np.ndarray, right: np.ndarray,
                     pts: np.ndarray) -> np.ndarray:
        """Sub-pixel disparity per query point; ``NaN`` where unmatched.

        ``pts`` is ``(N, 2)`` float pixel coordinates ``(u, v)`` in the left
        rectified image.
        """
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        us = np.round(pts[:, 0]).astype(np.int64)
        vs = np.round(pts[:, 1]).astype(np.int64)
        cfg = self.cfg
        L = left.astype(np.float64)
        R = right.astype(np.float64)
        # Forward pass: left feature -> right column (u - d).
        disp = _match_points(
            L, R, us, vs, cfg.patch_radius, cfg.min_disparity,
            cfg.max_disparity, cfg.v_radius, cfg.min_ncc, cfg.uniqueness,
            cfg.min_texture, cfg.subpixel, -1)
        if not cfg.lr_consistency:
            return disp
        # Reverse pass: take the matched right pixel and match it back into the
        # left image (left column = ur + d'). A correct match round-trips to the
        # same disparity; false matches in repetitive texture do not, so we drop
        # any point whose forward/return disparity disagree by > lr_max_diff.
        good = np.isfinite(disp)
        ur = np.round(us - disp).astype(np.int64)
        ur[~good] = 0
        back = _match_points(
            R, L, ur, vs, cfg.patch_radius, cfg.min_disparity,
            cfg.max_disparity, cfg.v_radius, cfg.min_ncc, cfg.uniqueness,
            cfg.min_texture, cfg.subpixel, +1)
        bad = good & (~np.isfinite(back)
                      | (np.abs(disp - back) > cfg.lr_max_diff))
        disp[bad] = np.nan
        return disp

    def depth_at(self, left: np.ndarray, right: np.ndarray,
                 pts: np.ndarray) -> np.ndarray:
        """Metric depth (m) per query point; ``0.0`` where unmatched/invalid."""
        disp = self.disparity_at(left, right, pts)
        depth = np.zeros_like(disp)
        good = np.isfinite(disp) & (disp > 1e-6)
        depth[good] = self.fx * self.baseline_m / disp[good]
        return depth.astype(np.float32)

    def sparse_depth_map(self, left: np.ndarray, right: np.ndarray,
                         pts: np.ndarray) -> np.ndarray:
        """An ``(H, W)`` depth map filled only at the rounded query pixels.

        Drop-in for the dense chip depth in the ``ours`` VIO: the odometry only
        reads ``depth_m[round(v), round(u)]`` at tracked points, so filling those
        cells is sufficient and keeps the existing sampling code unchanged.
        """
        h, w = left.shape[:2]
        out = np.zeros((h, w), dtype=np.float32)
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] == 0:
            return out
        depth = self.depth_at(left, right, pts)
        us = np.round(pts[:, 0]).astype(np.int64)
        vs = np.round(pts[:, 1]).astype(np.int64)
        ok = (depth > 0) & (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
        out[vs[ok], us[ok]] = depth[ok]
        return out


# ===========================================================================
# Dense semi-global matching (SGM) -- our own SGBM-equivalent, library-free.
#
# The sparse per-point block matcher above CANNOT reproduce the chip depth in
# low-parallax indoor scenes (measured: the global NCC peak is often not at the
# true disparity because the cost is flat/ambiguous over the few-pixel disparity
# range, and corners straddle depth edges). The fix is exactly what the chip's
# SGBM does: a GLOBAL smoothness prior. We implement Hirschmueller semi-global
# matching ourselves:
#
#   1. Census transform of both rectified images (robust to gain/offset; this is
#      the matching primitive SGBM-mode uses on the OAK-D too).
#   2. A Hamming-distance cost volume C(p, d) = popcount(censusL[p] ^ censusR[p-d]).
#   3. Path aggregation: along each of N directions r, accumulate
#         L_r(p,d) = C(p,d) + min( L_r(p-r,d),
#                                  L_r(p-r,d-1)+P1, L_r(p-r,d+1)+P1,
#                                  min_k L_r(p-r,k)+P2 ) - min_k L_r(p-r,k)
#      and S(p,d) = sum_r L_r(p,d). P1 penalises +/-1 disparity steps (slanted
#      surfaces), P2 >> P1 penalises larger jumps (depth discontinuities). This
#      is the global smoothness the sparse matcher lacks.
#   4. Winner-take-all over S, parabola sub-pixel, uniqueness gate, and a
#      left<->right consistency check derived from the same volume.
#
# Numba compiles our own explicit loops (same pattern as the sparse path); the
# pure-NumPy fallback runs the identical math, just slowly.
# ===========================================================================


@dataclass
class SGMConfig:
    """Tuning for the dense semi-global matcher (census + N-path SGM)."""

    census_radius: int = 3       # census window (2r+1)^2; r=3 -> 7x7 (48 bits)
    min_disparity: int = 0       # px
    num_disparities: int = 96    # disparity levels searched [dmin, dmin+ndisp)
    p1: int = 7                  # small-step (+/-1 disparity) smoothness penalty
    p2: int = 86                 # large-step (discontinuity) smoothness penalty
    num_paths: int = 8           # 4 (cardinal) or 8 (cardinal + diagonal)
    uniqueness: float = 0.10     # reject if 2nd-best cost within (1+u)*best
    subpixel: bool = True        # parabola fit on the aggregated-cost minimum
    lr_consistency: bool = True  # left<->right disparity agreement check
    lr_max_diff: float = 1.5     # px; reject if L/R disparity disagree by more
    min_depth: float = 0.1       # m; clamp output (reject closer)
    max_depth: float = 20.0      # m; clamp output (reject farther)
    downscale: int = 1           # compute at 1/N res then upsample (N in {1,2})
    speckle_window: int = 0      # remove connected disparity blobs <= this many
                                 # px (0 = off); kills salt-pepper noise
    speckle_range: float = 1.0   # px; neighbours join a blob if |disp| diff <=

    @classmethod
    def live(cls) -> "SGMConfig":
        """Faster preset for real-time use (live source / replay preview).

        Trades a little accuracy for speed: half-resolution cost volume
        (``downscale=2`` -> 1/4 the pixels and half the disparity range), 4
        cardinal paths instead of 8, and a smaller census window. Measured to
        keep the depth quality close to the full preset while fitting the live
        per-frame budget. The full default config stays the offline/accuracy
        reference.
        """
        return cls(census_radius=2, num_disparities=96, num_paths=4,
                   downscale=2)


# 8 SGM path directions (dv, du): 4 cardinal + 4 diagonal. ``num_paths`` selects
# the first 4 (cardinal only) or all 8.
_SGM_DIRS = np.array(
    [[0, 1], [0, -1], [1, 0], [-1, 0],
     [1, 1], [1, -1], [-1, 1], [-1, -1]],
    dtype=np.int64,
)


# ===========================================================================
# Stereo rectification (library-free) -- rectify the RIGHT frame into the LEFT
# rectified frame so block matching has aligned epipolar rows.
#
# Why this is needed: the recorder saves ``stereo.rectifiedLeft`` (rectified) but
# ``stereo.syncedRight`` (synced but NOT rectified) as the right image. Matching a
# rectified left against an unrectified right gives a per-pixel vertical offset
# that grows with image position (measured: ~48% of corners off by >=2 rows), so
# no row-search block matcher can work. We re-rectify the right frame ourselves
# from the raw calibration (both cameras' intrinsics + distortion + the
# left->right extrinsic), which is also exactly what a portable target platform
# must do from two raw frames.
#
# Convention (verified to 1e-7 against cv2.stereoRectify, which the OAK-D's
# rectification follows): split the inter-camera rotation in half
# (r_l = exp(+w/2), r_r = exp(-w/2)), build the rectifying basis from the
# half-rotated baseline, and KEEP THE LEFT INTRINSIC as the common rectified
# intrinsic (the chip does this -- its rectified-left fx equals the raw-left fx,
# not cv2's alpha-rescaled value). The right map then aligns with the chip's
# rectified-left, so disparity d gives metric depth Z = fx_left * baseline / d.
# Distortion is the OpenCV rational model (k1..k6, p1, p2, s1..s4); the tiny tilt
# terms (tau_x, tau_y ~ 1e-3) are negligible and omitted.
# ===========================================================================


def _rodrigues(om: np.ndarray) -> np.ndarray:
    """Axis-angle (3-vector) -> rotation matrix."""
    th = float(np.linalg.norm(om))
    if th < 1e-12:
        return np.eye(3)
    k = om / th
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle (3-vector)."""
    cos = (np.trace(R) - 1.0) * 0.5
    cos = max(-1.0, min(1.0, cos))
    th = np.arccos(cos)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]])
    return w * (th / (2.0 * np.sin(th)))


def rectify_rotations(R: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bouguet rectifying rotations ``(R1, R2)`` for the left/right cameras.

    ``R, T`` are the left->right extrinsic (``X_right = R @ X_left + T``). Matches
    ``cv2.stereoRectify`` to ~1e-7 (verified on the gold calibration).
    """
    om = _log_so3(np.asarray(R, dtype=np.float64))
    r_l = _rodrigues(0.5 * om)
    r_r = _rodrigues(-0.5 * om)
    t = r_r @ np.asarray(T, dtype=np.float64)
    e1 = -t / np.linalg.norm(t)
    e2 = np.array([-e1[1], e1[0], 0.0])
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    Rrect = np.vstack([e1, e2, e3])
    return Rrect @ r_l, Rrect @ r_r


def _distort_normalized(x: np.ndarray, y: np.ndarray,
                        D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV rational+tangential+thin-prism forward distortion (no tilt).

    Maps ideal normalized coordinates to distorted normalized coordinates.
    ``D`` is ``[k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, (taux, tauy)]``.
    """
    D = np.asarray(D, dtype=np.float64)
    k1, k2, p1, p2, k3 = D[0], D[1], D[2], D[3], D[4]
    k4 = D[5] if D.size > 5 else 0.0
    k5 = D[6] if D.size > 6 else 0.0
    k6 = D[7] if D.size > 7 else 0.0
    s1 = D[8] if D.size > 8 else 0.0
    s2 = D[9] if D.size > 9 else 0.0
    s3 = D[10] if D.size > 10 else 0.0
    s4 = D[11] if D.size > 11 else 0.0
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / \
             (1.0 + k4 * r2 + k5 * r4 + k6 * r6)
    a1 = 2.0 * x * y
    a2 = r2 + 2.0 * x * x
    a3 = r2 + 2.0 * y * y
    xd = x * radial + p1 * a1 + p2 * a2 + s1 * r2 + s2 * r4
    yd = y * radial + p1 * a3 + p2 * a1 + s3 * r2 + s4 * r4
    return xd, yd


@njit(cache=True, fastmath=True)
def _remap_bilinear(src, map_u, map_v):
    """Bilinear sample ``src`` at the per-pixel ``(map_u, map_v)`` coordinates."""
    H, W = src.shape
    out = np.zeros((H, W), dtype=np.float32)
    for v in range(H):
        for u in range(W):
            su = map_u[v, u]
            sv = map_v[v, u]
            if su < 0.0 or sv < 0.0 or su > W - 1.0 or sv > H - 1.0:
                continue
            u0 = int(su)
            v0 = int(sv)
            u1 = u0 + 1 if u0 < W - 1 else u0
            v1 = v0 + 1 if v0 < H - 1 else v0
            fu = su - u0
            fv = sv - v0
            a = src[v0, u0]
            b = src[v0, u1]
            cc = src[v1, u0]
            d = src[v1, u1]
            out[v, u] = (a * (1 - fu) * (1 - fv) + b * fu * (1 - fv)
                         + cc * (1 - fu) * fv + d * fu * fv)
    return out


class RightRectifier:
    """Precomputed warp that rectifies the raw right frame into the left frame.

    The map depends only on calibration, so it is built once and reused for every
    frame of a session. ``rectify(right_raw)`` returns the rectified right image,
    row-aligned with the chip's rectified-left and sharing the left intrinsic, so
    a same-row block match yields disparity ``d`` with ``Z = fx_left * B / d``.
    """

    def __init__(self, K_left: np.ndarray, K_right: np.ndarray,
                 dist_right: np.ndarray, R: np.ndarray, T: np.ndarray,
                 width: int, height: int):
        _, R2 = rectify_rotations(R, T)
        Kl = np.asarray(K_left, dtype=np.float64)
        Kr = np.asarray(K_right, dtype=np.float64)
        Kl_inv = np.linalg.inv(Kl)
        uu, vv = np.meshgrid(np.arange(width, dtype=np.float64),
                             np.arange(height, dtype=np.float64))
        ones = np.ones_like(uu)
        # Rectified pixel -> rectified normalized ray (intrinsic = K_left).
        pix = np.stack([uu, vv, ones], axis=0).reshape(3, -1)
        ray_rect = Kl_inv @ pix
        # Rectified -> raw-right camera coordinates (rectified = R2 @ raw_right).
        ray_raw = R2.T @ ray_rect
        x = ray_raw[0] / ray_raw[2]
        y = ray_raw[1] / ray_raw[2]
        xd, yd = _distort_normalized(x, y, dist_right)
        map_u = (Kr[0, 0] * xd + Kr[0, 1] * yd + Kr[0, 2]).reshape(height, width)
        map_v = (Kr[1, 1] * yd + Kr[1, 2]).reshape(height, width)
        self.map_u = np.ascontiguousarray(map_u.astype(np.float32))
        self.map_v = np.ascontiguousarray(map_v.astype(np.float32))

    @classmethod
    def from_calib(cls, calib) -> "RightRectifier":
        """Build from a :class:`ours.vio.reader.StereoCalib`."""
        T = calib.T_left_right
        return cls(calib.left.K, calib.right.K, calib.right.dist,
                   T[:3, :3], T[:3, 3], calib.left.width, calib.left.height)

    def rectify(self, right_raw: np.ndarray) -> np.ndarray:
        """Rectify a raw right frame; returns ``float32`` (same H, W)."""
        return _remap_bilinear(
            np.ascontiguousarray(right_raw.astype(np.float32)),
            self.map_u, self.map_v)


class LeftRectifier:
    """Precomputed warp that rectifies the raw LEFT frame into the rectified frame.

    Mirror of :class:`RightRectifier` but for the left camera: it uses the left
    rectifying rotation ``R1`` (the chip applies the same one to produce
    ``rectifiedLeft``) with the left intrinsics + distortion, and KEEPS the left
    intrinsic ``K_left`` as the common rectified intrinsic. So
    ``rectify(left_raw)`` reproduces the chip's ``rectifiedLeft`` from a raw
    frame, letting the whole stereo path run with NO VPU / depth library: rectify
    both raw frames ourselves, then SGM. Verified against
    ``cv2.initUndistortRectifyMap`` (left camera, R1, P=K_left).
    """

    def __init__(self, K_left: np.ndarray, dist_left: np.ndarray,
                 R: np.ndarray, T: np.ndarray, width: int, height: int):
        R1, _ = rectify_rotations(R, T)
        Kl = np.asarray(K_left, dtype=np.float64)
        Kl_inv = np.linalg.inv(Kl)
        uu, vv = np.meshgrid(np.arange(width, dtype=np.float64),
                             np.arange(height, dtype=np.float64))
        ones = np.ones_like(uu)
        # Rectified pixel -> rectified normalized ray (intrinsic = K_left).
        pix = np.stack([uu, vv, ones], axis=0).reshape(3, -1)
        ray_rect = Kl_inv @ pix
        # Rectified -> raw-left camera coordinates (rectified = R1 @ raw_left).
        ray_raw = R1.T @ ray_rect
        x = ray_raw[0] / ray_raw[2]
        y = ray_raw[1] / ray_raw[2]
        xd, yd = _distort_normalized(x, y, dist_left)
        map_u = (Kl[0, 0] * xd + Kl[0, 1] * yd + Kl[0, 2]).reshape(height, width)
        map_v = (Kl[1, 1] * yd + Kl[1, 2]).reshape(height, width)
        self.map_u = np.ascontiguousarray(map_u.astype(np.float32))
        self.map_v = np.ascontiguousarray(map_v.astype(np.float32))

    @classmethod
    def from_calib(cls, calib) -> "LeftRectifier":
        """Build from a :class:`ours.vio.reader.StereoCalib`."""
        T = calib.T_left_right
        return cls(calib.left.K, calib.left.dist,
                   T[:3, :3], T[:3, 3], calib.left.width, calib.left.height)

    def rectify(self, left_raw: np.ndarray) -> np.ndarray:
        """Rectify a raw left frame; returns ``float32`` (same H, W)."""
        return _remap_bilinear(
            np.ascontiguousarray(left_raw.astype(np.float32)),
            self.map_u, self.map_v)



@njit(cache=True, fastmath=True)
def _popcount64(x):
    """Population count of a uint64 (SWAR, constant-time)."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + \
        ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True, parallel=True, fastmath=True)
def _census(img, crad):
    """Census signature per pixel as a uint64 (window (2*crad+1)^2 - 1 bits)."""
    H, W = img.shape
    out = np.zeros((H, W), dtype=np.uint64)
    for v in prange(crad, H - crad):
        for u in range(crad, W - crad):
            c = img[v, u]
            bits = np.uint64(0)
            for dy in range(-crad, crad + 1):
                for dx in range(-crad, crad + 1):
                    if dy == 0 and dx == 0:
                        continue
                    bits = bits << np.uint64(1)
                    if img[v + dy, u + dx] < c:
                        bits = bits | np.uint64(1)
            out[v, u] = bits
    return out


@njit(cache=True, parallel=True, fastmath=True)
def _cost_volume(cl, cr, dmin, ndisp, max_cost):
    """Hamming-distance cost volume C(v,u,k) for d = dmin + k."""
    H, W = cl.shape
    vol = np.full((H, W, ndisp), max_cost, dtype=np.int32)
    for v in prange(H):
        for u in range(W):
            cval = cl[v, u]
            for k in range(ndisp):
                ur = u - (dmin + k)
                if ur < 0:
                    break            # larger k -> smaller ur, all out of range
                vol[v, u, k] = np.int32(_popcount64(cval ^ cr[v, ur]))
    return vol


@njit(cache=True, parallel=True, fastmath=True)
def _aggregate_dir(vol, agg, starts_v, starts_u, sy, sx, P1, P2):
    """Accumulate one SGM path direction into ``agg`` (in place), in parallel.

    ``starts_v/starts_u`` are the boundary pixels where a path in direction
    ``(sy, sx)`` begins. Every pixel of the image belongs to exactly one such
    path, so the paths are disjoint and can be walked concurrently -- each
    ``prange`` iteration writes a non-overlapping set of cells. The math is
    identical to the sequential scan.
    """
    H, W, ndisp = vol.shape
    ns = starts_v.shape[0]
    for s in prange(ns):
        Lprev = np.empty(ndisp, dtype=np.int32)
        Lcur = np.empty(ndisp, dtype=np.int32)
        v = starts_v[s]
        u = starts_u[s]
        first = True
        while 0 <= v < H and 0 <= u < W:
            if first:
                for k in range(ndisp):
                    c = vol[v, u, k]
                    Lcur[k] = c
                    agg[v, u, k] += c
                first = False
            else:
                mprev = Lprev[0]
                for k in range(1, ndisp):
                    if Lprev[k] < mprev:
                        mprev = Lprev[k]
                for k in range(ndisp):
                    best = Lprev[k]
                    if k > 0:
                        cc = Lprev[k - 1] + P1
                        if cc < best:
                            best = cc
                    if k < ndisp - 1:
                        cc = Lprev[k + 1] + P1
                        if cc < best:
                            best = cc
                    cc = mprev + P2
                    if cc < best:
                        best = cc
                    val = vol[v, u, k] + best - mprev
                    Lcur[k] = val
                    agg[v, u, k] += val
            for k in range(ndisp):
                Lprev[k] = Lcur[k]
            v += sy
            u += sx


def _path_starts(H: int, W: int, sy: int, sx: int) -> tuple[np.ndarray, np.ndarray]:
    """Boundary start pixels for an SGM path in direction ``(sy, sx)``.

    A pixel starts a path iff its predecessor ``(v - sy, u - sx)`` is outside the
    image. The union of all paths from these starts covers every pixel exactly
    once, which is what makes the parallel aggregation race-free. Vectorised so
    it costs O(boundary), not O(H*W).
    """
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pv = vv - sy
    pu = uu - sx
    mask = (pv < 0) | (pv >= H) | (pu < 0) | (pu >= W)
    vs, us = np.nonzero(mask)
    return vs.astype(np.int64), us.astype(np.int64)


@njit(cache=True, parallel=True, fastmath=True)
def _wta_lr(agg, dmin, uniq, subpixel, lr, lr_max):
    """Winner-take-all + sub-pixel + uniqueness + L/R consistency -> disparity.

    Returns an ``(H, W)`` float disparity map (``NaN`` where rejected). Rows are
    independent, so the winner search runs one row per thread.
    """
    H, W, ndisp = agg.shape
    disp = np.full((H, W), np.nan)

    # Right-image disparity from the SAME volume: for right pixel u_r the
    # candidate left pixel at disparity d is u_r + d, whose cost lives at
    # agg[v, u_r + d, k]. dispR[v, u_r] = argmin_d of those.
    dispR = np.full((H, W), -1.0)
    if lr:
        for v in prange(H):
            for ur in range(W):
                bc = np.int64(1) << np.int64(40)
                bk = -1
                for k in range(ndisp):
                    ul = ur + dmin + k
                    if ul >= W:
                        break
                    c = agg[v, ul, k]
                    if c < bc:
                        bc = c
                        bk = k
                if bk >= 0:
                    dispR[v, ur] = dmin + bk

    for v in prange(H):
        for u in range(W):
            bc = np.int64(1) << np.int64(40)
            bk = -1
            for k in range(ndisp):
                c = agg[v, u, k]
                if c < bc:
                    bc = c
                    bk = k
            if bk < 0:
                continue
            # Uniqueness: reject when a non-adjacent disparity is nearly as cheap.
            second = np.int64(1) << np.int64(40)
            for k in range(ndisp):
                if k < bk - 1 or k > bk + 1:
                    if agg[v, u, k] < second:
                        second = agg[v, u, k]
            if uniq > 0.0 and second < bc * (1.0 + uniq):
                continue

            d = float(dmin + bk)
            if subpixel and 0 < bk < ndisp - 1:
                cm = float(agg[v, u, bk - 1])
                c0 = float(agg[v, u, bk])
                cp = float(agg[v, u, bk + 1])
                den = cm - 2.0 * c0 + cp
                if den > 1e-9:
                    d += 0.5 * (cm - cp) / den

            if lr:
                ur = int(round(u - d))
                if ur < 0 or ur >= W:
                    continue
                dr = dispR[v, ur]
                if dr < 0.0 or abs(d - dr) > lr_max:
                    continue
            disp[v, u] = d
    return disp


def sgm_disparity(left: np.ndarray, right: np.ndarray,
                  cfg: SGMConfig) -> np.ndarray:
    """Dense sub-pixel disparity map via semi-global matching (NaN = invalid)."""
    ds = max(1, int(cfg.downscale))
    if ds > 1:
        # Box-average downsample both images, run SGM at the lower resolution
        # with a proportionally smaller disparity range, then upsample the
        # disparity map back to full resolution (values scale with resolution).
        Hf, Wf = left.shape[:2]
        h2, w2 = Hf // ds, Wf // ds
        Ld = _box_downsample(np.ascontiguousarray(left.astype(np.float64)), ds)
        Rd = _box_downsample(np.ascontiguousarray(right.astype(np.float64)), ds)
        sub = SGMConfig(census_radius=cfg.census_radius,
                        min_disparity=cfg.min_disparity // ds,
                        num_disparities=max(8, cfg.num_disparities // ds),
                        p1=cfg.p1, p2=cfg.p2, num_paths=cfg.num_paths,
                        uniqueness=cfg.uniqueness, subpixel=cfg.subpixel,
                        lr_consistency=cfg.lr_consistency,
                        lr_max_diff=cfg.lr_max_diff, downscale=1,
                        speckle_window=cfg.speckle_window,
                        speckle_range=cfg.speckle_range)
        disp_small = sgm_disparity(Ld[:h2, :w2], Rd[:h2, :w2], sub)
        return _upsample_disp(disp_small, Hf, Wf, ds)

    L = np.ascontiguousarray(left.astype(np.float64))
    R = np.ascontiguousarray(right.astype(np.float64))
    crad = cfg.census_radius
    ndisp = cfg.num_disparities
    max_cost = (2 * crad + 1) * (2 * crad + 1)  # max Hamming = #census bits + 1
    H, W = L.shape

    cl = _census(L, crad)
    cr = _census(R, crad)
    vol = _cost_volume(cl, cr, cfg.min_disparity, ndisp, np.int32(max_cost))

    agg = np.zeros_like(vol)
    npaths = 8 if cfg.num_paths >= 8 else 4
    for i in range(npaths):
        sy = int(_SGM_DIRS[i, 0])
        sx = int(_SGM_DIRS[i, 1])
        sv, su = _path_starts(H, W, sy, sx)
        _aggregate_dir(vol, agg, sv, su, sy, sx,
                       np.int32(cfg.p1), np.int32(cfg.p2))

    disp = _wta_lr(agg, cfg.min_disparity, float(cfg.uniqueness),
                   cfg.subpixel, cfg.lr_consistency, float(cfg.lr_max_diff))
    if cfg.speckle_window > 0:
        _speckle_filter(disp, int(cfg.speckle_window), float(cfg.speckle_range))
    return disp


@njit(cache=True)
def _speckle_filter(disp, max_size, max_diff):
    """Invalidate connected disparity blobs of <= ``max_size`` pixels (in place).

    4-connected flood fill (a la OpenCV ``filterSpeckles``): two neighbouring
    valid pixels join the same blob when their disparity differs by at most
    ``max_diff``. Any blob whose pixel count does not exceed ``max_size`` is set
    to ``NaN`` -- the isolated salt-pepper mismatches that survive the L/R check
    but corrupt depth at tracked points.
    """
    H, W = disp.shape
    label = np.zeros((H, W), dtype=np.uint8)   # 0 = unvisited, 1 = visited
    sv = np.empty(H * W, dtype=np.int32)
    su = np.empty(H * W, dtype=np.int32)
    mv = np.empty(max_size + 1, dtype=np.int32)
    mu = np.empty(max_size + 1, dtype=np.int32)
    for v0 in range(H):
        for u0 in range(W):
            if label[v0, u0] != 0 or np.isnan(disp[v0, u0]):
                continue
            top = 1
            sv[0] = v0
            su[0] = u0
            label[v0, u0] = 1
            count = 0
            small = True
            while top > 0:
                top -= 1
                v = sv[top]
                u = su[top]
                d = disp[v, u]
                if small:
                    mv[count] = v
                    mu[count] = u
                count += 1
                if count > max_size:
                    small = False
                if v > 0 and label[v - 1, u] == 0 and not np.isnan(disp[v - 1, u]) \
                        and abs(disp[v - 1, u] - d) <= max_diff:
                    label[v - 1, u] = 1
                    sv[top] = v - 1
                    su[top] = u
                    top += 1
                if v < H - 1 and label[v + 1, u] == 0 and not np.isnan(disp[v + 1, u]) \
                        and abs(disp[v + 1, u] - d) <= max_diff:
                    label[v + 1, u] = 1
                    sv[top] = v + 1
                    su[top] = u
                    top += 1
                if u > 0 and label[v, u - 1] == 0 and not np.isnan(disp[v, u - 1]) \
                        and abs(disp[v, u - 1] - d) <= max_diff:
                    label[v, u - 1] = 1
                    sv[top] = v
                    su[top] = u - 1
                    top += 1
                if u < W - 1 and label[v, u + 1] == 0 and not np.isnan(disp[v, u + 1]) \
                        and abs(disp[v, u + 1] - d) <= max_diff:
                    label[v, u + 1] = 1
                    sv[top] = v
                    su[top] = u + 1
                    top += 1
            if count <= max_size:
                for j in range(count):
                    disp[mv[j], mu[j]] = np.nan


def _box_downsample(img: np.ndarray, n: int) -> np.ndarray:
    """Average ``n x n`` blocks -> image at 1/n resolution (anti-aliased)."""
    h, w = img.shape[:2]
    h2, w2 = h // n, w // n
    cropped = img[:h2 * n, :w2 * n]
    return cropped.reshape(h2, n, w2, n).mean(axis=(1, 3))


def _upsample_disp(disp_small: np.ndarray, H: int, W: int, n: int) -> np.ndarray:
    """Nearest-neighbour upsample a 1/n disparity map and rescale values by n.

    Disparity is in pixels, so a value computed at 1/n resolution must be
    multiplied by ``n`` to become a full-resolution disparity. ``NaN`` (rejected)
    cells stay ``NaN``.
    """
    out = np.full((H, W), np.nan)
    hs, ws = disp_small.shape
    vv = np.minimum(np.arange(H) // n, hs - 1)
    uu = np.minimum(np.arange(W) // n, ws - 1)
    up = disp_small[np.ix_(vv, uu)] * float(n)
    out[:, :] = up
    return out


class SGMStereoMatcher:
    """Dense SGM stereo depth (drop-in for the chip depth in the ``ours`` VIO).

    Computes a full disparity map with our own semi-global matcher, then exposes
    the same query interface as :class:`StereoMatcher` (``depth_at`` /
    ``sparse_depth_map``) so the VIO can sample depth at its tracked pixels, plus
    ``dense_depth`` for a full metric map. Unlike the sparse matcher, the global
    smoothness prior disambiguates the low-parallax indoor disparities that made
    per-point block matching fail (see stereo_selftest for the measured gap).
    """

    def __init__(self, K: np.ndarray, baseline_m: float,
                 cfg: SGMConfig | None = None,
                 rectifier: "RightRectifier | None" = None,
                 left_rectifier: "LeftRectifier | None" = None):
        self.K = np.asarray(K, dtype=np.float64)
        self.fx = float(self.K[0, 0])
        self.baseline_m = float(baseline_m)
        self.cfg = cfg or SGMConfig()
        # Optional warp that rectifies the raw right frame into the left frame
        # before matching (see RightRectifier). Required whenever the right input
        # is not already rectified (the recorded gold ``syncedRight`` is raw).
        self.rectifier = rectifier
        # Optional warp that rectifies the raw LEFT frame too (see LeftRectifier).
        # Used by the fully VPU-free live path that feeds two RAW frames; left
        # stays None when the caller already provides a rectified left (the gold
        # ``rectifiedLeft`` from the chip), so the offline path is unchanged.
        self.left_rectifier = left_rectifier

    @classmethod
    def from_calib(cls, calib, cfg: SGMConfig | None = None,
                   rectify_left: bool = False) -> "SGMStereoMatcher":
        """Build a matcher that rectifies the raw right frame from ``calib``.

        ``calib`` is a :class:`ours.vio.reader.StereoCalib`. By default the
        matcher expects the chip's rectified-left and the **raw** right frame
        (exactly what the gold sessions store) and rectifies the right
        internally. Set ``rectify_left=True`` to ALSO rectify a raw left frame
        ourselves (the fully VPU-free live path that taps both raw cameras).
        """
        return cls(calib.left.K, calib.baseline_m, cfg,
                   rectifier=RightRectifier.from_calib(calib),
                   left_rectifier=(LeftRectifier.from_calib(calib)
                                   if rectify_left else None))

    def dense_disparity(self, left: np.ndarray,
                        right: np.ndarray) -> np.ndarray:
        """Full ``(H, W)`` sub-pixel disparity map (``NaN`` where invalid)."""
        if self.left_rectifier is not None:
            left = self.left_rectifier.rectify(left)
        if self.rectifier is not None:
            right = self.rectifier.rectify(right)
        return sgm_disparity(left, right, self.cfg)


    def dense_depth(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Full ``(H, W)`` metric depth map (``0.0`` where invalid)."""
        disp = self.dense_disparity(left, right)
        depth = np.zeros(disp.shape, dtype=np.float32)
        good = np.isfinite(disp) & (disp > 1e-6)
        z = self.fx * self.baseline_m / disp[good]
        z[(z < self.cfg.min_depth) | (z > self.cfg.max_depth)] = 0.0
        depth[good] = z
        return depth

    def dense_depth_rectified_left(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(rectified_left, depth)`` from two RAW frames.

        The returned depth is defined on the RECTIFIED-left pixel grid, so any
        consumer that samples depth at image coordinates (e.g. the KLT/PnP
        frontend) MUST track on the rectified left returned here -- NOT on the
        raw left -- or every feature's depth is read at the wrong pixel (the
        rectification warp, several px near the edges) and PnP degrades. This
        rectifies the left exactly once and reuses it for the disparity, so it
        costs the same as :meth:`dense_depth`.
        """
        left_rect = (self.left_rectifier.rectify(left)
                     if self.left_rectifier is not None else left)
        right_r = (self.rectifier.rectify(right)
                   if self.rectifier is not None else right)
        disp = sgm_disparity(left_rect, right_r, self.cfg)
        depth = np.zeros(disp.shape, dtype=np.float32)
        good = np.isfinite(disp) & (disp > 1e-6)
        z = self.fx * self.baseline_m / disp[good]
        z[(z < self.cfg.min_depth) | (z > self.cfg.max_depth)] = 0.0
        depth[good] = z
        return left_rect, depth

    def depth_at(self, left: np.ndarray, right: np.ndarray,
                 pts: np.ndarray) -> np.ndarray:
        """Metric depth (m) per query point; ``0.0`` where unmatched/invalid."""
        depth_map = self.dense_depth(left, right)
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        out = np.zeros(pts.shape[0], dtype=np.float32)
        if pts.shape[0] == 0:
            return out
        h, w = depth_map.shape
        us = np.round(pts[:, 0]).astype(np.int64)
        vs = np.round(pts[:, 1]).astype(np.int64)
        ok = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
        out[ok] = depth_map[vs[ok], us[ok]]
        return out

    def sparse_depth_map(self, left: np.ndarray, right: np.ndarray,
                         pts: np.ndarray) -> np.ndarray:
        """An ``(H, W)`` depth map filled only at the rounded query pixels.

        Computes the dense SGM depth once, then keeps only the cells the VIO
        will read (its tracked pixels). Drop-in for the chip depth map.
        """
        depth_map = self.dense_depth(left, right)
        h, w = depth_map.shape
        out = np.zeros((h, w), dtype=np.float32)
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] == 0:
            return out
        us = np.round(pts[:, 0]).astype(np.int64)
        vs = np.round(pts[:, 1]).astype(np.int64)
        ok = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
        out[vs[ok], us[ok]] = depth_map[vs[ok], us[ok]]
        return out
