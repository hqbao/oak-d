"""Library-free ORB: oriented FAST keypoints + steered BRIEF descriptors.

Our own replacement for the subset of ``cv2.ORB`` / ``cv2.BFMatcher`` /
``cv2.findFundamentalMat`` that the loop-closure frontend
(:mod:`slam.mathlib.loop.loopclosure`) needs, so the ``ours-slam`` path carries no cv2
runtime dependency (matching what :mod:`vio.mathlib.frontend.klt` /
:mod:`vio.mathlib.frontend.corners` did for the VO frontend).

Pipeline (pure NumPy):

1. **oriented FAST** -- FAST-9 corner test on the radius-3 Bresenham circle
   (a contiguous arc of >=9 pixels all brighter than ``Ip+t`` or all darker
   than ``Ip-t``), 3x3 non-max suppression on the corner score, then a Harris
   re-score to keep the strongest ``N`` (ORB's "HARRIS_SCORE" selection). A
   coarse image pyramid gives a measure of scale invariance.
2. **orientation** -- intensity-centroid angle ``atan2(m01, m10)`` over a
   radius-15 circular patch (Rosin moment, exactly ORB's rotation estimate).
3. **steered BRIEF-256** -- 256 intensity-comparison tests on a fixed Gaussian
   sampling pattern, the pattern *rotated* by the keypoint orientation before
   sampling (this is what makes BRIEF rotation-invariant -> "rBRIEF"), packed
   into a 32-byte (256-bit) binary descriptor. The patch is box-smoothed first
   (cheap stand-in for ORB's Gaussian blur) to make the bit tests stable.

Matching is Hamming brute force (``hamming_knn``) and the epipolar pre-filter
is a normalised 8-point fundamental-matrix RANSAC (``find_fundamental_ransac``).
The final metric verification reuses our own :func:`sky.front.pnp.solve_pnp_ransac`.

The descriptors are NOT bit-compatible with OpenCV's learned ``bit_pattern_31``
(we sample our own deterministic pattern), so ours descriptors only match ours
descriptors -- which is all loop closure needs. ``orb_selftest.py`` validates
the keypoints against the cv2 oracle (repeatability, not bit-equality).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# FAST-9 corner test on the radius-3 Bresenham circle (16 pixels, clockwise).
# ---------------------------------------------------------------------------
# (dx, dy) offsets, standard FAST circle ordering.
_CIRCLE = np.array(
    [(0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
     (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)],
    dtype=np.int64,
)
_ARC = 9          # contiguous-arc length for FAST-9
_PATCH = 31       # BRIEF patch size (odd)
_HALF = _PATCH // 2
_ORIENT_R = 15    # intensity-centroid radius (ORB uses 15)
_DESC_BITS = 256
_DESC_BYTES = _DESC_BITS // 8

# Popcount lookup table for uint8 (Hamming distance via XOR + table sum).
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass
class OrbConfig:
    n_features: int = 800        # keypoints kept (strongest by Harris score)
    fast_threshold: float = 20.0  # FAST intensity threshold t
    n_levels: int = 4            # pyramid levels (scale invariance)
    scale_factor: float = 1.2    # pyramid downscale per level
    harris_k: float = 0.04       # Harris score sensitivity
    edge_margin: int = _HALF + 1  # keep keypoints this far from the border


# ---------------------------------------------------------------------------
# Deterministic BRIEF sampling pattern (our own; not OpenCV's learned table).
# ---------------------------------------------------------------------------
def _make_pattern(seed: int = 0x012B) -> np.ndarray:
    """256 point-pairs sampled from an isotropic Gaussian in the patch.

    Returns an int array of shape (256, 4): ``(x1, y1, x2, y2)`` integer offsets
    inside the ``_PATCH`` x ``_PATCH`` window (the original BRIEF "G II"
    sampling: both points i.i.d. Gaussian, sigma = patch/5).
    """
    rng = np.random.default_rng(seed)
    sigma = _PATCH / 5.0
    pts = rng.normal(0.0, sigma, size=(_DESC_BITS, 4))
    pts = np.clip(np.round(pts), -_HALF, _HALF).astype(np.int64)
    return pts


_PATTERN = _make_pattern()
# Split into the two sample sets for convenience: (256,2) each.
_PAT_A = _PATTERN[:, 0:2].astype(np.float64)
_PAT_B = _PATTERN[:, 2:4].astype(np.float64)


def _box_blur(img: np.ndarray, r: int = 2) -> np.ndarray:
    """Separable box blur (window 2r+1) via integral image; edge-replicated."""
    f = img.astype(np.float64)
    pad = np.pad(f, ((r + 1, r), (r + 1, r)), mode="edge")
    ii = pad.cumsum(0).cumsum(1)
    h, w = img.shape
    # sum over [y-r, y+r] x [x-r, x+r] using the integral image
    A = ii[0:h, 0:w]
    B = ii[0:h, 2 * r + 1:2 * r + 1 + w]
    C = ii[2 * r + 1:2 * r + 1 + h, 0:w]
    D = ii[2 * r + 1:2 * r + 1 + h, 2 * r + 1:2 * r + 1 + w]
    area = (2 * r + 1) ** 2
    return (D - B - C + A) / area


def _fast_corners(gray: np.ndarray, t: float):
    """FAST-9 corner mask + score on a single image level.

    Returns ``(ys, xs, score)`` of detected corners (after 3x3 NMS on the FAST
    score = sum of absolute differences over the deciding arc).
    """
    h, w = gray.shape
    I = gray.astype(np.float64)
    # Stack the 16 circle neighbours (border pixels get an out-of-arc value).
    neigh = np.empty((16, h, w), np.float64)
    for k, (dx, dy) in enumerate(_CIRCLE):
        neigh[k] = np.roll(np.roll(I, -dy, axis=0), -dx, axis=1)
    brighter = neigh > (I + t)
    darker = neigh < (I - t)

    def _arc(mask: np.ndarray) -> np.ndarray:
        # any contiguous run of >= _ARC around the 16-circle (wrap-around)
        ext = np.concatenate([mask, mask[:_ARC - 1]], axis=0)   # (16+8,h,w)
        pref = np.concatenate(
            [np.zeros((1, h, w), np.int16), ext.cumsum(0).astype(np.int16)], 0)
        ws = pref[_ARC:_ARC + 16] - pref[0:16]                  # (16,h,w)
        return (ws == _ARC).any(0)

    is_corner = _arc(brighter) | _arc(darker)
    # zero a 3px border (the circle reads out of bounds there)
    is_corner[:3] = is_corner[-3:] = False
    is_corner[:, :3] = is_corner[:, -3:] = False
    if not is_corner.any():
        return np.empty(0, int), np.empty(0, int), np.empty(0)

    # FAST score: sum |Ip - Icircle| over the brighter-or-darker neighbours.
    diff = np.abs(neigh - I)
    score = np.where(brighter | darker, diff, 0.0).sum(0)
    score = np.where(is_corner, score, 0.0)

    # 3x3 non-max suppression on the score.
    keep = is_corner.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(score, dy, axis=0), dx, axis=1)
            keep &= score >= shifted
    ys, xs = np.nonzero(keep)
    return ys, xs, score[ys, xs]


def _harris_score(I: np.ndarray, ys: np.ndarray, xs: np.ndarray,
                  k: float) -> np.ndarray:
    """Harris corner response at the given pixels (7x7 window)."""
    gy, gx = np.gradient(I)
    Ixx = gx * gx
    Iyy = gy * gy
    Ixy = gx * gy
    # box-sum the structure tensor over a 7x7 window via integral images
    def bsum(M):
        return _box_blur(M, r=3) * 49.0
    Sxx, Syy, Sxy = bsum(Ixx), bsum(Iyy), bsum(Ixy)
    det = Sxx[ys, xs] * Syy[ys, xs] - Sxy[ys, xs] ** 2
    tr = Sxx[ys, xs] + Syy[ys, xs]
    return det - k * tr * tr


def _orientation(I: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Intensity-centroid orientation (rad) for each keypoint."""
    r = _ORIENT_R
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    circ = (xx * xx + yy * yy) <= r * r
    xw = (xx * circ).astype(np.float64)
    yw = (yy * circ).astype(np.float64)
    ang = np.empty(len(ys))
    for i in range(len(ys)):
        y0, x0 = ys[i], xs[i]
        patch = I[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
        m10 = float((patch * xw).sum())
        m01 = float((patch * yw).sum())
        ang[i] = np.arctan2(m01, m10)
    return ang


def _describe(I_blur: np.ndarray, ys: np.ndarray, xs: np.ndarray,
              ang: np.ndarray) -> np.ndarray:
    """Steered BRIEF-256 -> (N, 32) uint8 descriptors."""
    n = len(ys)
    desc = np.zeros((n, _DESC_BYTES), np.uint8)
    if n == 0:
        return desc
    h, w = I_blur.shape
    cos, sin = np.cos(ang), np.sin(ang)
    ax, ay = _PAT_A[:, 0], _PAT_A[:, 1]
    bx, by = _PAT_B[:, 0], _PAT_B[:, 1]
    for i in range(n):
        c, s = cos[i], sin[i]
        # rotate the sampling pattern by the keypoint orientation
        a_x = np.round(c * ax - s * ay).astype(np.int64) + xs[i]
        a_y = np.round(s * ax + c * ay).astype(np.int64) + ys[i]
        b_x = np.round(c * bx - s * by).astype(np.int64) + xs[i]
        b_y = np.round(s * bx + c * by).astype(np.int64) + ys[i]
        np.clip(a_x, 0, w - 1, out=a_x)
        np.clip(a_y, 0, h - 1, out=a_y)
        np.clip(b_x, 0, w - 1, out=b_x)
        np.clip(b_y, 0, h - 1, out=b_y)
        bits = (I_blur[a_y, a_x] < I_blur[b_y, b_x]).astype(np.uint8)
        desc[i] = np.packbits(bits)
    return desc


class ORB:
    """Oriented FAST + rotated BRIEF detector/descriptor (cv2.ORB stand-in)."""

    def __init__(self, cfg: OrbConfig | None = None):
        self.cfg = cfg or OrbConfig()

    def detect_and_compute(self, gray: np.ndarray):
        """Return ``(pts (N,2) float32, desc (N,32) uint8)`` for ``gray``.

        Keypoints are gathered across a coarse pyramid (coordinates mapped back
        to level-0 pixels), then the strongest ``n_features`` by Harris score
        are kept.
        """
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        cfg = self.cfg
        all_x, all_y, all_s = [], [], []
        scale = 1.0
        img = gray.astype(np.float64)
        for lvl in range(cfg.n_levels):
            if min(img.shape) < 2 * _PATCH:
                break
            ys, xs, _ = _fast_corners(img, cfg.fast_threshold)
            if len(ys):
                # drop keypoints too close to this level's border
                m = cfg.edge_margin
                ok = ((xs >= m) & (xs < img.shape[1] - m) &
                      (ys >= m) & (ys < img.shape[0] - m))
                ys, xs = ys[ok], xs[ok]
                if len(ys):
                    hs = _harris_score(img, ys, xs, cfg.harris_k)
                    all_x.append(xs * scale)
                    all_y.append(ys * scale)
                    all_s.append(hs)
            # downscale for the next pyramid level (simple 2x2-area decimation
            # via box blur + subsample at the configured scale factor)
            scale *= cfg.scale_factor
            new_h = int(round(gray.shape[0] / scale))
            new_w = int(round(gray.shape[1] / scale))
            if new_h < 2 * _PATCH or new_w < 2 * _PATCH:
                break
            img = _resize_area(gray.astype(np.float64), new_h, new_w)

        if not all_x:
            return np.empty((0, 2), np.float32), np.empty((0, 32), np.uint8)
        X = np.concatenate(all_x)
        Y = np.concatenate(all_y)
        S = np.concatenate(all_s)
        # keep the strongest n_features
        if len(X) > cfg.n_features:
            sel = np.argpartition(S, -cfg.n_features)[-cfg.n_features:]
            X, Y, S = X[sel], Y[sel], S[sel]

        # orientation + description are computed on the level-0 image so the
        # descriptor patch always has full resolution (keypoints from coarse
        # levels are described at their mapped level-0 location).
        I0 = gray.astype(np.float64)
        m = cfg.edge_margin
        xs0 = np.round(X).astype(np.int64)
        ys0 = np.round(Y).astype(np.int64)
        ok = ((xs0 >= m) & (xs0 < gray.shape[1] - m) &
              (ys0 >= m) & (ys0 < gray.shape[0] - m))
        xs0, ys0 = xs0[ok], ys0[ok]
        if len(xs0) == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 32), np.uint8)
        ang = _orientation(I0, ys0, xs0)
        I_blur = _box_blur(I0, r=2)
        desc = _describe(I_blur, ys0, xs0, ang)
        pts = np.stack([xs0, ys0], axis=1).astype(np.float32)
        return pts, desc


def _resize_area(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Light area-style downscale: box-blur then bilinear subsample."""
    h, w = img.shape
    sy, sx = h / new_h, w / new_w
    blur = _box_blur(img, r=max(1, int(min(sy, sx) / 2)))
    yy = (np.arange(new_h) + 0.5) * sy - 0.5
    xx = (np.arange(new_w) + 0.5) * sx - 0.5
    y0 = np.clip(np.floor(yy).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xx).astype(int), 0, w - 1)
    return blur[np.ix_(y0, x0)]


# ---------------------------------------------------------------------------
# Hamming brute-force kNN matcher (cv2.BFMatcher(NORM_HAMMING) stand-in).
# ---------------------------------------------------------------------------
def _hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between (Na,32) and (Nb,32) uint8 descriptors.

    Returns an (Na, Nb) int16 matrix. XOR every pair then sum the per-byte
    popcounts from the lookup table.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.int16)
    # (Na,1,32) ^ (1,Nb,32) -> (Na,Nb,32), table-lookup popcount, sum bytes.
    xor = np.bitwise_xor(a[:, None, :], b[None, :, :])
    return _POPCOUNT8[xor].sum(axis=2).astype(np.int16)


def hamming_knn(a: np.ndarray, b: np.ndarray, k: int = 2):
    """k nearest ``b`` descriptors for each ``a`` descriptor (Hamming).

    Returns ``idx (Na,k) int`` and ``dist (Na,k) int`` sorted ascending. Mirrors
    the part of ``cv2.BFMatcher.knnMatch`` the ratio test needs.
    """
    D = _hamming_matrix(a, b)
    nb = D.shape[1]
    k = min(k, nb)
    if k == 0:
        return (np.empty((len(a), 0), int), np.empty((len(a), 0), np.int16))
    idx = np.argpartition(D, kth=range(k), axis=1)[:, :k]
    dist = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(dist, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    dist = np.take_along_axis(dist, order, axis=1)
    return idx, dist


def match_ratio_mutual(desc_a: np.ndarray, desc_b: np.ndarray,
                       ratio: float = 0.75):
    """Lowe-ratio + mutual (cross-check) matches a->b.

    Returns a list of ``(ia, ib)`` index pairs, mirroring the behaviour of
    :meth:`slam.mathlib.loop.loopclosure.LoopDetector._good_matches`.
    """
    if len(desc_a) < 2 or len(desc_b) < 2:
        return []
    idx_ab, dist_ab = hamming_knn(desc_a, desc_b, k=2)
    idx_ba, _ = hamming_knn(desc_b, desc_a, k=1)
    best_ba = idx_ba[:, 0]                      # best a-index for each b
    good = []
    for ia in range(len(desc_a)):
        if dist_ab[ia, 1] <= 0:
            continue
        if dist_ab[ia, 0] < ratio * dist_ab[ia, 1]:
            ib = int(idx_ab[ia, 0])
            if best_ba[ib] == ia:               # mutual
                good.append((ia, ib))
    return good


# ---------------------------------------------------------------------------
# Fundamental-matrix RANSAC (cv2.findFundamentalMat(FM_RANSAC) stand-in).
# ---------------------------------------------------------------------------
def _normalize_pts(pts: np.ndarray):
    """Hartley isotropic normalisation. Returns ``(norm_pts (N,3), T (3,3))``."""
    c = pts.mean(0)
    d = np.sqrt(((pts - c) ** 2).sum(1)).mean()
    s = np.sqrt(2.0) / (d if d > 1e-12 else 1.0)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    return (ph @ T.T), T


def _fundamental_8pt(p1: np.ndarray, p2: np.ndarray):
    """Normalised 8-point fundamental matrix from >=8 correspondences."""
    n1, T1 = _normalize_pts(p1)
    n2, T2 = _normalize_pts(p2)
    x1, y1 = n1[:, 0], n1[:, 1]
    x2, y2 = n2[:, 0], n2[:, 1]
    A = np.stack([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2,
                  x1, y1, np.ones_like(x1)], axis=1)
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    F = Vt[-1].reshape(3, 3)
    # enforce rank 2
    U, S, Vt2 = np.linalg.svd(F)
    S[2] = 0.0
    F = U @ np.diag(S) @ Vt2
    F = T2.T @ F @ T1                            # denormalise
    if abs(F[2, 2]) > 1e-12:
        F = F / F[2, 2]
    return F


def _sampson(F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Sampson distance (px) of each correspondence to the epipolar geometry."""
    ph1 = np.hstack([p1, np.ones((len(p1), 1))])
    ph2 = np.hstack([p2, np.ones((len(p2), 1))])
    Fp1 = ph1 @ F.T                              # (N,3): lines in image 2
    Ftp2 = ph2 @ F                               # (N,3): lines in image 1
    num = (np.sum(ph2 * Fp1, axis=1)) ** 2
    den = Fp1[:, 0] ** 2 + Fp1[:, 1] ** 2 + Ftp2[:, 0] ** 2 + Ftp2[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        d = num / den
    d[~np.isfinite(d)] = np.inf
    return np.sqrt(d)


def find_fundamental_ransac(p1: np.ndarray, p2: np.ndarray,
                            thresh_px: float = 2.0, conf: float = 0.999,
                            max_iters: int = 2000,
                            rng: np.random.Generator | None = None):
    """RANSAC fundamental matrix. Returns ``(F (3,3), mask (N,) bool)`` or None.

    Drop-in for the subset of ``cv2.findFundamentalMat(..., cv2.FM_RANSAC, ...)``
    that loop closure uses as an epipolar pre-filter.
    """
    p1 = np.asarray(p1, np.float64)
    p2 = np.asarray(p2, np.float64)
    n = len(p1)
    if n < 8:
        return None
    if rng is None:
        rng = np.random.default_rng(0x5EED)
    best_mask = None
    best_cnt = 0
    it = 0
    iters = int(max_iters)
    thr2 = thresh_px
    while it < iters:
        it += 1
        idx = rng.choice(n, size=8, replace=False)
        F = _fundamental_8pt(p1[idx], p2[idx])
        if F is None:
            continue
        d = _sampson(F, p1, p2)
        mask = d < thr2
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_mask = mask
            # adaptive iteration count from the current inlier ratio
            w = cnt / n
            denom = 1.0 - w ** 8
            if denom > 1e-12:
                need = np.log(max(1e-12, 1.0 - conf)) / np.log(max(1e-12, denom))
                iters = min(iters, max(8, int(need)))
    if best_mask is None or best_cnt < 8:
        return None
    # refit on all inliers
    F = _fundamental_8pt(p1[best_mask], p2[best_mask])
    if F is None:
        return None
    d = _sampson(F, p1, p2)
    mask = d < thr2
    return F, mask
