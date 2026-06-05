"""Pure-NumPy Shi-Tomasi corner detection.

Our own replacement for ``cv2.goodFeaturesToTrack`` so the visual frontend
detects corners library-free (the KLT tracker in :mod:`oakd.vio.klt` already
replaced ``cv2.calcOpticalFlowPyrLK``).

It follows the standard Shi-Tomasi ("good features to track") recipe, the same
one OpenCV implements:

    1. Spatial gradients ``Ix, Iy`` (Sobel 3x3).
    2. Per-pixel structure-tensor products ``Ix^2, Iy^2, Ix·Iy`` summed over a
       ``block_size`` window (box filter) -> the windowed second-moment matrix
       ``M = [[Sxx, Sxy], [Sxy, Syy]]``.
    3. The corner response is the *smaller* eigenvalue of ``M``
       ``lambda_min = ((Sxx+Syy) - sqrt((Sxx-Syy)^2 + 4 Sxy^2)) / 2`` -- large
       only where the gradient is strong in *both* directions (a true corner).
    4. Keep pixels whose response exceeds ``quality_level * max(response)`` and
       that are a 3x3 local maximum (non-maximum suppression).
    5. Greedily pick the strongest first, enforcing a minimum Euclidean spacing
       ``min_distance`` (via an occupancy grid, like OpenCV).

A binary ``mask`` (0 = ignore) lets the caller exclude neighbourhoods of points
it already tracks, matching the ``cv2.goodFeaturesToTrack(mask=...)`` contract.
"""
from __future__ import annotations

import numpy as np

# Sobel 3x3 as separable kernels: smoothing [1 2 1] * derivative [-1 0 1].
_SMOOTH3 = np.array([1.0, 2.0, 1.0], dtype=np.float32)
_DERIV3 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)


def _conv3_rows(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """1-D convolution along rows (horizontal) with replicated borders."""
    p = np.pad(img, ((0, 0), (1, 1)), mode="edge")
    return k[0] * p[:, 0:-2] + k[1] * p[:, 1:-1] + k[2] * p[:, 2:]


def _conv3_cols(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """1-D convolution along columns (vertical) with replicated borders."""
    p = np.pad(img, ((1, 1), (0, 0)), mode="edge")
    return k[0] * p[0:-2, :] + k[1] * p[1:-1, :] + k[2] * p[2:, :]


def _sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradients (Ix, Iy) via separable smoothing * derivative."""
    Ix = _conv3_cols(_conv3_rows(img, _DERIV3), _SMOOTH3)
    Iy = _conv3_cols(_conv3_rows(img, _SMOOTH3), _DERIV3)
    return Ix, Iy


def _box_sum(img: np.ndarray, block: int) -> np.ndarray:
    """Sum over a ``block`` x ``block`` window (box filter), edge-padded.

    Uses an integral image so the cost is independent of ``block`` size.
    """
    r = block // 2
    p = np.pad(img, ((r + 1, r), (r + 1, r)), mode="edge")
    ii = np.cumsum(np.cumsum(p, axis=0), axis=1)
    # window [y-r, y+r] x [x-r, x+r] sum via the integral image corners
    H, W = img.shape
    y0 = np.arange(H)
    x0 = np.arange(W)
    # indices into ii (padded by r+1 on top/left)
    a = ii[np.ix_(y0, x0)]                       # top-left  (exclusive)
    b = ii[np.ix_(y0, x0 + block)]               # top-right
    c = ii[np.ix_(y0 + block, x0)]               # bottom-left
    d = ii[np.ix_(y0 + block, x0 + block)]       # bottom-right
    return d - b - c + a


def _dilate3(r: np.ndarray) -> np.ndarray:
    """3x3 grey dilation (local max) with -inf borders, for NMS."""
    p = np.pad(r, 1, mode="constant", constant_values=-np.inf)
    H, W = r.shape
    out = p[0:H, 0:W]
    for i in range(3):
        for j in range(3):
            out = np.maximum(out, p[i:i + H, j:j + W])
    return out


def good_features_to_track(
    gray: np.ndarray,
    max_corners: int,
    quality_level: float = 0.01,
    min_distance: float = 12.0,
    block_size: int = 7,
    mask: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Shi-Tomasi corners. Drop-in for ``cv2.goodFeaturesToTrack``.

    Returns an ``(N, 2) float32`` array of ``(x, y)`` pixel coordinates, sorted
    strongest-first, with at most ``max_corners`` points spaced at least
    ``min_distance`` apart. ``N`` may be 0.

    ``exclude`` is an optional ``(M, 2)`` set of already-tracked points; new
    corners are kept ``min_distance`` away from them too (this replaces drawing a
    ``cv2.circle`` mask around existing tracks).
    """
    if max_corners <= 0:
        return np.empty((0, 2), np.float32)
    img = gray.astype(np.float32)
    H, W = img.shape

    Ix, Iy = _sobel(img)
    Sxx = _box_sum(Ix * Ix, block_size)
    Syy = _box_sum(Iy * Iy, block_size)
    Sxy = _box_sum(Ix * Iy, block_size)

    # smaller eigenvalue of [[Sxx, Sxy], [Sxy, Syy]]
    tr = Sxx + Syy
    diff = Sxx - Syy
    disc = np.sqrt(np.maximum(diff * diff + 4.0 * Sxy * Sxy, 0.0))
    resp = 0.5 * (tr - disc)

    # ignore a border where the box window runs off the image
    b = block_size // 2 + 1
    border = np.zeros_like(resp, dtype=bool)
    border[b:H - b, b:W - b] = True
    resp = np.where(border, resp, 0.0)

    if mask is not None:
        resp = np.where(mask > 0, resp, 0.0)

    rmax = float(resp.max())
    if rmax <= 0.0:
        return np.empty((0, 2), np.float32)
    thresh = quality_level * rmax

    # non-maximum suppression: keep strict-ish 3x3 local maxima above threshold
    local_max = resp >= _dilate3(resp)
    keep = local_max & (resp > thresh)
    ys, xs = np.nonzero(keep)
    if ys.size == 0:
        return np.empty((0, 2), np.float32)
    vals = resp[ys, xs]
    order = np.argsort(vals)[::-1]
    ys, xs = ys[order], xs[order]

    # greedy min-distance enforcement via an occupancy grid (cells of side
    # min_distance; a new corner is rejected if any kept corner in the 3x3
    # neighbouring cells is closer than min_distance).
    md = max(float(min_distance), 1.0)
    md2 = md * md
    gw = int(W / md) + 1
    gh = int(H / md) + 1
    grid: list[list[tuple[float, float]]] = [[] for _ in range(gw * gh)]
    # pre-seed the grid with already-tracked points so fresh corners keep their
    # distance from them (replaces the cv2.circle exclusion mask).
    if exclude is not None:
        for ex, ey in np.asarray(exclude, dtype=np.float32).reshape(-1, 2):
            gx = int(ex / md)
            gy = int(ey / md)
            if 0 <= gx < gw and 0 <= gy < gh:
                grid[gy * gw + gx].append((float(ex), float(ey)))
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        cx = int(x / md)
        cy = int(y / md)
        ok = True
        for gy in range(max(0, cy - 1), min(gh, cy + 2)):
            for gx in range(max(0, cx - 1), min(gw, cx + 2)):
                for px, py in grid[gy * gw + gx]:
                    dx = px - x
                    dy = py - y
                    if dx * dx + dy * dy < md2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid[cy * gw + cx].append((float(x), float(y)))
            out_x.append(float(x))
            out_y.append(float(y))
            if len(out_x) >= max_corners:
                break

    return np.stack([out_x, out_y], axis=1).astype(np.float32)
