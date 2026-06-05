"""Pure-NumPy pyramidal Lucas-Kanade optical flow.

This is our own implementation of sparse pyramidal Lucas-Kanade (KLT) tracking,
written so the visual frontend does not depend on ``cv2.calcOpticalFlowPyrLK``.
It follows the standard Bouguet formulation (the same one OpenCV implements):

    For a point ``p`` we seek the displacement ``d`` minimising the windowed
    photometric error ``sum_w ( I(p + w) - J(p + d + w) )^2`` where ``I`` is the
    previous frame and ``J`` the current frame. A Gauss-Newton step gives

        G · η = b,
        G = sum_w [[Ix^2, Ix·Iy], [Ix·Iy, Iy^2]]   (the structure tensor of I)
        b = sum_w [Ix; Iy] · ( I(p+w) - J(p+d+w) )
        d ← d + η

    The spatial gradients ``Ix, Iy`` come from the *previous* frame and are
    therefore constant across the inner iterations (forward-additive LK), so the
    structure tensor ``G`` and its inverse are computed once per pyramid level.

A coarse-to-fine image pyramid lets it follow large motions: the displacement
solved at a coarse level is propagated down and refined at finer levels.

Everything is vectorised over *all* tracked points at once (the per-point window
patches are sampled with a single bilinear gather), so it stays reasonably fast
in pure NumPy. It is not as fast as OpenCV's hand-tuned C++/SIMD, but it tracks
the same corners to sub-pixel accuracy (validated against ``cv2`` in
``tools/klt_selftest.py``).
"""
from __future__ import annotations

import numpy as np

from .klt_numba import HAVE_NUMBA, _track_level

# Separable 5-tap Gaussian kernel ([1 4 6 4 1] / 16) used for pyramid
# downsampling -- the same low-pass OpenCV uses before decimating by 2.
_G5 = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0


def _sep_conv5(img: np.ndarray) -> np.ndarray:
    """Separable 5-tap Gaussian blur with reflected borders."""
    p = np.pad(img, ((0, 0), (2, 2)), mode="reflect")
    out = (_G5[0] * p[:, 0:-4] + _G5[1] * p[:, 1:-3] + _G5[2] * p[:, 2:-2]
           + _G5[3] * p[:, 3:-1] + _G5[4] * p[:, 4:])
    p = np.pad(out, ((2, 2), (0, 0)), mode="reflect")
    out = (_G5[0] * p[0:-4, :] + _G5[1] * p[1:-3, :] + _G5[2] * p[2:-2, :]
           + _G5[3] * p[3:-1, :] + _G5[4] * p[4:, :])
    return out


def build_pyramid(img: np.ndarray, max_level: int) -> list[np.ndarray]:
    """Gaussian image pyramid, level 0 = full resolution (float32)."""
    pyr = [img.astype(np.float32)]
    for _ in range(max_level):
        pyr.append(_sep_conv5(pyr[-1])[::2, ::2])
    return pyr


def _gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference spatial gradients (replicated borders)."""
    Ix = np.zeros_like(img)
    Iy = np.zeros_like(img)
    Ix[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    Ix[:, 0] = img[:, 1] - img[:, 0]
    Ix[:, -1] = img[:, -1] - img[:, -2]
    Iy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    Iy[0, :] = img[1, :] - img[0, :]
    Iy[-1, :] = img[-1, :] - img[-2, :]
    return Ix, Iy


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear-sample ``img`` at float coords; returns (values, valid_mask)."""
    H, W = img.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)
    wx = x - x0
    wy = y - y0
    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]
    val = (Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy)
           + Ic * (1 - wx) * wy + Id * wx * wy)
    return val, valid


def calc_optical_flow_pyr_lk(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    prev_pts: np.ndarray,
    win_size: int = 21,
    max_level: int = 3,
    iters: int = 30,
    eps: float = 0.01,
    min_eig: float = 1e-4,
    use_numba: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Track ``prev_pts`` from ``prev_gray`` into ``cur_gray`` with pyramidal LK.

    Drop-in for ``cv2.calcOpticalFlowPyrLK(prev, cur, prev_pts, None, ...)``:
    returns ``(next_pts, status)`` where ``next_pts`` is ``(N, 2)`` float32 and
    ``status`` is ``(N,)`` uint8 (1 = tracked, 0 = lost). Points whose structure
    tensor is rank-deficient (min eigenvalue below ``min_eig``) or that leave the
    image are marked lost.

    ``use_numba`` selects the per-point inner-loop backend: ``True`` uses the
    Numba-JIT scalar core (``klt_numba``), ``False`` the pure-NumPy vectorised
    path. ``None`` (default) auto-picks Numba when it is installed -- same
    algorithm and (to floating-point tolerance) same result, just ~10x faster.
    """
    if use_numba is None:
        use_numba = HAVE_NUMBA
    use_numba = use_numba and HAVE_NUMBA

    prev_pts = np.asarray(prev_pts, dtype=np.float32).reshape(-1, 2)
    N = prev_pts.shape[0]
    if N == 0:
        return prev_pts.copy(), np.zeros((0,), np.uint8)

    pyr_p = build_pyramid(prev_gray, max_level)
    pyr_c = build_pyramid(cur_gray, max_level)
    hw = win_size // 2

    if use_numba:
        return _calc_flow_numba(pyr_p, pyr_c, prev_pts, hw, max_level,
                                iters, eps, min_eig, prev_gray.shape)

    off = np.arange(-hw, hw + 1, dtype=np.float32)
    ox, oy = np.meshgrid(off, off)
    ox = ox.reshape(-1)[None, :]          # (1, P)
    oy = oy.reshape(-1)[None, :]
    P = ox.shape[1]

    guess = np.zeros((N, 2), dtype=np.float32)   # total flow in full-res coords
    bad = np.zeros(N, dtype=bool)

    for lvl in range(max_level, -1, -1):
        scale = 1.0 / (2 ** lvl)
        Ip = pyr_p[lvl]
        Ic = pyr_c[lvl]
        Ix, Iy = _gradients(Ip)

        p = prev_pts * scale                      # (N, 2) at this level
        px = p[:, 0:1] + ox                        # (N, P)
        py = p[:, 1:2] + oy

        # Previous patch + gradient patches (fixed across inner iterations).
        Ip_patch, _ = _bilinear(Ip, px, py)
        Ixp, _ = _bilinear(Ix, px, py)
        Iyp, _ = _bilinear(Iy, px, py)

        # Structure tensor per point + its (precomputed) inverse.
        Gxx = np.einsum("np,np->n", Ixp, Ixp)
        Gxy = np.einsum("np,np->n", Ixp, Iyp)
        Gyy = np.einsum("np,np->n", Iyp, Iyp)
        det = Gxx * Gyy - Gxy * Gxy
        tr = Gxx + Gyy
        min_eigen = (tr - np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))) * 0.5 / P
        bad |= min_eigen < min_eig
        safe = np.abs(det) > 1e-12
        inv_det = np.zeros_like(det)
        inv_det[safe] = 1.0 / det[safe]

        g = guess * scale                          # flow guess at this level
        # Active-set Gauss-Newton: iterate only over points that have not yet
        # converged (or been marked bad). Most points settle in a few
        # iterations, so the active set shrinks fast and the per-iteration cost
        # drops -- this is what makes the pure-NumPy loop tractable.
        active = (~bad) & safe
        eps2 = eps * eps
        for _ in range(iters):
            idx = np.flatnonzero(active)
            if idx.size == 0:
                break
            cx = p[idx, 0:1] + g[idx, 0:1] + ox
            cy = p[idx, 1:2] + g[idx, 1:2] + oy
            Ic_patch, _ = _bilinear(Ic, cx, cy)
            dI = Ip_patch[idx] - Ic_patch          # residual I - J(.+d)
            bx = np.einsum("np,np->n", Ixp[idx], dI)
            by = np.einsum("np,np->n", Iyp[idx], dI)
            # eta = G^{-1} b
            ex = (Gyy[idx] * bx - Gxy[idx] * by) * inv_det[idx]
            ey = (-Gxy[idx] * bx + Gxx[idx] * by) * inv_det[idx]
            g[idx, 0] += ex
            g[idx, 1] += ey
            converged = (ex * ex + ey * ey) < eps2
            active[idx[converged]] = False

        guess = g / scale                          # carry to next finer level

    nxt = prev_pts + guess
    H, W = prev_gray.shape
    in_bounds = ((nxt[:, 0] >= 0) & (nxt[:, 0] < W)
                 & (nxt[:, 1] >= 0) & (nxt[:, 1] < H))
    status = (~bad) & in_bounds
    return nxt.astype(np.float32), status.astype(np.uint8)


def _calc_flow_numba(pyr_p, pyr_c, prev_pts, hw, max_level,
                     iters, eps, min_eig, shape):
    """Numba-backed coarse-to-fine driver (same algorithm as the NumPy path).

    Builds per-level gradients with NumPy (whole-image, fast) then hands the
    sequential per-point Gauss-Newton refinement to the JIT-compiled
    :func:`klt_numba._track_level`, which is where the speed-up lives.
    """
    N = prev_pts.shape[0]
    prev_x = np.ascontiguousarray(prev_pts[:, 0], dtype=np.float64)
    prev_y = np.ascontiguousarray(prev_pts[:, 1], dtype=np.float64)
    guess_x = np.zeros(N, dtype=np.float64)
    guess_y = np.zeros(N, dtype=np.float64)
    bad = np.zeros(N, dtype=np.bool_)

    for lvl in range(max_level, -1, -1):
        scale = 1.0 / (2 ** lvl)
        Ip = np.ascontiguousarray(pyr_p[lvl], dtype=np.float64)
        Ic = np.ascontiguousarray(pyr_c[lvl], dtype=np.float64)
        Ix, Iy = _gradients(Ip)
        _track_level(Ip, Ic, np.ascontiguousarray(Ix), np.ascontiguousarray(Iy),
                     prev_x, prev_y, guess_x, guess_y, bad,
                     scale, hw, iters, eps, min_eig)

    nxt = np.empty((N, 2), dtype=np.float32)
    nxt[:, 0] = prev_x + guess_x
    nxt[:, 1] = prev_y + guess_y
    H, W = shape
    in_bounds = ((nxt[:, 0] >= 0) & (nxt[:, 0] < W)
                 & (nxt[:, 1] >= 0) & (nxt[:, 1] < H))
    status = (~bad) & in_bounds
    return nxt, status.astype(np.uint8)

