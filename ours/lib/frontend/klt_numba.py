"""Numba-JIT core for the per-point Lucas-Kanade inner loop.

The pure-NumPy :func:`ours.vio.klt.calc_optical_flow_pyr_lk` is *correct* (proved
by ``ours/tools/klt_selftest.py``) but ~10x too slow for live use, because the
coarse-to-fine Gauss-Newton iteration is inherently sequential: each step depends
on the previous displacement, so NumPy's whole-array vectorisation cannot hide
the per-iteration Python overhead or the temporary-array churn.

This module reimplements *only that hot loop* as explicit scalar loops and lets
Numba (an LLVM JIT) compile them to machine code. It is the same Bouguet
algorithm, byte-for-byte in intent, so the gold self-test still passes; Numba
just removes the interpreter overhead (and auto-vectorises the inner window
sums). The image pyramid + gradients are still built with NumPy in ``klt.py``
(whole-image ops, already fast) and passed in per level.

Numba is an *optional* dependency: ``klt.py`` imports :data:`HAVE_NUMBA` and
falls back to the pure-NumPy path when it is missing, so nothing here is required
to run. Importantly, we still own the whole algorithm -- Numba only accelerates
our own code, it does not implement the tracker for us (unlike calling cv2).
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange  # type: ignore
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only when numba is absent
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        """No-op fallback so the module imports without numba installed."""
        def wrap(fn):
            return fn
        # support both @njit and @njit(...) forms
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return wrap

    prange = range  # type: ignore


@njit(cache=True, fastmath=True)
def _bilinear_scalar(img, x, y):
    """Bilinear sample with edge clamping (matches klt._bilinear value path)."""
    H, W = img.shape
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    wx = x - x0
    wy = y - y0
    if x0 < 0:
        x0 = 0
    elif x0 > W - 1:
        x0 = W - 1
    if x1 < 0:
        x1 = 0
    elif x1 > W - 1:
        x1 = W - 1
    if y0 < 0:
        y0 = 0
    elif y0 > H - 1:
        y0 = H - 1
    if y1 < 0:
        y1 = 0
    elif y1 > H - 1:
        y1 = H - 1
    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]
    return (Ia * (1.0 - wx) * (1.0 - wy) + Ib * wx * (1.0 - wy)
            + Ic * (1.0 - wx) * wy + Id * wx * wy)


@njit(cache=True, fastmath=True, parallel=True)
def _track_level(Ip, Ic, Ix, Iy, prev_x, prev_y, guess_x, guess_y, bad,
                 scale, hw, iters, eps, min_eig):
    """One pyramid level: refine ``guess`` (full-res flow) for every point.

    Operates in place on ``guess_x``/``guess_y``/``bad``. ``prev_x``/``prev_y``
    are the full-resolution start coords; this level works at ``scale = 1/2**l``.
    Mirrors the per-level body of :func:`klt.calc_optical_flow_pyr_lk`.
    """
    N = prev_x.shape[0]
    P = (2 * hw + 1) * (2 * hw + 1)
    eps2 = eps * eps
    for n in prange(N):
        if bad[n]:
            continue
        px = prev_x[n] * scale
        py = prev_y[n] * scale

        # Fixed previous/gradient patches + structure tensor (computed once per
        # level, like the NumPy version which precomputes Ip_patch/Ixp/Iyp).
        ip_patch = np.empty(P)
        ixp = np.empty(P)
        iyp = np.empty(P)
        Gxx = 0.0
        Gxy = 0.0
        Gyy = 0.0
        k = 0
        for j in range(-hw, hw + 1):
            for i in range(-hw, hw + 1):
                sx = px + i
                sy = py + j
                ip_patch[k] = _bilinear_scalar(Ip, sx, sy)
                ix = _bilinear_scalar(Ix, sx, sy)
                iy = _bilinear_scalar(Iy, sx, sy)
                ixp[k] = ix
                iyp[k] = iy
                Gxx += ix * ix
                Gxy += ix * iy
                Gyy += iy * iy
                k += 1
        det = Gxx * Gyy - Gxy * Gxy
        tr = Gxx + Gyy
        disc = tr * tr - 4.0 * det
        if disc < 0.0:
            disc = 0.0
        min_eigen = (tr - np.sqrt(disc)) * 0.5 / P
        if min_eigen < min_eig or abs(det) <= 1e-12:
            bad[n] = True
            continue
        inv_det = 1.0 / det

        gx = guess_x[n] * scale
        gy = guess_y[n] * scale
        for _ in range(iters):
            bx = 0.0
            by = 0.0
            k = 0
            for j in range(-hw, hw + 1):
                for i in range(-hw, hw + 1):
                    jc = _bilinear_scalar(Ic, px + gx + i, py + gy + j)
                    dI = ip_patch[k] - jc
                    bx += ixp[k] * dI
                    by += iyp[k] * dI
                    k += 1
            ex = (Gyy * bx - Gxy * by) * inv_det
            ey = (-Gxy * bx + Gxx * by) * inv_det
            gx += ex
            gy += ey
            if ex * ex + ey * ey < eps2:
                break

        guess_x[n] = gx / scale
        guess_y[n] = gy / scale
