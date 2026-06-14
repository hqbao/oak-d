"""``sky.front.direct`` -- dense DIRECT RGB-D visual odometry (Stage-1 prototype).

WHY THIS EXISTS
---------------
At the 54x42 VL53-class ToF target the SPARSE corner/KLT VIO front-end suffers
**scale collapse** (measured Sim3 scale 0.23-0.63 against Basalt) and feature
dropouts (okfrac 0.54-0.81), giving ATE 50-98 cm vs 10-18 cm at full-res. The
root cause is feature starvation: only ~2300 px to find corners in. There simply
are not enough trackable corners for triangulation to fix the metric scale, so
the windowed BA / VO-translation-prior path under-estimates motion.

The research lever (Steinbrucker 2011; Kerl/Sturm/Cremers ICRA'13 "Robust
Odometry Estimation for RGB-D Cameras"; Whelan ICRA'13 "Robust Real-Time Visual
Odometry for Dense RGB-D Mapping") is **dense direct photometric alignment**:
align EVERY pixel that has a gradient (not just corners) using the ACCURATE
per-pixel ToF depth. Because the depth is given (metric), the pose is a pure
6-DoF SE(3) and the **scale is OBSERVED from the depth** rather than estimated
from feature triangulation -- which is exactly what should kill the scale
collapse, while using all gradient pixels kills the starvation.

This module is the from-scratch implementation of that estimator. It is a
research prototype: it estimates a single frame-to-keyframe relative pose, it
does NOT touch the frozen loose/tight live path, and it is exercised only by the
offline harness ``verification/direct_vo_bench.py``.

THE FORMULATION (so it can be checked by review)
------------------------------------------------
Unknown: ``T_cur_ref`` in SE(3) -- the rigid transform mapping a 3D point
expressed in the REFERENCE camera frame into the CURRENT camera frame.

For each reference pixel ``p = (u, v)`` that has a valid depth ``Z = depth_ref(p)``:

1. back-project:  ``P_ref = Z * K^{-1} [u, v, 1]^T``                  (ref frame)
2. transform:     ``P_cur = T_cur_ref @ P_ref``  (homogeneous)        (cur frame)
3. project:       ``w = pi(P_cur) = (fx X/Zc + cx, fy Y/Zc + cy)``    (cur pixels)
4. residual:      ``r(p) = I_cur(w) - I_ref(p)``   (bilinear-sampled intensities)

We minimise ``sum_p rho( w_p * r(p)^2 )`` over the SE(3) twist using
Gauss-Newton with a LEFT perturbation on the current estimate:

    ``T_cur_ref  <-  Exp(dxi) @ T_cur_ref``,   ``dxi = [rho(3); phi(3)]``.

The per-pixel Jacobian of the residual w.r.t. ``dxi`` (evaluated at ``dxi = 0``)
is the chain rule

    ``J_p = g_p^T  *  J_pi(P_cur)  *  J_warp``

where
  * ``g_p = [I_cur_x(w), I_cur_y(w)]`` -- image gradient of the CURRENT image at
    the warped location (px / px), obtained by Sobel + bilinear sampling.
  * ``J_pi`` -- the 2x3 Jacobian of the pinhole projection ``pi`` w.r.t. the 3D
    point ``P_cur = (X, Y, Z)``:
        ``[[fx/Z,    0,   -fx X / Z^2],
          [  0,   fy/Z,   -fy Y / Z^2]]``
  * ``J_warp`` -- the 3x6 Jacobian of ``P_cur`` w.r.t. the LEFT twist ``dxi``.
    For a left perturbation ``Exp(dxi) @ T`` acting on a point already at
    ``P_cur`` this is ``d(Exp(dxi) P_cur)/d dxi |_0 = [ I_3 | -skew(P_cur) ]``
    (translation-first twist order, matching :mod:`sky.math`). The minus sign on
    the rotation block is the derivative of ``skew(phi) @ P = -skew(P) @ phi``.

So ``J_p = g_p^T @ J_pi @ [I_3 | -skew(P_cur)]`` is a 1x6 row. The Gauss-Newton
normal equations accumulate ``H = sum_p w_p J_p^T J_p`` (6x6) and
``b = -sum_p w_p J_p r(p)`` (6x1); the update is ``dxi = solve(H, b)`` (with a
small Levenberg-Marquardt diagonal damping for conditioning), applied on the
left. We iterate to convergence per pyramid level, COARSE -> FINE.

THE GEOMETRIC (POINT-TO-PLANE) TERM -- Stage-2b (Whelan ICRA'13)
----------------------------------------------------------------
Photometric alignment weakly constrains the FORWARD (focus-of-expansion)
translation DoF on fast frames: moving along the optical axis barely changes the
image away from the FOE, so ``E_photo`` is nearly flat in that direction. Whelan
ICRA'13 fuses a point-to-plane ICP residual into the SAME Gauss-Newton solve --
``E = E_photo + w * E_geo`` -- because the geometric term reads that DoF straight
from the depth, and the two terms are robust on COMPLEMENTARY degeneracies
(``E_geo`` on flat-but-textured, ``E_photo`` on textured-but-flat).

The geometric term needs the CURRENT frame's depth too (hence ``estimate_pose_
direct`` now also takes ``depth_cur``). For each reference pixel with valid depth:

1. ``P_cur_est = T_cur_ref @ P_ref``   -- the SAME warped point the photometric
   term already computes (ref point pushed into the current frame).
2. project ``P_cur_est`` to a current pixel ``(u, v)`` (the same projection).
3. measured point: ``P_cur_meas = depth_cur(u, v) * K^{-1} [u, v, 1]^T`` -- back-
   project the CURRENT depth at the warped pixel (bilinear depth lookup).
4. surface normal ``n`` at ``(u, v)``: from the local depth gradient of the
   current depth map (cross-product of the two neighbouring back-projected rays;
   see :func:`_current_normals`), normalised. This is the plane the measured
   point lies on.
5. residual: ``r_geo = n . (P_cur_est - P_cur_meas)``  -- the signed distance of
   the warped reference point from the measured surface tangent plane.

Standard ICP linearisation freezes the correspondence per iteration: ``n`` and
``P_cur_meas`` are treated as CONSTANTS w.r.t. the pose (re-evaluated each
iteration after re-projecting), so ``r_geo`` depends on ``dxi`` only through
``P_cur_est``. With the LEFT perturbation ``Exp(dxi) @ T`` acting on the already-
warped point, ``d P_cur_est / d dxi = [ I_3 | -skew(P_cur_est) ]`` (the same warp
Jacobian the photometric term uses), so

    ``J_geo = n^T @ [ I_3 | -skew(P_cur_est) ]
            = [ n^T | (P_cur_est x n)^T ]``     (a 1x6 row)

using ``n^T (-skew(P)) = (skew(P) n)^T = (P x n)^T``. The two terms accumulate
into the SAME 6x6 normal equations, the geometric block scaled by the fused
weight ``w`` (``DirectConfig.geo_weight``, Whelan's ``w ~= 0.1`` metric-scale-
reconciling default): ``H = H_photo + w * H_geo``, ``b = b_photo + w * b_geo``.
The geo residual is robustly re-weighted too (its own t-scale, in METRES).

DIVERGENCE GUARD -- Stage-2b (protect the seed velocity)
--------------------------------------------------------
A sustained VO divergence poisons the IMU dead-reckoner's velocity (the
documented Stage-2a ``quick_motion`` failure). The estimator therefore reports a
``diverged`` signal and the raw numbers behind it so the CALLER (the harness)
can reject the VO pose and fall back to the IMU-only prediction for that frame --
crucially feeding the dead-reckoner the IMU-only prediction, NOT the diverged VO
fix, so a bad stretch cannot corrupt the velocity. ``diverged`` is True when, at
the finest level, the final fused residual was NOT reduced below the initial one
(GN ran uphill) OR the total applied pose step is implausibly large relative to
the seed (a runaway). The thresholds + the raw ``step_norm`` / ``rmse_ratio``
are exposed in ``info`` for the caller (and for honest sensitivity reporting).

ROBUST WEIGHTING (Kerl)
-----------------------
The photometric residual is heavy-tailed (occlusion, ToF depth holes, moving
edges). We use the iteratively-reweighted **Student-t** weight Kerl prescribes:
``w(r) = (nu + 1) / (nu + (r / sigma)^2)`` with ``nu = 5`` and ``sigma``
re-estimated each iteration from the residuals (the t-distribution scale). A
Huber weight is also provided as a fallback. Per-pixel depth validity (and the
warp falling inside the current image, with positive projected depth) gates which
pixels contribute at all. The geometric residual gets its OWN t-scale (it lives
in metres, not intensity units) so the two robust weightings do not interfere.

LEAF / PORT RULES
-----------------
Keeps ``sky.*`` a leaf: imports only ``numpy`` and :mod:`sky.math`. The image
gradients (Sobel) and the Gaussian image pyramid (pyrDown) are pure-NumPy
separable convolutions (:func:`_sobel3`, :func:`_pyr_down`) that reproduce the
matching ``cv2`` ops bit-for-bit (``BORDER_REFLECT_101``), so the flight runtime
needs NO OpenCV. ``sky.assert_import_clean()`` passes. No process / comms / io
module is reachable. Maps onto the C ``libskyfront`` layer alongside the KLT/PnP
front-end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sky.math import se3_exp

__all__ = [
    "DirectConfig",
    "estimate_pose_direct",
    "build_pyramid",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DirectConfig:
    """Tunables for :func:`estimate_pose_direct` (all with research defaults)."""

    levels: int = 3
    """Number of image/depth pyramid levels (level 0 = full resolution)."""

    max_iters: int = 30
    """Max Gauss-Newton iterations PER pyramid level."""

    min_grad: float = 4.0
    """Reference pixels whose CURRENT-image gradient magnitude (at the warp) is
    below this are kept but contribute little; we instead pre-select reference
    pixels by their gradient to focus the solve on informative pixels. This is
    the |grad| threshold (intensity units / px) for that pre-selection."""

    huber_delta: float = 4.0
    """Huber transition (intensity units). Only used when ``robust='huber'``."""

    t_dof: float = 5.0
    """Student-t degrees of freedom ``nu`` (Kerl uses 5). Only for ``robust='t'``."""

    robust: str = "t"
    """Robust weight: ``'t'`` (Student-t, Kerl -- default) or ``'huber'``."""

    convergence_eps: float = 1e-6
    """Stop a level early once ``||dxi||^2`` drops below this."""

    lm_damping: float = 1e-3
    """Levenberg-Marquardt diagonal damping factor ``lambda`` added as
    ``lambda * diag(H)`` for numerical conditioning of the 6x6 solve."""

    min_valid_frac: float = 0.02
    """If fewer than this fraction of selected pixels remain usable at a level
    (after validity + in-bounds gating), skip the level (too little signal)."""

    depth_min: float = 0.05
    """Minimum valid depth (m). Depths <= this are treated as invalid holes."""

    depth_max: float = 30.0
    """Maximum valid depth (m). Depths >= this are treated as invalid."""

    max_pixels: int = 6000
    """Cap on selected reference pixels per level (sub-sample the highest-gradient
    ones beyond this) to bound the per-iteration cost. 0 == no cap."""

    # Convenience: how many top-gradient pixels to *select* per level as a
    # fraction of the valid set (1.0 == use all valid). Kept conservative so the
    # solve is driven by textured regions, matching the direct-VO literature.
    grad_select_frac: float = 1.0

    # ---- Stage-2b: fused point-to-plane GEOMETRIC term (Whelan ICRA'13) ------ #
    geo_weight: float = 0.1
    """Fused weight ``w`` of the point-to-plane geometric block in
    ``E = E_photo + w * E_geo`` (Whelan's metric-scale-reconciling default ~0.1).
    Set to 0.0 to disable the geometric term entirely (pure photometric, the
    Stage-2a behaviour -- used to prove the photometric path is unregressed)."""

    geo_normal_max_angle_deg: float = 75.0
    """Reject a geometric correspondence whose estimated surface normal is more
    than this far from facing the camera (``|cos| < cos(angle)``): a grazing /
    ill-conditioned normal gives a meaningless point-to-plane distance."""

    geo_max_dist_m: float = 0.5
    """Hard gate (m) on the point-to-plane distance: a correspondence further than
    this from the measured surface is treated as an outlier (occlusion / wrong
    match) and dropped before the robust weighting, so a gross mismatch cannot
    dominate ``H_geo``."""

    # ---- Stage-2b: divergence GUARD (reported, the caller decides) ----------- #
    diverge_step_ratio: float = 8.0
    """Flag ``diverged`` when the total applied pose step at the finest level
    exceeds this multiple of the seed-implied step magnitude (a runaway GN). The
    seed step is the translation of ``init_T`` relative to identity; a small floor
    keeps a near-stationary seed from making the ratio explode spuriously."""

    diverge_seed_floor_m: float = 0.02
    """Floor (m) on the seed step used in the ``diverge_step_ratio`` test, so a
    near-zero seed translation does not make the ratio blow up on tiny real
    motion. Roughly one inter-frame translation at slow speed."""

    diverge_rmse_grow: float = 1.05
    """Flag ``diverged`` when the finest-level final photometric RMSE is MORE than
    this multiple of the level's initial RMSE (Gauss-Newton ran uphill -- the
    alignment got worse, the unmistakable signature of a divergent solve)."""


# --------------------------------------------------------------------------- #
# Pyramid construction (valid-aware depth reduction)
# --------------------------------------------------------------------------- #
def build_pyramid(
    gray: np.ndarray,
    depth: np.ndarray | None,
    K: np.ndarray,
    levels: int,
    *,
    depth_min: float = 0.05,
    depth_max: float = 30.0,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Build a ``levels``-level (gray, depth, K) pyramid, coarse handled by caller.

    Returned list is ordered FINE -> COARSE: index 0 is full resolution, index
    ``levels-1`` is the coarsest. Each entry is ``(gray_f32, depth_or_None, K_l)``.

    * GRAY is reduced with :func:`_pyr_down` (5-tap Gaussian + 2x decimation, a
      pure-NumPy bit-exact ``cv2.pyrDown`` reimplementation), the standard
      intensity reduction, returned as float32.
    * DEPTH is reduced VALID-AWARE: a 2x2 block reduces to the MEDIAN of its
      valid (in ``[depth_min, depth_max]``) members, or 0 (invalid) if the whole
      block is holes. This never blends a real depth with a 0-hole or across a
      depth discontinuity -- a naive blur/pyrDown would, corrupting the metric
      back-projection that gives this method its scale.
    * K is scaled per level: ``fx, fy, cx, cy`` are halved each downscale (the
      standard pinhole-under-decimation rule, with the +0.5/-0.5 pixel-centre
      convention folded in: ``c' = (c + 0.5)/2 - 0.5``).

    ``depth`` may be None (the CURRENT frame needs only intensity, not depth);
    then every level's depth entry is None.
    """
    g0 = np.ascontiguousarray(gray, dtype=np.float32)
    d0 = None if depth is None else np.ascontiguousarray(depth, dtype=np.float32)
    K0 = np.asarray(K, dtype=np.float64).copy()

    pyr: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = [(g0, d0, K0)]
    for _ in range(1, levels):
        g_prev, d_prev, K_prev = pyr[-1]
        # _pyr_down DEFINES the canonical level size ((W+1)//2 x (H+1)//2 for
        # odd dims, matching cv2.pyrDown). We reduce depth valid-aware to THAT
        # EXACT shape so the gray and depth grids never disagree (the 54x42
        # odd-dim trap).
        g_next = _pyr_down(g_prev)
        nh, nw = g_next.shape[:2]
        d_next = (None if d_prev is None
                  else _downsample_depth_valid(d_prev, nh, nw, depth_min, depth_max))
        # Pinhole K under 2x decimation with the pixel-centre convention.
        K_next = K_prev.copy()
        K_next[0, 0] *= 0.5                       # fx
        K_next[1, 1] *= 0.5                       # fy
        K_next[0, 2] = (K_prev[0, 2] + 0.5) * 0.5 - 0.5   # cx
        K_next[1, 2] = (K_prev[1, 2] + 0.5) * 0.5 - 0.5   # cy
        pyr.append((g_next, d_next, K_next))
    return pyr


def _downsample_depth_valid(
    depth: np.ndarray, out_h: int, out_w: int, dmin: float, dmax: float
) -> np.ndarray:
    """Valid-aware 2x depth reduction to an EXACT ``(out_h, out_w)`` target shape.

    Each output cell takes the MEDIAN of its source 2x2 block's VALID members
    (``dmin < d < dmax``); a block with no valid member maps to 0 (invalid). The
    target shape is taken from the matching ``cv2.pyrDown`` gray level so the two
    grids stay aligned even at odd dims (``out = (in+1)//2``): we pad the source
    by one row/col when needed so the 2x2 blocks tile exactly ``out_h x out_w``.
    This never blends a real depth with a 0-hole or across a depth edge -- a naive
    blur/pyrDown would, corrupting the metric back-projection.
    """
    h, w = depth.shape[:2]
    need_h, need_w = out_h * 2, out_w * 2
    d = depth.astype(np.float32)
    # Pad (edge-replicate) up to an exact 2x tiling of the target shape.
    if need_h > h or need_w > w:
        d = np.pad(d, ((0, max(0, need_h - h)), (0, max(0, need_w - w))),
                   mode="edge")
    d = d[:need_h, :need_w]
    valid = (d > dmin) & (d < dmax)
    # Stack the four 2x2-block members -> (out_h, out_w, 4).
    blocks = np.stack(
        [d[0::2, 0::2], d[0::2, 1::2], d[1::2, 0::2], d[1::2, 1::2]], axis=-1
    )
    vblocks = np.stack(
        [valid[0::2, 0::2], valid[0::2, 1::2],
         valid[1::2, 0::2], valid[1::2, 1::2]], axis=-1
    )
    # Median of the VALID members per block, computed warning-free (no nanmedian,
    # which emits an "All-NaN slice" RuntimeWarning on the all-holes blocks the ToF
    # depth is full of). Trick: push invalid members to +inf and SORT each block;
    # the valid values then occupy the first `cnt` slots in ascending order, so the
    # median is the average of the two middle valid slots (selected by `cnt`).
    cnt = vblocks.sum(axis=-1)                       # (out_h, out_w) valid count 0..4
    srt = np.sort(np.where(vblocks, blocks, np.inf), axis=-1)  # valids first, asc
    # Lower/upper middle indices of the valid run (clamped; only used where cnt>0).
    lo = np.clip((cnt - 1) // 2, 0, 3)
    hi = np.clip(cnt // 2, 0, 3)
    ii, jj = np.indices(cnt.shape)
    med = 0.5 * (srt[ii, jj, lo] + srt[ii, jj, hi])
    return np.where(cnt > 0, med, 0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Bilinear sampling + gradients
# --------------------------------------------------------------------------- #
def _bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Bilinear-sample ``img`` (HxW float) at floating (u, v); vectorised.

    Returns ``(values, valid)`` where ``valid`` is the boolean mask of samples
    whose 2x2 support lies fully inside the image. Out-of-bounds samples return 0
    (and ``valid=False``) so the caller can drop them.
    """
    h, w = img.shape[:2]
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1

    valid = (u0 >= 0) & (v0 >= 0) & (u1 <= w - 1) & (v1 <= h - 1)
    # Clamp indices so the gather is always in-range; invalid entries are masked
    # out by the caller via `valid`, so their clamped value is irrelevant.
    u0c = np.clip(u0, 0, w - 1)
    u1c = np.clip(u1, 0, w - 1)
    v0c = np.clip(v0, 0, h - 1)
    v1c = np.clip(v1, 0, h - 1)

    wu = u - u0
    wv = v - v0
    Ia = img[v0c, u0c]
    Ib = img[v0c, u1c]
    Ic = img[v1c, u0c]
    Id = img[v1c, u1c]
    top = Ia * (1.0 - wu) + Ib * wu
    bot = Ic * (1.0 - wu) + Id * wu
    val = top * (1.0 - wv) + bot * wv
    return val, valid


def _reflect101_pad(img: np.ndarray, pad: int) -> np.ndarray:
    """Pad ``img`` by ``pad`` on every side with ``BORDER_REFLECT_101``.

    Reflect-101 mirrors WITHOUT repeating the edge sample (``...c b | a b c ...``,
    NOT ``...b a | a b c``), which is OpenCV's default border for both
    ``cv2.Sobel`` and ``cv2.pyrDown``. NumPy's ``mode="reflect"`` is exactly this
    convention, so the convolutions below match cv2 at the borders too.
    """
    return np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")


# Separable Sobel-3 kernels: gx = smooth_y [1,2,1] outer diff_x [-1,0,1] (and the
# transpose for gy). The /8 normalisation makes the operator a central difference
# in magnitude (cv2.Sobel(...) / 8), matching the prior behaviour exactly.
_SOBEL_SMOOTH = np.array([1.0, 2.0, 1.0], dtype=np.float64)
_SOBEL_DIFF = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
# pyrDown 5-tap binomial Gaussian (normalised), applied separably; cv2 uses the
# same [1,4,6,4,1]/16 kernel before dropping the odd rows/cols.
_PYR_KERNEL = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0


def _sep_conv(img: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Separable 2-D correlation: kernel ``ky`` down the rows, ``kx`` across cols.

    Reflect-101 padding + a sliding-window dot makes this a drop-in for the cv2
    separable filters used here. ``img`` is treated as float64 internally for
    accuracy and returned float32 by the callers.
    """
    pad = len(kx) // 2
    p = _reflect101_pad(np.asarray(img, dtype=np.float64), pad)
    h, w = img.shape
    # Convolve along columns (x) first, then rows (y); both via shifted-add over
    # the small (3- or 5-tap) kernels -- cheap and exact, no scipy dependency.
    tmp = np.zeros((h + 2 * pad, w), dtype=np.float64)
    for i, c in enumerate(kx):
        if c != 0.0:
            tmp += c * p[:, i:i + w]
    out = np.zeros((h, w), dtype=np.float64)
    for j, c in enumerate(ky):
        if c != 0.0:
            out += c * tmp[j:j + h, :]
    return out


def _sobel3(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Pure-NumPy ``cv2.Sobel(img, CV_32F, dx, dy, ksize=3)`` (one of dx/dy is 1)."""
    if dx == 1:
        return _sep_conv(img, _SOBEL_DIFF, _SOBEL_SMOOTH)
    return _sep_conv(img, _SOBEL_SMOOTH, _SOBEL_DIFF)


def _pyr_down(img: np.ndarray) -> np.ndarray:
    """Pure-NumPy ``cv2.pyrDown(img)``: 5-tap Gaussian blur then 2x decimation.

    Blurs with the separable ``[1,4,6,4,1]/16`` binomial kernel (reflect-101
    border) and keeps the even rows/cols, giving the canonical ``((H+1)//2,
    (W+1)//2)`` output shape -- bit-exact with cv2.pyrDown on these float images.
    """
    blurred = _sep_conv(img, _PYR_KERNEL, _PYR_KERNEL)
    return np.ascontiguousarray(blurred[0::2, 0::2], dtype=np.float32)


def _image_gradients(img: np.ndarray):
    """Central-difference image gradients ``(gx, gy)`` (intensity / px), via Sobel.

    Uses a normalised Sobel (:func:`_sobel3` / 8, a pure-NumPy ``cv2.Sobel``
    reimplementation) so the result matches a central difference in magnitude.
    ``img`` is float32; returns two float32 arrays.
    """
    gx = (_sobel3(img, 1, 0) / 8.0).astype(np.float32)
    gy = (_sobel3(img, 0, 1) / 8.0).astype(np.float32)
    return gx, gy


def _current_normals(depth: np.ndarray, K: np.ndarray, dmin: float, dmax: float):
    """Per-pixel surface normals of a depth map, in the CAMERA frame.

    For the point-to-plane geometric term we need, at every current-frame pixel,
    the local surface-tangent plane the measured ToF point lies on. We obtain its
    normal the standard organised-point-cloud way (KinectFusion / Whelan): back-
    project the whole depth grid to a 3D point map ``P(u,v)``, then take the cross
    product of the two in-plane tangent vectors estimated by central differences
    along the image axes,

        ``n(u,v) = (P_{u+1} - P_{u-1}) x (P_{v+1} - P_{v-1})``  (then normalise),

    oriented to FACE the camera (``n_z < 0`` in the optical frame where +z points
    away from the camera, so we flip any normal whose z-component is positive).

    Returns ``(normals, normal_valid)`` where ``normals`` is ``(H, W, 3)`` and
    ``normal_valid`` is the ``(H, W)`` bool mask of pixels whose own depth AND
    all four cross-difference neighbours are valid (so the tangent vectors are
    real surface directions, not depth-edge / hole artefacts) and whose normal is
    non-degenerate. Invalid pixels carry a zero normal.
    """
    h, w = depth.shape[:2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Back-project the whole grid to a (H, W, 3) organised point map.
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    Z = depth.astype(np.float64)
    P = np.empty((h, w, 3), dtype=np.float64)
    P[..., 0] = (uu - cx) * Z / fx
    P[..., 1] = (vv - cy) * Z / fy
    P[..., 2] = Z

    valid = (Z > dmin) & (Z < dmax)

    # Central-difference tangents along the image x/y axes (in 3D). The interior
    # slices keep the (H, W, 3) shape; the 1px border is left invalid below.
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1, :] = P[:, 2:, :] - P[:, :-2, :]
    dv[1:-1, :, :] = P[2:, :, :] - P[:-2, :, :]

    n = np.cross(du, dv)                       # (H, W, 3), unnormalised normal
    norm = np.linalg.norm(n, axis=-1)          # (H, W)
    good = norm > 1e-9
    inv = np.zeros_like(norm)
    inv[good] = 1.0 / norm[good]
    n *= inv[..., None]

    # Orient toward the camera: optical +z points away, so a camera-facing surface
    # has n_z < 0; flip any normal pointing away.
    flip = n[..., 2] > 0.0
    n[flip] *= -1.0

    # A normal is trustworthy only where its own depth AND every neighbour used in
    # the two central differences is valid, and the cross product was non-degenerate.
    nbr_valid = np.zeros((h, w), dtype=bool)
    nbr_valid[1:-1, 1:-1] = (
        valid[1:-1, 1:-1]
        & valid[1:-1, 2:] & valid[1:-1, :-2]      # left/right (du)
        & valid[2:, 1:-1] & valid[:-2, 1:-1]      # up/down   (dv)
    )
    normal_valid = nbr_valid & good
    n[~normal_valid] = 0.0
    return n, normal_valid


# --------------------------------------------------------------------------- #
# Robust weights (Kerl Student-t, or Huber)
# --------------------------------------------------------------------------- #
def _t_scale_and_weights(r: np.ndarray, nu: float, n_iters: int = 5):
    """Student-t IRLS: estimate scale ``sigma`` then per-residual weights.

    Implements Kerl ICRA'13 eq. (7)-(9): the t-distribution variance is solved by
    the fixed-point iteration
        ``sigma^2 <- mean( r^2 * (nu + 1) / (nu + (r/sigma)^2) )``
    seeded from the residual MAD, then the weights are
        ``w(r) = (nu + 1) / (nu + (r/sigma)^2)``.
    Returns ``(weights, sigma)``.
    """
    if r.size == 0:
        return np.ones_like(r), 1.0
    # Robust seed for sigma (MAD -> Gaussian-consistent stdev).
    sigma2 = max(np.median(np.abs(r)) * 1.4826, 1e-3) ** 2
    for _ in range(n_iters):
        w = (nu + 1.0) / (nu + (r * r) / sigma2)
        new = float(np.mean(w * r * r))
        if new <= 1e-12:
            break
        if abs(new - sigma2) / max(sigma2, 1e-12) < 1e-3:
            sigma2 = new
            break
        sigma2 = new
    sigma = float(np.sqrt(max(sigma2, 1e-12)))
    w = (nu + 1.0) / (nu + (r * r) / (sigma * sigma))
    return w, sigma


def _huber_weights(r: np.ndarray, delta: float):
    """Huber IRLS weights ``w = 1`` for ``|r| <= delta`` else ``delta / |r|``."""
    a = np.abs(r)
    w = np.ones_like(r)
    big = a > delta
    w[big] = delta / np.maximum(a[big], 1e-12)
    return w


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
@dataclass
class _LevelCache:
    """Pre-computed, pose-independent reference quantities for one pyramid level."""

    P_ref: np.ndarray   # (M, 3) back-projected reference points (ref frame)
    I_ref: np.ndarray   # (M,)   reference intensities at the selected pixels
    K: np.ndarray       # (3, 3) this level's intrinsics
    shape: tuple        # (H, W) of the current image at this level


def estimate_pose_direct(
    gray_ref: np.ndarray,
    depth_ref: np.ndarray,
    gray_cur: np.ndarray,
    K: np.ndarray,
    *,
    depth_cur: np.ndarray | None = None,
    init_T: np.ndarray | None = None,
    levels: int = 3,
    max_iters: int = 30,
    cfg: DirectConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """Dense direct RGB-D odometry: estimate ``T_cur_ref`` (4x4 SE(3)).

    Aligns the CURRENT image to the REFERENCE frame's geometry by minimising the
    FUSED photometric + point-to-plane geometric residual over every selected
    reference pixel with valid depth, by coarse-to-fine Gauss-Newton on the SE(3)
    left twist (see the module docstring for the full formulation):
    ``E = E_photo + cfg.geo_weight * E_geo``.

    Parameters
    ----------
    gray_ref, gray_cur : (H, W) arrays (uint8 or float) -- reference & current
        intensity images, SAME resolution.
    depth_ref : (H, W) float -- metric depth (m) for the REFERENCE frame; 0 (or
        outside ``[cfg.depth_min, cfg.depth_max]``) marks invalid pixels.
    depth_cur : optional (H, W) float -- metric depth (m) for the CURRENT frame.
        Required for the Stage-2b point-to-plane GEOMETRIC term; when None (or
        ``cfg.geo_weight == 0``) the solve is pure photometric (Stage-2a). At the
        VL53 ToF target the current frame's depth is available, so the geometric
        term is on by default.
    K : (3, 3) pinhole intrinsics for this resolution.
    init_T : optional 4x4 SE(3) seed for ``T_cur_ref`` (e.g. an IMU/gyro prior);
        identity if None.
    levels, max_iters : pyramid depth and per-level GN iteration cap. If ``cfg``
        is given its ``levels`` / ``max_iters`` take precedence only when the
        positional args are left at the defaults; otherwise the explicit args win.
    cfg : optional :class:`DirectConfig` for the robust weight / thresholds.

    Returns
    -------
    (T_cur_ref, info) where ``info`` carries convergence + DIVERGENCE diagnostics:
        ``converged`` (bool), ``final_rmse`` (intensity units, finest level),
        ``iters`` (total GN steps over all levels), ``valid_frac`` (fraction of
        selected finest-level pixels that stayed usable), ``n_pixels`` (finest
        level), ``per_level`` (list of dicts), ``sigma`` (final photometric
        t-scale), ``geo_sigma`` (final geometric t-scale, m), ``n_geo`` (finest-
        level geometric correspondences used), and the Stage-2b divergence guard:
        ``diverged`` (bool -- the CALLER decides what to do), ``step_norm`` (m,
        total finest-level translation step the solve applied beyond the seed),
        ``step_ratio`` (``step_norm`` / seed step, vs ``cfg.diverge_step_ratio``),
        ``rmse_ratio`` (finest-level final/initial photometric RMSE, vs
        ``cfg.diverge_rmse_grow``).
    """
    cfg = cfg or DirectConfig()
    # Explicit positional args win over cfg defaults when the caller set them.
    if levels != 3:
        cfg = _with(cfg, levels=levels)
    if max_iters != 30:
        cfg = _with(cfg, max_iters=max_iters)
    L = int(cfg.levels)

    gref = np.ascontiguousarray(gray_ref, dtype=np.float32)
    gcur = np.ascontiguousarray(gray_cur, dtype=np.float32)
    dref = np.ascontiguousarray(depth_ref, dtype=np.float32)

    pyr_ref = build_pyramid(gref, dref, K, L,
                            depth_min=cfg.depth_min, depth_max=cfg.depth_max)
    # The CURRENT frame needs depth only when the geometric term is active; build
    # its valid-aware depth pyramid then (same reduction as the reference depth).
    use_geo = cfg.geo_weight > 0.0 and depth_cur is not None
    if use_geo:
        dcur = np.ascontiguousarray(depth_cur, dtype=np.float32)
        pyr_cur = build_pyramid(gcur, dcur, K, L,
                                depth_min=cfg.depth_min, depth_max=cfg.depth_max)
    else:
        pyr_cur = build_pyramid(gcur, None, K, L)

    # Working estimate of T_cur_ref (left-perturbed during the solve). Keep the
    # seed so the divergence guard can measure how far the solve moved from it.
    T_seed = np.eye(4) if init_T is None else np.asarray(init_T, dtype=np.float64)
    T = T_seed.copy()

    total_iters = 0
    per_level: list[dict] = []
    final_rmse = float("nan")
    final_valid_frac = 0.0
    final_n = 0
    final_sigma = float("nan")
    final_geo_sigma = float("nan")
    final_n_geo = 0
    final_rmse_ratio = float("nan")
    any_converged = False

    # COARSE -> FINE: iterate from the coarsest level (last in the FINE->COARSE
    # pyramid list) down to the finest (index 0).
    for lvl in range(L - 1, -1, -1):
        g_ref_l, d_ref_l, K_l = pyr_ref[lvl]
        g_cur_l, d_cur_l, _ = pyr_cur[lvl]
        gx_l, gy_l = _image_gradients(g_cur_l)
        # Pre-compute the current-frame surface normals + point map ONCE per level
        # for the point-to-plane term (pose-independent: they live in the current
        # camera frame, looked up at the warped pixel each iteration).
        geo_l = None
        if use_geo and d_cur_l is not None:
            n_cur_l, nvalid_l = _current_normals(
                d_cur_l, K_l, cfg.depth_min, cfg.depth_max)
            geo_l = (d_cur_l, n_cur_l, nvalid_l)

        cache = _select_reference_pixels(g_ref_l, d_ref_l, K_l, cfg)
        if cache is None:
            per_level.append({"level": lvl, "skipped": True, "reason": "no_valid_px"})
            continue

        T, lvl_info = _solve_level(
            T, cache, g_cur_l, gx_l, gy_l, K_l, cfg, geo_l)
        lvl_info["level"] = lvl
        per_level.append(lvl_info)
        total_iters += lvl_info["iters"]
        any_converged = any_converged or lvl_info["converged"]
        if lvl == 0:
            final_rmse = lvl_info["rmse"]
            final_valid_frac = lvl_info["valid_frac"]
            final_n = lvl_info["n_pixels"]
            final_sigma = lvl_info["sigma"]
            final_geo_sigma = lvl_info["geo_sigma"]
            final_n_geo = lvl_info["n_geo"]
            final_rmse_ratio = lvl_info["rmse_ratio"]

    # ---- Stage-2b divergence guard (report; the caller decides) ------------- #
    # step_norm: how far (m) the finest-level solve translated the pose AWAY from
    # the seed -- i.e. the magnitude of the correction GN applied on top of the
    # IMU/gyro prior. A runaway GN produces a step far larger than the plausible
    # inter-frame motion the seed implies.
    seed_step = float(np.linalg.norm(T_seed[:3, 3] - np.eye(4)[:3, 3]))
    step_norm = float(np.linalg.norm(T[:3, 3] - T_seed[:3, 3]))
    seed_ref = max(seed_step, cfg.diverge_seed_floor_m)
    step_ratio = step_norm / seed_ref
    rmse_ratio = final_rmse_ratio
    diverged = bool(
        (step_ratio > cfg.diverge_step_ratio)
        or (np.isfinite(rmse_ratio) and rmse_ratio > cfg.diverge_rmse_grow)
    )

    info = {
        "converged": bool(any_converged),
        "final_rmse": final_rmse,
        "iters": total_iters,
        "valid_frac": final_valid_frac,
        "n_pixels": final_n,
        "sigma": final_sigma,
        "geo_sigma": final_geo_sigma,
        "n_geo": final_n_geo,
        "per_level": per_level,
        # Stage-2b divergence guard signals.
        "diverged": diverged,
        "step_norm": step_norm,
        "step_ratio": step_ratio,
        "rmse_ratio": rmse_ratio,
    }
    return T, info


def _with(cfg: DirectConfig, **kw) -> DirectConfig:
    """Return a copy of ``cfg`` with the given fields overridden (dataclass replace)."""
    from dataclasses import replace
    return replace(cfg, **kw)


def _select_reference_pixels(
    g_ref: np.ndarray, d_ref: np.ndarray, K: np.ndarray, cfg: DirectConfig
) -> _LevelCache | None:
    """Pick informative reference pixels (valid depth + texture) and back-project.

    Returns a :class:`_LevelCache` with the back-projected 3D points (ref frame)
    and the reference intensities, or None if too few valid pixels exist. Pixel
    SELECTION is driven by the REFERENCE-image gradient (texture there is what
    makes the photometric error informative) and by depth validity.
    """
    h, w = g_ref.shape[:2]
    if d_ref is None:
        return None

    gx, gy = _image_gradients(g_ref)
    gmag = np.sqrt(gx * gx + gy * gy)

    valid_depth = (d_ref > cfg.depth_min) & (d_ref < cfg.depth_max)
    textured = gmag >= cfg.min_grad
    mask = valid_depth & textured
    n_valid = int(mask.sum())
    if n_valid < max(8, int(cfg.min_valid_frac * h * w)):
        # Fall back to ALL valid-depth pixels (low-texture scene): better a noisy
        # solve than no solve, and the robust weight will down-weight flat ones.
        mask = valid_depth
        n_valid = int(mask.sum())
        if n_valid < 8:
            return None

    vs, us = np.nonzero(mask)
    Z = d_ref[vs, us].astype(np.float64)
    I = g_ref[vs, us].astype(np.float64)

    # Optional: keep only the highest-gradient subset, and/or cap the count.
    gsel = gmag[vs, us]
    if 0.0 < cfg.grad_select_frac < 1.0 and us.size > 16:
        keep = max(16, int(cfg.grad_select_frac * us.size))
        idx = np.argpartition(gsel, -keep)[-keep:]
        us, vs, Z, I = us[idx], vs[idx], Z[idx], I[idx]
        gsel = gsel[idx]
    if cfg.max_pixels and us.size > cfg.max_pixels:
        idx = np.argpartition(gsel, -cfg.max_pixels)[-cfg.max_pixels:]
        us, vs, Z, I = us[idx], vs[idx], Z[idx], I[idx]

    # Back-project: P = Z * K^{-1} [u, v, 1]^T (reference camera frame).
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (us.astype(np.float64) - cx) * Z / fx
    Y = (vs.astype(np.float64) - cy) * Z / fy
    P_ref = np.stack([X, Y, Z], axis=1)  # (M, 3)

    return _LevelCache(P_ref=P_ref, I_ref=I, K=np.asarray(K, np.float64),
                       shape=(h, w))


def _solve_level(
    T: np.ndarray,
    cache: _LevelCache,
    g_cur: np.ndarray,
    gx_cur: np.ndarray,
    gy_cur: np.ndarray,
    K: np.ndarray,
    cfg: DirectConfig,
    geo: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict]:
    """One pyramid level of robust Gauss-Newton (fused photometric + geometric).

    ``geo`` (when not None) is ``(depth_cur, normals_cur, normal_valid)`` for the
    Stage-2b point-to-plane term: the current-frame depth map, its per-pixel
    surface normals and the normal-validity mask, all in the CURRENT camera frame
    (pose-independent; looked up at the warped pixel each iteration). When ``geo``
    is None or ``cfg.geo_weight == 0`` the solve is pure photometric and the
    normal-equation accumulation is BYTE-IDENTICAL to the pre-Stage-2b path.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    P_ref = cache.P_ref          # (M, 3) reference-frame points
    I_ref = cache.I_ref          # (M,)
    M = P_ref.shape[0]

    use_geo = geo is not None and cfg.geo_weight > 0.0
    if use_geo:
        depth_cur, normals_cur, normal_valid_cur = geo
        cos_min = float(np.cos(np.radians(cfg.geo_normal_max_angle_deg)))

    converged = False
    iters_done = 0
    rmse = float("nan")
    rmse0 = float("nan")     # the FIRST iteration's photometric RMSE (for the guard)
    valid_frac = 0.0
    sigma = float("nan")
    geo_sigma = float("nan")
    n_geo = 0

    for it in range(cfg.max_iters):
        iters_done = it + 1
        R = T[:3, :3]
        t = T[:3, 3]
        # Transform reference points into the CURRENT camera frame.
        P_cur = (R @ P_ref.T).T + t      # (M, 3)
        Xc, Yc, Zc = P_cur[:, 0], P_cur[:, 1], P_cur[:, 2]

        in_front = Zc > cfg.depth_min
        # Project (guard the divide; out-of-front pixels are masked out anyway).
        Zc_safe = np.where(in_front, Zc, 1.0)
        u = fx * Xc / Zc_safe + cx
        v = fy * Yc / Zc_safe + cy

        I_cur, in_img = _bilinear_sample(g_cur, u, v)
        gxs, _ = _bilinear_sample(gx_cur, u, v)
        gys, _ = _bilinear_sample(gy_cur, u, v)

        usable = in_front & in_img
        n_use = int(usable.sum())
        valid_frac = n_use / max(M, 1)
        if n_use < 8:
            break

        # Photometric residual on the usable subset.
        r = (I_cur - I_ref)[usable]

        # Robust weights (Kerl Student-t or Huber).
        if cfg.robust == "huber":
            wts = _huber_weights(r, cfg.huber_delta)
            sigma = float(np.median(np.abs(r)) * 1.4826)
        else:
            wts, sigma = _t_scale_and_weights(r, cfg.t_dof)

        rmse = float(np.sqrt(np.mean(r * r)))
        if it == 0:
            rmse0 = rmse

        # ---- Per-pixel 1x6 Jacobians (left twist, translation-first) -------- #
        # J = g^T @ J_pi @ [I3 | -skew(P_cur)]
        # J_pi (2x3) per pixel:
        #   [[fx/Z, 0, -fx X/Z^2],
        #    [0, fy/Z, -fy Y/Z^2]]
        Xu = Xc[usable]
        Yu = Yc[usable]
        Zu = Zc[usable]
        invZ = 1.0 / Zu
        invZ2 = invZ * invZ
        gxu = gxs[usable]
        gyu = gys[usable]

        # row = g^T @ J_pi  (1x3): the image-gradient-weighted projection Jacobian.
        # a = d(residual)/dX, b = d/dY, c = d/dZ  (in current camera coords).
        a = gxu * (fx * invZ)
        b = gyu * (fy * invZ)
        c = -(gxu * fx * Xu + gyu * fy * Yu) * invZ2
        # Now J_warp = [I3 | -skew(P_cur)]; multiply row=[a,b,c] by it:
        #   translation block (I3):           [a, b, c]
        #   rotation block (-skew(P_cur)):
        #     -skew(P) = [[0, Z, -Y], [-Z, 0, X], [Y, -X, 0]]
        #     row @ (-skew(P)) = [ -b*Z + c*Y,  a*Z - c*X,  -a*Y + b*X ]
        J = np.empty((n_use, 6), dtype=np.float64)
        J[:, 0] = a
        J[:, 1] = b
        J[:, 2] = c
        J[:, 3] = -b * Zu + c * Yu
        J[:, 4] = a * Zu - c * Xu
        J[:, 5] = -a * Yu + b * Xu

        # Weighted normal equations: H = J^T W J, g = -J^T W r.
        WJ = J * wts[:, None]
        H = J.T @ WJ                      # (6, 6)
        grad = -(J.T @ (wts * r))         # (6,)

        # ---- Stage-2b: fused point-to-plane GEOMETRIC block ---------------- #
        # E = E_photo + w * E_geo  ->  H += w * H_geo, grad += w * grad_geo.
        # Built on the SAME warped points P_cur (no re-projection); see the module
        # docstring for the residual + Jacobian derivation.
        if use_geo:
            H_g, grad_g, geo_sigma, n_geo = _geo_normal_equations(
                P_cur, usable, u, v, depth_cur, normals_cur, normal_valid_cur,
                fx, fy, cx, cy, cos_min, cfg)
            if n_geo > 0:
                H = H + cfg.geo_weight * H_g
                grad = grad + cfg.geo_weight * grad_g

        # Levenberg-Marquardt diagonal damping for conditioning.
        H[np.diag_indices(6)] += cfg.lm_damping * np.diag(H)
        # Tiny absolute floor so an all-zero column (degenerate DoF) is solvable.
        H[np.diag_indices(6)] += 1e-9

        try:
            dxi = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break

        # Apply on the LEFT: T <- Exp(dxi) @ T  (dxi = [rho; phi]).
        T = se3_exp(dxi) @ T

        if float(dxi @ dxi) < cfg.convergence_eps:
            converged = True
            break

    # Ratio of final to initial photometric RMSE at this level: > 1 means GN ran
    # UPHILL (got worse) -- the divergence-guard's "alignment degraded" signal.
    rmse_ratio = (rmse / rmse0) if (np.isfinite(rmse0) and rmse0 > 1e-9) else float("nan")

    return T, {
        "iters": iters_done,
        "converged": converged,
        "rmse": rmse,
        "rmse_ratio": rmse_ratio,
        "valid_frac": valid_frac,
        "n_pixels": M,
        "sigma": sigma,
        "geo_sigma": geo_sigma,
        "n_geo": n_geo,
    }


def _geo_normal_equations(
    P_cur: np.ndarray,
    usable: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    depth_cur: np.ndarray,
    normals_cur: np.ndarray,
    normal_valid_cur: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    cos_min: float,
    cfg: DirectConfig,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Point-to-plane geometric block ``(H_geo, grad_geo, sigma_geo, n_geo)``.

    Implements the Stage-2b geometric term (Whelan ICRA'13). For each USABLE warped
    reference point ``P_cur_est`` (already in the current camera frame) projected
    to the current pixel ``(u, v)``:

      * measured point ``P_cur_meas = depth_cur(u,v) * K^{-1}[u,v,1]^T`` (back-
        project the CURRENT depth at the warped pixel),
      * surface normal ``n = normals_cur(u,v)`` (nearest-neighbour lookup of the
        pre-computed current-frame normals; the normal/measured point are FROZEN
        per iteration, standard ICP correspondence linearisation),
      * residual ``r_geo = n . (P_cur_est - P_cur_meas)``  (signed plane distance),
      * Jacobian ``J_geo = [ n^T | (P_cur_est x n)^T ]``  (1x6, left twist).

    Outliers are gated BEFORE robust weighting: the measured depth must be valid,
    the normal must be valid and facing the camera (``|n . ray_dir| >= cos_min``),
    and the plane distance must be ``<= cfg.geo_max_dist_m``. The surviving
    residuals get their own Student-t / Huber robust weights (scale in METRES).
    Returns zero matrices + ``n_geo = 0`` when nothing survives.
    """
    Pu = P_cur[usable]                          # (n_use, 3) warped points (cur frame)
    uu = u[usable]
    vu = v[usable]
    h, w = depth_cur.shape[:2]

    # Nearest current pixel for the depth + normal lookup (round-to-nearest; the
    # warp already gated in-bounds, but clamp defensively). The depth map and the
    # normal map are organised per integer pixel, so the measured point must be
    # back-projected at the SAME integer pixel ``(ui, vi)`` the depth was sampled
    # at -- using the float warp coords here would put the measured point off its
    # own depth ray by up to half a pixel.
    ui = np.clip(np.round(uu).astype(np.int64), 0, w - 1)
    vi = np.clip(np.round(vu).astype(np.int64), 0, h - 1)

    Zmeas = depth_cur[vi, ui].astype(np.float64)
    nrm = normals_cur[vi, ui]                    # (n_use, 3)
    nvalid = normal_valid_cur[vi, ui]

    valid_meas = (Zmeas > cfg.depth_min) & (Zmeas < cfg.depth_max) & nvalid
    if not valid_meas.any():
        return np.zeros((6, 6)), np.zeros(6), float("nan"), 0

    Pu = Pu[valid_meas]
    ui = ui[valid_meas]
    vi = vi[valid_meas]
    Zmeas = Zmeas[valid_meas]
    nrm = nrm[valid_meas]

    # Measured 3D point: back-project the current depth at the (integer) pixel it
    # was sampled at, so ``Pmeas`` lies exactly on its own depth ray.
    Pmeas = np.empty_like(Pu)
    Pmeas[:, 0] = (ui.astype(np.float64) - cx) * Zmeas / fx
    Pmeas[:, 1] = (vi.astype(np.float64) - cy) * Zmeas / fy
    Pmeas[:, 2] = Zmeas

    # Grazing-angle gate: the viewing ray dir to the measured point vs the normal.
    # A near-parallel (grazing) surface gives an ill-conditioned plane distance.
    ray = Pmeas / np.maximum(np.linalg.norm(Pmeas, axis=1, keepdims=True), 1e-9)
    cos_face = np.abs(np.sum(nrm * ray, axis=1))
    facing = cos_face >= cos_min

    # Point-to-plane signed distance r = n . (P_est - P_meas).
    diff = Pu - Pmeas
    r_geo_all = np.sum(nrm * diff, axis=1)

    keep = facing & (np.abs(r_geo_all) <= cfg.geo_max_dist_m)
    n_geo = int(keep.sum())
    if n_geo < 6:
        return np.zeros((6, 6)), np.zeros(6), float("nan"), 0

    Pk = Pu[keep]
    nk = nrm[keep]
    r_geo = r_geo_all[keep]

    # Robust weights for the geometric residual (own scale, in METRES).
    if cfg.robust == "huber":
        # Reuse a metric Huber transition tied to the outlier gate (10% of it),
        # so a near-outlier is softly down-weighted before the hard cut.
        wts_g = _huber_weights(r_geo, 0.1 * cfg.geo_max_dist_m)
        geo_sigma = float(np.median(np.abs(r_geo)) * 1.4826)
    else:
        wts_g, geo_sigma = _t_scale_and_weights(r_geo, cfg.t_dof)

    # Jacobian rows: J_geo = [ n^T | (P_est x n)^T ]  (left twist, trans-first).
    #   n^T (-skew(P)) = (skew(P) n)^T = (P x n)^T   -> rotation block = P_est x n.
    Jg = np.empty((n_geo, 6), dtype=np.float64)
    Jg[:, 0:3] = nk
    Jg[:, 3:6] = np.cross(Pk, nk)

    WJg = Jg * wts_g[:, None]
    H_g = Jg.T @ WJg                            # (6, 6)
    grad_g = -(Jg.T @ (wts_g * r_geo))          # (6,)
    return H_g, grad_g, geo_sigma, n_geo
