"""Stereo camera-calibration solve (Phase 3 -- the calibration math core).

Takes the per-view left/right checkerboard corners collected by
:class:`ui.mathlib.calib.collector.StereoCheckerboardCollector` and recovers the
two intrinsics, their distortion, and the rigid left->right extrinsic.

WIDE-FOV (fisheye) MODEL -- why the default 5-coeff model is NOT enough
----------------------------------------------------------------------
The real target is the OAK-D **W** (a ~95-110 deg WIDE lens). OpenCV's default
``cv2.calibrateCamera`` distortion model (5 coeffs ``k1,k2,p1,p2,k3``) CANNOT
represent that much barrel distortion: the optimizer compensates by inflating the
focal length (we saw ``fx`` run away to ~1884 px on a 640-wide image whose true
``fx`` is ~285), which then poisons ``stereoCalibrate`` (run with
``CALIB_FIX_INTRINSIC`` on those wrong intrinsics) into a 1e13-px divergence and a
nonsense ~960 mm baseline. So we:

* fit with :data:`cv2.CALIB_RATIONAL_MODEL` -- 8 coeffs ``k1..k6,p1,p2`` -- which the
  OAK-D factory calib (a 14-coeff rational + thin-prism + tilt model whose thin-prism
  and tilt terms are ~0) is a near-exact superset of, so the rational model captures
  essentially all of this lens's distortion while keeping the calib.json dist length
  at a value ``calib_check`` recognises (8 is in its known set);
* SEED a sane intrinsic guess (``cx,cy`` = image centre, ``fx=fy`` from an assumed
  wide HFOV) with :data:`cv2.CALIB_USE_INTRINSIC_GUESS`, so the optimiser starts near
  the true wide-FOV focal length instead of running away;
* REJECT per-view reprojection outliers (a single mis-detected board corrupts the
  whole fit) before re-fitting on the clean views;
* SANITY-FLOOR the result and flag it FAILED (rather than save garbage) when the
  recovered focal length or the stereo RMS is implausible.

T_left_right CONVENTION (get this right -- a flipped extrinsic is a silent bug)
------------------------------------------------------------------------------
The project's :class:`imu_camera.io.reader.StereoCalib` defines ``T_left_right`` as
the 4x4 transform that maps a point from the LEFT camera frame to the RIGHT camera
frame (in metres); the baseline is ``||T_left_right[:3,3]||``.

:func:`cv2.stereoCalibrate(objectPoints, imagePoints1, imagePoints2, ...)` returns
``R, T`` such that, for a point ``X``::

    X_cam2 = R @ X_cam1 + T

i.e. it maps camera-1's frame to camera-2's frame. We pass the LEFT corners as
``imagePoints1`` and the RIGHT corners as ``imagePoints2``, so camera 1 = left and
camera 2 = right, and cv2's ``(R, T)`` is therefore exactly LEFT->RIGHT -- it drops
straight into ``T_left_right`` with NO inversion or transpose. The self-test proves
this by feeding a known left->right ground-truth transform through
``cv2.projectPoints`` and asserting the recovered ``T_left_right`` matches it (not
its inverse).

cv2 POLICY
----------
cv2 is lazy-imported inside :func:`solve_stereo`; importing this module (or the
package) does not load OpenCV, keeping the flight path cv2-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Wide-FOV solve tuning -- the knobs the OAK-D W fix turns. All gathered here so
# the math is auditable in one place (and the self-test can reference them).
# --------------------------------------------------------------------------- #
#: Assumed horizontal FOV (deg) for the SEED intrinsic guess. The OAK-D W is a wide
#: ~95-110 deg lens; 100 deg gives ``fx = width / (2*tan(50deg)) ~= 0.42*width`` (=> ~268
#: for a 640-wide image, right on the ~285 truth), so the optimiser starts NEAR the
#: real wide-FOV focal length instead of running away to ~1884.
_SEED_HFOV_DEG = 100.0
#: Per-view reprojection-outlier gate (MAD-based -- robust to legitimate high-tilt
#: views that carry slightly higher RMS but PIN the focal length). Drop a view whose
#: per-view error exceeds ``max(_OUTLIER_RMS_FLOOR_PX, median + _OUTLIER_RMS_MAD_K *
#: 1.4826 * MAD)``. A bare ``median-multiplier`` gate (the old ``2.5x median``)
#: preferentially deletes those high-tilt views; the MAD (scaled by 1.4826 to a
#: Gaussian-sigma estimate, K=3) instead drops only true statistical outliers while
#: the absolute floor still protects a uniformly clean-but-noisy dataset. The gate runs
#: as ONE drop -> refit pass (not iterated to convergence) AFTER the intrinsic-guess fit
#: so the residuals it scores are honest.
_OUTLIER_RMS_FLOOR_PX = 1.5
_OUTLIER_RMS_MAD_K = 3.0
#: 1.4826 rescales the median-absolute-deviation to an unbiased Gaussian-sigma estimate.
_MAD_TO_SIGMA = 1.4826
#: Minimum number of views that must SURVIVE outlier rejection. Below this the fit is
#: under-constrained; we fail honestly ("recapture") rather than emit garbage.
_MIN_CLEAN_VIEWS = 8
#: Sanity floor on the recovered focal length, as a fraction of image WIDTH (so the
#: bound is resolution-relative, not OAK-D-pixel-hardcoded). ``fx/width`` for a lens of
#: HFOV ``h`` is ``1/(2*tan(h/2))``: HFOV 120 deg -> ~0.29, HFOV 90 deg -> ~0.50, so a
#: plausible wide/normal lens lands in ~[0.29, 0.50]. We widen to [0.25, 0.60] for
#: per-camera spread + mild distortion. The runaway failure produced ``fx/width ~=
#: 1884/640 ~= 2.9`` -- far above this ceiling -- so it is flagged; a wide OAK-D W at
#: ~260-285/640 ~= 0.41-0.45 sits comfortably inside.
_FX_OVER_WIDTH_LO = 0.25
_FX_OVER_WIDTH_HI = 0.60
#: Sanity bounds on the stereo BASELINE (millimetres). Unlike focal length, baseline is
#: a PHYSICAL rig dimension and does NOT scale with image resolution, so a millimetre
#: window is the correct (not brittle) form here. The OAK-D's nominal baseline is ~75 mm;
#: [60, 90] mm brackets it with margin while flagging the ~961 mm corner-flip divergence
#: and the ~0 mm collapse a fully mis-corresponded solve produces.
_BASELINE_MM_LO = 60.0
_BASELINE_MM_HI = 90.0
#: Maximum plausible inter-camera rotation (``||log(R)||``, radians). A stereo rig is
#: near-parallel; >~5 deg means a flipped/mis-corresponded extrinsic (the corner-order
#: bug rotated R to ~168 deg). 5 deg = 0.0873 rad.
_INTERCAM_ROT_MAX_RAD = np.deg2rad(5.0)
#: Sanity ceiling on the joint stereo reprojection RMS (px). A converged stereo fit
#: is sub-pixel-to-low-single-digit; the divergence we are guarding against was ~1e13.
_STEREO_RMS_MAX_PX = 5.0


@dataclass(frozen=True)
class StereoCalibResult:
    """Solved stereo calibration: intrinsics, distortion, extrinsic, RMS errors.

    All transforms are in METRES (``square_size_m`` sets the world scale).

    Diagnostics
    -----------
    ``n_views_used`` (views surviving outlier rejection), the per-camera mono RMS,
    the OpenCV ``calibrate_flags`` actually used, and the boolean ``ok`` verdict (with
    a human ``failure_reason`` when ``ok`` is False) are surfaced so the wizard can
    show an HONEST "did not converge -- recapture" message instead of saving garbage.
    """

    K_l: np.ndarray        # 3x3 left intrinsics
    dist_l: np.ndarray     # (k,) left distortion coefficients (OpenCV order)
    rms_l: float           # left mono reprojection RMS (pixels)
    K_r: np.ndarray        # 3x3 right intrinsics
    dist_r: np.ndarray     # (k,) right distortion coefficients
    rms_r: float           # right mono reprojection RMS (pixels)
    R: np.ndarray          # 3x3 rotation, LEFT->RIGHT
    T: np.ndarray          # (3,) translation, LEFT->RIGHT, metres
    stereo_rms: float      # joint stereo reprojection RMS (pixels)
    # -- diagnostics (defaulted so existing positional construction stays valid) --
    n_views_used: int = 0          # views kept after per-view outlier rejection
    calibrate_flags: int = 0       # OpenCV calibrate flags actually used
    ok: bool = True                # False => the solve did NOT converge / is implausible
    failure_reason: str = ""       # human reason when ok is False ("" when ok)

    @property
    def T_left_right(self) -> np.ndarray:
        """4x4 homogeneous LEFT->RIGHT transform (metres) -- the project convention."""
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = self.R
        M[:3, 3] = self.T
        return M

    @property
    def baseline_m(self) -> float:
        """Stereo baseline = ``||T||`` (metres)."""
        return float(np.linalg.norm(self.T))


def _object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    """Planar board's 3D inner-corner coordinates (Z=0), scaled to metres.

    Row-major over ``(rows, cols)`` so corner index ``j*cols + i`` sits at world
    ``(i, j, 0) * square_size_m`` -- the SAME raster order
    :func:`cv2.findChessboardCorners` returns its image points in, so object[k]
    corresponds to image[k] across every view.
    """
    obj = np.zeros((rows * cols, 3), dtype=np.float32)
    # mgrid gives (i across cols, j down rows); ravel in C-order to match the
    # detector's left-to-right, top-to-bottom corner ordering.
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # (rows*cols, 2) as (i, j)
    obj[:, :2] = grid
    obj *= float(square_size_m)
    return obj


def _seed_camera_matrix(image_size: tuple[int, int]) -> np.ndarray:
    """Initial intrinsic GUESS for a wide-FOV lens (the runaway-focal antidote).

    ``cx,cy`` = image centre; ``fx=fy = width / (2*tan(HFOV/2))`` for the assumed wide
    :data:`_SEED_HFOV_DEG`. Seeding this with ``CALIB_USE_INTRINSIC_GUESS`` starts the
    optimiser near the true wide focal length (~0.42*width), so it no longer inflates
    ``fx`` to fake the barrel distortion away. Same guess for BOTH cameras (an OAK-D
    stereo pair shares a lens design).
    """
    w, h = float(image_size[0]), float(image_size[1])
    f = w / (2.0 * np.tan(np.deg2rad(_SEED_HFOV_DEG) / 2.0))
    return np.array([[f, 0.0, w / 2.0],
                     [0.0, f, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _calibrate_one(cv2, obj_pts, img_pts, image_size, flags, K0):
    """One camera's ``cv2.calibrateCameraExtended`` with the wide-FOV flags + seed guess.

    Returns ``(rms, K, dist, rvecs, tvecs, per_view_rms)`` where ``per_view_rms`` is the
    HONEST per-view reprojection RMS (px) cv2 computes from the SAME optimised model --
    used by the MAD outlier gate. The extended variant gives ``perViewErrors`` directly,
    so we no longer re-project by hand. The seed ``K0`` is COPIED in (cv2 mutates the
    matrix in place under ``CALIB_USE_INTRINSIC_GUESS``).
    """
    rms, K, dist, rvecs, tvecs, _, _, per_view = cv2.calibrateCameraExtended(
        obj_pts, img_pts, image_size, K0.copy(), None, flags=flags)
    # perViewErrors comes back as (V, 1); flatten to a plain (V,) float array.
    return rms, K, dist, rvecs, tvecs, np.asarray(per_view, dtype=np.float64).ravel()


def _mad_threshold(per_view_rms: np.ndarray) -> float:
    """MAD-based outlier cut for per-view reprojection RMS.

    ``max(_OUTLIER_RMS_FLOOR_PX, median + K * 1.4826 * MAD)`` where ``MAD`` is the
    median absolute deviation about the median. The ``1.4826`` factor rescales MAD to an
    unbiased Gaussian-sigma estimate, so the cut is "median + K-sigma" of a robust spread
    -- it drops only genuine statistical outliers and, unlike a bare ``median-multiplier``
    gate, does NOT preferentially delete the legitimate high-tilt views that carry a
    slightly elevated (but still in-family) RMS yet pin the focal length. The absolute
    floor keeps a uniformly clean-but-noisy dataset from rejecting good views.
    """
    med = float(np.median(per_view_rms))
    mad = float(np.median(np.abs(per_view_rms - med)))
    return max(_OUTLIER_RMS_FLOOR_PX, med + _OUTLIER_RMS_MAD_K * _MAD_TO_SIGMA * mad)


def _clean_view_mask(rms_l: np.ndarray, rms_r: np.ndarray) -> np.ndarray:
    """Boolean keep-mask: drop views whose L or R per-view RMS is a MAD outlier.

    A view is kept only if BOTH its cameras reproject within the MAD cut
    (:func:`_mad_threshold`) computed per camera. Robust to a single bad view (the median
    and MAD are unmoved by one outlier) while preserving legitimate high-tilt views.
    """
    return (rms_l <= _mad_threshold(rms_l)) & (rms_r <= _mad_threshold(rms_r))


def solve_stereo(
    views: list[tuple[np.ndarray, np.ndarray]],
    pattern_cols: int,
    pattern_rows: int,
    square_size_m: float,
    image_size: tuple[int, int],
    *,
    dump_path: str | Path | None = None,
) -> StereoCalibResult:
    """Solve the stereo calibration from per-view left/right corner sets.

    Parameters
    ----------
    views:
        List of ``(corners_left, corners_right)`` pairs, each an ``(N,2)`` float
        array with ``N == pattern_cols * pattern_rows`` corners in the detector's
        raster order (as produced by
        :class:`~ui.mathlib.calib.collector.StereoCheckerboardCollector`).
    pattern_cols, pattern_rows:
        INNER-corner counts (OpenCV ``patternSize``).
    square_size_m:
        Physical edge length of one board square, in METRES -- sets the world scale
        so the recovered baseline is in metres.
    image_size:
        ``(width, height)`` in pixels (cv2 convention).
    dump_path:
        Optional ``.npz`` path. When given, the captured per-view data (object
        points, L/R image points, board geometry, image size) and the solved result
        are saved there for offline debugging of a real-device capture (see
        :func:`dump_views`). The path is returned via the log; a dump failure never
        breaks the solve.

    Returns
    -------
    StereoCalibResult
        Intrinsics + distortion per camera, the LEFT->RIGHT extrinsic, the mono +
        stereo reprojection RMS errors, and the diagnostics (``n_views_used``,
        ``calibrate_flags``, ``ok`` / ``failure_reason``). A non-converged or
        implausible solve is returned with ``ok=False`` and a reason -- it is NOT
        raised, so the wizard can show an honest "recapture" message and still dump
        the data.

    L<->R corner-ORDER prerequisite
    -------------------------------
    The per-view ``(corners_left, corners_right)`` pairs MUST already correspond
    index-for-index (``object[k]``/``left[k]``/``right[k]`` = the same board point). The
    180-degree corner-order ambiguity that :func:`cv2.findChessboardCorners` exposes is
    reconciled UPSTREAM by :func:`ui.mathlib.calib.detect.reconcile_lr` (called by the
    collector when it accepts a view) -- a mismatched (reversed) right order would
    otherwise make this solve diverge to a ~1 m baseline / ~168-degree rotation.

    Pipeline
    --------
    1. Per camera, :func:`cv2.calibrateCameraExtended` with a SEEDED wide-FOV intrinsic
       guess (:data:`cv2.CALIB_USE_INTRINSIC_GUESS`) recovers ``K`` + ``dist`` + the
       honest per-view reprojection errors (``perViewErrors``).
    2. Per-view OUTLIER rejection (MAD gate, one drop->refit): drop views whose per-view
       RMS is a MAD outlier in either camera, then RE-FIT both cameras on the surviving
       views (require :data:`_MIN_CLEAN_VIEWS`).
    3. :func:`cv2.stereoCalibrate` (``CALIB_FIX_INTRINSIC``) solves only the relative
       pose on those clean, well-conditioned intrinsics -- the standard, stable
       two-stage stereo calibration.
    4. SANITY-FLOOR: a runaway focal length, an implausible baseline, an excessive
       inter-camera rotation, or a diverged stereo RMS is flagged ``ok=False`` rather
       than returned as a usable (but garbage) calibration.

    Distortion-model SELECTION (wide lens vs mild lens, automatic)
    -------------------------------------------------------------
    The fit is attempted FIRST with the 8-coeff :data:`cv2.CALIB_RATIONAL_MODEL` -- the
    only model rich enough for the OAK-D **W**'s wide barrel distortion (the default
    5-coeff model would runaway ``fx`` and trip the sanity floor). If that wide fit is
    implausible (``ok=False`` -- which happens when the lens is actually MILD, so the
    extra rational ``k4..k6`` terms are unobservable and over-fit the noise), we FALL
    BACK to the standard 5-coeff model, which is well-conditioned on mild data. The
    sanity verdict thus self-selects the right model per lens with NO operator input.
    """
    # Lazy import: keeps `import ui.mathlib.calib` cv2-free for the flight path.
    import cv2

    if len(views) < 2:
        raise ValueError(
            f"stereo calibration needs >= 2 views, got {len(views)}")
    n_corners = pattern_cols * pattern_rows

    obj1 = _object_points(pattern_cols, pattern_rows, square_size_m)
    object_points: list[np.ndarray] = []
    img_left: list[np.ndarray] = []
    img_right: list[np.ndarray] = []
    for k, (cl, cr) in enumerate(views):
        cl = np.asarray(cl, dtype=np.float32)
        cr = np.asarray(cr, dtype=np.float32)
        if cl.shape != (n_corners, 2) or cr.shape != (n_corners, 2):
            raise ValueError(
                f"view {k}: expected ({n_corners}, 2) corners, got "
                f"L={cl.shape} R={cr.shape}")
        object_points.append(obj1.copy())
        # cv2 wants (N,1,2) float32 image-point buffers.
        img_left.append(cl.reshape(-1, 1, 2))
        img_right.append(cr.reshape(-1, 1, 2))

    # Try the wide-FOV (8-coeff rational) model first; fall back to the standard
    # 5-coeff model only if the rational fit is implausible (a MILD lens over-fits the
    # rational terms). A seeded intrinsic guess is used for BOTH attempts.
    K0 = _seed_camera_matrix(image_size)
    rational_flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_USE_INTRINSIC_GUESS
    standard_flags = cv2.CALIB_USE_INTRINSIC_GUESS

    result, dump_args = _solve_with_flags(
        cv2, object_points, img_left, img_right, image_size, rational_flags, K0)
    if not result.ok and "too few clean views" not in result.failure_reason:
        # The wide model did not converge on this data -- retry with the standard
        # 5-coeff model (well-conditioned for a MILD lens, which over-fits the rational
        # terms). We adopt the standard attempt: it either converges (the mild-lens
        # case) or, if it also fails, gives the more honest non-diverged numbers than
        # the rational over-fit. A "too few clean views" failure is NOT retried -- a
        # different distortion model cannot conjure clean views out of bad data.
        result, dump_args = _solve_with_flags(
            cv2, object_points, img_left, img_right, image_size, standard_flags, K0)

    _maybe_dump(dump_path, *dump_args, pattern_cols, pattern_rows,
                square_size_m, image_size, result)
    return result


def _solve_with_flags(cv2, object_points, img_left, img_right, image_size,
                      calib_flags, K0):
    """Run the full per-camera + gating + stereo + sanity pipeline for ONE model.

    Returns ``(result, (obj_clean, il_clean, ir_clean))`` -- the second element is the
    surviving-view buffers the caller dumps. Factored out of :func:`solve_stereo` so
    the rational and the fallback standard model share one well-tested code path.
    """
    n_views = len(object_points)

    # 1. First-pass per-camera intrinsics on ALL views (the SEEDED intrinsic-guess fit),
    #    capturing cv2's honest per-view reprojection RMS to score each view.
    rms_l0, K_l, dist_l, _, _, pv_l = _calibrate_one(
        cv2, object_points, img_left, image_size, calib_flags, K0)
    rms_r0, K_r, dist_r, _, _, pv_r = _calibrate_one(
        cv2, object_points, img_right, image_size, calib_flags, K0)

    # 2. Per-view outlier rejection (MAD gate, ONE drop -> refit -- not iterated). The
    #    gate runs AFTER the intrinsic-guess fit above, so the per-view residuals it
    #    scores are honest. Drop the mis-detected boards, then RE-FIT on the survivors.
    keep = _clean_view_mask(pv_l, pv_r)
    n_used = int(np.count_nonzero(keep))

    # Honest failure (not garbage) when too few clean views survive.
    if n_used < min(_MIN_CLEAN_VIEWS, n_views):
        reason = (f"too few clean views after outlier rejection: {n_used} kept "
                  f"(need >= {min(_MIN_CLEAN_VIEWS, n_views)}) -- recapture")
        result = _failed_result(K_l, dist_l, rms_l0, K_r, dist_r, rms_r0,
                                n_used, calib_flags, reason)
        return result, (object_points, img_left, img_right)

    if n_used < n_views:
        # Re-run the per-camera calibration on the surviving (clean) views only.
        obj_c = [object_points[i] for i in range(n_views) if keep[i]]
        il_c = [img_left[i] for i in range(n_views) if keep[i]]
        ir_c = [img_right[i] for i in range(n_views) if keep[i]]
        rms_l, K_l, dist_l, _, _, _ = _calibrate_one(
            cv2, obj_c, il_c, image_size, calib_flags, K0)
        rms_r, K_r, dist_r, _, _, _ = _calibrate_one(
            cv2, obj_c, ir_c, image_size, calib_flags, K0)
    else:
        obj_c, il_c, ir_c = object_points, img_left, img_right
        rms_l, rms_r = rms_l0, rms_r0

    # 3. Stereo extrinsic with FIXED intrinsics (two-stage, stable). LEFT is camera 1
    #    and RIGHT is camera 2, so cv2's returned (R, T) maps LEFT->RIGHT -- exactly
    #    the StereoCalib.T_left_right convention (no inversion needed). The clean
    #    intrinsics already carry the chosen distortion model, so the extrinsic fit
    #    inherits it without re-solving the distortion.
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    stereo_rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        obj_c, il_c, ir_c,
        K_l, dist_l, K_r, dist_r, image_size,
        criteria=criteria, flags=flags)

    result = StereoCalibResult(
        K_l=np.asarray(K_l, dtype=np.float64),
        dist_l=np.asarray(dist_l, dtype=np.float64).ravel(),
        rms_l=float(rms_l),
        K_r=np.asarray(K_r, dtype=np.float64),
        dist_r=np.asarray(dist_r, dtype=np.float64).ravel(),
        rms_r=float(rms_r),
        R=np.asarray(R, dtype=np.float64),
        T=np.asarray(T, dtype=np.float64).ravel(),
        stereo_rms=float(stereo_rms),
        n_views_used=n_used,
        calibrate_flags=int(calib_flags),
        ok=True,
        failure_reason="",
    )

    # 4. Sanity-floor: catch the runaway-focal / diverged-stereo failure modes and
    #    flag them rather than handing back a usable-looking but garbage calibration.
    result = _sanity_floor(result, image_size)
    return result, (obj_c, il_c, ir_c)


def _failed_result(K_l, dist_l, rms_l, K_r, dist_r, rms_r, n_used, flags,
                   reason: str) -> StereoCalibResult:
    """Build a FAILED result (identity extrinsic) carrying the failure reason.

    Used when the solve cannot proceed to a stereo fit (too few clean views). The
    intrinsics are whatever the first pass produced; the extrinsic is identity so the
    baseline is 0 -- the ``ok=False`` flag + reason are what the wizard reads.
    """
    return StereoCalibResult(
        K_l=np.asarray(K_l, dtype=np.float64),
        dist_l=np.asarray(dist_l, dtype=np.float64).ravel(),
        rms_l=float(rms_l),
        K_r=np.asarray(K_r, dtype=np.float64),
        dist_r=np.asarray(dist_r, dtype=np.float64).ravel(),
        rms_r=float(rms_r),
        R=np.eye(3, dtype=np.float64),
        T=np.zeros(3, dtype=np.float64),
        stereo_rms=float("inf"),
        n_views_used=int(n_used),
        calibrate_flags=int(flags),
        ok=False,
        failure_reason=reason,
    )


def _rotation_angle_rad(R: np.ndarray) -> float:
    """Geodesic angle ``||log(R)||`` of a rotation matrix (radians).

    The rotation angle of ``R`` is ``arccos((trace(R) - 1) / 2)``; the trace argument is
    clamped to ``[-1, 1]`` so floating-point drift just past the bound does not produce a
    NaN. Used to assert the inter-camera rotation of a stereo rig is small (near-parallel).
    """
    cos_theta = (float(np.trace(R)) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def _sanity_floor(result: StereoCalibResult,
                  image_size: tuple[int, int]) -> StereoCalibResult:
    """Flag an implausible solve ``ok=False`` (cheap physical-plausibility assertions).

    Catches every reported real-device failure mode rather than handing back a
    usable-looking but garbage calibration. We do NOT mutate the numbers (the wizard
    still shows them) -- we only set ``ok=False`` + a reason so the operator recaptures:

    * runaway FOCAL length -- ``fx/width`` leaves the wide/normal-lens band
      ``[_FX_OVER_WIDTH_LO, _FX_OVER_WIDTH_HI]`` (the wide-lens bug inflated it to ~2.9);
    * implausible BASELINE -- ``||T||`` leaves ``[_BASELINE_MM_LO, _BASELINE_MM_HI]`` mm
      (the corner-order bug produced ~961 mm; a fully mis-corresponded solve collapses
      toward ~0 mm);
    * excessive INTER-CAMERA ROTATION -- ``||log(R)|| > _INTERCAM_ROT_MAX_RAD`` (a stereo
      rig is near-parallel; the corner-order bug rotated R to ~168 deg);
    * diverged stereo RMS -- ``> _STEREO_RMS_MAX_PX`` (the divergence was ~1e13).

    The focal bound is width-relative; the baseline bound is a millimetre window (a
    physical rig dimension that does not scale with resolution); the rotation bound is in
    radians -- all three flag the corner-order divergence the prior fix missed.
    """
    import dataclasses

    w = float(image_size[0])
    fx_l = float(result.K_l[0, 0]) / w
    fx_r = float(result.K_r[0, 0]) / w
    baseline_mm = result.baseline_m * 1000.0
    rot_rad = _rotation_angle_rad(result.R)
    reasons: list[str] = []
    if not (_FX_OVER_WIDTH_LO <= fx_l <= _FX_OVER_WIDTH_HI):
        reasons.append(f"left fx/width={fx_l:.3f} outside "
                       f"[{_FX_OVER_WIDTH_LO},{_FX_OVER_WIDTH_HI}]")
    if not (_FX_OVER_WIDTH_LO <= fx_r <= _FX_OVER_WIDTH_HI):
        reasons.append(f"right fx/width={fx_r:.3f} outside "
                       f"[{_FX_OVER_WIDTH_LO},{_FX_OVER_WIDTH_HI}]")
    if not (_BASELINE_MM_LO <= baseline_mm <= _BASELINE_MM_HI):
        reasons.append(f"baseline={baseline_mm:.1f} mm outside "
                       f"[{_BASELINE_MM_LO:.0f},{_BASELINE_MM_HI:.0f}] mm")
    if rot_rad > _INTERCAM_ROT_MAX_RAD:
        reasons.append(f"inter-camera rotation={np.rad2deg(rot_rad):.1f} deg > "
                       f"{np.rad2deg(_INTERCAM_ROT_MAX_RAD):.0f} deg")
    if not np.isfinite(result.stereo_rms) or result.stereo_rms > _STEREO_RMS_MAX_PX:
        reasons.append(f"stereo RMS={result.stereo_rms:.3g} px > {_STEREO_RMS_MAX_PX}")
    if reasons:
        return dataclasses.replace(
            result, ok=False,
            failure_reason="implausible solve (did not converge): "
                           + "; ".join(reasons))
    return result


# --------------------------------------------------------------------------- #
# Debug dump: persist the operator's REAL captured corners so a failing on-device
# calibration can be reproduced offline (with REAL data, not blind guesses).
# --------------------------------------------------------------------------- #
def dump_views(path: str | Path,
               object_points: list[np.ndarray],
               img_left: list[np.ndarray],
               img_right: list[np.ndarray],
               pattern_cols: int,
               pattern_rows: int,
               square_size_m: float,
               image_size: tuple[int, int],
               result: StereoCalibResult | None = None) -> Path:
    """Save the captured per-view calibration data (+ result) to an ``.npz``.

    Writes the object points, the left + right image points per view, the board
    geometry ``(cols, rows, square_size_m)``, the ``image_size``, and -- when given --
    the solved result fields. The operator can send us this single file to reproduce
    a real-device failure exactly. Pure numpy I/O (no cv2).

    Returns the resolved path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Stack the per-view (N,2) corner sets into (V, N, 2) arrays for compact storage.
    obj = np.stack([np.asarray(o, dtype=np.float64).reshape(-1, 3)
                    for o in object_points], axis=0)
    il = np.stack([np.asarray(p, dtype=np.float64).reshape(-1, 2)
                   for p in img_left], axis=0)
    ir = np.stack([np.asarray(p, dtype=np.float64).reshape(-1, 2)
                   for p in img_right], axis=0)
    payload: dict[str, np.ndarray] = {
        "object_points": obj,
        "img_left": il,
        "img_right": ir,
        "pattern_cols": np.asarray(pattern_cols),
        "pattern_rows": np.asarray(pattern_rows),
        "square_size_m": np.asarray(float(square_size_m)),
        "image_size": np.asarray(image_size, dtype=np.int64),
    }
    if result is not None:
        payload.update({
            "K_l": result.K_l, "dist_l": result.dist_l,
            "K_r": result.K_r, "dist_r": result.dist_r,
            "R": result.R, "T": result.T,
            "rms_l": np.asarray(result.rms_l),
            "rms_r": np.asarray(result.rms_r),
            "stereo_rms": np.asarray(result.stereo_rms),
            "n_views_used": np.asarray(result.n_views_used),
            "calibrate_flags": np.asarray(result.calibrate_flags),
            "ok": np.asarray(result.ok),
            "failure_reason": np.asarray(result.failure_reason),
        })
    np.savez(out, **payload)
    return out.resolve()


def _maybe_dump(dump_path, object_points, img_left, img_right, pattern_cols,
                pattern_rows, square_size_m, image_size, result) -> None:
    """Dump the views to ``dump_path`` if given; LOG the path; never break the solve.

    A calibration is infrequent, so the dump is always-on when a path is supplied. A
    dump failure (full disk, bad path) is logged but never propagated -- it must not
    sink an otherwise-good solve.
    """
    if dump_path is None:
        return
    import logging
    log = logging.getLogger(__name__)
    try:
        written = dump_views(dump_path, object_points, img_left, img_right,
                             pattern_cols, pattern_rows, square_size_m,
                             image_size, result)
        log.info("calib debug dump written: %s (%d views, ok=%s)",
                 written, len(object_points), result.ok)
    except Exception as exc:                                          # noqa: BLE001
        log.warning("calib debug dump to %s failed: %s", dump_path, exc)
