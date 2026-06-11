"""Checkerboard corner detection (Phase 3 -- the calibration math core).

The operator points the camera at the printed/displayed board from Phase 1
(:mod:`sky.calib.checkerboard`); this module turns one grayscale frame into the
subpixel inner-corner coordinates the stereo solve consumes.

cv2 POLICY -- the flight runtime (VIO / SLAM / depth) is cv2-free and STAYS so
-----------------------------------------------------------------------------
The calibration wizard is an operator/dev tool, NOT a flight path, so OpenCV is
acceptable here. To keep importing :mod:`sky.calib` from EVER pulling cv2
into the flight processes, ``cv2`` is **lazy-imported inside the functions** that
need it -- never at module top. Merely importing this module (or the package) must
not load OpenCV; the self-test asserts ``cv2`` stays out of ``sys.modules`` on a
bare package import.

CONVENTION -- inner corners, NOT squares (matches the Phase-1 generator)
-----------------------------------------------------------------------
``pattern_cols`` / ``pattern_rows`` are the number of INNER corners (where four
squares meet), i.e. OpenCV's ``patternSize`` one-to-one. A "9x6" board therefore
yields ``9 * 6 = 54`` corner points.
"""
from __future__ import annotations

import numpy as np

# Subpixel-refinement window: half-side of the search neighbourhood, in pixels.
# (5,5) -> an 11x11 window, the OpenCV-tutorial default; large enough to lock onto
# a corner at the OAK-D's resolution, small enough not to swallow a neighbour.
_SUBPIX_WIN = (5, 5)
# (-1,-1) disables the dead zone in the centre of the search window.
_SUBPIX_ZERO_ZONE = (-1, -1)


def detect_corners(
    gray: np.ndarray,
    pattern_cols: int,
    pattern_rows: int,
) -> np.ndarray | None:
    """Detect + subpixel-refine the inner checkerboard corners in one frame.

    Runs :func:`cv2.findChessboardCorners` with the robust flag set (adaptive
    threshold + intensity normalisation, which together handle uneven lighting and
    contrast across the board) and then :func:`cv2.cornerSubPix` to refine the
    integer corners to subpixel accuracy -- subpixel refinement is what makes the
    intrinsics fit converge to a sub-pixel reprojection RMS.

    Parameters
    ----------
    gray:
        ``(H, W)`` ``uint8`` grayscale image. A non-2-D or non-``uint8`` array is a
        programming error and raises ``ValueError`` (cv2 would otherwise fail with
        an opaque message).
    pattern_cols, pattern_rows:
        INNER-corner counts (OpenCV ``patternSize`` convention). A correct
        detection returns exactly ``pattern_cols * pattern_rows`` corners.

    Returns
    -------
    numpy.ndarray | None
        ``(N, 2)`` ``float32`` array of subpixel ``(x, y)`` corner coordinates with
        ``N == pattern_cols * pattern_rows``, or ``None`` if the board was not found
        (partially occluded, too oblique, out of frame, ...).
    """
    # Lazy import: keeps `import sky.calib` cv2-free for the flight path.
    import cv2

    if gray.ndim != 2:
        raise ValueError(
            f"detect_corners expects a 2-D grayscale image, got shape {gray.shape}")
    if gray.dtype != np.uint8:
        # cv2.findChessboardCorners wants 8-bit; refuse silently-wrong inputs.
        raise ValueError(
            f"detect_corners expects a uint8 image, got dtype {gray.dtype}")

    pattern_size = (pattern_cols, pattern_rows)
    # ADAPTIVE_THRESH: tolerate lighting gradients across the board.
    # NORMALIZE_IMAGE: equalise before thresholding for low-contrast captures.
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found or corners is None:
        return None

    # Refine to subpixel. Terminate on either 30 iterations or a 0.001-px move,
    # whichever comes first (the standard, well-behaved criteria).
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )
    # cornerSubPix refines IN PLACE on a float32 (N,1,2) buffer; copy so we never
    # mutate the caller's array, and so the returned data owns its memory.
    refined = corners.astype(np.float32).copy()
    cv2.cornerSubPix(gray, refined, _SUBPIX_WIN, _SUBPIX_ZERO_ZONE, criteria)

    # Collapse the cv2 (N,1,2) layout to a plain (N,2) array for our solve/collector.
    return refined.reshape(-1, 2)


# --------------------------------------------------------------------------- #
# L<->R corner-ORDER reconciliation -- the PRIMARY real-device garbage root cause.
#
# `cv2.findChessboardCorners` returns the inner corners in raster order, but a
# `cols x rows` board has a 180-degree corner-order AMBIGUITY: depending on which
# way round the detector locked onto the board, the SAME physical grid can come back
# either as the list `c` or as its full reversal `c[::-1]` (a 180-degree rotation of
# the index map -- corner[k] <-> corner[N-1-k]). The detector is run INDEPENDENTLY on
# the left and right images, so a board imaged at different in-plane rotations on the
# two cameras can return OPPOSITE orderings. If those mismatched arrays reach
# `cv2.stereoCalibrate`, object[k]/left[k] and right[k] no longer name the same board
# point: the solve "explains" the 180-degree swap as a ~180-degree inter-camera
# rotation and a runaway baseline (~1 m instead of ~75 mm) -- exactly the real-device
# failure (`baseline 961 mm`, `R ~ 168 deg`). The wide-FOV model fix does NOT cure
# this; the L/R arrays must be made to correspond BEFORE the solve.
#
# The two valid orderings differ by a full reversal, so reconciliation is a binary
# choice: keep R as-is, or reverse it. We pick the orientation whose grid runs the
# SAME way as the left grid (see `_grid_axes` / `reconcile_lr`).
# --------------------------------------------------------------------------- #
def _grid_axes(corners: np.ndarray, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """The two in-image grid-axis vectors of a raster-order corner set.

    ``corners`` is the ``(rows*cols, 2)`` row-major grid. ``corner[1]-corner[0]``
    steps one column (the fast/inner axis), ``corner[cols]-corner[0]`` steps one row
    (the slow/outer axis). Returned as ``(col_axis, row_axis)`` unit-ish vectors -- the
    full board span, not normalised, so a longer board dominates the dot product used
    to compare orientations.
    """
    col_axis = corners[1] - corners[0]
    row_axis = corners[cols] - corners[0]
    return col_axis, row_axis


def reconcile_lr(
    corners_l: np.ndarray,
    corners_r: np.ndarray,
    cols: int,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a stereo pair's L and R corners name the SAME physical board points.

    Resolves the 180-degree corner-order ambiguity (see the module note above): if the
    right array is reversed relative to the left, this returns the right array reversed
    back so ``corners_l[k]`` and the returned ``corners_r[k]`` are the same board
    corner for every ``k``. The LEFT array is returned UNCHANGED (it is the reference);
    the object points are tied to the left raster order, so only the right needs to
    follow it.

    Detection (robust to in-plane board rotation, no homography needed)
    ------------------------------------------------------------------
    A full reversal ``c -> c[::-1]`` negates BOTH grid-axis vectors
    (:func:`_grid_axes`), so a handedness/cross-product test is blind to it (the cross
    product is bilinear and keeps its sign). Instead we compare the grid-axis
    DIRECTIONS between the two cameras. For a normal stereo rig the cameras differ only
    by a small rotation + a horizontal baseline, so the board's row/column axes point
    in nearly the SAME image direction in both views; a 180-degree reversal flips both
    of R's axes to point OPPOSITE the left's. We therefore reverse R iff that opposite
    orientation is the better-aligned one::

        align(c)  = dot(col_axis_L, col_axis_c) + dot(row_axis_L, row_axis_c)

    and keep whichever of ``corners_r`` / ``corners_r[::-1]`` gives the larger
    ``align`` (reversing R negates both its axes, so the two scores are exact
    negatives -- the sign of ``align(corners_r)`` alone decides). This is exact for a
    fronto-parallel-ish rig and stays correct under the moderate inter-camera rotation
    and the per-view board tilts a real OAK-D capture sees, because a true 180-degree
    swap changes the alignment by ~2x the board span -- far larger than any tilt
    perturbs it.

    Parameters
    ----------
    corners_l, corners_r:
        ``(N, 2)`` raster-order corner sets from :func:`detect_corners`, with
        ``N == cols * rows``.
    cols, rows:
        INNER-corner counts (OpenCV ``patternSize``). Used to index the row axis.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(corners_l, corners_r_reconciled)`` -- the left unchanged, the right either
        as-is or fully reversed so it corresponds to the left index-for-index.
    """
    n = cols * rows
    if corners_l.shape != (n, 2) or corners_r.shape != (n, 2):
        raise ValueError(
            f"reconcile_lr expects ({n}, 2) corner sets, got "
            f"L={corners_l.shape} R={corners_r.shape}")

    col_l, row_l = _grid_axes(corners_l, cols)
    col_r, row_r = _grid_axes(corners_r, cols)
    # Alignment of R's grid axes with L's. Reversing R negates col_r and row_r, so the
    # reversed alignment is exactly -align; a negative align means the reversed R is the
    # better match and R must be flipped to agree with L.
    align = float(np.dot(col_l, col_r) + np.dot(row_l, row_r))
    if align < 0.0:
        return corners_l, corners_r[::-1].copy()
    return corners_l, corners_r
