"""Pure checkerboard-target generator for camera calibration.

The operator prints this board (or shines it fullscreen on a monitor) and points
the camera at it; later phases capture frames + solve intrinsics/extrinsics.

CONVENTION -- inner corners, NOT squares (the #1 source of calibration confusion)
---------------------------------------------------------------------------------
``patternSize`` in OpenCV (``cv2.findChessboardCorners(img, patternSize)``) is the
number of **inner corners**, i.e. the points where four squares meet -- NOT the
number of squares. A board described as "9x6" for calibration has

    cols = 9, rows = 6   inner corners
    => 10 x 7            squares  (squares = inner_corners + 1 along each axis)
    => 9 * 6 = 54        detected corner points

:func:`make_checkerboard` takes ``(cols, rows)`` as those inner-corner counts, to
match OpenCV one-to-one.

WHY THE WHITE BORDER (quiet zone)
---------------------------------
``cv2.findChessboardCorners`` needs the board surrounded by a light quiet zone; a
checkerboard that runs to the image edge with no margin fails to detect. We always
draw the board on a white canvas with a configurable margin (default = one square)
on every side.

TOP-LEFT SQUARE COLOR
---------------------
The top-left square is WHITE by the standard convention used by the common
generators (and by OpenCV's own ``gen_pattern`` reference output), which keeps
detection orientation stable. A square at board grid ``(i, j)`` (i = column,
j = row, both 0-based) is therefore BLACK when ``(i + j)`` is odd.

PURITY (the leaf rule)
----------------------
This module is the PURE half of the generator: it imports only ``numpy`` and
returns plain ``uint8`` arrays, so it lives in the shared :mod:`sky.calib` leaf
library. The IMPURE wrapper that saves a PNG (via the project's pure-Python codec)
and the optional Qt fullscreen preview + CLI stay per-project in
``ui/mathlib/calib/checkerboard.py``, which calls :func:`make_checkerboard` here.
OpenCV appears only in the self-tests as a detection oracle.
"""
from __future__ import annotations

import numpy as np

# Pixel intensities for the two square colors (8-bit grayscale).
_WHITE = np.uint8(255)
_BLACK = np.uint8(0)


def make_checkerboard(
    cols: int,
    rows: int,
    square_px: int = 100,
    margin_squares: float = 1.0,
) -> np.ndarray:
    """Render a calibration checkerboard to a ``uint8`` grayscale image.

    Parameters
    ----------
    cols, rows:
        INNER-CORNER counts (OpenCV ``patternSize`` convention). The drawn board
        has ``cols + 1`` squares horizontally and ``rows + 1`` squares vertically,
        and a correct detection yields exactly ``cols * rows`` corners. Both must
        be ``>= 2`` (you need at least a 2x2 corner grid to calibrate).
    square_px:
        Side length of each square in pixels (``>= 1``).
    margin_squares:
        Width of the white quiet-zone border on every side, expressed in squares.
        Must be ``>= 1`` so ``cv2.findChessboardCorners`` can lock on. The border
        is rounded to a whole number of pixels (``round(margin_squares *
        square_px)``).

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` ``uint8`` array, where::

            margin_px = round(margin_squares * square_px)
            W = (cols + 1) * square_px + 2 * margin_px
            H = (rows + 1) * square_px + 2 * margin_px

        The top-left square is white; squares at grid ``(i, j)`` with ``(i + j)``
        odd are black.
    """
    if cols < 2 or rows < 2:
        raise ValueError(
            f"cols/rows are INNER-CORNER counts and must be >= 2; "
            f"got cols={cols}, rows={rows}")
    if square_px < 1:
        raise ValueError(f"square_px must be >= 1, got {square_px}")
    if margin_squares < 1:
        raise ValueError(
            f"margin_squares must be >= 1 (the quiet zone is REQUIRED for "
            f"cv2.findChessboardCorners), got {margin_squares}")

    # Squares span = inner corners + 1 along each axis.
    n_sq_x = cols + 1
    n_sq_y = rows + 1
    margin_px = int(round(margin_squares * square_px))

    width = n_sq_x * square_px + 2 * margin_px
    height = n_sq_y * square_px + 2 * margin_px

    # Start from an all-white canvas: this gives the quiet-zone border for free
    # and means we only have to paint the black squares.
    img = np.full((height, width), _WHITE, dtype=np.uint8)

    for j in range(n_sq_y):           # board row (top -> bottom)
        for i in range(n_sq_x):       # board column (left -> right)
            # Top-left square (i=j=0) is white; flip on every step => black when
            # (i + j) is odd. This is the standard, detection-stable parity.
            if (i + j) % 2 == 1:
                y0 = margin_px + j * square_px
                x0 = margin_px + i * square_px
                img[y0:y0 + square_px, x0:x0 + square_px] = _BLACK
    return img


def square_px_from_mm(square_mm: float, dpi: float) -> int:
    """Pixels per square for PRINTING ``square_mm`` squares at ``dpi``.

    ``25.4`` mm per inch, so ``square_px = round(square_mm / 25.4 * dpi)``. This is
    accurate only when the PNG is printed at 100% scale (no "fit to page"). On a
    SCREEN it is meaningless -- the real square size then depends on the monitor's
    pixel pitch and must be measured with a ruler.
    """
    if square_mm <= 0:
        raise ValueError(f"square_mm must be > 0, got {square_mm}")
    if dpi <= 0:
        raise ValueError(f"dpi must be > 0, got {dpi}")
    return int(round(square_mm / 25.4 * dpi))
