"""Pure checkerboard-target generator for camera calibration (Phase 1).

The operator prints this board (or shines it fullscreen on a monitor) and points
the OAK-D at it; later phases capture frames + solve intrinsics/extrinsics.

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

PURITY
------
No OpenCV, no Qt in this module -- it returns a plain ``numpy`` ``uint8`` array and
saves through the project's pure-Python PNG codec
(:func:`ui.comms.lib.misc.pngio.imwrite_gray`). OpenCV appears only in the
self-test as a detection oracle; Qt appears only behind the optional ``--show``
flag (lazy-imported in :func:`_show_fullscreen`).
"""
from __future__ import annotations

from pathlib import Path

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


def save_checkerboard(path: str | Path, img: np.ndarray) -> None:
    """Save a generated board as an 8-bit grayscale PNG (pure-Python codec).

    Thin wrapper over :func:`ui.comms.lib.misc.pngio.imwrite_gray` so callers do
    not need OpenCV / Pillow. PNG is lossless, so the saved pixels are byte-exact.
    """
    # Imported here (not at module top) only to keep the dependency surface of the
    # pure array generator obvious; the import itself is still pure-Python.
    from ui.comms.lib.misc.pngio import imwrite_gray

    imwrite_gray(path, img)


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


def _show_fullscreen(path: str | Path) -> None:
    """Open ``path`` fullscreen via Qt (optional ``--show`` workflow).

    Qt is LAZY-imported here so the core generator + the CLI save path stay
    headless / testable on a machine with no display. Blocks until the window is
    closed (press Esc or close it).
    """
    # Local imports: nothing here is reachable unless --show is passed. The
    # project is on PyQt6, where enums are SCOPED (Qt.AlignmentFlag.*,
    # Qt.Key.*), QShortcut lives in QtGui (not QtWidgets), and exec() has no
    # trailing underscore.
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeySequence, QPixmap, QShortcut
    from PyQt6.QtWidgets import QApplication, QLabel

    app = QApplication.instance() or QApplication([])
    label = QLabel()
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet("background-color: white;")
    pix = QPixmap(str(path))
    if pix.isNull():
        raise RuntimeError(f"Qt could not load the PNG for --show: {path}")
    label.setPixmap(pix)
    # Esc closes the fullscreen window so the operator is not stuck.
    QShortcut(QKeySequence(Qt.Key.Key_Escape), label, activated=label.close)
    label.showFullScreen()
    app.exec()


def _build_arg_parser() -> "object":
    """Construct the CLI argument parser (factored out for clarity)."""
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        prog="python -m ui.mathlib.calib.checkerboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Generate a printable / displayable checkerboard calibration target.

            cols/rows are INNER CORNERS (OpenCV patternSize convention), NOT
            squares: 9x6 inner corners == a 10x7-square board == 54 corners.

            PRINT vs SCREEN accuracy:
              * PRINT: use --square-mm + --dpi to size the squares, then print the
                PNG at 100% (turn OFF "fit to page"). The printed square then
                really is square-mm wide -- use that value as the calibration scale.
              * SCREEN ("roi vao man hinh"): pixel->mm depends on your monitor's
                pixel pitch, so --square-mm/--dpi do NOT give the true on-screen
                size. After displaying fullscreen you MUST measure one displayed
                square with a ruler and feed that measured mm value to the solver.
            """),
    )
    parser.add_argument("--cols", type=int, default=9,
                        help="inner corners across (default: 9)")
    parser.add_argument("--rows", type=int, default=6,
                        help="inner corners down (default: 6)")
    parser.add_argument(
        "--square-px", type=int, default=100,
        help="pixels per square (default: 100; ignored if --square-mm given)")
    parser.add_argument(
        "--margin-squares", type=float, default=1.0,
        help="white quiet-zone border, in squares (default: 1; min 1)")
    parser.add_argument(
        "--square-mm", type=float, default=None,
        help="physical square size in mm for PRINTING; with --dpi sets square-px")
    parser.add_argument(
        "--dpi", type=float, default=300.0,
        help="printer DPI used with --square-mm (default: 300)")
    parser.add_argument("--out", type=str, default="checkerboard.png",
                        help="output PNG path (default: checkerboard.png)")
    parser.add_argument(
        "--show", action="store_true",
        help="after saving, open the PNG fullscreen via Qt (needs a display)")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    """CLI entry point: render the board, save a PNG, print the physical size.

    Returns a process exit code (0 = success). Kept import-light: ``argparse`` and
    (optionally) Qt are imported lazily so merely importing this module stays pure.
    """
    args = _build_arg_parser().parse_args(argv)

    # Resolve pixels-per-square: --square-mm (+ --dpi) wins, else --square-px.
    if args.square_mm is not None:
        square_px = square_px_from_mm(args.square_mm, args.dpi)
        # Echo the exact printed size: round-tripping mm->px->mm exposes any
        # quantization so the operator uses the TRUE printed value, not the asked.
        printed_mm = square_px / args.dpi * 25.4
        size_note = (
            f"each square = {square_px} px "
            f"= {printed_mm:.3f} mm when PRINTED at {args.dpi:g} DPI (100% scale)")
    else:
        square_px = args.square_px
        size_note = f"each square = {square_px} px (no physical size requested)"

    img = make_checkerboard(
        cols=args.cols,
        rows=args.rows,
        square_px=square_px,
        margin_squares=args.margin_squares,
    )
    save_checkerboard(args.out, img)

    out_path = Path(args.out).resolve()
    h, w = img.shape
    n_sq_x, n_sq_y = args.cols + 1, args.rows + 1
    print(f"Saved {out_path}")
    print(
        f"  pattern : {args.cols}x{args.rows} INNER CORNERS "
        f"({n_sq_x}x{n_sq_y} squares, {args.cols * args.rows} corners)")
    print(f"  image   : {w}x{h} px (margin {int(round(args.margin_squares * square_px))} px)")
    print(f"  size    : {size_note}")
    print(
        "  SCREEN  : if you display this fullscreen, MEASURE one square with a "
        "ruler and use that mm value for calibration -- the px/DPI figure above "
        "is the PRINT size only.")

    if args.show:
        _show_fullscreen(out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI, not tests
    raise SystemExit(main())
