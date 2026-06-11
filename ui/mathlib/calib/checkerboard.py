"""Per-project I/O wrapper around the shared checkerboard generator.

The PURE board math -- :func:`~sky.calib.checkerboard.make_checkerboard` and
:func:`~sky.calib.checkerboard.square_px_from_mm` -- lives in the shared
:mod:`sky.calib` leaf library (numpy-only, so it stays movable). This module is
the IMPURE half that genuinely needs the ``ui`` process: it saves the board to a
PNG through the project's pure-Python PNG codec
(:func:`ui.comms.lib.misc.pngio.imwrite_gray`), optionally previews it fullscreen
via Qt, and exposes the operator CLI. Those edges (``ui.comms`` / PyQt6) are why
the wrapper stays per-project -- moving them into ``sky.*`` would break the leaf
rule (see ``docs/CONSOLIDATION_PLAN.md``).

The pure generators are re-exported here so existing ``ui`` call-sites and the CLI
keep importing them from this module unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Re-export the pure generators from the shared library so this module remains the
# single ``ui``-side entry point for the checkerboard workflow.
from sky.calib.checkerboard import make_checkerboard, square_px_from_mm

__all__ = ["make_checkerboard", "square_px_from_mm", "save_checkerboard"]


def save_checkerboard(path: str | Path, img: np.ndarray) -> None:
    """Save a generated board as an 8-bit grayscale PNG (pure-Python codec).

    Thin wrapper over :func:`ui.comms.lib.misc.pngio.imwrite_gray` so callers do
    not need OpenCV / Pillow. PNG is lossless, so the saved pixels are byte-exact.
    This is the ``ui``-coupled half of the generator (see the module docstring).
    """
    # Imported here (not at module top) only to keep the dependency surface of the
    # save path obvious; the import itself is still pure-Python.
    from ui.comms.lib.misc.pngio import imwrite_gray

    imwrite_gray(path, img)


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
