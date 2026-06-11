#!/usr/bin/env python3
"""Self-test for the pure checkerboard calibration-target generator.

The generator (:mod:`ui.mathlib.calib.checkerboard`) is cv2-free / Qt-free: it
returns a ``uint8`` grayscale board and saves it via the project's pure-Python PNG
codec. Per the repo's "cv2 survives only as a dev-time oracle in self-tests"
stance, OpenCV appears HERE ONLY -- as the detection oracle that is the REAL gate:

    cv2.findChessboardCorners(board, (cols, rows)) -> found is True
    and exactly cols * rows corners.

A geometry-only check is not enough: a board that detects in OpenCV is the actual
deliverable, so we prove detection on the canonical 9x6 board AND on parameter
variations (so the result is not hard-coded), plus we round-trip the saved PNG
through the pure decoder to confirm the file an operator prints/displays is the
same array we detected.

Run:  .venv/bin/python -m ui.tests.checkerboard_selftest
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2          # dev/test ORACLE only -- never imported by the generator
import numpy as np

from ui.comms.lib.misc.pngio import imread_gray
from ui.mathlib.calib.checkerboard import (
    make_checkerboard,
    save_checkerboard,
    square_px_from_mm,
)


def _assert_geometry(img: np.ndarray, cols: int, rows: int,
                     square_px: int, margin_squares: float) -> None:
    """Shape, white border, and interior-alternation invariants."""
    margin_px = int(round(margin_squares * square_px))
    exp_h = (rows + 1) * square_px + 2 * margin_px
    exp_w = (cols + 1) * square_px + 2 * margin_px
    assert img.dtype == np.uint8, f"dtype {img.dtype} != uint8"
    assert img.shape == (exp_h, exp_w), f"shape {img.shape} != {(exp_h, exp_w)}"

    # The whole quiet-zone border ring must be pure white (255).
    assert (img[:margin_px, :] == 255).all(), "top margin not white"
    assert (img[-margin_px:, :] == 255).all(), "bottom margin not white"
    assert (img[:, :margin_px] == 255).all(), "left margin not white"
    assert (img[:, -margin_px:] == 255).all(), "right margin not white"

    # Top-left square is white; its diagonal neighbour (1,1) is also white;
    # the squares at (1,0) and (0,1) are black -- this is the (i+j)-odd parity.
    def square_mean(i: int, j: int) -> float:
        y0 = margin_px + j * square_px
        x0 = margin_px + i * square_px
        return float(img[y0:y0 + square_px, x0:x0 + square_px].mean())

    assert square_mean(0, 0) == 255.0, "top-left square must be white"
    assert square_mean(1, 1) == 255.0, "square (1,1) must be white (parity)"
    assert square_mean(1, 0) == 0.0, "square (1,0) must be black (parity)"
    assert square_mean(0, 1) == 0.0, "square (0,1) must be black (parity)"


def _assert_detects(img: np.ndarray, cols: int, rows: int) -> int:
    """REAL gate: OpenCV must find exactly cols*rows inner corners."""
    found, corners = cv2.findChessboardCorners(img, (cols, rows))
    assert found, f"cv2.findChessboardCorners failed on {cols}x{rows} board"
    n = 0 if corners is None else corners.shape[0]
    assert n == cols * rows, f"expected {cols * rows} corners, got {n}"
    return n


def test_canonical_9x6() -> None:
    """The spec's canonical case: 9x6 inner corners, 80px squares, 1-sq margin."""
    cols, rows, sq, margin = 9, 6, 80, 1.0
    img = make_checkerboard(cols, rows, square_px=sq, margin_squares=margin)
    _assert_geometry(img, cols, rows, sq, margin)
    n = _assert_detects(img, cols, rows)
    print(f"[ok] 9x6 @80px: shape={img.shape}, cv2 found={True}, corners={n}")


def test_param_variations() -> None:
    """Different cols/rows/square_px prove the generator is not hard-coded."""
    cases = [
        (7, 5, 60, 1.0),
        (10, 7, 40, 2.0),   # wider margin
        (5, 4, 100, 1.5),
    ]
    for cols, rows, sq, margin in cases:
        img = make_checkerboard(cols, rows, square_px=sq, margin_squares=margin)
        _assert_geometry(img, cols, rows, sq, margin)
        n = _assert_detects(img, cols, rows)
        print(f"[ok] {cols}x{rows} @{sq}px margin={margin}: corners={n}")


def test_png_roundtrip_detects() -> None:
    """The SAVED PNG (what the operator uses) decodes back and still detects."""
    cols, rows, sq = 9, 6, 80
    img = make_checkerboard(cols, rows, square_px=sq, margin_squares=1.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "board.png"
        save_checkerboard(path, img)
        decoded = imread_gray(path)
        # Pure-Python PNG is lossless: byte-for-byte identical to the source.
        assert np.array_equal(decoded, img), "PNG round-trip changed pixels"
        n = _assert_detects(decoded, cols, rows)
    print(f"[ok] PNG round-trip lossless + detects: corners={n}")


def test_square_px_from_mm() -> None:
    """Print-sizing helper: 25mm @300DPI -> round(25/25.4*300) = 295 px."""
    assert square_px_from_mm(25.0, 300.0) == 295
    assert square_px_from_mm(30.0, 96.0) == round(30.0 / 25.4 * 96.0)
    print("[ok] square_px_from_mm: 25mm@300dpi -> 295 px")


def test_validation() -> None:
    """Guard rails: bad params must raise (especially the required quiet zone)."""
    bad = [
        dict(cols=1, rows=6),                       # < 2 inner corners
        dict(cols=9, rows=6, square_px=0),          # non-positive square
        dict(cols=9, rows=6, margin_squares=0.0),   # missing quiet zone
    ]
    for kwargs in bad:
        try:
            make_checkerboard(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    print("[ok] validation: cols<2 / square_px<1 / margin<1 all raise")


def main() -> int:
    test_canonical_9x6()
    test_param_variations()
    test_png_roundtrip_detects()
    test_square_px_from_mm()
    test_validation()
    print("\nALL CHECKERBOARD SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
