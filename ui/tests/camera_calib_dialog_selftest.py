#!/usr/bin/env python3
"""Offscreen Qt selftest for the Phase-4 stereo camera-calibration WIZARD.

NO device, NO real photos: a FAKE stereo stream (a stub exposing the wizard's four
touch-points -- ``start`` / ``stop`` / ``.error`` / ``.device_id``) hands the wizard
synthetic checkerboard image PAIRS. The pairs are REAL Phase-1 boards perspective-
warped to varied tilts (the way the Phase-3 collector test builds them), so the
wizard's collector runs the GENUINE ``detect_corners`` cv2 path end-to-end -- this
is an integration test of the dialog glue, not a re-test of the math core.

We never enter the Qt event loop: the test calls the wizard's own ``_feed_pair``
(the recv-thread hook -- here called directly) and ticks ``_drain`` (the UI timer's
slot), exactly the split that makes the dialog offline-testable. The off-thread
solve worker IS started (real ``QThread``); we pump ``QCoreApplication`` events +
``wait()`` on it so its ``done`` signal is delivered without a running event loop.

Gates
-----
1. Capture advances and reaches ``complete`` ONLY after diverse + tilted coverage
   (a fronto-parallel-only sweep stays incomplete with a "tilt the board" guidance).
2. The off-thread solve runs and yields a sane result (baseline in a plausible band,
   low RMS), the live preview + corner overlay render (a non-trivial pixmap), and the
   calib_check verdict is PASS.
3. "Save" writes a calib.json that ``StereoCalib.from_json`` reloads with the right
   baseline and that ``calib_check`` PASSes.
4. The mono-guard path: a stream that reports ``.error`` surfaces it in the wizard.

Run::

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m ui.tests.camera_calib_dialog_selftest
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Force headless Qt BEFORE any Qt import (mirrors the other offscreen selftests).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

from imu_camera.io.reader import StereoCalib
from imu_camera.tools.calib_check import FAIL, run_checks
from ui.calib.checkerboard import make_checkerboard
from sky.calib.collector import CoverageStatus, FrameStatus
from ui.qt import theme
from ui.qt.camera_calib_dialog import CameraCalibWizard

# The wizard's Save also writes to this per-device store (used opt-in via
# --use-camera-calib); the store selftest covers it in isolation, here we assert the
# wizard wires it in.
from imu_camera.device import camera_calib_store

# --------------------------------------------------------------------------- #
# Synthetic STEREO board frames via a GROUND-TRUTH pinhole rasterizer.
# --------------------------------------------------------------------------- #
# The wizard's collector runs the genuine ``detect_corners`` cv2 path on whatever
# images we hand it, and the solve is only sane if those images are GEOMETRICALLY
# CONSISTENT stereo pairs (a real baseline, square pixels). A simple 2D image warp
# is NOT a consistent pinhole projection across views, so it gives a degenerate
# (~0 baseline, non-square) solve. Instead we RASTERIZE each view through a real GT
# pinhole model: for a PLANAR board at Z=0, the canonical board-pixel -> image-pixel
# map IS the homography ``H = K [r1 r2 t]``. Warping the Phase-1 board image by that
# H (per camera, with the GT LEFT->RIGHT extrinsic) yields detectable images whose
# corners encode the SAME GT geometry -- so the solve recovers GT intrinsics + a real
# ~75 mm baseline at sub-pixel RMS, exactly like a real capture would.
PATTERN_COLS, PATTERN_ROWS = 9, 6
SQUARE_MM = 25.0
IMAGE_SIZE = (640, 400)               # (width, height) -- matches the wizard's W/H

# GT rig (OAK-D-like): fx~=fy~=285, centred principal point, 7.5 cm baseline.
_GT_K_L = np.array([[285.0, 0.0, 320.0],
                    [0.0, 285.0, 200.0],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
_GT_K_R = np.array([[285.0, 0.0, 320.0],
                    [0.0, 285.0, 200.0],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
_GT_BASELINE_M = 0.075
# Near-parallel rig, right camera translated along -X (right cam sits to the right).
_GT_R_LR = np.eye(3, dtype=np.float64)
_GT_T_LR = np.array([-_GT_BASELINE_M, 0.0, 0.0], dtype=np.float64)

# Board-image pixels-per-square: the canonical Phase-1 board we warp.
_BOARD_SQ_PX = 30
_BOARD_MARGIN_SQ = 1.0
# Board metric scale: each board square is SQUARE_MM in the world.
_SQUARE_M = SQUARE_MM / 1000.0


def _euler_xyz_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Rotation from XYZ Euler tilts (deg) -- board orientation in the LEFT frame."""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _board_pixel_to_world_H() -> np.ndarray:
    """Homography mapping canonical board-IMAGE pixels -> board-PLANE metres (Z=0).

    The Phase-1 board image has its top-left INNER corner at pixel
    ``(margin_px + sq_px, margin_px + sq_px)`` and corners spaced ``sq_px`` apart;
    the solve's world board has inner corner ``(i, j)`` at ``(i, j) * square_m``.
    This 3x3 (affine) homography sends a board-image pixel to its world XY in metres,
    so composing it with the camera projection gives the full image->image warp.
    """
    margin_px = int(round(_BOARD_MARGIN_SQ * _BOARD_SQ_PX))
    # Board-image pixel of inner corner (0,0) and the per-square pixel step.
    origin = margin_px + _BOARD_SQ_PX
    s = _SQUARE_M / _BOARD_SQ_PX        # metres per board-image pixel
    # world_x = (px - origin) * s ; world_y = (py - origin) * s
    return np.array([[s, 0.0, -origin * s],
                     [0.0, s, -origin * s],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _project_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Planar projection homography: world board-plane (Z=0) metres -> image pixels.

    For ``X = (x, y, 0)`` the pinhole maps it by ``K [r1 r2 t] (x, y, 1)^T`` -- the
    classic planar homography. ``t`` is the board origin in the camera frame.
    """
    H = K @ np.column_stack((R[:, 0], R[:, 1], t))
    return H


def _render_stereo_pair(rx, ry, rz, depth, xo, yo
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize ONE board pose into a geometrically-consistent LEFT/RIGHT image pair.

    Composes board-image-pixel -> world (``_board_pixel_to_world_H``) with world ->
    image (``_project_homography``) for each camera, then ``warpPerspective``-s the
    canonical board into the 640x400 frame. The RIGHT camera uses the GT LEFT->RIGHT
    extrinsic, so the pair has a real ~75 mm baseline; tilt (rx, ry) gives the skew
    the collector's coverage gate needs.
    """
    import cv2          # dev/test ORACLE only (lazy: QApplication must exist first)

    board = make_checkerboard(PATTERN_COLS, PATTERN_ROWS, square_px=_BOARD_SQ_PX,
                              margin_squares=_BOARD_MARGIN_SQ)
    bh, bw = board.shape

    R_bl = _euler_xyz_to_R(rx, ry, rz)         # board -> left camera
    # Place the board centre near (xo, yo, depth) in the left frame.
    bx = (PATTERN_COLS - 1) * _SQUARE_M * 0.5
    by = (PATTERN_ROWS - 1) * _SQUARE_M * 0.5
    t_bl = np.array([xo - bx, yo - by, depth], dtype=np.float64)

    Hb2w = _board_pixel_to_world_H()
    out = []
    for K, R_lr, t_lr in ((_GT_K_L, np.eye(3), np.zeros(3)),
                          (_GT_K_R, _GT_R_LR, _GT_T_LR)):
        # board -> this camera: chain board->left then left->this-cam.
        R_bc = R_lr @ R_bl
        t_bc = R_lr @ t_bl + t_lr
        Hw2i = _project_homography(K, R_bc, t_bc)
        H = Hw2i @ Hb2w                          # board-image px -> camera image px
        H = H / H[2, 2]
        img = cv2.warpPerspective(
            board, H, IMAGE_SIZE, borderValue=255,
            flags=cv2.INTER_LINEAR)
        out.append(img)
    return out[0], out[1]


# (rx, ry, rz, depth, xo, yo): tilt (rx/ry) for skew coverage + varied distance /
# in-frame translation so every view also clears the collector's novelty gate.
# Each pose differs from the others on >=2 axes (tilt + translate + zoom) so none is
# rejected as a near-duplicate -- the sweep accepts all 8 and clears the tilt gate.
_TILTED_SPECS = [
    (0, 0, 0, 0.50, 0.00, 0.00),
    (22, -10, 3, 0.42, -0.07, -0.05),
    (-20, 14, -5, 0.58, 0.08, -0.04),
    (15, 20, 6, 0.66, -0.06, 0.06),
    (-22, -16, -4, 0.40, 0.06, 0.06),
    (25, -22, 8, 0.70, 0.09, -0.07),
    (-18, 22, -10, 0.62, -0.09, 0.07),
    (28, 8, -6, 0.46, 0.02, -0.08),
]
# Fronto-parallel sweep (rx=ry=0 -> skew ~0): novel on distance/translation only,
# so it reaches the count but NEVER the tilt-coverage gate. Each pose combines a
# large in-frame shift AND a depth change so it clears the novelty gate without tilt.
_FLAT_SPECS = [
    (0, 0, 0, 0.45, 0.00, 0.00),
    (0, 0, 0, 0.70, 0.12, 0.00),
    (0, 0, 0, 0.40, -0.12, 0.00),
    (0, 0, 0, 0.62, 0.00, 0.09),
    (0, 0, 0, 0.50, 0.00, -0.09),
    (0, 0, 0, 0.66, 0.11, 0.07),
    (0, 0, 0, 0.43, -0.11, -0.07),
    (0, 0, 0, 0.55, 0.12, -0.06),
]


# --------------------------------------------------------------------------- #
# Fake stereo stream stub (the wizard's four touch-points only).
# --------------------------------------------------------------------------- #
class _FakeStereoStream:
    """Duck-type of IpcStereoRawSource: start/stop/.error/.device_id, no device.

    The wizard calls ``start(cb)`` (we just retain ``cb``; the TEST pushes frames),
    polls ``.error`` each drain, and reads ``.device_id`` -- nothing else. We can
    pre-set ``.error`` to exercise the mono-guard / connect-failure path.
    """

    def __init__(self, device_id: str = "selftest-cam",
                 error: str | None = None) -> None:
        self.device_id = device_id
        self.error = error
        self.started = False
        self.stopped = False
        self._cb = None

    def start(self, callback) -> None:
        self._cb = callback
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    # Test helper: deliver one synthetic pair through the wizard's recv hook.
    def emit(self, gray_left, gray_right, seq: int = 0) -> None:
        if self._cb is not None:
            self._cb(seq, seq * 1000, gray_left, gray_right)


# --------------------------------------------------------------------------- #
# One shared QApplication for the offscreen run.
# --------------------------------------------------------------------------- #
def _app() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv or ["selftest"])


def _settle_detection(wizard: CameraCalibWizard) -> None:
    """Block until the wizard's OFF-thread detection of the current pair lands.

    Detection now runs on a background :class:`_DetectWorker` (mirroring the solve
    worker), so ``_drain`` only ARMS it; the collector is fed asynchronously when the
    worker's ``done`` signal is delivered to ``_on_detected``. With no running event
    loop in this offscreen test we reproduce the timer's behaviour deterministically:
    wait for the worker, then pump queued slot invocations so ``done`` -> feed fires.
    """
    det = wizard._detector
    if det is not None:
        assert det.wait(20000), "detect worker did not finish in time"
    QCoreApplication.processEvents()    # deliver the queued done -> _on_detected


def _drive(wizard: CameraCalibWizard, stream: _FakeStereoStream, specs) -> None:
    """Push each spec's STEREO pair through emit()+_drain, settling the off-thread
    detection each step, until complete (or the specs run out)."""
    for i, spec in enumerate(specs):
        left, right = _render_stereo_pair(*spec)
        stream.emit(left, right, seq=i)
        wizard._drain()                 # UI-thread tick: arm off-thread detect + paint
        _settle_detection(wizard)       # wait out the detect + deliver its result
        if wizard._coll is not None and wizard._coll.complete:
            break


# --------------------------------------------------------------------------- #
# Gate 1: fronto-parallel sweep reaches the count but NOT complete (tilt guidance).
# --------------------------------------------------------------------------- #
def test_fronto_parallel_not_complete() -> None:
    _app()
    stream = _FakeStereoStream()
    wiz = CameraCalibWizard(None, device_id="selftest-cam",
                            width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], stream=stream)
    # Use a small target so the short sweep reaches the count.
    wiz._on_start()
    wiz._coll.cfg.n_target = 6
    assert stream.started, "wizard must start the injected stream on START"

    _drive(wiz, stream, _FLAT_SPECS)

    coll = wiz._coll
    assert coll.count_ok, f"fronto sweep should reach the count: {coll.accepted_count}"
    assert not coll.skew_ok, "fronto-parallel set must NOT satisfy tilt coverage"
    assert not coll.complete, "all-fronto-parallel data must NOT report complete"
    # The wizard's status line must carry the operator's tilt guidance.
    assert "tilt" in wiz._status.text().lower(), wiz._status.text()
    # No solve was kicked off (no result, save disabled).
    assert wiz._result is None and not wiz._save_btn.isEnabled()

    # --- B1: count bar shows RAW counts ("7/15"), not a bare percentage. The
    #     format string carries %v/%m and the range maps 0..n_target so %v is the
    #     real accepted count and %m the target (count_ok here -> value == max). ---
    assert wiz._count_bar.format() == "views %v/%m", wiz._count_bar.format()
    assert wiz._count_bar.maximum() == coll.cfg.n_target, (
        f"count bar max must equal n_target: {wiz._count_bar.maximum()} vs "
        f"{coll.cfg.n_target}")
    assert wiz._count_bar.value() == min(coll.accepted_count, coll.cfg.n_target), (
        f"count bar value must equal accepted count: {wiz._count_bar.value()} vs "
        f"{coll.accepted_count}")
    assert wiz._count_bar.text() == f"views {wiz._count_bar.value()}/" \
        f"{coll.cfg.n_target}", wiz._count_bar.text()
    # Count gate is satisfied here -> the count bar chunk reads GREEN.
    assert theme.GOOD in wiz._count_bar.styleSheet(), wiz._count_bar.styleSheet()

    # --- B1: the tilt bar must NOT read complete while skew_ok is False -- the
    #     label says "need more" and the fill is capped below full (the honest-
    #     semantics fix for the "one mild tilt = 100% hang" misread). ---
    assert wiz._tilt_bar.format() == "tilt: need more", wiz._tilt_bar.format()
    assert wiz._tilt_bar.value() < 100, (
        f"incomplete tilt bar must not read full: {wiz._tilt_bar.value()}")
    assert wiz._tilt_bar.value() <= 80, (
        f"incomplete tilt fill must be capped below full: {wiz._tilt_bar.value()}")
    # Tilt gate unmet -> the tilt bar chunk reads AMBER (not green).
    assert theme.WARN in wiz._tilt_bar.styleSheet(), wiz._tilt_bar.styleSheet()
    print(f"[ok] fronto-parallel: {coll.accepted_count} accepted, count_ok="
          f"{coll.count_ok} skew_ok={coll.skew_ok} -> NOT complete; "
          f"count_bar={wiz._count_bar.text()!r} (green={theme.GOOD in wiz._count_bar.styleSheet()}) "
          f"tilt_bar={wiz._tilt_bar.text()!r} fill={wiz._tilt_bar.value()}% "
          f"status guidance = {wiz._status.text()!r}")
    wiz.close()


# --------------------------------------------------------------------------- #
# Gate 2 + 3: tilted sweep completes, solve runs off-thread, sane result + PASS,
# preview renders, and Save writes a reloadable calib.json that calib_check PASSes.
# --------------------------------------------------------------------------- #
def _solve_tilted_wizard(stream: _FakeStereoStream,
                         device_id: str = "selftest-cam") -> CameraCalibWizard:
    """Drive a wizard to a COMPLETED, PASSING solve off the tilted synthetic sweep.

    Factored out so both the solve/save test and the per-device-store test exercise
    the SAME real capture -> off-thread solve path (no duplicated drive). Returns the
    wizard with ``_result`` set + a PASS verdict + Save enabled.
    """
    wiz = CameraCalibWizard(None, device_id=device_id,
                            width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], stream=stream)
    wiz._square_spin.setValue(SQUARE_MM)
    wiz._on_start()
    # Diverse-but-bounded synthetic sweep: the 8 tilted poses reliably yield 5
    # accepted (the rest fall just under the novelty gate, like a real operator's
    # near-repeats), so target 5 -- enough for a well-conditioned solve.
    wiz._coll.cfg.n_target = 5
    _drive(wiz, stream, _TILTED_SPECS)
    assert wiz._coll.complete, "tilted data must complete"
    assert wiz._solver is not None, "completion must start the off-thread solver"
    assert wiz._solver.wait(20000), "solve worker did not finish in time"
    QCoreApplication.processEvents()
    return wiz


def test_tilted_completes_solves_and_saves() -> None:
    _app()
    stream = _FakeStereoStream()
    wiz = CameraCalibWizard(None, device_id="selftest-cam",
                            width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], stream=stream)
    wiz._square_spin.setValue(SQUARE_MM)
    wiz._on_start()
    # Diverse-but-bounded synthetic sweep: the 8 tilted poses reliably yield 5
    # accepted (the rest fall just under the novelty gate, like a real operator's
    # near-repeats), so target 5 -- enough for a well-conditioned solve.
    wiz._coll.cfg.n_target = 5

    _drive(wiz, stream, _TILTED_SPECS)

    coll = wiz._coll
    assert coll.complete, (
        f"tilted data must complete: {coll.accepted_count} accepted, "
        f"skew_ok={coll.skew_ok}")
    assert coll.skew_ok, "tilted sweep must satisfy the tilt-coverage gate"
    # Reaching complete in _drain stops the stream and kicks the off-thread solve.
    assert not wiz._running, "stream must stop once the dataset is complete"
    assert wiz._solver is not None, "completion must start the off-thread solver"

    # --- B1: with skew_ok now True the tilt bar reads OK (full + green), no longer
    #     the "need more" amber state -- the honest-semantics flip on satisfaction. ---
    assert wiz._tilt_bar.format() == "tilt: OK", wiz._tilt_bar.format()
    assert wiz._tilt_bar.value() == 100, wiz._tilt_bar.value()
    assert theme.GOOD in wiz._tilt_bar.styleSheet(), wiz._tilt_bar.styleSheet()

    # --- B2: the final accepted frame armed the "view banked" confirmation latch,
    #     turning the preview border green for ~400 ms (it survives several ticks
    #     rather than a single ~33 ms frame). Ticking it down clears it cleanly. ---
    assert wiz._accept_hold > 0, "accept of the final view must arm the banked latch"
    assert theme.GOOD in wiz._preview.styleSheet(), (
        f"preview border must flash green on accept: {wiz._preview.styleSheet()!r}")
    # Decay the latch the way the UI timer would: feed status-less ticks until it
    # reverts, then assert the border drops back to the neutral (non-green) state.
    held = wiz._accept_hold
    for _ in range(held + 1):
        wiz._tick_accept_confirmation(None)
    assert wiz._accept_hold == 0, "banked latch must decay to 0 after its hold window"
    assert theme.GOOD not in wiz._preview.styleSheet(), (
        f"preview border must revert to neutral after the hold: "
        f"{wiz._preview.styleSheet()!r}")

    # The live preview must have rendered the LEFT frame with the corner overlay
    # (a non-null, non-trivial pixmap proves _show_left ran on the green markers).
    pix = wiz._preview.pixmap()
    assert pix is not None and not pix.isNull(), "preview pixmap must be rendered"
    assert pix.width() > 1 and pix.height() > 1, "preview pixmap must be non-trivial"

    # Pump the off-thread solve to completion WITHOUT a running event loop: wait()
    # on the worker, then drain queued slot invocations so `done` -> `_on_solved`
    # fires on this thread.
    assert wiz._solver.wait(20000), "solve worker did not finish in time"
    QCoreApplication.processEvents()

    result = wiz._result
    assert result is not None, "solve must produce a result (done signal delivered)"

    # --- Debug dump: the always-on .npz of the operator's REAL captured corners was
    #     written to a stable per-view-count path, and that path is shown to the
    #     operator (so they can send us the file). Verify both. ---
    assert wiz._dump_path is not None, "solve must set a debug-dump path"
    assert wiz._dump_path == f"/tmp/oakd_calib_views_{coll.accepted_count}.npz", (
        f"dump path must be stable + suffixed by the view count: {wiz._dump_path}")
    dump_p = Path(wiz._dump_path)
    assert dump_p.exists(), f"the debug dump must be written: {dump_p}"
    dump = np.load(dump_p, allow_pickle=False)
    assert dump["img_left"].shape[0] == coll.accepted_count, dump["img_left"].shape
    assert list(dump["image_size"]) == list(IMAGE_SIZE)
    dump_p.unlink()                                   # tidy the /tmp artefact
    # Sane numbers: GT-rasterized 25 mm squares on a ~285 px-focal rig -> the
    # detect+solve recovers a near-GT fit, so the RMS is sub-pixel and the baseline
    # sits right on the GT ~75 mm (in the plausible stereo band).
    assert result.stereo_rms < 1.0, f"stereo RMS too high: {result.stereo_rms:.3f} px"
    assert 0.02 <= result.baseline_m <= 0.30, (
        f"baseline {result.baseline_m * 1000:.2f} mm outside plausible 20-300 mm band")
    assert wiz._verdict is not None and wiz._verdict[0] == "PASS", (
        f"calib_check verdict should be PASS, got {wiz._verdict}")
    assert wiz._save_btn.isEnabled(), "save must be enabled after a good solve"

    # --- B3: the results line ECHOES the entered square size, so a 25.00-vs-25.40
    #     typo is visible after solving. The square was set to SQUARE_MM above. ---
    res_txt = wiz._result_lbl.text()
    assert f"square={SQUARE_MM:.2f} mm" in res_txt, (
        f"results line must echo the entered square size: {res_txt!r}")
    # --- H3: the EXPECTED baseline hint is echoed next to the measured one. ---
    assert "expect ~75 for OAK-D" in res_txt, (
        f"results line must echo the expected baseline: {res_txt!r}")
    # --- H3: the GT-rasterized solve lands a sub-pixel RMS and a ~75 mm baseline,
    #     so both key metrics colour GREEN (in-band) on the results line. ---
    assert theme.GOOD in res_txt, (
        f"in-band metrics must colour green on the results line: {res_txt!r}")
    print(f"[ok] tilted+solve: complete at {coll.accepted_count} views, "
          f"RMS L/R={result.rms_l:.3f}/{result.rms_r:.3f} stereo={result.stereo_rms:.3f} px, "
          f"baseline={result.baseline_m * 1000:.2f} mm, verdict={wiz._verdict[0]}; "
          f"results echoes square={SQUARE_MM:.2f} mm + expected baseline")

    # --- Save: write to a temp path, reload via the pipeline loader, assert the
    #     baseline round-trips and calib_check PASSes on the written file. ---
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "calib.json"
        # Drive the writer directly (the file-dialog path is operator-only); this is
        # the SAME call _on_save makes after the path is chosen.
        from sky.calib.writer import write_calib_json
        write_calib_json(result, wiz._image_size, out)
        assert out.exists()

        calib = StereoCalib.from_json(json.loads(out.read_text()))
        # cm/m round-trip: reloaded baseline == solved baseline to 1e-6 m.
        assert abs(calib.baseline_m - result.baseline_m) < 1e-6, (
            f"baseline round-trip broken: {calib.baseline_m * 1000:.6f} mm vs "
            f"{result.baseline_m * 1000:.6f} mm")
        checks = run_checks(calib, reader=None)
        fails = [c for c in checks if c.status == FAIL]
        assert not fails, "calib_check FAILED on the saved file:\n" + "\n".join(
            f"  {c.name}: {c.measured} -- {c.note}" for c in fails)
        print(f"[ok] save round-trip: reloaded baseline "
              f"{calib.baseline_m * 1000:.4f} mm (== solved), calib_check "
              f"{len(checks)} checks / 0 FAIL")
    wiz.close()


# --------------------------------------------------------------------------- #
# Gate 3b: the wizard's Save ALSO writes the calib to the per-device STORE, so the
# LIVE pipeline can apply it when run with --use-camera-calib (factory is the
# default). We drive the REAL _on_save (monkeypatching only the operator-facing file
# picker -> Cancel, so no file is written) with the store redirected to a TMP path,
# then assert load_camera_calib() returns the saved calib -- proving the wizard wires
# the store save in, WITHOUT polluting the real .cache/camera_calib.json.
# --------------------------------------------------------------------------- #
def test_save_writes_to_per_device_store() -> None:
    _app()
    from PyQt6.QtWidgets import QFileDialog

    device_id = "selftest-store-cam"
    stream = _FakeStereoStream(device_id=device_id)
    wiz = _solve_tilted_wizard(stream, device_id=device_id)
    assert wiz._result is not None and wiz._verdict[0] == "PASS", wiz._verdict
    result = wiz._result

    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "camera_calib.json"
        # Redirect the store to a throwaway path so the REAL .cache is untouched, and
        # force the file picker to Cancel so _on_save persists ONLY to the store. The
        # store save runs BEFORE the picker, so a cancelled export still persists.
        orig_default = camera_calib_store._DEFAULT_PATH
        orig_picker = QFileDialog.getSaveFileName
        camera_calib_store._DEFAULT_PATH = store_path
        QFileDialog.getSaveFileName = staticmethod(  # type: ignore[assignment]
            lambda *a, **k: ("", ""))               # operator hit Cancel
        try:
            # Pre-state: nothing saved for this device yet.
            assert camera_calib_store.load_camera_calib(device_id) is None
            wiz._on_save()                          # the REAL Save handler

            saved = camera_calib_store.load_camera_calib(device_id)
            assert isinstance(saved, StereoCalib), (
                f"wizard Save must persist to the per-device store: {saved!r}")
            # The persisted calib matches the solved one (baseline round-trips cm->m).
            assert abs(saved.baseline_m - result.baseline_m) < 1e-6, (
                f"stored baseline {saved.baseline_m * 1000:.6f} mm != solved "
                f"{result.baseline_m * 1000:.6f} mm")
            assert abs(saved.left.fx - result.K_l[0, 0]) < 1e-6, (
                f"stored fx {saved.left.fx} != solved {result.K_l[0, 0]}")
            # It went to the TMP store, not the real cache.
            assert store_path.exists(), "store save must write the redirected path"
            assert camera_calib_store._DEFAULT_PATH == store_path
            # The status reflects the new opt-in semantics: factory is the default,
            # this saved calib applies only with --use-camera-calib.
            status = wiz._status.text()
            assert device_id in status, status
            assert "FACTORY calib by default" in status, status
            assert "--use-camera-calib" in status, status
            print(f"[ok] wizard Save -> store: load_camera_calib({device_id!r}) "
                  f"returns a StereoCalib (baseline {saved.baseline_m * 1000:.2f} mm "
                  f"== solved); status={status!r}; wrote only the tmp store")
        finally:
            camera_calib_store._DEFAULT_PATH = orig_default
            QFileDialog.getSaveFileName = staticmethod(orig_picker)  # type: ignore[assignment]
    wiz.close()


# --------------------------------------------------------------------------- #
# Gate 4: mono-guard / stream error surfaces in the wizard.
# --------------------------------------------------------------------------- #
def test_mono_guard_surfaces_error() -> None:
    _app()
    err = ("stereo calibration needs a stereo capture; this stream has no "
           "right frame")
    stream = _FakeStereoStream(error=err)
    wiz = CameraCalibWizard(None, device_id="selftest-cam",
                            width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], stream=stream)
    wiz._on_start()
    # First drain tick reads stream.error and surfaces it (stops capture).
    wiz._drain()
    assert not wiz._running, "wizard must stop capture on a stream error"
    txt = wiz._status.text()
    assert "right frame" in txt or "stereo" in txt.lower(), txt
    assert "⚠" in txt, f"error must be flagged to the operator: {txt!r}"
    print(f"[ok] mono guard: stream .error surfaced in wizard -> {txt!r}")
    wiz.close()


# --------------------------------------------------------------------------- #
# Gate 4b: a NON-CONVERGED solve (the wide-FOV runaway / diverged-stereo failure)
# must show an HONEST "did not converge — recapture" message and leave Save DISABLED,
# never persist garbage. We unit-test the wizard's _on_solved handler directly with a
# StereoCalibResult flagged ok=False (the same object the solver emits when its sanity
# floor trips), so this gate does not depend on synthesising a runaway capture.
# --------------------------------------------------------------------------- #
def test_non_converged_solve_blocks_save() -> None:
    _app()
    from sky.calib.solve import StereoCalibResult

    stream = _FakeStereoStream()
    wiz = CameraCalibWizard(None, device_id="selftest-cam",
                            width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], stream=stream)
    wiz._dump_path = "/tmp/oakd_calib_views_unittest.npz"   # what _begin_solve would set
    # A FAILED result: runaway focal (fx/width ~= 2.9, the ~1884 symptom) + diverged
    # stereo RMS, flagged ok=False with the honest reason the sanity floor produces.
    bad = StereoCalibResult(
        K_l=np.diag([1884.0, 1779.0, 1.0]), dist_l=np.zeros(8), rms_l=2.0,
        K_r=np.diag([1880.0, 1775.0, 1.0]), dist_r=np.zeros(8), rms_r=2.0,
        R=np.eye(3), T=np.array([-0.96, 0.0, 0.0]), stereo_rms=1.5e13,
        n_views_used=15, calibrate_flags=0, ok=False,
        failure_reason="implausible solve (did not converge): left fx/width=2.944 "
                       "outside [0.2,2.0]; stereo RMS=1.5e+13 px > 5.0")
    wiz._on_solved(bad)

    # Save must STAY disabled -- a non-converged calib must never be persisted.
    assert not wiz._save_btn.isEnabled(), "Save must be disabled for a failed solve"
    assert wiz._verdict == ("FAIL", bad.failure_reason), wiz._verdict
    status = wiz._status.text()
    assert "did not converge" in status.lower(), status
    assert "recapture" in status.lower(), status
    # The debug-dump path is surfaced so the operator can send us the file.
    assert wiz._dump_path in status, status
    verdict_txt = wiz._verdict_lbl.text()
    assert "did not converge" in verdict_txt.lower(), verdict_txt
    print(f"[ok] non-converged solve: Save stays DISABLED, status={status!r}")
    wiz.close()


# --------------------------------------------------------------------------- #
# Gate 6: the FREEZE repro -- a fast stream of board-LESS frames with a SLOW,
# FAILING detect must NOT saturate the UI thread (the on-device hang). This is the
# regression test for the decoupled frame-pump (drain-to-latest + off-thread detect).
# --------------------------------------------------------------------------- #
class _SlowBoardlessCollector:
    """A collector whose ``feed`` is SLOW and ALWAYS fails to find a board.

    This is the exact pathological case the operator hits before aiming the board:
    ``cv2.findChessboardCorners`` grinds for 100-300 ms on a board-less frame and
    returns nothing. We model it with a ``sleep`` + a board-less ``FrameStatus`` so
    the test reproduces the freeze WITHOUT cv2 or a device. It exposes only the
    surface the wizard touches: ``feed`` / ``complete`` / ``pattern_cols/rows`` /
    ``cfg`` / ``accepted_count`` and a ``feed_calls`` counter so the test can prove
    how many detections actually ran.
    """

    def __init__(self, n_target: int = 15, feed_sleep_s: float = 0.15) -> None:
        self.pattern_cols, self.pattern_rows = PATTERN_COLS, PATTERN_ROWS
        self.complete = False
        self.accepted_count = 0
        self.feed_calls = 0
        self._sleep = feed_sleep_s

        class _Cfg:
            pass

        self.cfg = _Cfg()
        self.cfg.n_target = n_target
        self.cfg.min_skew_spread = 5e-3

    def feed(self, gray_left, gray_right) -> FrameStatus:
        # SLOW, like a real board-less findChessboardCorners x2 -- this is what made
        # the old UI-thread `while self._queue` loop never return.
        self.feed_calls += 1
        time.sleep(self._sleep)
        cov = CoverageStatus(0.0, 0.0, 0.0, 0.0, 0.0)
        return FrameStatus(
            found_left=False, found_right=False, accepted=False, accepted_count=0,
            n_target=self.cfg.n_target, reason="Board not found in EITHER camera.",
            novelty=0.0, complete=False, coverage=cov, skew_ok=False, count_ok=False)


def test_fast_stream_slow_boardless_detect_does_not_freeze() -> None:
    _app()
    stream = _FakeStereoStream()
    wiz = CameraCalibWizard(None, device_id="selftest-cam",
                            width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], stream=stream)
    wiz._on_start()
    # Swap in the pathological collector AFTER start (real geometry is irrelevant --
    # detection always fails here; this is purely the frame-pump under load).
    slow = _SlowBoardlessCollector(n_target=15, feed_sleep_s=0.15)
    wiz._coll = slow

    # The placeholder text proves the preview has NOT been painted yet.
    assert "live preview appears here" in wiz._preview.text(), wiz._preview.text()

    # A cheap solid-gray frame is enough -- detection fails on it regardless.
    blank = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0]), 128, dtype=np.uint8)

    # --- CONTRAST: prove the OLD `_drain` logic (the `while self._queue:` loop that
    #     fed the collector INLINE on the UI thread) would HANG on this exact load.
    #     We replay the old loop against a small backlog of the SAME slow collector
    #     and measure that it scales with the backlog (N * feed_time) -- i.e. it
    #     blocks the UI thread for as long as there are queued frames, which is the
    #     freeze. The NEW _drain (asserted below) is O(1) regardless of backlog. ---
    old_backlog = [(blank.copy(), blank.copy()) for _ in range(4)]
    t0 = time.monotonic()
    for gl, gr in old_backlog:                      # the OLD `while self._queue:` body
        slow.feed(gl, gr)
    old_dt = time.monotonic() - t0
    assert old_dt >= 4 * 0.15 * 0.9, (              # ~0.6 s for 4 frames @150 ms
        f"sanity: the old inline loop should scale with the backlog, got {old_dt:.3f}s")
    slow.feed_calls = 0                             # reset so the NEW-path count is clean
    blank_idx_note = old_dt                         # carried into the final printout

    # --- Flood the queue FASTER than detection can run: 80 board-less pairs pushed
    #     in a burst (like the ~20 fps stream while the operator is still aiming). ---
    N_FLOOD = 80
    for i in range(N_FLOOD):
        stream.emit(blank.copy(), blank.copy(), seq=i)
    assert wiz._frames_received == N_FLOOD, wiz._frames_received
    # The thread-safe queue is BOUNDED (maxlen) -- a flood can never grow it without
    # limit even before a single drain runs (the old code had no such guarantee on
    # processing: it looped over the WHOLE backlog on the UI thread).
    assert len(wiz._queue) <= wiz._queue.maxlen, (
        f"queue must stay bounded by maxlen: {len(wiz._queue)} > {wiz._queue.maxlen}")

    # --- (a) _drain must return PROMPTLY and arm at most ONE detection per tick,
    #     NOT loop over the backlog. Time a single tick: even though each detect
    #     sleeps 150 ms OFF-thread, the UI-thread tick itself must be ~instant. ---
    t0 = time.monotonic()
    wiz._drain()
    drain_dt = time.monotonic() - t0
    assert drain_dt < 0.05, (
        f"_drain blocked the UI thread for {drain_dt * 1000:.0f} ms -- it must NOT "
        f"run detection inline (the freeze). Expected a near-instant return.")

    # --- (d) drain-to-latest: ONE tick drains the ENTIRE backlog out of the queue
    #     (keeping only the newest, which it handed to the worker), so the queue is
    #     now empty -- stale frames were DROPPED, not processed one-by-one. ---
    assert len(wiz._queue) == 0, (
        f"drain-to-latest must empty the queue in one tick: {len(wiz._queue)} left")
    # At most ONE detect was armed by that single tick (bounded, not 80).
    assert wiz._detector is not None, "the tick should have armed exactly one detect"
    assert slow.feed_calls <= 1, (
        f"a single tick must run at most ONE detection, not the backlog: "
        f"{slow.feed_calls} feeds")

    # --- (b) the preview WAS updated even with detection failing/slow: the live gray
    #     shows so the operator can aim. The placeholder text is gone and a non-null
    #     pixmap is painted -- this is the smooth live view the freeze denied. ---
    assert wiz._preview.text() == "", (
        f"preview placeholder must be replaced by the live gray: {wiz._preview.text()!r}")
    pix = wiz._preview.pixmap()
    assert pix is not None and not pix.isNull(), (
        "preview pixmap must be painted every tick, independent of detection")
    assert pix.width() > 1 and pix.height() > 1, "preview pixmap must be non-trivial"

    # --- Pump the worker to completion + re-arm a few more ticks; assert detection
    #     stays BOUNDED (one in flight at a time) rather than processing all 80. ---
    for _ in range(5):
        det = wiz._detector
        if det is not None:
            assert det.wait(5000), "detect worker did not finish"
        QCoreApplication.processEvents()       # deliver done -> _on_detected
        # Re-arm from a fresh frame the way the timer would.
        stream.emit(blank.copy(), blank.copy())
        t0 = time.monotonic()
        wiz._drain()
        assert time.monotonic() - t0 < 0.05, "a later _drain tick also blocked the UI"
    # After flooding 80 + ~5 frames, only a HANDFUL of detections ran (one per
    # settled tick) -- the old while-loop would have run ~80 inline and hung.
    assert slow.feed_calls <= 8, (
        f"detection must be throttled to one-in-flight, not the backlog: "
        f"{slow.feed_calls} feeds for {N_FLOOD}+ frames")

    # The in-dialog liveness readout shows the operator the stream is alive even
    # though no board is ever found (received N frames + a detections/sec figure).
    live = wiz._liveness_lbl.text()
    assert "received" in live and "detections/s" in live, live

    # --- (c) the watchdog .error path still surfaces PROMPTLY now that _drain
    #     returns: set the stream error (as the 5 s watchdog would) and tick once;
    #     the wizard must surface it and stop -- the old freeze never re-read it. ---
    stream.error = "no frames for 5.0 s — capture stream appears dead (watchdog)"
    wiz._drain()
    assert not wiz._running, "watchdog error must stop capture"
    txt = wiz._status.text()
    assert "⚠" in txt and "watchdog" in txt, f"watchdog error must surface: {txt!r}"

    print(f"[ok] freeze repro: OLD inline loop took {blank_idx_note:.2f}s for just 4 "
          f"frames (scales w/ backlog = the hang); NEW: flooded {N_FLOOD} board-less "
          f"pairs @150ms/detect -> _drain stayed <50 ms/tick, queue drained-to-latest "
          f"(0 left), only {slow.feed_calls} detections ran (NOT {N_FLOOD}), preview "
          f"painted, liveness={live!r}, watchdog .error surfaced promptly")
    wiz.close()


# --------------------------------------------------------------------------- #
# Gate 5: a None stream is a programming error (the device-free contract).
# --------------------------------------------------------------------------- #
def test_none_stream_rejected() -> None:
    _app()
    try:
        CameraCalibWizard(None, width=IMAGE_SIZE[0], height=IMAGE_SIZE[1],
                          stream=None)
    except ValueError as exc:
        assert "stream" in str(exc).lower()
        print(f"[ok] None-stream guard: {exc}")
        return
    raise AssertionError("a None stream must raise (device-free contract)")


def test_show_board_window_stays_open() -> None:
    """'Show checkerboard' must NOT flash-and-close (the operator's live report).

    The old path ran a NESTED ``app.exec()`` and dropped its QLabel reference, so
    inside the running wizard the board flashed on-screen and vanished instantly.
    This reproduces the fix headlessly: the board window must be RETAINED (a strong
    ref, no nested exec -> if ``_on_show_board`` blocked, this test would hang),
    shown fullscreen, with the square aspect PRESERVED (a stretched board biases
    the recovered focal lengths), replaced cleanly on re-show, and torn down when
    the wizard closes (no orphan window left covering the screen).
    """
    _app()
    from ui.calib.checkerboard import make_checkerboard
    from ui.qt.camera_calib_dialog import _BoardWindow

    wiz = CameraCalibWizard(None, device_id="selftest-cam",
                            width=IMAGE_SIZE[0], height=IMAGE_SIZE[1],
                            stream=_FakeStereoStream())
    assert wiz._board_win is None, "should start with no board window"

    wiz._on_show_board()                      # must RETURN (no nested exec)
    bw = wiz._board_win
    assert isinstance(bw, _BoardWindow), "board window ref not retained (GC flash)"
    pm = bw.pixmap()
    assert pm is not None and not pm.isNull(), "board pixmap missing"
    assert bw.isVisible(), "board window not shown"
    # 2nd reported bug: must be a NORMAL resizable window, NOT forced fullscreen.
    assert not bw.isFullScreen(), "board window must be resizable, not fullscreen"

    src = make_checkerboard(int(wiz._cols_spin.value()),
                            int(wiz._rows_spin.value()),
                            square_px=100, margin_squares=1.0)
    src_ar = src.shape[1] / src.shape[0]
    pm_ar = pm.width() / pm.height()
    assert abs(src_ar - pm_ar) < 0.02, f"board distorted: {src_ar:.4f} vs {pm_ar:.4f}"

    # Resizing the window rescales the board, STILL preserving the square aspect.
    bw.resize(500, 700)
    bw._rescale()
    pm2 = bw.pixmap()
    pm2_ar = pm2.width() / pm2.height()
    assert abs(src_ar - pm2_ar) < 0.02, f"resized board distorted: {pm2_ar:.4f}"
    assert pm2.width() <= 500 and pm2.height() <= 700, "rescale must fit the window"

    prev = bw
    wiz._on_show_board()                      # re-show replaces cleanly
    assert wiz._board_win is not None and wiz._board_win is not prev

    wiz.close()                               # closeEvent tears the board down
    assert wiz._board_win is None, "closeEvent must release the board window"
    print(f"[ok] show-checkerboard: retained + RESIZABLE (not fullscreen) + aspect "
          f"{pm_ar:.3f} preserved on resize, replaced on re-show, torn down on "
          f"close (no nested exec, no GC-flash)")


def main() -> int:
    # Construct the QApplication FIRST and hold a strong Python reference for the
    # whole run BEFORE any cv2 import or QWidget construction (the offscreen pattern
    # the other UI selftests use): without a live reference the wrapper is collected
    # and the next QWidget trips "Must construct a QApplication before a QWidget".
    app = _app()
    test_none_stream_rejected()
    test_show_board_window_stays_open()
    test_mono_guard_surfaces_error()
    test_non_converged_solve_blocks_save()
    test_fast_stream_slow_boardless_detect_does_not_freeze()
    test_fronto_parallel_not_complete()
    test_tilted_completes_solves_and_saves()
    test_save_writes_to_per_device_store()
    # Flush any deferred deleteLater() from the closed dialogs through `app` (this
    # also makes the strong reference an explicit, used one -- pyflakes-clean).
    app.processEvents()
    print("\nALL CAMERA CALIB WIZARD SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
