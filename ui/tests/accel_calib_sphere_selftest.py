#!/usr/bin/env python3
"""Offscreen Qt selftest for the LIVE gravity-sphere in the accel-calib wizard.

NO device: a FAKE IMU stream stub (the dialog's four touch-points --
``start`` / ``stop`` / ``.error`` / ``.device_id``) lets the TEST push synthetic
six-face IMU samples straight through the dialog's own
:meth:`AccelCalibDialog._feed_sample` (the recv-thread hook) + :meth:`_drain`
(the UI-timer slot) -- the exact split that makes the dialog offline-testable
(no event loop, no device), mirroring ``camera_calib_dialog_selftest``.

The synthetic faces are built from a KNOWN distorted sensor model (a bias + a
scale/misalignment matrix, like the offline tool's ``--demo``) so the SAME
production ``SixFaceCollector`` + ``solve_accel_calibration`` the dialog drives
recover a real, non-trivial calibration -- this is an integration test of the
live-sphere glue, not a re-test of the math core.

Gates
-----
1. The sphere pixmap is rendered and GROWS as faces are captured: blank before
   any capture, non-blank after the first face, and the pixmap CHANGES (different
   bytes) between an early face count and a later one (more RED dots landed).
2. After the 6th face + solve the dialog shows the SNAP: the post-solve pixmap
   differs from the pre-solve (5-face) one (the GREEN calibrated dots + bias +
   annotations appear), the residual line is populated, and Save is enabled.
3. The render is THROTTLED to face-set changes: a steady "still holding" drain
   tick that captures nothing must NOT re-render the sphere (the cache key holds).

Run::

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m ui.tests.accel_calib_sphere_selftest
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Force headless Qt BEFORE any Qt import (mirrors the other offscreen selftests).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from sky.sensors.accel_calib import SIX_FACES
from ui.qt.calib_dialogs import AccelCalibDialog

# Known injected distortion (a plausible MEMS) -- a real bias + per-axis scale +
# a few-degrees misalignment, so the solver has something genuine to recover and
# the RAW dots sit visibly OFF the sphere (the ellipsoid the picture teaches).
_BIAS = np.array([0.18, -0.27, 0.11])              # m/s^2 raw zero offset
_SENSOR = np.array([                                # raw = SENSOR @ true + bias
    [1.018, 0.020, -0.013],
    [-0.015, 0.987, 0.022],
    [0.011, -0.018, 1.009],
])
_G = 9.80665


# --------------------------------------------------------------------------- #
# Fake IMU stream stub (the dialog's four touch-points only).
# --------------------------------------------------------------------------- #
class _FakeImuStream:
    """Duck-type of IpcImuRawSource: start/stop/.error/.device_id, no device.

    The dialog calls ``start(cb)`` (we just retain ``cb``; the TEST pushes
    samples), polls ``.error`` each drain, and reads ``.device_id`` -- nothing
    else.
    """

    def __init__(self, device_id: str = "selftest-imu") -> None:
        self.device_id = device_id
        self.error: str | None = None
        self.started = False
        self.stopped = False
        self._cb = None

    def start(self, callback) -> None:
        self._cb = callback
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def emit(self, gyro, accel, t_s) -> None:
        if self._cb is not None:
            self._cb(np.asarray(gyro), np.asarray(accel), float(t_s))


def _app() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv or ["selftest"])


def _raw_face(face_idx: int) -> np.ndarray:
    """Synthetic RAW accel mean for a perfectly-square hold of ``face_idx``.

    The true resting specific force is ``g * dir`` along that face; the distorted
    sensor reads ``SENSOR @ (g*dir) + bias`` -- the same forward model the offline
    demo uses, so the production solver recovers a real (T, b).
    """
    true_sf = _G * SIX_FACES[face_idx]
    return _SENSOR @ true_sf + _BIAS


def _hold_face(dlg: AccelCalibDialog, stream: _FakeImuStream, face_idx: int,
               t0: float) -> float:
    """Push enough still + square samples to CAPTURE one face, return the next t.

    Feeds a clean still streak (gyro 0, accel = the face's raw reading) long
    enough to clear the stillness gate (window_s + min_samples), then a single
    MOVING sample (large gyro) so the collector's "rotate to the next face" latch
    clears before the next face. Drives the dialog the way the live UI does:
    emit -> _drain.
    """
    accel = _raw_face(face_idx)
    t = t0
    dt = 0.01                                        # 100 Hz, like the real IMU
    # 0.6 s window + a margin of samples (min 30) -> 80 still samples is plenty.
    for _ in range(80):
        stream.emit(np.zeros(3), accel, t)
        t += dt
    dlg._drain()                                     # one UI tick consumes the streak
    # A clear "moving" sample so the next still face is accepted (clears the latch).
    stream.emit(np.array([1.0, 1.0, 1.0]), accel, t)
    t += dt
    dlg._drain()
    return t


def _pix_bytes(dlg: AccelCalibDialog) -> bytes | None:
    """Raw bytes of the sphere pixmap (None if blank), for change detection."""
    pix = dlg._sphere.pixmap()
    if pix is None or pix.isNull():
        return None
    img = pix.toImage()
    ptr = img.bits()
    ptr.setsize(img.sizeInBytes())
    return bytes(ptr)


def main() -> int:
    app = _app()
    stream = _FakeImuStream()
    dlg = AccelCalibDialog(None, device_id="selftest-imu", stream=stream)
    dlg._on_start()
    assert stream.started, "dialog must start the injected stream on START"

    # Before any capture: sphere is blank (just the START hint), no pixmap.
    assert _pix_bytes(dlg) is None, "sphere must be blank before the first capture"

    # --- Gate 1: capture faces one by one; the sphere fills (non-blank + grows). ---
    t = 0.0
    pix_after = {}                                   # face-count -> pixmap bytes
    capture_order = [0, 1, 2, 3, 4, 5]              # +X,-X,+Y,-Y,+Z,-Z (any order)
    for n, face in enumerate(capture_order, start=1):
        t = _hold_face(dlg, stream, face, t)
        assert len(dlg._coll.captured_faces) == n, (
            f"expected {n} faces captured, got {dlg._coll.captured_faces}")
        b = _pix_bytes(dlg)
        assert b is not None, f"sphere must be non-blank after {n} face(s)"
        pix_after[n] = b

    # The pixmap CHANGED as more RED dots landed: an early count differs from a
    # later one (different drawn content, not a frozen first render).
    assert pix_after[1] != pix_after[3], (
        "sphere must change as faces accumulate (1 vs 3 faces identical)")
    assert pix_after[3] != pix_after[5], (
        "sphere must change as faces accumulate (3 vs 5 faces identical)")
    print(f"[ok] live fill: sphere non-blank from face 1, content grows 1->3->5 "
          f"(bytes {len(pix_after[1])}/{len(pix_after[3])}/{len(pix_after[5])})")

    # --- Gate 2: 6th face + solve -> the SNAP appears (post-solve != 5-face). ---
    assert dlg._coll.complete, "all six faces must be captured"
    assert dlg._coll.calibration is not None, "the 6th face must trigger the solve"
    pix_solved = pix_after[6]
    assert pix_solved != pix_after[5], (
        "post-solve sphere (GREEN snap + bias + annotations) must differ from the "
        "5-face pre-solve render")
    # The solve produced a sane, low-residual fit (the known model is recovered).
    cal = dlg._coll.calibration
    assert cal.residual_g < 0.1, (
        f"solver should recover the known model to a tight residual, got "
        f"{cal.residual_g:.4f} m/s^2")
    # The dialog's result line + Save reflect the completed, accepted calibration.
    assert "residual" in dlg._result.text() and "—" not in dlg._result.text(), (
        f"result line must show the residual: {dlg._result.text()!r}")
    assert dlg._coll.verdict().ok, "a clean six-face solve must pass the verdict"
    assert dlg._save_btn.isEnabled(), "Save must be enabled after a good solve"
    print(f"[ok] the SNAP: 6th face + solve re-rendered (GREEN calibrated dots), "
          f"residual={cal.residual_g:.4f} m/s^2, result={dlg._result.text()!r}, "
          f"Save enabled={dlg._save_btn.isEnabled()}")

    # --- Gate 3: render is THROTTLED -- a still tick that captures nothing must
    #     NOT re-render (the face-set/solve cache key is unchanged). ---
    before = _pix_bytes(dlg)
    key_before = dlg._sphere_key
    # A short still streak on an already-captured face -> no new capture, no solve
    # change. (Complete collector ignores feeds anyway; key must hold.)
    accel = _raw_face(0)
    for _ in range(10):
        stream.emit(np.zeros(3), accel, t)
        t += 0.01
    dlg._drain()
    assert dlg._sphere_key == key_before, (
        "a no-capture tick must not change the sphere cache key (no re-render)")
    assert _pix_bytes(dlg) == before, (
        "a no-capture tick must not change the rendered sphere pixmap")
    print(f"[ok] throttle: a no-capture drain tick did NOT re-render the sphere "
          f"(cache key held = {key_before})")

    dlg.close()
    app.processEvents()
    print("\nALL ACCEL CALIB SPHERE SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
