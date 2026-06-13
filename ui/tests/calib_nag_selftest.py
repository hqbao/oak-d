#!/usr/bin/env python3
"""Offscreen Qt selftest for the STARTUP calibration notification
(:func:`ui.main.install_calib_nag`).

NO device, NO IPC, NO real cache: builds a real ``QMainWindow`` + ``QToolBar``
offscreen and calls ``install_calib_nag`` with STUB status dicts (the
``calibration_status`` shape) and a STUB ``open_dialog`` callback. Asserts the
NON-BLOCKING contract:

1. INCOMPLETE status -> a persistent "⚠ CALIB INCOMPLETE" indicator is created and
   added to the toolbar (stashed on ``win._calib_nag_btn``); the status-bar shows a
   message that NAMES the missing items and the inaccuracy risk; NO modal blocks
   (the call returns immediately).
2. Clicking the indicator fires ``open_dialog`` (opens the status dialog).
3. ALL-✓ status -> NO nag indicator (``win._calib_nag_btn`` is None, return None),
   and a brief "calibration OK" confirmation is shown instead (no nag).
4. INTEGRATION with the REAL ``calibration_status`` (loaders stubbed, no cache):
   * camera-only "missing" (gyro+accel cached, camera store empty) -> the camera is
     factory-by-default, so ``calibration_status`` reports it calibrated and the nag
     does NOT fire,
   * gyro/accel genuinely uncalibrated -> the nag DOES fire and names them but NOT
     camera.

Run::

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m ui.tests.calib_nag_selftest
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Force headless Qt BEFORE any Qt import (mirrors the other offscreen selftests).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMainWindow, QToolBar      # noqa: E402

from ui.main import install_calib_nag                               # noqa: E402


# --------------------------------------------------------------------------- #
def _status(missing: list[str]) -> dict:
    """Minimal calibration_status-shaped dict (install_calib_nag only reads
    ``all_calibrated`` + ``missing``)."""
    return {"all_calibrated": not missing, "missing": list(missing)}


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    assert app is not None

    # ---- Gate 1: INCOMPLETE -> indicator created + message names the items --- #
    win = QMainWindow()
    tb = QToolBar()
    win.addToolBar(tb)
    opened = {"n": 0}
    nag = install_calib_nag(win, tb, _status(["accel", "camera"]),
                            lambda: opened.__setitem__("n", opened["n"] + 1))
    assert nag is not None, "expected a nag indicator for an incomplete status"
    assert win._calib_nag_btn is nag, "nag not stashed on win for lifetime/test hook"
    assert "CALIB INCOMPLETE" in nag.text(), nag.text()
    # The indicator is part of the toolbar (a persistent, visible control).
    assert nag in tb.findChildren(type(nag)), "nag not added to the toolbar"
    # The status-bar message names the missing items + the inaccuracy risk.
    msg = win.statusBar().currentMessage()
    assert "accel" in msg and "camera" in msg, msg
    assert "inaccurate" in msg.lower(), msg
    print("[1] incomplete -> persistent ⚠ indicator + naming status message   OK")

    # ---- Gate 2: clicking the indicator opens the dialog -------------------- #
    nag.click()
    assert opened["n"] == 1, opened
    print("[2] clicking the indicator fires open_dialog (opens status view)    OK")

    # ---- Gate 3: ALL-✓ -> no nag, just a brief confirmation ----------------- #
    win2 = QMainWindow()
    tb2 = QToolBar()
    win2.addToolBar(tb2)
    ret = install_calib_nag(win2, tb2, _status([]), lambda: None)
    assert ret is None, "expected NO indicator when fully calibrated"
    assert win2._calib_nag_btn is None, win2._calib_nag_btn
    # No "CALIB INCOMPLETE" button anywhere on the toolbar.
    from PyQt6.QtWidgets import QPushButton
    assert not [b for b in tb2.findChildren(QPushButton)
                if "INCOMPLETE" in b.text()], "nag shown despite full calibration"
    msg2 = win2.statusBar().currentMessage()
    assert "OK" in msg2, msg2
    print("[3] all-✓ -> NO nag indicator, brief 'calibration OK' message       OK")

    # ---- Gate 4: INTEGRATION with the real calibration_status --------------- #
    # Camera is factory-by-default (opt-in via --use-camera-calib), so an empty
    # camera store must NEVER make the nag fire. Stub the three loaders the status
    # module imported so no real cache is touched.
    from imu_camera.device import calib_status as cs
    orig = (cs.load_gyro_bias, cs.load_accel_calib, cs.load_camera_calib)

    def _stub(present):
        return (lambda _dev: object()) if present else (lambda _dev: None)
    try:
        # 4a. gyro+accel cached, camera store EMPTY -> no nag (camera doesn't count).
        cs.load_gyro_bias = _stub(True)          # type: ignore[assignment]
        cs.load_accel_calib = _stub(True)        # type: ignore[assignment]
        cs.load_camera_calib = _stub(False)      # type: ignore[assignment]
        win3 = QMainWindow()
        tb3 = QToolBar()
        win3.addToolBar(tb3)
        ret3 = install_calib_nag(win3, tb3, cs.calibration_status("dev"),
                                 lambda: None)
        assert ret3 is None, "camera-only-empty must NOT nag (factory is default)"
        assert win3._calib_nag_btn is None, win3._calib_nag_btn
        print("[4a] camera store empty (gyro/accel ok) -> NO nag for camera       OK")

        # 4b. gyro+accel UNcalibrated, camera store empty -> nag DOES fire and
        #     names gyro+accel but NOT camera.
        cs.load_gyro_bias = _stub(False)         # type: ignore[assignment]
        cs.load_accel_calib = _stub(False)       # type: ignore[assignment]
        cs.load_camera_calib = _stub(False)      # type: ignore[assignment]
        win4 = QMainWindow()
        tb4 = QToolBar()
        win4.addToolBar(tb4)
        nag4 = install_calib_nag(win4, tb4, cs.calibration_status("dev"),
                                 lambda: None)
        assert nag4 is not None, "gyro/accel missing must nag"
        msg4 = win4.statusBar().currentMessage()
        assert "gyro" in msg4 and "accel" in msg4, msg4
        assert "camera" not in msg4, f"camera must not be nagged: {msg4!r}"
        print("[4b] gyro/accel missing -> nag fires, names gyro+accel not camera  OK")
    finally:
        cs.load_gyro_bias, cs.load_accel_calib, cs.load_camera_calib = orig

    print("\nALL startup calib-nag CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
