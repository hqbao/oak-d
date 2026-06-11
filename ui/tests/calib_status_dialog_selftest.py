#!/usr/bin/env python3
"""Offscreen Qt selftest for :class:`ui.qt.calib_status_dialog.CalibrationStatusDialog`.

NO device, NO real cache: the dialog is driven by a STUB ``status_provider`` (a
plain dict in the ``calibration_status`` shape) and STUB ``openers`` (callables that
record they fired). We never enter the Qt event loop -- we construct the dialog,
inspect its per-row widgets, fire the buttons directly, and simulate a re-show by
calling ``showEvent``. The dialog import path is cv2/depthai-free (asserted too).

Gates
-----
1. ALL-✓ status: every badge reads the green ✓ (FieldGood) and the done detail; the
   ``all_calibrated`` case is rendered without error.
2. PARTIAL status (accel missing): the gyro/camera badges read ✓ (FieldGood), the
   accel badge reads ✗ (FieldBad), and the accel detail carries the missing wording.
3. "Open wizard" buttons are WIRED: clicking each row's button fires exactly that
   item's opener (and only it).
4. RE-SHOW re-queries: a provider whose result FLIPS between calls is re-read on
   ``showEvent`` / ``refresh`` (badge state changes; refresh_count increments).
5. The dialog module import pulls in NEITHER cv2 NOR depthai.

Run::

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m ui.tests.calib_status_dialog_selftest
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Force headless Qt BEFORE any Qt import (mirrors the other offscreen selftests).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QShowEvent                                # noqa: E402
from PyQt6.QtWidgets import QApplication                          # noqa: E402

from ui.qt.calib_status_dialog import CalibrationStatusDialog     # noqa: E402


# --------------------------------------------------------------------------- #
def _status(gyro_ok: bool, accel_ok: bool, camera_ok: bool) -> dict:
    """Build a calibration_status-shaped dict (the contract the dialog consumes)."""
    def _it(name, ok, done, miss):
        return {"name": name, "calibrated": ok, "detail": done if ok else miss}
    items = [
        _it("gyro", gyro_ok, "bias cached", "not yet measured — auto-measured"),
        _it("accel", accel_ok, "6-position done", "not calibrated — raw accel"),
        _it("camera", camera_ok, "user stereo calib", "not calibrated — factory"),
    ]
    missing = [it["name"] for it in items if not it["calibrated"]]
    return {"device_id": "dev", "items": items,
            "all_calibrated": not missing, "missing": missing}


def _badge_state(dlg: CalibrationStatusDialog) -> dict:
    """Map item name -> (badge text, badge objectName) for assertions."""
    return {name: (b.text(), b.objectName())
            for name, b in dlg._badges.items()}


# --------------------------------------------------------------------------- #
def main() -> int:
    # A QApplication must exist before any QWidget is constructed; hold a reference
    # so it isn't garbage-collected mid-test.
    app = QApplication.instance() or QApplication(sys.argv[:1])
    assert app is not None

    # Track which openers fired (and how many times) -- a list per item.
    fired: dict[str, int] = {"gyro": 0, "accel": 0, "camera": 0}

    def _mk_opener(name: str):
        def _fn():
            fired[name] += 1
        return _fn

    openers = {n: _mk_opener(n) for n in ("gyro", "accel", "camera")}

    # ---- Gate 1: ALL-✓ ------------------------------------------------------ #
    dlg_all = CalibrationStatusDialog(
        None, status_provider=lambda: _status(True, True, True), openers=openers)
    st = _badge_state(dlg_all)
    assert all(txt == "✓" for txt, _ in st.values()), st
    assert all(obj == "FieldGood" for _, obj in st.values()), st
    assert dlg_all._details["gyro"].text() == "bias cached"
    print("[1] all-✓ status -> every badge green ✓ (FieldGood), done detail   OK")

    # ---- Gate 2: PARTIAL (accel missing) ------------------------------------ #
    # A mutable holder so we can flip the provider's result for the re-show gate.
    state = {"cur": _status(True, False, True)}
    dlg = CalibrationStatusDialog(
        None, status_provider=lambda: state["cur"], openers=openers)
    st = _badge_state(dlg)
    assert st["gyro"] == ("✓", "FieldGood"), st
    assert st["camera"] == ("✓", "FieldGood"), st
    assert st["accel"] == ("✗", "FieldBad"), st
    assert "raw accel" in dlg._details["accel"].text(), dlg._details["accel"].text()
    print("[2] partial (accel ✗) -> gyro/camera ✓, accel red ✗, miss detail  OK")

    # ---- Gate 3: "Open wizard" buttons wired -------------------------------- #
    # Find each row's "Open wizard" button. findChildren returns them in
    # construction order, which the dialog builds in item order (gyro, accel,
    # camera). Click each and assert ONLY that item's opener fired.
    from PyQt6.QtWidgets import QPushButton
    open_btns = [b for b in dlg.findChildren(QPushButton)
                 if b.text() == "Open wizard"]
    assert len(open_btns) == 3, [b.text() for b in dlg.findChildren(QPushButton)]
    for name, btn in zip(("gyro", "accel", "camera"), open_btns):
        before = dict(fired)
        btn.click()
        delta = {k: fired[k] - before[k] for k in fired}
        assert delta[name] == 1 and sum(delta.values()) == 1, (name, delta)
    print("[3] each row's 'Open wizard' fires exactly that item's opener      OK")

    # ---- Gate 4: re-show re-queries ----------------------------------------- #
    rc_before = dlg.refresh_count
    state["cur"] = _status(True, True, True)          # accel now calibrated
    dlg.showEvent(QShowEvent())                       # simulate a re-show
    assert dlg.refresh_count == rc_before + 1, (dlg.refresh_count, rc_before)
    st = _badge_state(dlg)
    assert st["accel"] == ("✓", "FieldGood"), st       # flipped to ✓ on re-query
    print("[4] re-show re-queries provider; accel flips ✓ (refresh_count++)   OK")

    # ---- Gate 5: import path is cv2/depthai-free ---------------------------- #
    assert "cv2" not in sys.modules, "cv2 leaked into the dialog import path"
    assert "depthai" not in sys.modules, "depthai leaked into the dialog import path"
    print("[5] dialog import path has no cv2 / no depthai                     OK")

    print("\nALL CalibrationStatusDialog CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
