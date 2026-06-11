"""The ONE place that shows all three calibrations + launches each wizard.

Today gyro / accel / camera calibration are three separate menu wizards with no
single status view. This dialog folds them into one panel: one row per calibration
with a ✓/✗ badge, the name, the detail string, and an "Open wizard" button that
launches the matching existing wizard. It is the first item under the "Calibration"
menu ("Calibration status…").

Device-agnostic + cv2/depthai-free import path
----------------------------------------------
The dialog takes its data through a ``status_provider`` callable (re-invoked on every
:meth:`showEvent`, so a just-finished calibration is reflected the next time the
dialog is shown) and its wizard launchers through an ``openers`` mapping. It imports
ONLY PyQt6 + the theme -- no cv2, no depthai, no device code -- so it stays
multi-chip-generic and offline-unit-testable (a test passes a stub provider + stub
openers; see ``ui/tests/calib_status_dialog_selftest.py``).

The provider returns the dict from
:func:`imu_camera.mathlib.device.calib_status.calibration_status` (an ``items`` list
of ``{"name", "calibrated", "detail"}`` rows). The caller keys it by the resolved
``dev_id`` and wires the three openers to the existing
``_open_gyro_calib`` / ``_open_accel_calib`` / ``_open_camera_calib`` handlers.
"""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from . import theme

# Badge glyphs + the theme object names that colour them (green GOOD / red BAD via
# the shared QSS). A calibrated item reads a green ✓; a missing one a red ✗.
_BADGE_OK = "✓"
_BADGE_MISSING = "✗"


class CalibrationStatusDialog(QDialog):
    """Single status view for gyro + accel + camera calibration.

    Parameters
    ----------
    status_provider:
        Zero-arg callable returning the
        :func:`~imu_camera.mathlib.device.calib_status.calibration_status` dict.
        Re-invoked on every show so the rows reflect the latest on-disk state.
    openers:
        Maps each item ``name`` ("gyro" / "accel" / "camera") to a zero-arg callable
        that launches that item's wizard. An item with no opener gets a disabled
        button (defensive -- the caller always supplies all three).
    """

    def __init__(self, parent=None, *,
                 status_provider: Callable[[], dict],
                 openers: dict[str, Callable[[], None]]) -> None:
        super().__init__(parent)
        self._status_provider = status_provider
        self._openers = dict(openers)
        # Per-row widgets, keyed by item name, so :meth:`refresh` re-syncs them
        # in place instead of rebuilding the layout on every show.
        self._badges: dict[str, QLabel] = {}
        self._names: dict[str, QLabel] = {}
        self._details: dict[str, QLabel] = {}
        # Count of refreshes -- lets a test assert that re-showing re-queries.
        self.refresh_count = 0

        self.setStyleSheet(theme.QSS)
        self.setWindowTitle("Calibration status")
        self.setMinimumWidth(560)

        root = QVBoxLayout(self)

        title = QLabel("CALIBRATION STATUS")
        title.setObjectName("PanelTitle")
        root.addWidget(title)

        hint = QLabel(
            "Calibrate any item marked ✗ before flying — uncalibrated sensors "
            "make the pose drift. Use “Open wizard” for each."
        )
        hint.setObjectName("DialogHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # One row per item, built ONCE from the first query (its order also fixes the
        # display order). Each row: [badge] [name] [detail ...] [Open wizard].
        grid = QGridLayout()
        grid.setColumnStretch(2, 1)          # detail column eats the slack
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        for r, item in enumerate(self._status_provider().get("items", [])):
            name = item["name"]

            badge = QLabel(_BADGE_MISSING)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setFixedWidth(18)
            grid.addWidget(badge, r, 0)
            self._badges[name] = badge

            name_lab = QLabel(name.capitalize())
            name_lab.setObjectName("DialogMono")
            grid.addWidget(name_lab, r, 1)
            self._names[name] = name_lab

            detail_lab = QLabel(item.get("detail", ""))
            detail_lab.setObjectName("DialogHint")
            detail_lab.setWordWrap(True)
            grid.addWidget(detail_lab, r, 2)
            self._details[name] = detail_lab

            btn = QPushButton("Open wizard")
            opener = self._openers.get(name)
            if opener is None:
                btn.setEnabled(False)
            else:
                # Bind `opener` per-iteration; `_c` swallows clicked(bool).
                btn.clicked.connect(lambda _c=False, fn=opener: fn())
            grid.addWidget(btn, r, 3)

        wrap = QWidget()
        wrap.setLayout(grid)
        root.addWidget(wrap)

        # Bottom row: a re-check + a close. "Re-check" lets the operator refresh
        # without re-opening (e.g. after finishing a wizard launched from here).
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        recheck = QPushButton("Re-check")
        recheck.clicked.connect(lambda _c=False: self.refresh())
        btn_row.addWidget(recheck)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        btn_row.addWidget(close)
        root.addLayout(btn_row)

        # Initial sync so the rows show real state before the first show.
        self.refresh()

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        """Re-query the provider and re-sync every row's badge + detail.

        Cheap (three tiny JSON reads behind the provider); called on construction,
        on every :meth:`showEvent`, and by the "Re-check" button.
        """
        self.refresh_count += 1
        status = self._status_provider()
        for item in status.get("items", []):
            name = item["name"]
            ok = bool(item["calibrated"])
            badge = self._badges.get(name)
            if badge is not None:
                badge.setText(_BADGE_OK if ok else _BADGE_MISSING)
                # FieldGood / FieldBad set the green / red colour via the shared QSS.
                badge.setObjectName("FieldGood" if ok else "FieldBad")
                # Re-polish so the changed objectName actually re-applies the style.
                badge.style().unpolish(badge)
                badge.style().polish(badge)
            detail = self._details.get(name)
            if detail is not None:
                detail.setText(item.get("detail", ""))

    # ------------------------------------------------------------------ #
    def showEvent(self, event) -> None:                      # noqa: N802 (Qt name)
        """Re-query on every show so a just-finished wizard is reflected."""
        self.refresh()
        super().showEvent(event)
