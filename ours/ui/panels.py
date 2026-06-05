"""Telemetry side-panels — numeric readouts that follow the live pose."""
from __future__ import annotations

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget,
)

from ..lib.misc.pose import Pose, PoseHistory


def _panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("Panel")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(8, 6, 8, 8)
    lay.setSpacing(4)
    t = QLabel(title.upper())
    t.setObjectName("PanelTitle")
    lay.addWidget(t)
    return f, lay


def _row(grid: QGridLayout, row: int, label: str) -> QLabel:
    lab = QLabel(label)
    lab.setObjectName("FieldLabel")
    val = QLabel("--")
    val.setObjectName("FieldValue")
    val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    grid.addWidget(lab, row, 0)
    grid.addWidget(val, row, 1)
    return val


class TelemetryPanel(QWidget):
    """Stacks Position, Velocity, Attitude and Status sub-panels."""

    def __init__(self, history: PoseHistory, source_fps_getter, parent=None) -> None:
        super().__init__(parent)
        self.history = history
        self._fps_get = source_fps_getter

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ---- position (NED) ----------------------------------------------
        pf, pl = _panel("Position (NED, m)")
        pg = QGridLayout(); pg.setHorizontalSpacing(12); pg.setVerticalSpacing(2)
        self.pos_n = _row(pg, 0, "north")
        self.pos_e = _row(pg, 1, "east")
        self.pos_d = _row(pg, 2, "down")
        self.dist_total = _row(pg, 3, "path len")
        pl.addLayout(pg)

        # ---- velocity ----------------------------------------------------
        vf, vl = _panel("Velocity (m/s)")
        vg = QGridLayout(); vg.setHorizontalSpacing(12); vg.setVerticalSpacing(2)
        self.vel_n = _row(vg, 0, "vN")
        self.vel_e = _row(vg, 1, "vE")
        self.vel_d = _row(vg, 2, "vD")
        self.vel_mag = _row(vg, 3, "speed")
        vl.addLayout(vg)

        # ---- attitude ----------------------------------------------------
        af, al = _panel("Attitude (deg)")
        ag = QGridLayout(); ag.setHorizontalSpacing(12); ag.setVerticalSpacing(2)
        self.att_r = _row(ag, 0, "roll")
        self.att_p = _row(ag, 1, "pitch")
        self.att_y = _row(ag, 2, "yaw")
        self.att_ar = _row(ag, 3, "accel roll")
        self.att_ap = _row(ag, 4, "accel pitch")
        al.addLayout(ag)

        # ---- status ------------------------------------------------------
        sf, sl = _panel("Tracking")
        sg = QGridLayout(); sg.setHorizontalSpacing(12); sg.setVerticalSpacing(2)
        self.st_state = _row(sg, 0, "state")
        self.st_fps = _row(sg, 1, "src fps")
        self.st_samples = _row(sg, 2, "samples")
        self.st_uptime = _row(sg, 3, "uptime")
        sl.addLayout(sg)

        for w in (pf, vf, af, sf):
            root.addWidget(w)
        root.addStretch(1)

        self._cum_dist = 0.0
        self._prev_pos: np.ndarray | None = None

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)                                  # 20 Hz UI refresh
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ----------------------------------------------------------------------

    def _refresh(self) -> None:
        traj, _flags, latest = self.history.snapshot()

        if latest is not None:
            p = latest.pos_ned
            v = latest.vel_ned
            r, pi, y = latest.rpy_deg

            self.pos_n.setText(f"{p[0]:+8.3f}")
            self.pos_e.setText(f"{p[1]:+8.3f}")
            self.pos_d.setText(f"{p[2]:+8.3f}")

            self.vel_n.setText(f"{v[0]:+7.3f}")
            self.vel_e.setText(f"{v[1]:+7.3f}")
            self.vel_d.setText(f"{v[2]:+7.3f}")
            self.vel_mag.setText(f"{float(np.linalg.norm(v)):7.3f}")

            self.att_r.setText(f"{r:+7.2f}")
            self.att_p.setText(f"{pi:+7.2f}")
            self.att_y.setText(f"{y:+7.2f}")

            arp = latest.accel_rpy_deg
            if arp is not None:
                self.att_ar.setText(f"{arp[0]:+7.2f}")
                self.att_ap.setText(f"{arp[1]:+7.2f}")
            else:
                self.att_ar.setText("--")
                self.att_ap.setText("--")

            ok = latest.tracking_ok
            self.st_state.setText("OK" if ok else "LOST")
            self.st_state.setObjectName("FieldGood" if ok else "FieldBad")
            self.st_state.style().unpolish(self.st_state)
            self.st_state.style().polish(self.st_state)

            # cumulative path length
            if self._prev_pos is not None:
                self._cum_dist += float(np.linalg.norm(p - self._prev_pos))
            self._prev_pos = p.copy()
            self.dist_total.setText(f"{self._cum_dist:8.3f}")

        self.st_fps.setText(f"{self._fps_get():6.1f}")
        self.st_samples.setText(f"{traj.shape[0]:>6d}")
        self.st_uptime.setText(f"{self.history.uptime_s:7.1f} s")
