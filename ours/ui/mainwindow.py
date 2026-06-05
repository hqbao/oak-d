"""Top-level QMainWindow: header bar, view-preset toolbar, viewport, side panel."""
from __future__ import annotations

from collections.abc import Callable

from PyQt6 import QtCore
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QStatusBar,
    QToolBar, QVBoxLayout, QWidget,
)

from ..lib.pose import PoseHistory
from ..sources.base import PoseSource
from . import theme
from .panels import TelemetryPanel
from .viewer3d import VIEW_PRESETS, Viewer3D


def _header(title: str, subtitle: str) -> QWidget:
    w = QWidget()
    w.setObjectName("Header")
    lay = QHBoxLayout(w)
    lay.setContentsMargins(14, 8, 14, 8)
    t = QLabel(title)
    t.setObjectName("HeaderTitle")
    s = QLabel(subtitle)
    s.setObjectName("HeaderSub")
    lay.addWidget(t)
    lay.addSpacing(16)
    lay.addWidget(s)
    lay.addStretch(1)
    return w


class MainWindow(QMainWindow):
    def __init__(
        self,
        history: PoseHistory,
        source: PoseSource,
        source_name: str = "fake",
    ) -> None:
        super().__init__()
        self.history = history
        self.source = source

        self.setWindowTitle(f"OAK-D Pose Viewer  ·  source: {source_name}")
        self.resize(1400, 860)
        self.setStyleSheet(theme.QSS)

        # ---- view-preset toolbar -----------------------------------------
        tb = QToolBar("Views")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(16, 16))

        # Start/Stop: the user levels the drone, then presses START to seed the
        # world frame from the current (gravity-leveled) attitude and begin.
        self.run_act = QAction("START", self)
        self.run_act.setCheckable(True)
        self.run_act.toggled.connect(self._toggle_run)
        tb.addAction(self.run_act)
        tb.addSeparator()

        for name in ("ISO", "TOP", "FRONT", "BACK", "LEFT", "RIGHT"):
            act = QAction(name, self)
            act.triggered.connect(lambda _checked=False, n=name: self._goto_view(n))
            tb.addAction(act)

        tb.addSeparator()

        self.follow_act = QAction("FOLLOW", self)
        self.follow_act.setCheckable(True)
        self.follow_act.toggled.connect(self._toggle_follow)
        tb.addAction(self.follow_act)

        self.clear_act = QAction("CLEAR TRAIL", self)
        self.clear_act.triggered.connect(self.history.clear)
        tb.addAction(self.clear_act)

        # Wipe the SLAM keyframe map (only when the source keeps one). Handy for
        # restarting a loop-closure test without relaunching the pipeline.
        if hasattr(source, "clear_slam_map"):
            self.clear_kf_act = QAction("CLEAR KEYFRAMES", self)
            self.clear_kf_act.triggered.connect(source.clear_slam_map)
            tb.addAction(self.clear_kf_act)

        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tb)

        # ---- central layout ----------------------------------------------
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(_header("OAK-D · POSE 3D", f"SRC = {source_name.upper()}"))

        body = QWidget()
        bh = QHBoxLayout(body)
        bh.setContentsMargins(6, 6, 6, 6)
        bh.setSpacing(6)

        self.viewer = Viewer3D(history)
        # Live SLAM map overlay (keyframe dots + loop-closure links), only when
        # the source publishes one (backend='slam'); harmless no-op otherwise.
        if hasattr(source, "slam_overlay_snapshot"):
            self.viewer.set_overlay_source(source.slam_overlay_snapshot)
        self.panel = TelemetryPanel(history, source_fps_getter=lambda: source.fps)
        self.panel.setFixedWidth(260)

        bh.addWidget(self.viewer, stretch=1)
        bh.addWidget(self.panel, stretch=0)

        root.addWidget(body, stretch=1)
        self.setCentralWidget(central)

        sb = QStatusBar()
        sb.showMessage("Ready.")
        self.setStatusBar(sb)

        # Poll the source for an abort reason (e.g. bad startup attitude) so we
        # can surface it and reset the START button when the worker bails out.
        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(300)
        self._poll.timeout.connect(self._poll_source)
        self._poll.start()

    # ----------------------------------------------------------------------

    def _toggle_run(self, on: bool) -> None:
        if on:
            self.run_act.setText("STOP")
            self.history.clear()
            self.source.start(self.history.push)
            self.statusBar().showMessage("Running — level held, world frame seeded.", 3000)
        else:
            self.run_act.setText("START")
            self.source.stop()
            self.statusBar().showMessage("Stopped.", 2000)

    def _poll_source(self) -> None:
        # Worker aborted with a reason (bad attitude, device error, ...).
        if self.source.error and self.run_act.isChecked():
            msg = self.source.error
            self.run_act.blockSignals(True)
            self.run_act.setChecked(False)
            self.run_act.setText("START")
            self.run_act.blockSignals(False)
            self.statusBar().showMessage(f"⚠ {msg}", 0)
            return
        # Worker exited on its own (end of stream) — reset the button.
        if self.run_act.isChecked() and not self.source.is_running():
            self.run_act.blockSignals(True)
            self.run_act.setChecked(False)
            self.run_act.setText("START")
            self.run_act.blockSignals(False)
            self.statusBar().showMessage("Source stopped.", 2000)

    # ----------------------------------------------------------------------

    def _goto_view(self, name: str) -> None:
        self.viewer.set_view(name)
        self.statusBar().showMessage(f"View: {name}", 1500)

    def _toggle_follow(self, on: bool) -> None:
        self.viewer.set_follow(on)
        self.statusBar().showMessage(f"Follow camera: {'ON' if on else 'OFF'}", 1500)

    def closeEvent(self, event) -> None:                              # noqa: N802
        try:
            self.source.stop()
        finally:
            super().closeEvent(event)
