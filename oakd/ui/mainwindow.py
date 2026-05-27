"""Top-level QMainWindow: header bar, view-preset toolbar, viewport, side panel."""
from __future__ import annotations

from collections.abc import Callable

from PyQt6 import QtCore
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QStatusBar,
    QToolBar, QVBoxLayout, QWidget,
)

from ..pose import PoseHistory
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
        self.panel = TelemetryPanel(history, source_fps_getter=lambda: source.fps)
        self.panel.setFixedWidth(260)

        bh.addWidget(self.viewer, stretch=1)
        bh.addWidget(self.panel, stretch=0)

        root.addWidget(body, stretch=1)
        self.setCentralWidget(central)

        sb = QStatusBar()
        sb.showMessage("Ready.")
        self.setStatusBar(sb)

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
