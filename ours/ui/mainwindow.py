"""Top-level QMainWindow: header bar, view-preset toolbar, viewport, side panel."""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from PyQt6 import QtCore
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QStatusBar,
    QToolBar, QVBoxLayout, QWidget,
)

from ..lib.misc.pose import PoseHistory
from .source import PoseSource
from . import theme
from .panels import TelemetryPanel
from .viewer3d import VIEW_PRESETS, Viewer3D

_REPO_ROOT = Path(__file__).resolve().parents[2]


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
        default_view: str = "ISO",
    ) -> None:
        super().__init__()
        self.history = history
        self.source = source

        self.setWindowTitle(f"OAK-D Pose Viewer  ·  source: {source_name}")
        self.resize(1400, 860)
        self.setStyleSheet(theme.QSS)

        # ---- minimal toolbar: just the primary START/STOP action ----------
        # Everything else now lives in the feature menu bar (the on-screen
        # controls had outgrown a single toolbar), keeping the toolbar to the
        # one action the operator reaches for most.
        tb = QToolBar("Run")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(16, 16))
        self.run_act = QAction("START", self)
        self.run_act.setCheckable(True)
        self.run_act.toggled.connect(self._toggle_run)
        tb.addAction(self.run_act)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tb)

        # ---- feature menu bar --------------------------------------------
        self._build_menus(source)

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

        self.viewer = Viewer3D(history, default_view=default_view)
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

    def _build_menus(self, source: PoseSource) -> None:
        """Organise the features into a menu bar (View / Calibration / Visualize)."""
        mbar = self.menuBar()

        view_menu = mbar.addMenu("View")
        for name in VIEW_PRESETS:
            act = QAction(name.title(), self)
            act.triggered.connect(lambda _c=False, n=name: self._goto_view(n))
            view_menu.addAction(act)
        view_menu.addSeparator()
        self.follow_act = QAction("Follow Camera", self)
        self.follow_act.setCheckable(True)
        self.follow_act.toggled.connect(self._toggle_follow)
        view_menu.addAction(self.follow_act)
        clear_act = QAction("Clear Trail", self)
        clear_act.triggered.connect(self.history.clear)
        view_menu.addAction(clear_act)
        if hasattr(source, "clear_slam_map"):
            kf_act = QAction("Clear Keyframes", self)
            kf_act.triggered.connect(source.clear_slam_map)
            view_menu.addAction(kf_act)

        cal_menu = mbar.addMenu("Calibration")
        gyro_act = QAction("Gyroscope Bias…", self)
        gyro_act.triggered.connect(self._open_gyro_calib)
        cal_menu.addAction(gyro_act)
        accel_act = QAction("Accelerometer (6-position)…", self)
        accel_act.triggered.connect(self._open_accel_calib)
        cal_menu.addAction(accel_act)

        vis_menu = mbar.addMenu("Visualize")
        imucam_act = QAction("Camera + IMU (synced, live)…", self)
        imucam_act.triggered.connect(self._open_imucam)
        vis_menu.addAction(imucam_act)
        triplet_act = QAction("Camera + Depth + IMU (triplet)…", self)
        triplet_act.triggered.connect(self._launch_triplet)
        vis_menu.addAction(triplet_act)
        stereo_act = QAction("Stereo Depth…", self)
        stereo_act.triggered.connect(self._launch_stereo)
        vis_menu.addAction(stereo_act)

    def _release_device(self, reason: str) -> bool:
        """Stop the live source so a calibration/visualize tool can open the device.

        The OAK-D is single-client: a wizard or external viewer cannot connect
        while the VIO source is streaming. If the source is running we stop it
        (and reset the START button) after confirming with the operator.
        """
        if not self.source.is_running():
            return True
        ans = QMessageBox.question(
            self, "Release device",
            f"{reason}\n\nThis needs exclusive access to the OAK-D, so the live "
            "pipeline will be stopped. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ans != QMessageBox.StandardButton.Yes:
            return False
        self.run_act.blockSignals(True)
        self.run_act.setChecked(False)
        self.run_act.setText("START")
        self.run_act.blockSignals(False)
        self.source.stop()
        self.statusBar().showMessage("Live pipeline stopped for device access.",
                                     2000)
        return True

    def _open_gyro_calib(self) -> None:
        if not self._release_device("Gyroscope bias calibration"):
            return
        from .calib_dialogs import GyroCalibDialog
        GyroCalibDialog(self).exec()

    def _open_accel_calib(self) -> None:
        if not self._release_device("Six-position accelerometer calibration"):
            return
        from .calib_dialogs import AccelCalibDialog
        AccelCalibDialog(self).exec()

    def _open_imucam(self) -> None:
        """Open the synced camera/IMU view inside the app (live, our own UI)."""
        if not self._release_device("Synced camera + IMU view"):
            return
        from .imucam_window import ImuCamWindow
        # A member ref keeps the window alive; reuse one instance so repeated
        # opens don't stack streams on the single-client device.
        win = getattr(self, "_imucam_win", None)
        if win is None:
            win = ImuCamWindow(parent=self)
            self._imucam_win = win
        win.show()
        win.raise_()
        win.activateWindow()
        win.ensure_started()          # retry on every open (e.g. after replug)
        self.statusBar().showMessage("Synced camera + IMU view opened.", 2500)

    def _launch_triplet(self) -> None:
        """Open the synced image+depth+IMU triplet inside the app (our own UI)."""
        if not self._release_device("Camera + Depth + IMU triplet view"):
            return
        from .synced_window import SyncedViewWindow
        # Reuse one instance so repeated opens don't stack streams on the
        # single-client device (mirrors _open_imucam).
        win = getattr(self, "_triplet_win", None)
        if win is None:
            win = SyncedViewWindow(parent=self)
            self._triplet_win = win
        win.show()
        win.raise_()
        win.activateWindow()
        win.ensure_started()
        self.statusBar().showMessage("Camera + Depth + IMU triplet opened.", 2500)

    def _launch_stereo(self) -> None:
        self._launch_tool(["-m", "ours.tools.stereo_view", "--live", "--fast"],
                          "Stereo depth")

    def _launch_tool(self, args: list[str], label: str) -> None:
        """Launch a proven live viewer tool in its own process (real data)."""
        if not self._release_device(f"{label} viewer"):
            return
        env = dict(os.environ)
        env["PYTHONPATH"] = (str(_REPO_ROOT) + os.pathsep
                             + env.get("PYTHONPATH", ""))
        try:
            subprocess.Popen([sys.executable, *args], cwd=str(_REPO_ROOT),
                             env=env)
            self.statusBar().showMessage(f"Launched {label} viewer.", 2500)
        except OSError as e:
            QMessageBox.warning(self, "Launch failed",
                                f"Could not start {label}:\n{e}")

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
