"""Qt calibration wizards: gyroscope bias and six-position accelerometer.

Both dialogs drive the *tested* state machines from
:mod:`ours.lib.imu.calib_collect` (the stillness gate + six-face collector) with
live IMU samples from :class:`ours.flows.capture.imu_stream.ImuStream`, and persist the
result through :mod:`ours.lib.imu.calib_store`.

Threading: the IMU stream calls :meth:`_feed_sample` on its background thread,
which only appends to a thread-safe queue. A UI-thread ``QTimer`` then drains the
queue and feeds the collector, so ALL collector access stays on one thread (no
locks, no races). Splitting it this way also makes the dialogs unit-testable
offline: a test can call :meth:`_feed_sample` with synthetic samples and tick
:meth:`_drain` directly, no device required.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QDialog, QVBoxLayout, QWidget,
)

from ..flows.capture.imu_stream import ImuStream
from ..lib.imu.calib_collect import (
    SixFaceCollector,
    StaticCollector,
    StaticCollectorConfig,
    face_name,
)
from ..lib.imu.calib_store import save_accel_calib, save_gyro_bias
from . import theme


class _CalibDialogBase(QDialog):
    """Shared plumbing: an IMU stream feeding a queue drained by a UI timer."""

    def __init__(self, parent=None, rate_hz: int = 200,
                 stream: ImuStream | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(theme.QSS)
        self.setMinimumWidth(440)
        self._queue: deque = deque(maxlen=4000)
        self._stream = stream if stream is not None else ImuStream(rate_hz)
        self._owns_stream = stream is None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)               # ~30 Hz UI drain
        self._timer.timeout.connect(self._drain)
        self._running = False

    # -- stream lifecycle -------------------------------------------------- #
    def _start_stream(self) -> None:
        if self._running:
            return
        self._queue.clear()
        self._stream.start(self._feed_sample)
        self._timer.start()
        self._running = True

    def _stop_stream(self) -> None:
        self._timer.stop()
        if self._owns_stream:
            self._stream.stop()
        self._running = False

    def _feed_sample(self, gyro, accel, t_s) -> None:
        """Called on the IMU thread -- queue only, no collector access here."""
        self._queue.append((np.asarray(gyro), np.asarray(accel), float(t_s)))

    def _drain(self) -> None:
        """UI-thread: feed all queued samples to the collector, refresh view."""
        if self._stream.error:
            self._on_error(self._stream.error)
            return
        while self._queue:
            gyro, accel, t_s = self._queue.popleft()
            self._on_sample(gyro, accel, t_s)
        self._refresh()

    # -- subclass hooks ---------------------------------------------------- #
    def _on_sample(self, gyro, accel, t_s) -> None:        # pragma: no cover
        raise NotImplementedError

    def _refresh(self) -> None:                            # pragma: no cover
        raise NotImplementedError

    def _on_error(self, msg: str) -> None:
        self._stop_stream()

    def closeEvent(self, event) -> None:                              # noqa: N802
        self._stop_stream()
        super().closeEvent(event)


class GyroCalibDialog(_CalibDialogBase):
    """Estimate the gyro zero-rate bias from one motionless window."""

    def __init__(self, parent=None, device_id: str | None = None,
                 stream: ImuStream | None = None) -> None:
        super().__init__(parent, stream=stream)
        self.setWindowTitle("Calibrate Gyroscope")
        self._device_id = device_id
        self._coll = StaticCollector(StaticCollectorConfig(
            gyro_thresh=0.05, accel_dev_thresh=0.4, window_s=1.0,
            min_samples=80))
        self._bias: np.ndarray | None = None

        root = QVBoxLayout(self)
        title = QLabel("GYROSCOPE BIAS CALIBRATION")
        title.setObjectName("PanelTitle")
        root.addWidget(title)
        hint = QLabel("Place the camera on a flat, completely still surface, "
                      "then press START and do not touch it.")
        hint.setObjectName("DialogHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        root.addWidget(self._bar)
        self._status = QLabel("Idle.")
        self._status.setObjectName("DialogHint")
        root.addWidget(self._status)
        self._result = QLabel("bias = —")
        self._result.setObjectName("DialogMono")
        root.addWidget(self._result)

        btns = QHBoxLayout()
        self._start_btn = QPushButton("START")
        self._start_btn.clicked.connect(self._on_start)
        self._save_btn = QPushButton("SAVE")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        close_btn = QPushButton("CLOSE")
        close_btn.clicked.connect(self.close)
        btns.addWidget(self._start_btn)
        btns.addWidget(self._save_btn)
        btns.addStretch(1)
        btns.addWidget(close_btn)
        root.addLayout(btns)

    def _on_start(self) -> None:
        self._coll.reset()
        self._bias = None
        self._save_btn.setEnabled(False)
        self._result.setText("bias = —")
        self._status.setText("Hold still…")
        self._start_stream()

    def _on_sample(self, gyro, accel, t_s) -> None:
        if self._bias is not None:
            return
        self._coll.feed(gyro, accel, t_s)
        if self._coll.ready:
            self._bias = self._coll.gyro_mean.copy()
            if self._device_id is None:
                self._device_id = self._stream.device_id
            self._stop_stream()

    def _refresh(self) -> None:
        if self._bias is not None:
            b = self._bias
            self._bar.setValue(100)
            self._status.setText(f"Done ({self._coll.n} samples). "
                                 "Review and SAVE.")
            self._result.setText(
                f"bias = [{b[0]:+.5f}, {b[1]:+.5f}, {b[2]:+.5f}] rad/s")
            self._save_btn.setEnabled(True)
            self._start_btn.setText("REDO")
        else:
            self._bar.setValue(int(self._coll.progress * 100))
            if self._running:
                self._status.setText(
                    f"Hold still…  {self._coll.n} samples, "
                    f"{self._coll.progress * 100:.0f}%")

    def _on_save(self) -> None:
        if self._bias is None:
            return
        dev = self._device_id or "default"
        save_gyro_bias(dev, self._bias, self._coll.n)
        self._status.setText(f"Saved for device {dev}.")
        self._save_btn.setEnabled(False)

    def _on_error(self, msg: str) -> None:
        super()._on_error(msg)
        self._status.setText(f"⚠ {msg}")


class AccelCalibDialog(_CalibDialogBase):
    """Six-position accelerometer calibration (bias + scale + misalignment)."""

    def __init__(self, parent=None, device_id: str | None = None,
                 stream: ImuStream | None = None) -> None:
        super().__init__(parent, stream=stream)
        self.setWindowTitle("Calibrate Accelerometer · 6-position")
        self.setMinimumWidth(480)
        self._device_id = device_id
        self._coll = SixFaceCollector()

        root = QVBoxLayout(self)
        title = QLabel("ACCELEROMETER 6-POSITION CALIBRATION")
        title.setObjectName("PanelTitle")
        root.addWidget(title)
        hint = QLabel("Press START, then hold the camera still with each face "
                      "up and down in turn (any order). Each face is captured "
                      "automatically once it is steady; rotate to the next "
                      "after the tick.")
        hint.setObjectName("DialogHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # Six face indicators.
        grid = QGridLayout()
        self._face_lbls: list[QLabel] = []
        for i in range(6):
            lab = QLabel(f"○ {face_name(i)}")
            lab.setObjectName("FaceTodo")
            self._face_lbls.append(lab)
            grid.addWidget(lab, i // 3, i % 3)
        wrap = QWidget()
        wrap.setLayout(grid)
        root.addWidget(wrap)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        root.addWidget(self._bar)
        self._status = QLabel("Idle.")
        self._status.setObjectName("DialogHint")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        self._result = QLabel("residual = —")
        self._result.setObjectName("DialogMono")
        root.addWidget(self._result)

        btns = QHBoxLayout()
        self._start_btn = QPushButton("START")
        self._start_btn.clicked.connect(self._on_start)
        self._save_btn = QPushButton("SAVE")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        close_btn = QPushButton("CLOSE")
        close_btn.clicked.connect(self.close)
        btns.addWidget(self._start_btn)
        btns.addWidget(self._save_btn)
        btns.addStretch(1)
        btns.addWidget(close_btn)
        root.addLayout(btns)
        self._last_status = None

    def _on_start(self) -> None:
        self._coll.reset()
        self._save_btn.setEnabled(False)
        self._result.setText("residual = —")
        self._last_status = None
        for i, lab in enumerate(self._face_lbls):
            lab.setText(f"○ {face_name(i)}")
            lab.setObjectName("FaceTodo")
            lab.setStyleSheet(theme.QSS)
        self._start_stream()

    def _on_sample(self, gyro, accel, t_s) -> None:
        if self._coll.complete:
            return
        self._last_status = self._coll.feed(gyro, accel, t_s)
        if self._coll.complete:
            if self._device_id is None:
                self._device_id = self._stream.device_id
            self._stop_stream()

    def _refresh(self) -> None:
        st = self._last_status
        captured = set(self._coll.captured_faces)
        for i, lab in enumerate(self._face_lbls):
            done = i in captured
            lab.setText(("● " if done else "○ ") + face_name(i))
            lab.setObjectName("FaceDone" if done else "FaceTodo")
            lab.setStyleSheet(theme.QSS)
        if st is not None:
            self._bar.setValue(int(st.progress * 100))
            self._status.setText(st.message)
        if self._coll.complete and self._coll.calibration is not None:
            cal = self._coll.calibration
            self._result.setText(
                f"residual = {cal.residual_g:.4f} m/s²   "
                f"(6/6 faces, lower is better)")
            self._save_btn.setEnabled(True)
            self._start_btn.setText("REDO")
            self._bar.setValue(100)

    def _on_save(self) -> None:
        cal = self._coll.calibration
        if cal is None:
            return
        dev = self._device_id or "default"
        save_accel_calib(dev, cal, len(self._coll.captured_faces))
        self._status.setText(f"Saved for device {dev}.")
        self._save_btn.setEnabled(False)

    def _on_error(self, msg: str) -> None:
        super()._on_error(msg)
        self._status.setText(f"⚠ {msg}")
