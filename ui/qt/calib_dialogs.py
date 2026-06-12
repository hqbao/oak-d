"""Qt calibration wizards: gyroscope bias and six-position accelerometer.

Both dialogs drive the *tested* state machines from
:mod:`sky.sensors.calib_collect` (the stillness gate + six-face collector) with
live IMU samples from an injected stream, and persist the result through
:mod:`sky.sensors.calib_store`.

Stream contract (device-free)
-----------------------------
The dialog drives any object exposing ``start(callback)`` / ``stop()`` / ``.error``
/ ``.device_id``. In the 4-process proc4 UI the caller ALWAYS injects
:class:`ui.modules.ipc_sources.IpcImuRawSource` (capture's RAW ``imu.raw`` over
IPC) -- the UI never opens a device. The single-process default IMU stream
(``ImuStream``, which opens the device) is therefore never used here; it is
imported LAZILY only on the fallback path (``stream is None``), so importing this
module pulls neither depthai nor that single-process stream.

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
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QDialog, QVBoxLayout, QWidget,
)

from sky.sensors.calib_collect import (
    SixFaceCollector,
    StaticCollector,
    StaticCollectorConfig,
    face_name,
    gyro_bias_verdict,
)
from sky.sensors.calib_store import save_accel_calib, save_gyro_bias

from . import theme

# Per-face unit gravity directions (face k's expected specific-force direction),
# matching sky.sensors.calib_collect's face indexing -- used only to label the
# live sphere's captured dots with their direction; mirrors the collector's own
# _FACE_DIRS without importing that private constant.
_FACE_DIRS = np.array([
    [+1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, +1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, +1.0], [0.0, 0.0, -1.0],
])


class _CalibDialogBase(QDialog):
    """Shared plumbing: an IMU stream feeding a queue drained by a UI timer."""

    def __init__(self, parent=None, stream: object | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(theme.QSS)
        self.setMinimumWidth(440)
        self._queue: deque = deque(maxlen=4000)
        # The dialog drives ANY object with start()/stop()/.error/.device_id. In
        # proc4 the caller ALWAYS injects IpcImuRawSource (capture's RAW imu.raw
        # over IPC) -- the UI is device-free, so there is no in-process default
        # stream to fall back to. A None stream is therefore a programming error
        # here; surface it clearly rather than opening a device this project
        # cannot. (The single-process build injects its own device ImuStream.)
        if stream is None:
            raise ValueError(
                "calib dialog needs an injected `stream` in the device-free "
                "proc4 UI (e.g. IpcImuRawSource); the in-process default IMU "
                "stream is not available in the `ui` project.")
        self._stream = stream
        self._owns_stream = False
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
                 stream: object | None = None) -> None:
        super().__init__(parent, stream=stream)
        self.setWindowTitle("Calibrate Gyroscope")
        self._device_id = device_id
        self._coll = StaticCollector(StaticCollectorConfig(
            gyro_thresh=0.05, accel_dev_thresh=0.4, window_s=1.0,
            min_samples=80))
        self._bias: np.ndarray | None = None
        self._verdict = None

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
        self._verdict = None
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
            self._verdict = gyro_bias_verdict(
                self._coll.gyro_std_max, self._coll.n)
            if self._device_id is None:
                self._device_id = self._stream.device_id
            self._stop_stream()

    def _refresh(self) -> None:
        if self._bias is not None:
            b = self._bias
            v = self._verdict
            self._bar.setValue(100)
            self._result.setText(
                f"bias = [{b[0]:+.5f}, {b[1]:+.5f}, {b[2]:+.5f}] rad/s"
                + (f"   (noise {v.metric:.4f})" if v is not None else ""))
            self._start_btn.setText("REDO")
            if v is not None and v.ok:
                self._status.setText(f"Done ({self._coll.n} samples). "
                                     "Review and SAVE.")
                self._save_btn.setEnabled(True)
            else:
                self._status.setText(
                    "⚠ " + (v.message if v is not None else "Rejected."))
                self._save_btn.setEnabled(False)
        else:
            self._bar.setValue(int(self._coll.progress * 100))
            if self._running:
                self._status.setText(
                    f"Hold still…  {self._coll.n} samples, "
                    f"{self._coll.progress * 100:.0f}%")

    def _on_save(self) -> None:
        if self._bias is None:
            return
        if self._verdict is not None and not self._verdict.ok:
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
                 stream: object | None = None) -> None:
        super().__init__(parent, stream=stream)
        self.setWindowTitle("Calibrate Accelerometer · 6-position")
        self.setMinimumWidth(480)
        self._device_id = device_id
        self._coll = SixFaceCollector()
        # Live-sphere render cadence: matplotlib is heavy, so we re-render ONLY when
        # the captured-face SET changes (each new face) or completion flips -- never
        # on every ~30 Hz drain tick. ``_sphere_key`` caches what was last drawn.
        self._sphere_key: tuple | None = None

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

        # Live gravity-sphere view: the captured raw vectors land on the sphere as
        # the operator tumbles (RED), and SNAP onto the g-sphere (GREEN) after the
        # 6th + solve. Rendered by the SAME shared core the offline tool uses
        # (imu_camera.tools.gravity_sphere.render_gravity_sphere), shown as a pixmap.
        self._sphere = QLabel("Press START — captured faces appear on the gravity "
                              "sphere here.")
        self._sphere.setObjectName("DialogHint")
        self._sphere.setWordWrap(True)
        self._sphere.setMinimumHeight(360)
        self._sphere.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._sphere)

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
        # Reset the live sphere to its empty (sphere-only) state so a re-run does
        # not carry over the previous run's dots.
        self._sphere_key = None
        self._sphere.setPixmap(QPixmap())
        self._sphere.setText("Tumble through the six faces — captured vectors land "
                             "on the gravity sphere as you go.")
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
            v = self._coll.verdict()
            self._result.setText(
                f"residual = {cal.residual_g:.4f} m/s²   "
                f"(6/6 faces, lower is better)")
            self._start_btn.setText("REDO")
            self._bar.setValue(100)
            if v.ok:
                self._save_btn.setEnabled(True)
            else:
                self._save_btn.setEnabled(False)
                self._status.setText("⚠ " + v.message)
        # Update the live gravity sphere (throttled: only on a face-set / solve change).
        self._refresh_sphere()

    def _captured_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """The raw mean accel vectors captured so far + their face directions.

        ``SixFaceCollector`` stores one mean per captured face in its ``_caps``
        dict (face index -> raw vector) and discards them after solving; the dialog
        owns that collector LOCALLY, so reading the captured means here needs no
        IPC / comms change. Returns ``(k, 3)`` raw vectors and the matching ``(k, 3)``
        unit gravity directions, ordered by face index for a stable render.
        """
        caps = self._coll._caps                      # local collector state (no IPC)
        idx = sorted(caps)
        if not idx:
            return np.empty((0, 3)), np.empty((0, 3))
        raw = np.array([caps[i] for i in idx], dtype=np.float64)
        dirs = _FACE_DIRS[idx]
        return raw, dirs

    def _refresh_sphere(self) -> None:
        """Re-render the live gravity sphere when (and only when) the picture changed.

        The render is a full matplotlib-Agg figure (~tenths of a second), so we key
        it on the captured-face set + whether a calibration exists and skip the work
        on the steady ~30 Hz drain ticks where nothing new landed. Imports are LAZY
        so opening this dialog is the only time matplotlib/this tool load.
        """
        raw, dirs = self._captured_raw()
        cal = self._coll.calibration
        # Cache key: which faces are captured + solved-or-not. A new face capture or
        # the post-6th solve flips this; a plain "still holding" tick does not.
        key = (tuple(sorted(self._coll.captured_faces)), cal is not None)
        if key == self._sphere_key:
            return
        self._sphere_key = key
        if raw.shape[0] == 0:
            return                                   # nothing captured yet: keep the hint

        # Lazy import: matplotlib + the shared renderer load only once the dialog is
        # actually drawing a sphere (keeps the flight / import path clean).
        from imu_camera.tools.gravity_sphere import render_gravity_sphere

        img = render_gravity_sphere(raw, cal, directions=dirs)
        h, w, _ = img.shape
        # Agg gives a contiguous RGB888 buffer; wrap it as a QImage then copy into a
        # pixmap (copy() detaches from the soon-freed numpy buffer).
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())
        self._sphere.setText("")
        self._sphere.setPixmap(pix.scaled(
            self._sphere.width(), self._sphere.height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))

    def _on_save(self) -> None:
        cal = self._coll.calibration
        if cal is None:
            return
        if not self._coll.verdict().ok:
            return
        dev = self._device_id or "default"
        save_accel_calib(dev, cal, len(self._coll.captured_faces))
        self._status.setText(f"Saved for device {dev}.")
        self._save_btn.setEnabled(False)

    def _on_error(self, msg: str) -> None:
        super()._on_error(msg)
        self._status.setText(f"⚠ {msg}")
