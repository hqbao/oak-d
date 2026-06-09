"""In-app Qt window: the gyro-fusion strip chart (ALGORITHMS.md #5).

The one view that explains WHY the gyro-fused VIO stays straight where pure-vision
(``pose.vo``, grey) drifts during fast yaw. It subscribes VIO's per-frame
``frame.gyrofuse`` diagnostic (a REAL odometry output, not a re-derivation) and
draws a scrolling two-lane strip chart:

* TOP -- the RAW vision inter-frame rotation (grey) vs the gyro inter-frame
  rotation (cyan) in deg/frame, the disagreement between them shaded, and two
  reference lines: the gate ("gyro starts taking over") and gate+span
  ("full gyro");
* BOTTOM -- the resulting correction gain (vision weight 1->0) and the
  translation-trust (0..1).

When the grey trace pulls away from the cyan one and crosses the gate, the gain
collapses toward the gyro -- that IS the fast-yaw mechanism that keeps the fused
trajectory straight, made visible.

Source model (injected ``source_factory``)
------------------------------------------
The window taps a duck-typed gyro-fusion stream built by an injected zero-arg
factory. In the 4-process proc4 UI the factory is ALWAYS the IPC adapter
(:class:`~ui.modules.ipc_sources.IpcGyroFuseSource`), which subscribes VIO's
``frame.gyrofuse`` over IPC -- the UI never opens a device. The source pushes each
record from its IPC recv thread; the window's QTimer renders the chart on the GUI
thread, so the buffer between them is guarded by a lock.

cv2 is pulled lazily via :mod:`ui.viz.gyrofuse_render` (only when this window
opens); depthai is never imported (the UI is device-free by contract).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from . import theme

#: Zero-arg factory returning an object with ``start(callback)`` / ``stop()`` /
#: ``.error`` that streams per-frame ``FrameGyroFuse`` records to ``callback``.
SourceFactory = Callable[[], object]


class GyroFuseWindow(QWidget):
    """Scrolling gyro-fusion strip chart (vision vs gyro rotation + gain)."""

    def __init__(self, source_factory: SourceFactory, *,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self._make_source = source_factory

        self.setWindowTitle("Gyro fusion")
        self.setObjectName("GyroFuseWindow")
        self.resize(840, 560)
        self.setMinimumSize(620, 420)
        self.setStyleSheet(theme.QSS)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(self._build_header())

        self._view = QLabel("starting…  (waiting for gyro-fused frames)")
        self._view.setObjectName("Raster")
        self._view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._view.setMinimumHeight(360)
        self._view.setStyleSheet(
            f"background:{theme.BG}; color:{theme.TEXT_DIM};")
        root.addWidget(self._view, stretch=1)

        hint = QLabel(
            "VISION (eyes) = grey, accurate when slow but under-rotates on fast "
            "yaw · GYRO (inner ear) = cyan, always right · amber-shaded frames = "
            "the gyro took over (eyes & gyro disagreed past the gate)")
        hint.setObjectName("ScaleTick")
        hint.setWordWrap(True)
        root.addWidget(hint, stretch=0)

        self._status = QLabel("—")
        self._status.setObjectName("ImuCamStatus")
        root.addWidget(self._status, stretch=0)

        # The renderer is created lazily on start (pulls cv2 only then).
        self._chart = None
        self._source = None
        self._buf: np.ndarray | None = None
        self._lock = threading.Lock()
        self._pending: list = []                 # records from the recv thread
        self._running = False
        self._failed = False
        self._first_seen = False
        self._t_start = 0.0
        self._startup_timeout_s = 20.0

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)              # ~30 Hz redraw
        self._timer.timeout.connect(self._on_tick)

    # -- construction helpers --------------------------------------------- #
    def _build_header(self) -> QWidget:
        from PyQt6.QtWidgets import QFrame

        bar = QFrame()
        bar.setObjectName("Panel")
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(8)
        title = QLabel("GYRO FUSION")
        title.setObjectName("HeaderTitle")
        sub = QLabel("why the fused VIO stays straight under fast yaw")
        sub.setObjectName("HeaderSub")
        h.addWidget(title)
        h.addWidget(sub)
        h.addStretch(1)
        self._pill = QLabel("IPC")
        self._pill.setObjectName("FieldValue")
        h.addWidget(self._pill)
        return bar

    # -- lifecycle -------------------------------------------------------- #
    def ensure_started(self) -> None:
        if self._running and not self._failed:
            return
        self._teardown()
        self.start()

    def start(self) -> None:
        if self._running:
            return
        from ui.viz.gyrofuse_render import GyroFuseChart

        self._chart = GyroFuseChart()
        self._source = self._make_source()
        with self._lock:
            self._pending.clear()
        self._source.start(self._on_record)      # IPC recv thread -> _pending
        self._running = True
        self._failed = False
        self._first_seen = False
        self._t_start = time.monotonic()
        self._view.setText("starting…  (waiting for gyro-fused frames)")
        self._set_status("connecting…", theme.TEXT_DIM)
        self._timer.start()

    def stop(self) -> None:
        self._teardown()

    def _teardown(self) -> None:
        self._timer.stop()
        if self._source is not None:
            try:
                self._source.stop()
            except Exception:
                pass
        self._source = None
        self._running = False

    # -- data flow -------------------------------------------------------- #
    def _on_record(self, msg) -> None:
        """IPC recv-thread callback: stash one FrameGyroFuse record (cheap)."""
        with self._lock:
            self._pending.append(msg)

    def _drain(self) -> list:
        with self._lock:
            if not self._pending:
                return []
            out = self._pending
            self._pending = []
            return out

    # -- per-tick update -------------------------------------------------- #
    def _on_tick(self) -> None:
        from ui.viz.gyrofuse_render import GyroFuseSample

        records = self._drain()
        if records:
            self._first_seen = True
            for m in records:
                self._chart.add(GyroFuseSample(
                    vision_rot_deg=float(m.vision_rot_deg),
                    gyro_rot_deg=float(m.gyro_rot_deg),
                    disagree_deg=float(m.disagree_deg),
                    gain=float(m.gain), t_trust=float(m.t_trust),
                    gate_deg=float(m.gate_deg), span_deg=float(m.span_deg)))
            self._show_rgb(self._chart.render())
            last = records[-1]
            self._set_status(
                f"vision {last.vision_rot_deg:5.2f}°  gyro "
                f"{last.gyro_rot_deg:5.2f}°  disagree {last.disagree_deg:5.2f}°  "
                f"·  gain {last.gain:4.2f}  t_trust {last.t_trust:4.2f}  "
                f"·  n={self._chart.sample_count}",
                theme.TEXT)
        elif self._first_seen:
            # No new fused frames this tick: keep the last chart up (the source
            # legitimately goes quiet when the camera is not rotating much).
            return
        else:
            self._maybe_report_no_frame()

    def _show_rgb(self, rgb: np.ndarray) -> None:
        g = np.ascontiguousarray(rgb)
        self._buf = g
        self._blit(self._view, QImage(g.data, g.shape[1], g.shape[0],
                                      3 * g.shape[1],
                                      QImage.Format.Format_RGB888))

    @staticmethod
    def _blit(label: QLabel, img: QImage) -> None:
        pix = QPixmap.fromImage(img)
        target = label.size()
        if target.width() > 1 and target.height() > 1:
            pix = pix.scaled(target, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                             QtCore.Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(pix)

    # -- failure handling ------------------------------------------------- #
    def _maybe_report_no_frame(self) -> None:
        err = self._source.error if self._source is not None else None
        timed_out = (time.monotonic() - self._t_start) > self._startup_timeout_s
        if err:
            self._fail(err)
        elif timed_out:
            self._fail("no gyro-fused frames — is VIO running WITH gyro? "
                       "(gyro fusion only runs when the session has IMU "
                       "extrinsics and --no-gyro is NOT set)")

    def _fail(self, message: str) -> None:
        if self._failed:
            return
        self._failed = True
        self._view.setStyleSheet(
            f"color: {theme.WARN}; font-size: 14px; font-weight: bold;"
            f" background:{theme.BG};")
        self._view.setText(f"⚠  {message}")
        self._set_status("not streaming — reopen from the Visualize menu to retry",
                         theme.BAD)
        self._teardown()

    def _set_status(self, text: str, color: str) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color};")

    # -- Qt events -------------------------------------------------------- #
    def showEvent(self, event) -> None:                            # noqa: N802
        super().showEvent(event)
        self.ensure_started()

    def closeEvent(self, event) -> None:                           # noqa: N802
        try:
            self.stop()
        finally:
            super().closeEvent(event)
