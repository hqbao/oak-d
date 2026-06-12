"""In-app Qt window: the "Pose Graph" before/after visualiser (ALGORITHMS.md §4.3).

Shows the SLAM pose-graph "aha" on actual data: keyframe poses as graph NODES,
odometry edges chaining consecutive keyframes, the LOOP edge(s) connecting the
two revisited keyframes, and a BEFORE/AFTER toggle that swaps the raw/drifted VIO
node positions for the pose-graph-optimised ones -- so you SEE the loop close and
the drift correction redistribute SMOOTHLY along the whole trajectory (per-node
correction arrows make "spread along the path" literal).

It is a PURE CONSUMER of existing topics (no new IPC field): the source
(:class:`~ui.modules.ipc_sources.IpcPoseGraphSource`) joins VIO's raw ``pose.odom``
(BEFORE), SLAM's ``slam.map`` corrected poses (AFTER) and SLAM's ``slam.loop``
edges, and renders a 2D top-down (world X-Z) image via
:func:`ui.viz.posegraph_render.render_pose_graph`.

Two modes, one widget (the slider state machine), MIRRORING :mod:`ui.qt.ba_window`
------------------------------------------------------------------------------
A single "Follow latest" checkbox selects the mode:

* **LIVE** (constructor ``live=True`` -> checkbox default ON): each new snapshot
  (one per loop closure) advances the slider to the head and renders it.
* **REPLAY** (``live=False`` -> checkbox default OFF): the slider SCRUBS the
  buffered loop-closure snapshots; ``valueChanged`` reads ``source.snapshot_at(i)``
  under the source lock and renders that historical closure.

Loop closures are sporadic, so the window stays patiently on its "waiting" frame
until the first one arrives (NOT an error), exactly like :mod:`ui.qt.loop_window`.

cv2 is pulled lazily via :mod:`ui.viz.posegraph_render` (only when this window
opens); depthai is never imported (the UI is device-free by contract). The widget
structure mirrors :mod:`ui.qt.ba_window`.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget,
)

from . import theme

#: Zero-arg factory returning an object with ``start(callback)`` / ``stop()`` /
#: ``.error`` + ``snapshot_count()`` / ``snapshot_at(i)`` (the buffered deque the
#: slider scrubs). See :class:`~ui.modules.ipc_sources.IpcPoseGraphSource`.
SourceFactory = Callable[[], object]


class PoseGraphWindow(QWidget):
    """The pose-graph before/after visualiser (nodes + edges + correction arrows)."""

    def __init__(self, source_factory: SourceFactory, *, live: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self._make_source = source_factory
        self._live = bool(live)

        self.setWindowTitle("Pose Graph")
        self.setObjectName("PoseGraphWindow")
        self.resize(1160, 760)
        self.setMinimumSize(840, 560)
        self.setStyleSheet(theme.QSS)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(self._build_header())

        self._view = QLabel("waiting for a loop closure …")
        self._view.setObjectName("Raster")
        self._view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._view.setMinimumHeight(480)
        self._view.setStyleSheet(f"background:{theme.BG}; color:{theme.TEXT_DIM};")
        root.addWidget(self._view, stretch=1)

        root.addWidget(self._build_controls())

        hint = QLabel(
            "top-down X-Z · dots = keyframe poses · grey lines = odometry edges · "
            "magenta = loop edge (kf revisits kf) · before = raw/drifted (loop "
            "open) → after = pose-graph optimised (loop closed) · amber arrows = "
            "per-node correction, spread smoothly along the whole path")
        hint.setObjectName("ScaleTick")
        hint.setWordWrap(True)
        root.addWidget(hint, stretch=0)

        self._status = QLabel("—")
        self._status.setObjectName("ImuCamStatus")
        root.addWidget(self._status, stretch=0)

        self._source = None
        self._buf: np.ndarray | None = None
        self._lock = threading.Lock()
        self._pending = None                       # latest snapshot (latest-wins)
        self._running = False
        self._failed = False
        self._first_seen = False
        self._t_start = 0.0
        self._cur_count = 0                         # last-known slider range

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)                # ~20 Hz; closures are rare
        self._timer.timeout.connect(self._on_tick)

    # -- construction helpers --------------------------------------------- #
    def _build_header(self) -> QWidget:
        from PyQt6.QtWidgets import QFrame

        bar = QFrame()
        bar.setObjectName("Panel")
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(8)
        title = QLabel("POSE GRAPH")
        title.setObjectName("HeaderTitle")
        sub = QLabel("loop closure spreads drift correction along the whole path")
        sub.setObjectName("HeaderSub")
        h.addWidget(title)
        h.addWidget(sub)
        h.addStretch(1)
        self._pill = QLabel("LIVE" if self._live else "REPLAY")
        self._pill.setObjectName("FieldValue")
        h.addWidget(self._pill)
        return bar

    def _build_controls(self) -> QWidget:
        from PyQt6.QtWidgets import QFrame

        bar = QFrame()
        bar.setObjectName("Panel")
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(10)

        self._follow_cb = QCheckBox("Follow latest")
        # LIVE -> follow ON (rolling head); REPLAY -> follow OFF (scrub).
        self._follow_cb.setChecked(self._live)
        self._follow_cb.toggled.connect(self._on_follow_toggled)
        h.addWidget(self._follow_cb)

        self._before_cb = QCheckBox("Show before (raw / drifted)")
        self._before_cb.setChecked(False)
        self._before_cb.toggled.connect(self._on_before_toggled)
        h.addWidget(self._before_cb)

        h.addWidget(QLabel("loop closures"))
        self._slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        h.addWidget(self._slider, stretch=1)

        self._idx_lbl = QLabel("0 / 0")
        self._idx_lbl.setObjectName("FieldValue")
        h.addWidget(self._idx_lbl)
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
        from ui.viz.posegraph_render import render_pose_graph

        self._render = render_pose_graph
        self._source = self._make_source()
        with self._lock:
            self._pending = None
        self._source.start(self._on_snapshot)      # IPC recv thread -> _pending
        self._running = True
        self._failed = False
        self._first_seen = False
        self._cur_count = 0
        self._t_start = time.monotonic()
        self._show_rgb(self._render(None, show_before=self._before_cb.isChecked()))
        self._set_status("connecting…  (a loop must close first)", theme.TEXT_DIM)
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
    def _on_snapshot(self, snap) -> None:
        """IPC recv-thread callback: latch the freshest snapshot (cheap)."""
        with self._lock:
            self._pending = snap

    def _take_pending(self):
        with self._lock:
            snap, self._pending = self._pending, None
            return snap

    # -- per-tick update -------------------------------------------------- #
    def _on_tick(self) -> None:
        snap = self._take_pending()
        # Keep the slider range in sync with the source buffer (grows per closure).
        count = self._source.snapshot_count() if self._source is not None else 0
        if count != self._cur_count:
            self._cur_count = count
            self._sync_slider_range(count)

        if snap is not None:
            self._first_seen = True
            if self._follow_cb.isChecked():
                # Follow mode: jump the slider to the head and render the newest.
                self._set_slider_silently(max(0, count - 1))
                self._render_snapshot(snap)
            # else (scrub mode): the buffer grew but we hold the displayed index.
        elif not self._first_seen:
            self._maybe_report_no_frame()
        # else: keep the last rendered snapshot up (no new data this tick).

    # -- slider state machine --------------------------------------------- #
    def _sync_slider_range(self, count: int) -> None:
        hi = max(0, count - 1)
        self._slider.setMaximum(hi)
        if self._follow_cb.isChecked():
            self._set_slider_silently(hi)
        self._idx_lbl.setText(f"{self._slider.value() + (1 if count else 0)} "
                              f"/ {count}")

    def _set_slider_silently(self, value: int) -> None:
        """Set the slider WITHOUT firing valueChanged (avoid a re-render loop)."""
        self._slider.blockSignals(True)
        self._slider.setValue(value)
        self._slider.blockSignals(False)

    def _on_slider(self, value: int) -> None:
        """Scrub: render the buffered snapshot at ``value`` (replay/manual)."""
        if self._source is None:
            return
        snap = self._source.snapshot_at(int(value))
        count = self._source.snapshot_count()
        self._idx_lbl.setText(f"{value + (1 if count else 0)} / {count}")
        if snap is not None:
            self._render_snapshot(snap)

    def _on_follow_toggled(self, on: bool) -> None:
        if on and self._source is not None:
            # Resume following: jump to the head and render the newest snapshot.
            count = self._source.snapshot_count()
            self._set_slider_silently(max(0, count - 1))
            snap = self._source.snapshot_at(-1)
            if snap is not None:
                self._render_snapshot(snap)

    def _on_before_toggled(self, _on: bool) -> None:
        # Re-render the CURRENTLY-shown snapshot with the new before/after mode.
        if self._source is None:
            self._show_rgb(self._render(None,
                                        show_before=self._before_cb.isChecked()))
            return
        snap = self._source.snapshot_at(self._slider.value())
        if snap is None:
            snap = self._source.snapshot_at(-1)
        if snap is not None:
            self._render_snapshot(snap)
        else:
            self._show_rgb(self._render(None,
                                        show_before=self._before_cb.isChecked()))

    # -- render ----------------------------------------------------------- #
    def _render_snapshot(self, snap) -> None:
        self._show_rgb(self._render(snap, show_before=self._before_cb.isChecked()))
        n_kf = int(getattr(snap, "n_kf", 0))
        n_edges = len(getattr(snap, "loop_edges", ()))
        n_loops = int(getattr(snap, "n_loops", 0))
        # The headline number: how far PGO moved the nodes (max correction delta).
        before = np.asarray(snap.kf_before_xz, np.float64).reshape(-1, 2)
        after = np.asarray(snap.kf_after_xz, np.float64).reshape(-1, 2)
        if len(before) and len(before) == len(after):
            max_corr = float(np.linalg.norm(after - before, axis=1).max())
        else:
            max_corr = float("nan")
        corr_txt = (f"{max_corr:.3f} m" if np.isfinite(max_corr) else "n/a")
        self._set_status(
            f"loops {n_loops} · nodes {n_kf} · loop edges {n_edges} · "
            f"max node correction {corr_txt}", theme.TEXT)

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
        if err:
            self._fail(err)
        # No timeout-to-failure: a healthy SLAM that simply has not closed a loop
        # yet is NOT an error -- the "waiting" frame stays up. Only a real
        # connect/runtime error (surfaced on the source) fails the window.

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
