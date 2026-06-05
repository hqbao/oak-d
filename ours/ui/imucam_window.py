"""In-app Qt window for the split camera/IMU front-end's synchronised output.

This is the "visualise live, on our own UI" view: it runs the REAL split
acquisition flows -- :class:`~ours.flows.cam_reader.CamReaderFlow` and
:class:`~ours.flows.imu_reader.ImuReaderFlow` -- over a private
:class:`~ours.lib.flow.pubsub.Bus` and renders every
:class:`~ours.lib.flow.messages.ImuCamPacket` they publish straight into a Qt
widget. No cv2 window, no subprocess: the synced view lives inside the pose
viewer application.

The drawing is the same honest renderer the dev tool uses
(:mod:`ours.lib.viz.imucam_render`) -- nothing is computed in a parallel pipeline
to fill the panels; each panel is exactly what the packet carries. cv2 is pulled
only by that renderer (when this window is opened), so the base UI stays cv2-free.

* **Live** (default): ``LiveCamSource`` + ``LiveImuSource`` -- the OAK-D is
  single-client, so the caller must release the VIO device first (the menu does
  this via ``MainWindow._release_device``). Validated on the bench.
* **Replay** (``source_factory`` injected): drives the same window from a
  recorded session with no device -- this is what the offscreen self-test uses.
"""
from __future__ import annotations

import queue
import time
from collections.abc import Callable

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..flows.cam_reader import CamReaderFlow
from ..flows.cam_reader.sources import CamSource
from ..flows.imu_reader import ImuReaderFlow
from ..flows.imu_reader.sources import ImuSource
from ..lib.flow import Bus, Flow, topics
from ..lib.viz.imucam_render import GyroChart, compose
from . import theme

# (cam source, imu source) factory -- injected so the window runs live (default)
# or off a recorded session (self-test) with identical rendering.
SourceFactory = Callable[[], tuple[CamSource, ImuSource]]


def live_source_factory(width: int = 640, height: int = 400,
                        fps: int = 20) -> SourceFactory:
    """Default factory: the OAK-D cameras + IMU (depthai pulled lazily here)."""
    def _make() -> tuple[CamSource, ImuSource]:
        from ..flows.cam_reader.sources import LiveCamSource
        from ..flows.imu_reader.sources import LiveImuSource
        return (LiveCamSource(width=width, height=height, fps=fps),
                LiveImuSource(rate_hz=200))
    return _make


class _QueueSink(Flow):
    """Drop each ImuCamPacket into a queue for the Qt (display) thread."""

    def __init__(self, bus: Bus, out: "queue.Queue") -> None:
        super().__init__("imucam-ui-sink", bus)
        self._out = out
        self.on(topics.IMUCAM_SAMPLE, [self._Enqueue(out)])

    class _Enqueue:
        name = "enqueue"

        def __init__(self, out: "queue.Queue") -> None:
            self._out = out

        def run(self, ctx, msg):
            try:
                self._out.put_nowait(msg)
            except queue.Full:
                pass                        # drop to stay realtime on the UI
            return None

    def on_end(self) -> None:
        try:
            self._out.put_nowait(None)      # sentinel: stream finished
        except queue.Full:
            pass


class ImuCamWindow(QWidget):
    """Live synced camera/IMU view embedded in the pose-viewer application."""

    def __init__(self, source_factory: SourceFactory | None = None, *,
                 fps: int = 20, parent: QWidget | None = None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self._make_sources = source_factory or live_source_factory(fps=fps)
        self._fps = max(1, int(fps))

        self.setWindowTitle("Camera + IMU — synced (live)")
        self.setObjectName("ImuCamWindow")
        self.resize(1280, 420)
        self.setStyleSheet(theme.QSS)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)
        self._view = QLabel("starting…")
        self._view.setObjectName("ImuCamView")
        self._view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._view.setMinimumHeight(360)
        self._status = QLabel("—")
        self._status.setObjectName("ImuCamStatus")
        root.addWidget(self._view, stretch=1)
        root.addWidget(self._status, stretch=0)

        self._chart = GyroChart()
        self._queue: "queue.Queue" = queue.Queue(maxsize=8)
        self._bus: Bus | None = None
        self._cam: CamReaderFlow | None = None
        self._imu: ImuReaderFlow | None = None
        self._sink: _QueueSink | None = None
        self._running = False
        self._ended = False
        self._first_seen = False
        self._failed = False
        self._t_start = 0.0
        self._buf: np.ndarray | None = None   # keep QImage backing alive

        # If no frame arrives within this window (and nothing reports an error),
        # assume the device is unreachable/stalled rather than hanging forever.
        self._startup_timeout_s = 12.0

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._on_tick)

    # ------------------------------------------------------------------ #
    def ensure_started(self) -> None:
        """(Re)start streaming. Retries cleanly after a previous failure.

        Called every time the view is opened from the menu: if it is already
        streaming it is a no-op (just raise the window); if it failed before
        (e.g. the OAK-D was unplugged) it tears the dead graph down and starts
        fresh, so plugging the device in and reopening retries.
        """
        if self._running and not self._failed:
            return
        self._teardown()
        self.start()

    def start(self) -> None:
        """Build the flow graph and begin streaming into the widget."""
        if self._running:
            return
        self._clear_queue()                  # drop any stale END sentinel
        cam_src, imu_src = self._make_sources()
        self._bus = Bus()
        self._sink = _QueueSink(self._bus, self._queue)
        self._sink.expected_ends = 1
        self._imu = ImuReaderFlow(self._bus, imu_src)
        self._cam = CamReaderFlow(self._bus, cam_src, fps=self._fps,
                                  realtime=True)
        # Start consumers before producers; the IMU source must be filling the
        # buffer before the first camera trigger drains it.
        self._sink.start()
        self._imu.start()
        self._cam.start()
        self._running = True
        self._ended = False
        self._first_seen = False
        self._failed = False
        self._t_start = time.monotonic()
        self._view.setText("starting…  (opening the OAK-D)")
        self._timer.start()

    def stop(self) -> None:
        self._teardown()

    def _teardown(self) -> None:
        self._timer.stop()
        for f in (self._cam, self._imu, self._sink):
            if f is not None:
                try:
                    f.stop()
                except Exception:
                    pass
        self._cam = self._imu = self._sink = self._bus = None
        self._running = False

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------ #
    def _on_tick(self) -> None:
        packet = self._drain_latest()
        if packet is None:
            self._maybe_report_no_frame()
            return
        self._first_seen = True
        row = compose(packet, self._chart)               # BGR uint8 (H, W, 3)
        self._show(row)
        self._status.setText(
            f"seq={packet.seq}   imu samples={packet.imu_ts.size}   "
            f"left {packet.gray_left.shape[1]}×{packet.gray_left.shape[0]}")

    def _maybe_report_no_frame(self) -> None:
        """Surface a clean error/end state instead of hanging on 'starting…'."""
        if self._first_seen:
            if self._ended:
                self._status.setText("stream ended")
                self._timer.stop()
            return
        # No frame yet: decide whether the stream failed or simply finished.
        reason = self._failure_reason()
        threads_dead = (self._cam is not None and not self._cam.is_alive()
                        and self._imu is not None and not self._imu.is_alive())
        timed_out = (time.monotonic() - self._t_start) > self._startup_timeout_s
        if reason or self._ended or threads_dead or timed_out:
            self._fail(reason or
                       ("no frames — is the OAK-D connected and free? "
                        "(nothing else may hold the device)"))

    def _fail(self, message: str) -> None:
        if self._failed:
            return
        self._failed = True
        self._view.setText(f"⚠  {message}")
        self._status.setText("not streaming — reopen from the Visualize menu to retry")
        self._teardown()          # release the dead graph so a reopen retries

    def _failure_reason(self) -> str | None:
        """The first concrete error reported by the camera or IMU source."""
        if self._cam is not None and getattr(self._cam, "error", None):
            return self._cam.error
        imu_src = getattr(self._imu, "source", None)
        if imu_src is not None and getattr(imu_src, "error", None):
            return f"IMU open failed: {imu_src.error}"
        return None

    def _drain_latest(self):
        """Return the most recent packet, dropping stale ones to stay realtime."""
        latest = None
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:                              # END sentinel
                self._ended = True
                break
            latest = item
        return latest

    def _show(self, bgr: np.ndarray) -> None:
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])       # BGR -> RGB
        self._buf = rgb                                   # keep alive for QImage
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img)
        target = self._view.size()
        if target.width() > 1 and target.height() > 1:
            pix = pix.scaled(target, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                             QtCore.Qt.TransformationMode.SmoothTransformation)
        self._view.setPixmap(pix)

    # ------------------------------------------------------------------ #
    def showEvent(self, event) -> None:                              # noqa: N802
        super().showEvent(event)
        self.ensure_started()

    def closeEvent(self, event) -> None:                             # noqa: N802
        try:
            self.stop()
        finally:
            super().closeEvent(event)
