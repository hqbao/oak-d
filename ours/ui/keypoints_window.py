"""In-app Qt window: camera frame + detected keypoints, coloured by depth.

The headline view for inspecting OUR visual frontend. Each frame shows the
rectified-left image with every live
:class:`~ours.lib.frontend.frontend.KLTFrontend` track drawn on it:

* the dot **colour** = that keypoint's metric depth (the SAME fixed khaki
  0.3-8.0 m map + scale-bar legend as the depth panel), so colour means the same
  distance everywhere; keypoints with no stereo return are hollow grey rings, not
  a faked colour;
* a faint **trail** per track id shows where the *same* keypoint moved over the
  last ``TRAIL_LEN`` (20) frames -- the persistent id is what links the dots.

Honest-data: the keypoints + ids are the REAL frontend output **subscribed from
the running pipeline** -- the odometry flow's
:class:`~ours.flows.odometry.publish_tracks.PublishTracks` publishes the very
``{id: pixel}`` its motion estimate consumes on ``frame.tracks``, and this window
is just a sink for it (no parallel detector, no second frontend). The trail is the
UI buffering each id's recent positions. Nothing is invented. See
:mod:`ours.lib.viz.keypoint_overlay`.

The window runs the same flow graph the VIO runs (``ours.app.build_live`` /
``build_replay``) with a :class:`~ours.flows.ui.tracks.UiTracksFlow` sink, exactly
as :class:`~ours.ui.live_source.FlowPoseSource` drives the 3D viewer off
``pose.odom``. Two data sources drive the identical window through an injected
worker factory:

* **Live** (:class:`LiveKeypointWorker`, default): wires the live OAK-D graph and
  taps ``frame.tracks`` -- bench-only (no device in CI).
* **Replay** (:class:`ReplayKeypointWorker`): wires the recorded-session graph and
  taps ``frame.tracks``, fully offline -- this is what the self-test drives.

cv2 / depthai are imported lazily (only when this window opens).
"""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from . import theme
from .synced_window import _DepthScaleBar, _panel, _raster_label


# --------------------------------------------------------------------------- #
# Sample + worker model
# --------------------------------------------------------------------------- #
@dataclass
class KeypointSample:
    """One frame's finished overlay + the honest stats behind it.

    The producer renders the overlay (so the trails stay continuous regardless of
    UI frame drops); the UI just blits ``rgb`` and prints the stats. The raw
    ``ids`` / ``points`` / ``depths`` ride along for tests + future hover-inspect.
    """

    rgb: np.ndarray                # (H, W, 3) uint8, finished overlay
    ids: np.ndarray                # (N,) int64 persistent track ids
    points: np.ndarray             # (N, 2) float32 pixel coords
    depths: np.ndarray             # (N,) float64 metres, 0 == invalid
    seq: int
    t_s: float
    n_tracks: int
    n_valid: int
    mean_age: float
    new_count: int

    @property
    def valid_pct(self) -> float:
        return 100.0 * self.n_valid / self.n_tracks if self.n_tracks else 0.0


class KeypointWorker(threading.Thread):
    """Base producer: subscribe ``frame.tracks``, render, queue samples.

    Runs the flow graph (built by the subclass' :meth:`_drive`) with a
    :class:`~ours.flows.ui.tracks.UiTracksFlow` sink. For each subscribed
    :class:`~ours.lib.flow.messages.FrameTracks` it advances the per-id
    :class:`~ours.lib.viz.keypoint_overlay.TrackTrails`, renders the overlay and
    ships a finished :class:`KeypointSample` (then ``None``). It runs NO frontend
    itself -- the tracks come straight off the pipeline -- so it needs no numba
    parallel lock.
    """

    mode = "—"
    #: live realtime view builds a latest-only graph (bounded latency); replay
    #: keeps the full-fidelity FIFO graph (offline, deterministic, every frame).
    latest_only = False

    def __init__(self, maxsize: int = 4) -> None:
        super().__init__(daemon=True)
        self.queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self.error: str | None = None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:                                          # noqa: D401
        from ..flows.ui.tracks import UiTracksFlow
        from ..lib.flow.pubsub import Bus
        from ..lib.viz.keypoint_overlay import (
            TrackTrails, draw_overlay, sample_depths,
        )

        trails = TrackTrails()
        t0_ns: list[int | None] = [None]

        def on_tracks(msg) -> None:
            ids = np.asarray(msg.ids, dtype=np.int64).reshape(-1)
            pts = np.asarray(msg.points, dtype=np.float32).reshape(-1, 2)
            # Always advance the trails (cheap dict ops) so per-id history stays
            # continuous even on frames we don't render.
            trails.update(ids, pts)
            if t0_ns[0] is None:
                t0_ns[0] = msg.ts_ns
            # If the UI hasn't drained the last overlays, skip the expensive
            # draw_overlay for this frame: rendering a frame the UI will only drop
            # would back the bus inbox up and grow latency without showing more.
            if self.queue.full():
                return
            depths = sample_depths(msg.depth_m, pts)
            rgb = draw_overlay(msg.gray_left, msg.depth_m, ids, pts, trails)
            sample = KeypointSample(
                rgb=rgb, ids=ids, points=pts, depths=depths,
                seq=int(msg.seq), t_s=(msg.ts_ns - t0_ns[0]) * 1e-9,
                n_tracks=int(ids.shape[0]),
                n_valid=int(np.count_nonzero(depths > 1e-6)),
                mean_age=trails.mean_age(), new_count=trails.new_count)
            try:
                self.queue.put_nowait(sample)
            except queue.Full:
                pass                       # drop to stay realtime

        bus = Bus()
        try:
            self._drive(bus, UiTracksFlow(bus, on_tracks,
                                          latest_only=self.latest_only))
        except Exception as exc:           # surface, don't crash the UI
            self.error = str(exc)
        finally:
            try:
                self.queue.put_nowait(None)    # END sentinel
            except queue.Full:
                pass

    def _drive(self, bus, sink) -> None:
        """Build + run the flow graph feeding ``sink`` until stopped/drained."""
        raise NotImplementedError


class ReplayKeypointWorker(KeypointWorker):
    """Drive the recorded-session graph and tap ``frame.tracks`` (fully offline)."""

    mode = "REPLAY"

    def __init__(self, session_dir, fps: float = 20.0,
                 max_frames: int | None = None) -> None:
        super().__init__()
        self._session_dir = session_dir
        self._max_frames = max_frames
        # ``fps`` is accepted for call-site compatibility; the replay graph drives
        # the cam flow itself (full speed), so there is no UI-side throttle here.

    def _drive(self, bus, sink) -> None:
        from ..app import build_replay
        from ..lib.io.reader import SessionReader

        reader = SessionReader(Path(self._session_dir))
        if len(reader) == 0:
            self.error = f"no frames in {self._session_dir}"
            return
        (cam_flow, imu_flow), flows, _ = build_replay(
            bus, reader, depth_fast=True,
            max_frames=int(self._max_frames or 0), ui=sink,
            with_backend_slam=False)
        for f in flows:
            f.start()
        imu_flow.start()
        cam_flow.start()
        try:
            # Let the camera produce every frame (or stop early on request).
            while not self._stop.is_set() and cam_flow.is_alive():
                self._stop.wait(0.02)
            # Drain the whole graph BEFORE tearing flows down, so every frame's
            # tracks reach the sink (mirrors ours.app.run_replay's ordering).
            if not self._stop.is_set():
                sink.done.wait(timeout=10.0)
        finally:
            cam_flow.stop()
            imu_flow.stop()
            for f in flows:
                f.stop()


class LiveKeypointWorker(KeypointWorker):
    """Drive the live OAK-D graph and tap ``frame.tracks`` -- bench-only.

    Wires the SAME live acquisition + odometry front-end the VIO runs
    (:func:`ours.app.build_live`) off the one shared device and subscribes
    ``frame.tracks``, so the keypoints shown are exactly what the running odometry
    frontend tracked. It builds the graph WITHOUT the back-end/SLAM flows
    (``with_backend_slam=False``) -- those don't affect the tracks and would
    otherwise compete for CPU and make the live view fall seconds behind. Not
    exercised in CI (needs hardware); confirm on the bench.
    """

    mode = "LIVE"
    latest_only = True

    def __init__(self, width: int = 640, height: int = 400, fps: int = 20,
                 fast: bool = False) -> None:
        super().__init__()
        self._w, self._h, self._fps, self._fast = width, height, int(fps), fast

    def _drive(self, bus, sink) -> None:
        from ..app import build_live

        device, (cam_flow, imu_flow), flows, _ = build_live(
            bus, width=self._w, height=self._h, fps=self._fps,
            depth_fast=self._fast, ui=sink, with_backend_slam=False,
            realtime_latest=True)
        for f in flows:
            f.start()
        imu_flow.start()
        cam_flow.start()
        try:
            while not self._stop.is_set() and cam_flow.is_alive():
                self._stop.wait(0.05)
        finally:
            cam_flow.stop()
            imu_flow.stop()
            sink.done.wait(timeout=5.0)
            for f in flows:
                f.stop()
            device.release()



WorkerFactory = Callable[[], KeypointWorker]


def live_worker_factory(width: int = 640, height: int = 400, fps: int = 20,
                        fast: bool = False) -> WorkerFactory:
    return lambda: LiveKeypointWorker(width=width, height=height, fps=fps,
                                      fast=fast)


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #
class KeypointTrackWindow(QWidget):
    """Polished in-app frame + depth-coloured keypoints + trails view."""

    def __init__(self, worker_factory: WorkerFactory | None = None, *,
                 fps: int = 20, parent: QWidget | None = None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self._make_worker = worker_factory or live_worker_factory(fps=fps)

        self.setWindowTitle("Keypoint depth tracker (live)")
        self.setObjectName("KeypointTrackWindow")
        self.resize(1100, 820)
        self.setMinimumSize(720, 560)
        self.setStyleSheet(theme.QSS)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(self._build_header())

        panel, lay, _ = _panel("FRAME · KLT TRACKS (depth shaded)")
        rasters = QHBoxLayout()
        rasters.setContentsMargins(0, 0, 0, 0)
        rasters.setSpacing(4)
        self._view = _raster_label()
        rasters.addWidget(self._view, stretch=1)
        rasters.addWidget(_DepthScaleBar(), stretch=0)
        lay.addLayout(rasters, stretch=1)
        hint = QLabel("colour = depth · hollow grey = no stereo · amber = fresh "
                      "track · trail = last 20 frames")
        hint.setObjectName("ScaleTick")
        lay.addWidget(hint, stretch=0)
        root.addWidget(panel, stretch=1)

        self._status = QLabel("—")
        self._status.setObjectName("ImuCamStatus")
        root.addWidget(self._status, stretch=0)

        self._worker: KeypointWorker | None = None
        self._buf: np.ndarray | None = None
        self._running = False
        self._ended = False
        self._first_seen = False
        self._failed = False
        self._t_start = 0.0
        self._startup_timeout_s = 18.0       # SGM JIT warmup can be slow

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._on_tick)

    # -- construction helpers --------------------------------------------- #
    def _build_header(self) -> QWidget:
        from PyQt6.QtWidgets import QFrame

        bar = QFrame()
        bar.setObjectName("Panel")
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(8)
        title = QLabel("KEYPOINT DEPTH TRACKER")
        title.setObjectName("HeaderTitle")
        sub = QLabel("rect-left · OUR KLT frontend")
        sub.setObjectName("HeaderSub")
        h.addWidget(title)
        h.addWidget(sub)
        h.addStretch(1)
        self._mode_pill = QLabel("—")
        self._mode_pill.setObjectName("FieldValue")
        h.addWidget(self._mode_pill)
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
        self._worker = self._make_worker()
        self._mode_pill.setText(self._worker.mode)
        self._mode_pill.setStyleSheet(
            f"color: {theme.GOOD if self._worker.mode == 'LIVE' else theme.TEXT_DIM};")
        self._worker.start()
        self._running = True
        self._ended = False
        self._first_seen = False
        self._failed = False
        self._t_start = time.monotonic()
        self._view.setStyleSheet("")
        self._view.setText("starting…  (opening the OAK-D / compiling SGM)")
        self._set_status("connecting…", theme.TEXT_DIM)
        self._timer.start()

    def stop(self) -> None:
        self._teardown()

    def _teardown(self) -> None:
        self._timer.stop()
        if self._worker is not None:
            try:
                self._worker.stop()
            except Exception:
                pass
        self._worker = None
        self._running = False

    # -- per-frame update ------------------------------------------------- #
    def _on_tick(self) -> None:
        sample = self._drain_latest()
        if sample is None:
            self._maybe_report_no_frame()
            return
        self._first_seen = True
        self._show_rgb(sample.rgb)
        self._update_footer(sample)

    def _update_footer(self, s: KeypointSample) -> None:
        pct = s.valid_pct
        if pct >= 80.0:
            col = theme.GOOD
        elif pct >= 50.0:
            col = theme.WARN
        else:
            col = theme.BAD
        self._status.setText(
            f"trk {s.n_tracks}  ·  valid-z {s.n_valid} ({pct:.0f}%)  ·  "
            f"mean-age {s.mean_age:.1f} f  ·  new {s.new_count}  ·  "
            f"SEQ {s.seq}  ·  t {s.t_s:5.1f}s")
        self._status.setStyleSheet(f"color: {col};")

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

    # -- failure / end handling ------------------------------------------- #
    def _maybe_report_no_frame(self) -> None:
        if self._first_seen:
            if self._ended:
                self._set_status("stream ended", theme.WARN)
                self._timer.stop()
            return
        err = self._worker.error if self._worker is not None else None
        dead = self._worker is not None and not self._worker.is_alive()
        timed_out = (time.monotonic() - self._t_start) > self._startup_timeout_s
        if err or self._ended or dead or timed_out:
            self._fail(err or
                       ("no frames — is the OAK-D connected and free? "
                        "(nothing else may hold the device)"))

    def _fail(self, message: str) -> None:
        if self._failed:
            return
        self._failed = True
        self._view.setStyleSheet(
            f"color: {theme.WARN}; font-size: 15px; font-weight: bold;")
        self._view.setText(f"⚠  {message}")
        self._set_status("not streaming — reopen from the Visualize menu to retry",
                         theme.BAD)
        self._teardown()

    def _drain_latest(self):
        latest = None
        if self._worker is None:
            return None
        while True:
            try:
                item = self._worker.queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._ended = True
                break
            latest = item
        return latest

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
