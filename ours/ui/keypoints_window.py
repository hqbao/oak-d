"""In-app Qt window: camera frame + detected keypoints, coloured by depth.

The headline view for inspecting OUR visual frontend. Each frame shows the
rectified-left image with every live :class:`~ours.lib.frontend.frontend.KLTFrontend`
track drawn on it:

* the dot **colour** = that keypoint's metric depth (the SAME fixed TURBO
  0.3-8.0 m map + scale-bar legend as the depth panel), so colour means the same
  distance everywhere; keypoints with no stereo return are hollow grey rings, not
  a faked colour;
* a faint **trail** per track id shows where the *same* keypoint moved over the
  last ``TRAIL_LEN`` (20) frames -- the persistent id is what links the dots.

Honest-data: the keypoints + ids are the real frontend output (the same code the
odometry runs); the trail is the UI buffering each id's recent positions. Nothing
is invented. See :mod:`ours.lib.viz.keypoint_overlay`.

Two data sources drive the identical window through an injected worker factory:

* **Live** (:class:`LiveKeypointWorker`, default): taps the two RAW OAK-D cameras,
  rectifies the left frame, runs our SGM for depth and our KLT frontend for the
  tracks -- bench-only (no device in CI).
* **Replay** (:class:`ReplayKeypointWorker`): runs the frontend over a recorded
  session's stored frames + depth, fully offline -- this is what the self-test
  drives.

cv2 / depthai are imported lazily (only when this window opens).
"""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

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
    """Base producer: pushes :class:`KeypointSample` (then ``None``) onto a queue.

    Owns the persistent :class:`KLTFrontend` + :class:`TrackTrails`, advances both
    once per produced frame (so tracking + trails stay continuous), renders the
    overlay and ships the finished sample. Subclasses implement :meth:`_frames`,
    yielding ``(seq, t_s, gray_left, depth_m)``.
    """

    mode = "—"

    def __init__(self, maxsize: int = 4) -> None:
        super().__init__(daemon=True)
        self.queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self.error: str | None = None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:                                          # noqa: D401
        from ..lib.flow.runtime import NUMBA_PARALLEL_LOCK
        from ..lib.frontend.frontend import FrontendConfig, KLTFrontend
        from ..lib.viz.keypoint_overlay import (
            TrackTrails, draw_overlay, sample_depths,
        )

        frontend = KLTFrontend(FrontendConfig())
        trails = TrackTrails()
        try:
            for seq, t_s, gray, depth_m in self._frames():
                if self._stop.is_set():
                    break
                # KLT uses numba parallel=True, which is not threadsafe across
                # Python threads -- serialize it like the odometry flow's
                # ProcessVO does so concurrent frontends (e.g. two windows, or a
                # window + the VIO source) can't abort the numba runtime.
                with NUMBA_PARALLEL_LOCK:
                    state = frontend.process(gray)
                ids = np.asarray(state.ids, dtype=np.int64).reshape(-1)
                pts = np.asarray(state.points, dtype=np.float32).reshape(-1, 2)
                trails.update(ids, pts)
                depths = sample_depths(depth_m, pts)
                rgb = draw_overlay(gray, depth_m, ids, pts, trails)
                sample = KeypointSample(
                    rgb=rgb, ids=ids, points=pts, depths=depths,
                    seq=int(seq), t_s=float(t_s),
                    n_tracks=int(ids.shape[0]),
                    n_valid=int(np.count_nonzero(depths > 1e-6)),
                    mean_age=trails.mean_age(), new_count=trails.new_count)
                try:
                    self.queue.put_nowait(sample)
                except queue.Full:
                    pass                       # drop to stay realtime
        except Exception as exc:               # surface, don't crash the UI
            self.error = str(exc)
        finally:
            try:
                self.queue.put_nowait(None)    # END sentinel
            except queue.Full:
                pass

    def _frames(self):
        raise NotImplementedError


class ReplayKeypointWorker(KeypointWorker):
    """Run the frontend over a recorded session's stored frames (fully offline)."""

    mode = "REPLAY"

    def __init__(self, session_dir, fps: float = 20.0,
                 max_frames: int | None = None) -> None:
        super().__init__()
        self._session_dir = session_dir
        self._fps = max(float(fps), 1e-3)
        self._max_frames = max_frames

    def _frames(self):
        from ..lib import SessionReader

        reader = SessionReader(self._session_dir)
        if len(reader) == 0:
            self.error = f"no frames in {self._session_dir}"
            return
        period = 1.0 / self._fps
        n = len(reader) if self._max_frames is None \
            else min(len(reader), self._max_frames)
        for i in range(n):
            if self._stop.is_set():
                return
            t0 = time.perf_counter()
            fr = reader.load_frame(i)
            yield fr.seq, fr.ts_s, fr.gray_left, fr.depth_m
            dt = period - (time.perf_counter() - t0)
            if dt > 0:
                self._stop.wait(dt)


class LiveKeypointWorker(KeypointWorker):
    """Live frontend over a connected OAK-D -- bench-only.

    Taps the two RAW cameras, rectifies the left frame and runs our SGM on the
    host for depth (mirrors :class:`ours.ui.synced_window.LiveTripletWorker`'s
    camera + SGM setup, minus the IMU -- this view needs only image + depth). Not
    exercised in CI (needs hardware); confirm on the bench.
    """

    mode = "LIVE"

    def __init__(self, width: int = 640, height: int = 400, fps: int = 20,
                 fast: bool = False) -> None:
        super().__init__()
        self._w, self._h, self._fps, self._fast = width, height, int(fps), fast

    def _frames(self):
        import cv2
        import depthai as dai

        from ..lib import SGMStereoMatcher, StereoCalib
        from ..lib.config.resolution import ResolutionProfile

        left_socket = dai.CameraBoardSocket.CAM_B
        right_socket = dai.CameraBoardSocket.CAM_C
        res = ResolutionProfile.for_resolution(self._w, self._h)
        cfg = res.sgm(fast=self._fast)

        with dai.Pipeline() as p:
            left = p.create(dai.node.Camera).build(left_socket,
                                                   sensorFps=self._fps)
            right = p.create(dai.node.Camera).build(right_socket,
                                                    sensorFps=self._fps)
            left_out = left.requestOutput((self._w, self._h))
            right_out = right.requestOutput((self._w, self._h))
            q_left = left_out.createOutputQueue(maxSize=4, blocking=False)
            q_right = right_out.createOutputQueue(maxSize=4, blocking=False)
            p.start()

            ch = p.getDefaultDevice().readCalibration()

            def _intr(sock):
                Ki = np.array(ch.getCameraIntrinsics(sock, self._w, self._h),
                              dtype=np.float64)
                dist = list(ch.getDistortionCoefficients(sock))
                return {"fx": float(Ki[0, 0]), "fy": float(Ki[1, 1]),
                        "cx": float(Ki[0, 2]), "cy": float(Ki[1, 2]),
                        "dist": [float(x) for x in dist],
                        "width": int(self._w), "height": int(self._h)}

            T_lr = np.array(ch.getCameraExtrinsics(left_socket, right_socket),
                            dtype=np.float64).reshape(4, 4)
            calib = StereoCalib.from_json({
                "intrinsics_left": _intr(left_socket),
                "intrinsics_right": _intr(right_socket),
                "T_left_right": T_lr.tolist(),
            })
            matcher = SGMStereoMatcher.from_calib(calib, cfg, rectify_left=True)
            dummy = np.zeros((self._h, self._w), np.uint8)
            matcher.dense_depth(dummy, dummy)          # one-time JIT warmup

            def _as_gray(msg):
                g = msg.getCvFrame()
                if g.ndim == 3:
                    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
                return g

            pend_l: dict[int, np.ndarray] = {}
            pend_r: dict[int, np.ndarray] = {}
            t0 = time.monotonic()
            while p.isRunning() and not self._stop.is_set():
                got = False
                while True:
                    m = q_left.tryGet()
                    if m is None:
                        break
                    pend_l[m.getSequenceNum()] = _as_gray(m)
                    got = True
                while True:
                    m = q_right.tryGet()
                    if m is None:
                        break
                    pend_r[m.getSequenceNum()] = _as_gray(m)
                    got = True

                common = pend_l.keys() & pend_r.keys()
                if common:
                    seq = max(common)
                    gl, gr = pend_l[seq], pend_r[seq]
                    pend_l = {k: v for k, v in pend_l.items() if k > seq}
                    pend_r = {k: v for k, v in pend_r.items() if k > seq}
                    rect_left, depth = matcher.dense_depth_rectified_left(gl, gr)
                    yield (int(seq), time.monotonic() - t0,
                           np.clip(rect_left, 0, 255).astype(np.uint8), depth)
                if not got:
                    self._stop.wait(0.002)


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

        panel, lay, _ = _panel("FRAME · KLT TRACKS (TURBO depth)")
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
