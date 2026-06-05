"""In-app Qt window for the live (image | depth | IMU) triplet.

This is the polished, in-application replacement for the old cv2 ``np.hstack``
triplet dump (:mod:`ours.tools.synced_view`). It reuses the exact design
language the user already approved in :class:`ours.ui.imucam_window.ImuCamWindow`
-- themed :class:`QFrame` panels, the auto-scaling :class:`~ours.ui.imu_panels.GyroPlot`
and the interactive 3D :class:`~ours.ui.imu_panels.Accel3DView` -- and adds a real
**depth** panel between the camera image and the IMU.

Three honest panels, each showing exactly what the capture produces (no parallel
pipeline)::

    [ IMAGE · RECT-LEFT | DEPTH · KHAKI (+scale bar) | GYRO chart / ACCEL 3D ]

Two data sources drive the identical window through an injected worker factory:

* **Live** (:class:`LiveTripletWorker`, default): taps the two RAW OAK-D cameras
  + the IMU, rectifies the left frame and runs our own SGM on the host -- the
  same real building blocks as :func:`ours.tools.synced_view.run_live`. Bench-only
  (no device in CI); the offscreen self-test uses the replay worker instead.
* **Replay** (:class:`ReplayTripletWorker`): replays a recorded session's stored
  depth + IMU with no device -- fully offline, this is what the self-test drives.

cv2 / pyqtgraph / depthai are all imported lazily (only when this window opens),
so the base UI stays lightweight.
"""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from . import theme
from .imu_panels import Accel3DView, GyroPlot

_G = 9.80665
# 0.7x the ImuCamWindow accel zoom (its _VIEW_DIST is 4.6): a larger camera
# distance renders the vector smaller so it fits this wider triplet panel.
_ACCEL_VIEW_DIST = 4.6 / 0.7


# --------------------------------------------------------------------------- #
# Sample + worker model
# --------------------------------------------------------------------------- #
@dataclass
class TripletSample:
    """One synced (image, depth, IMU) unit -- exactly what the capture emits.

    ``gyro_rows`` / ``accel_rows`` are the RAW IMU samples (body frame) that fell
    in this frame's interval -- the same rows :class:`GyroPlot` / :class:`Accel3DView`
    consume in the approved ImuCam window. Either may be empty (no IMU this frame).
    """

    gray_left: np.ndarray          # (H, W) uint8
    depth_m: np.ndarray            # (H, W) float32, metres, 0 == invalid
    gyro_rows: np.ndarray          # (M, 3) rad/s, body frame (may be empty)
    accel_rows: np.ndarray         # (M, 3) m/s^2, body frame (may be empty)
    seq: int
    t_s: float
    frame_label: str = "IMU frame"
    imu_calibrated: bool = False   # True once a per-device calibration applied

    @property
    def imu_n(self) -> int:
        return int(max(np.shape(self.gyro_rows)[0] if np.size(self.gyro_rows)
                       else 0,
                       np.shape(self.accel_rows)[0] if np.size(self.accel_rows)
                       else 0))


class TripletWorker(threading.Thread):
    """Base producer: pushes :class:`TripletSample` (then ``None``) onto a queue.

    Subclasses implement :meth:`_produce`, yielding samples; the base handles the
    queue (drop-newest when full to stay realtime), the stop flag and the END
    sentinel. ``error`` carries the first fatal reason for the UI to surface.
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
        try:
            for sample in self._produce():
                if self._stop.is_set():
                    break
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

    def _produce(self):
        raise NotImplementedError


class ReplayTripletWorker(TripletWorker):
    """Replay a recorded session's stored depth + IMU (no device, fully offline)."""

    mode = "REPLAY"

    def __init__(self, session_dir, fps: float = 20.0,
                 max_frames: int | None = None, calibration=None) -> None:
        super().__init__()
        self._session_dir = session_dir
        self._fps = max(float(fps), 1e-3)
        self._max_frames = max_frames
        self._calib = calibration

    def _produce(self):
        from ..lib import SessionReader, slice_imu

        reader = SessionReader(self._session_dir)
        if len(reader) == 0:
            self.error = f"no frames in {self._session_dir}"
            return
        calib = self._calib
        calibrated = calib is not None and not calib.is_identity
        imu = reader.load_imu()
        ts_i, gyro, accel = imu["ts_ns"], imu["gyro"], imu["accel"]
        frame_ts = [int(r["ts_ns"]) for r in reader._frames]
        period = 1.0 / self._fps
        n = len(reader) if self._max_frames is None \
            else min(len(reader), self._max_frames)
        for i in range(n):
            if self._stop.is_set():
                return
            t0 = time.perf_counter()
            fr = reader.load_frame(i)
            t_prev = frame_ts[0] if i == 0 else frame_ts[i - 1]
            seg = slice_imu(ts_i, gyro, accel, t_prev, frame_ts[i],
                            bracket=False)
            grows = np.asarray(seg.gyro, dtype=np.float64)
            arows = np.asarray(seg.accel, dtype=np.float64)
            if calibrated:
                grows, arows = calib.apply(grows, arows)
            yield TripletSample(
                gray_left=fr.gray_left, depth_m=fr.depth_m,
                gyro_rows=grows, accel_rows=arows,
                seq=fr.seq, t_s=fr.ts_s, frame_label="IMU frame",
                imu_calibrated=calibrated)
            dt = period - (time.perf_counter() - t0)
            if dt > 0:
                self._stop.wait(dt)


class LiveTripletWorker(TripletWorker):
    """Live (image, depth, IMU) from a connected OAK-D -- bench-only.

    Taps the two RAW cameras + IMU, rectifies the left frame and runs our SGM on
    the host (no chip StereoDepth), exactly as :func:`ours.tools.synced_view.run_live`
    does -- so the triplet shown is the real VIO pipeline input. Not exercised in
    CI (needs hardware); confirm on the bench.
    """

    mode = "LIVE"

    def __init__(self, width: int = 640, height: int = 400, fps: int = 20,
                 fast: bool = False) -> None:
        super().__init__()
        self._w, self._h, self._fps, self._fast = width, height, int(fps), fast

    def _produce(self):
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
            imu = p.create(dai.node.IMU)
            imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW,
                                 dai.IMUSensor.GYROSCOPE_RAW], 200)
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)
            left_out = left.requestOutput((self._w, self._h))
            right_out = right.requestOutput((self._w, self._h))
            q_left = left_out.createOutputQueue(maxSize=4, blocking=False)
            q_right = right_out.createOutputQueue(maxSize=4, blocking=False)
            q_imu = imu.out.createOutputQueue(maxSize=50, blocking=False)
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

            # Load this device's cached IMU calibration (gyro bias + accel
            # affine) so the synced view shows CALIBRATED IMU -- the same
            # correction the imucam.sample packet carries. Missing -> raw.
            from ..lib.imu.imu_calib import ImuCalibration
            from ..lib.oak_live import _read_device_id
            imu_calib = ImuCalibration.load(_read_device_id(p))
            imu_calibrated = not imu_calib.is_identity

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

                grows: list = []
                arows: list = []
                msg = q_imu.tryGet()
                while msg is not None:
                    for pkt in msg.packets:
                        a, g = pkt.acceleroMeter, pkt.gyroscope
                        av = [a.x, a.y, a.z]
                        gv = [g.x, g.y, g.z]
                        if np.all(np.isfinite(av)):
                            arows.append(av)
                        if np.all(np.isfinite(gv)):
                            grows.append(gv)
                    msg = q_imu.tryGet()

                common = pend_l.keys() & pend_r.keys()
                if common:
                    seq = max(common)
                    gl, gr = pend_l[seq], pend_r[seq]
                    pend_l = {k: v for k, v in pend_l.items() if k > seq}
                    pend_r = {k: v for k, v in pend_r.items() if k > seq}
                    rect_left, depth = matcher.dense_depth_rectified_left(gl, gr)
                    g_arr = np.asarray(grows, dtype=np.float64)
                    a_arr = np.asarray(arows, dtype=np.float64)
                    if imu_calibrated:
                        g_arr, a_arr = imu_calib.apply(g_arr, a_arr)
                    yield TripletSample(
                        gray_left=np.clip(rect_left, 0, 255).astype(np.uint8),
                        depth_m=depth,
                        gyro_rows=g_arr,
                        accel_rows=a_arr,
                        seq=int(seq), t_s=time.monotonic() - t0,
                        frame_label="IMU frame", imu_calibrated=imu_calibrated)
                if not got:
                    self._stop.wait(0.002)


WorkerFactory = Callable[[], TripletWorker]


def live_worker_factory(width: int = 640, height: int = 400, fps: int = 20,
                        fast: bool = False) -> WorkerFactory:
    return lambda: LiveTripletWorker(width=width, height=height, fps=fps,
                                     fast=fast)


# --------------------------------------------------------------------------- #
# Depth colormap scale-bar widget (static legend -- range is fixed)
# --------------------------------------------------------------------------- #
class _DepthScaleBar(QWidget):
    """Vertical khaki-ramp gradient + fixed tick labels for the depth colormap."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from ..lib.viz.depth_render import D_MAX, D_MIN, depth_scale_bar

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self._bar_bgr = depth_scale_bar(256, 14)          # keep buffer alive
        rgb = np.ascontiguousarray(self._bar_bgr[:, :, ::-1])
        self._buf = rgb
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        grad = QLabel()
        grad.setFixedWidth(14)
        grad.setScaledContents(True)                       # stretch vertically
        grad.setPixmap(QPixmap.fromImage(img))
        row.addWidget(grad)

        ticks = QVBoxLayout()
        ticks.setContentsMargins(0, 0, 0, 0)
        ticks.setSpacing(0)
        mid = 0.5 * (D_MIN + D_MAX)
        for text, stretch in ((f"{D_MIN:.1f} m", 0), (None, 1),
                              (f"{mid:.0f} m", 0), (None, 1),
                              (f"{D_MAX:.1f} m", 0)):
            if text is None:
                ticks.addStretch(1)
                continue
            lab = QLabel(text)
            lab.setObjectName("ScaleTick")
            ticks.addWidget(lab)
        row.addLayout(ticks)


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #
def _panel(title: str) -> tuple[QWidget, QVBoxLayout, QLabel]:
    """A themed ``QFrame#Panel`` with a ``PanelTitle`` header; returns
    ``(frame, body_layout, title_label)`` so callers fill the body."""
    from PyQt6.QtWidgets import QFrame

    frame = QFrame()
    frame.setObjectName("Panel")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(6, 6, 6, 6)
    lay.setSpacing(4)
    title_lab = QLabel(title)
    title_lab.setObjectName("PanelTitle")
    lay.addWidget(title_lab)
    return frame, lay, title_lab


def _raster_label() -> QLabel:
    lab = QLabel("…")
    lab.setObjectName("ImuCamView")
    lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    lab.setMinimumSize(160, 120)
    # Ignored size policy breaks the pixmap->label size feedback loop (see the
    # long note in imucam_window.py): the splitter fixes the size, the pixmap
    # just fits inside it.
    lab.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
    return lab


class SyncedViewWindow(QWidget):
    """Polished in-app live (image | depth | IMU) triplet view."""

    def __init__(self, worker_factory: WorkerFactory | None = None, *,
                 fps: int = 20, parent: QWidget | None = None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self._make_worker = worker_factory or live_worker_factory(fps=fps)

        self.setWindowTitle("Synced view — image · depth · IMU (live)")
        self.setObjectName("SyncedViewWindow")
        self.resize(1280, 820)
        self.setMinimumSize(900, 620)
        self.setStyleSheet(theme.QSS)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(self._build_header())

        # Layout mirrors ImuCamWindow: cameras on top (image | depth), the IMU
        # panels below spanning the full width (gyro chart | 3D accel vector).
        body = QSplitter(QtCore.Qt.Orientation.Vertical)
        body.setChildrenCollapsible(False)
        body.setHandleWidth(6)

        cams = QSplitter(QtCore.Qt.Orientation.Horizontal)
        cams.setChildrenCollapsible(False)
        cams.setHandleWidth(6)
        img_panel, img_lay, _ = _panel("IMAGE · RECT-LEFT")
        self._image = _raster_label()
        img_lay.addWidget(self._image, stretch=1)
        cams.addWidget(img_panel)
        cams.addWidget(self._build_depth_panel())
        cams.setSizes([1, 1])

        body.addWidget(cams)
        body.addWidget(self._build_imu_panel())
        body.setSizes([440, 360])
        root.addWidget(body, stretch=1)

        self._status = QLabel("—")
        self._status.setObjectName("ImuCamStatus")
        root.addWidget(self._status, stretch=0)

        self._worker: TripletWorker | None = None
        self._buf_img: np.ndarray | None = None
        self._buf_depth: np.ndarray | None = None
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
        title = QLabel("SYNCED VIEW")
        title.setObjectName("HeaderTitle")
        sub = QLabel("IMAGE · DEPTH · IMU")
        sub.setObjectName("HeaderSub")
        h.addWidget(title)
        h.addWidget(sub)
        h.addStretch(1)
        self._mode_pill = QLabel("—")
        self._mode_pill.setObjectName("FieldValue")
        h.addWidget(self._mode_pill)
        return bar

    def _build_depth_panel(self) -> QWidget:
        panel, lay, title = self._depth_title_row()
        rasters = QHBoxLayout()
        rasters.setContentsMargins(0, 0, 0, 0)
        rasters.setSpacing(4)
        self._depth = _raster_label()
        rasters.addWidget(self._depth, stretch=1)
        rasters.addWidget(_DepthScaleBar(), stretch=0)
        lay.addLayout(rasters, stretch=1)
        hint = QLabel("black = no stereo return")
        hint.setObjectName("ScaleTick")
        lay.addWidget(hint, stretch=0)
        return panel

    def _depth_title_row(self):
        from PyQt6.QtWidgets import QFrame

        panel = QFrame()
        panel.setObjectName("Panel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("DEPTH · KHAKI")
        title.setObjectName("PanelTitle")
        row.addWidget(title)
        row.addStretch(1)
        self._valid = QLabel("valid —")
        self._valid.setObjectName("FieldValue")
        row.addWidget(self._valid)
        lay.addLayout(row)
        return panel, lay, title

    def _build_imu_panel(self) -> QWidget:
        from PyQt6.QtWidgets import QFrame

        panel = QFrame()
        panel.setObjectName("Panel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        # Title carries the honest calibration state (CALIBRATED vs RAW), set
        # per-frame from the sample so the operator always knows which they see.
        self._imu_title = QLabel("IMU · RAW")
        self._imu_title.setObjectName("PanelTitle")
        lay.addWidget(self._imu_title)

        # Gyro chart | 3D accel vector, side by side and spanning the full
        # width (same arrangement as ImuCamWindow's bottom row).
        row = QSplitter(QtCore.Qt.Orientation.Horizontal)
        row.setChildrenCollapsible(False)
        row.setHandleWidth(6)
        self._gyro = GyroPlot()
        row.addWidget(self._gyro)

        accel_box = QWidget()
        av = QVBoxLayout(accel_box)
        av.setContentsMargins(0, 0, 0, 0)
        av.setSpacing(2)
        # 0.7x the ImuCamWindow zoom so the whole vector fits in this wider
        # panel; passed per-instance so the ImuCamWindow accel is unchanged.
        self._accel = Accel3DView(view_dist=_ACCEL_VIEW_DIST)
        av.addWidget(self._accel, stretch=1)
        self._imu_readout = QLabel("tilt — · |a| —")
        self._imu_readout.setObjectName("ImuCamStatus")
        av.addWidget(self._imu_readout, stretch=0)
        row.addWidget(accel_box)
        # Equal halves: large equal sizes force a stable 50/50 split regardless
        # of the two children's differing size hints (the GL view hints larger).
        row.setSizes([10_000, 10_000])
        row.setStretchFactor(0, 1)
        row.setStretchFactor(1, 1)
        lay.addWidget(row, stretch=1)
        return panel

    # -- lifecycle -------------------------------------------------------- #
    def ensure_started(self) -> None:
        if self._running and not self._failed:
            return
        self._teardown()
        self.start()

    def start(self) -> None:
        if self._running:
            return
        self._gyro.clear_history()
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
        self._image.setStyleSheet("")
        self._image.setText("starting…  (opening the OAK-D / compiling SGM)")
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
        self._show_gray(self._image, sample.gray_left)
        self._show_depth(sample.depth_m)
        if sample.imu_n > 0:
            self._gyro.add(sample.gyro_rows)
            self._accel.set_accel(sample.accel_rows)
        self._update_imu_title(sample)
        self._update_imu_readout(sample)
        self._update_footer(sample)

    def _update_imu_title(self, s: TripletSample) -> None:
        # Match the global PanelTitle selector so this color override wins over
        # the app QSS (equal specificity, but the widget's own sheet is local).
        if s.imu_calibrated:
            self._imu_title.setText("IMU · CALIBRATED")
            self._imu_title.setStyleSheet(
                f"QLabel#PanelTitle {{ color: {theme.GOOD}; }}")
        else:
            self._imu_title.setText("IMU · RAW")
            self._imu_title.setStyleSheet(
                f"QLabel#PanelTitle {{ color: {theme.ACCENT}; }}")

    def _update_imu_readout(self, s: TripletSample) -> None:
        if s.imu_n == 0:
            self._imu_readout.setText("— no IMU this frame —")
            self._imu_readout.setStyleSheet(f"color: {theme.WARN};")
            return
        a = np.asarray(s.accel_rows, dtype=np.float64).mean(axis=0)
        mag = float(np.linalg.norm(a))
        if mag > 1e-6:
            tilt = float(np.degrees(np.arccos(
                np.clip(-a[1] / mag, -1.0, 1.0))))
            tilt_txt = f"{tilt:4.1f}°"
        else:
            tilt_txt = "—"
        self._imu_readout.setText(
            f"tilt {tilt_txt} (derived) · |a| {mag:5.2f} m/s²")
        self._imu_readout.setStyleSheet(f"color: {self._mag_color(mag)};")

    def _update_footer(self, s: TripletSample) -> None:
        imu_col = theme.WARN if s.imu_n == 0 else theme.TEXT_DIM
        self._status.setText(
            f"SEQ {s.seq}  ·  t {s.t_s:5.1f}s  ·  {s.frame_label}  ·  "
            f"imu_n {s.imu_n}")
        self._status.setStyleSheet(f"color: {imu_col};")

    @staticmethod
    def _mag_color(mag: float) -> str:
        d = abs(mag - _G)
        if d <= 0.5:
            return theme.GOOD
        if d <= 1.5:
            return theme.WARN
        return theme.BAD

    @staticmethod
    def _valid_color(pct: float) -> str:
        if pct >= 60.0:
            return theme.GOOD
        if pct >= 30.0:
            return theme.WARN
        return theme.BAD

    def _show_depth(self, depth_m: np.ndarray) -> None:
        from ..lib.viz.depth_render import colorize_depth

        pct = float((depth_m > 1e-6).mean()) * 100.0
        self._valid.setText(f"valid {pct:.0f}%")
        self._valid.setStyleSheet(
            f"color: {self._valid_color(pct)}; font-weight: bold;")
        bgr = colorize_depth(depth_m)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        self._buf_depth = rgb
        self._blit(self._depth, QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                                       3 * rgb.shape[1],
                                       QImage.Format.Format_RGB888))

    def _show_gray(self, label: QLabel, gray: np.ndarray) -> None:
        g = np.ascontiguousarray(gray)
        self._buf_img = g
        self._blit(label, QImage(g.data, g.shape[1], g.shape[0], g.shape[1],
                                 QImage.Format.Format_Grayscale8))

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
        self._image.setStyleSheet(
            f"color: {theme.WARN}; font-size: 15px; font-weight: bold;")
        self._image.setText(f"⚠  {message}")
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
