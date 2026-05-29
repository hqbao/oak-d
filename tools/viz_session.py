#!/usr/bin/env python3
"""Visualize a recorded session — one tab per pipeline checkpoint.

Usage::

    ./tools/viz_session.py sessions/2026-05-29_loop1
    ./tools/viz_session.py /tmp/oakd_rec_smoke

Tabs
----
- Overview   : meta.json + calib summary + record counts
- C0 Frame   : rectified-left, rectified-right, depth colormap
- C1 IMU     : 6-channel gyro + accel time series
- C2/C3 Pose : 3D trajectory (VIO vs SLAM overlay) + pos/quat timeseries
- C5/C6 Evts : loop closures (odom_correction jumps) + tracking-lost gaps

Playback bar at the bottom drives all tabs from a shared timeline
(seconds since the earliest sample across every stream).
Hotkeys: SPACE play/pause, LEFT/RIGHT step ±0.1s, HOME jump to start.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtCore import Qt, QTimer, pyqtSignal              # noqa: E402
from PyQt6.QtGui import (                                    # noqa: E402
    QImage, QKeySequence, QPixmap, QShortcut,
)
from PyQt6.QtWidgets import (                                # noqa: E402
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSlider, QSplitter, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg                                       # noqa: E402
import pyqtgraph.opengl as gl                                # noqa: E402

from oakd.ui import theme                                    # noqa: E402


# ============================================================================
# Loaders
# ============================================================================

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _stack_poses(records: list[dict]) -> dict[str, np.ndarray]:
    if not records:
        return {
            "ts_s": np.zeros(0),
            "pos": np.zeros((0, 3)),
            "quat": np.zeros((0, 4)),
            "tracking_ok": np.zeros(0, dtype=bool),
        }
    ts = np.array([r["ts_ns"] for r in records], dtype=np.float64) / 1e9
    pos = np.array([r["pos"] for r in records], dtype=np.float64)
    quat = np.array([r["quat_wxyz"] for r in records], dtype=np.float64)
    ok = np.array([r.get("tracking_ok", True) for r in records], dtype=bool)
    return {"ts_s": ts, "pos": pos, "quat": quat, "tracking_ok": ok}


def _stack_imu(records: list[dict]) -> dict[str, np.ndarray]:
    if not records:
        return {"ts_s": np.zeros(0), "gyro": np.zeros((0, 3)),
                "accel": np.zeros((0, 3))}
    ts = np.array([r["ts_ns"] for r in records], dtype=np.float64) / 1e9
    gyro = np.array([r["gyro"] for r in records], dtype=np.float64)
    accel = np.array([r["accel"] for r in records], dtype=np.float64)
    return {"ts_s": ts, "gyro": gyro, "accel": accel}


def _frame_ts_array(frames: list[dict]) -> np.ndarray:
    return (np.array([r["ts_ns"] for r in frames], dtype=np.float64) / 1e9
            if frames else np.zeros(0))


def _load_features(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Returns (ts_s array, list of Nx2 px arrays — one per frame)."""
    recs = _load_jsonl(path)
    if not recs:
        return np.zeros(0), []
    ts = np.array([r["ts_ns"] for r in recs], dtype=np.float64) / 1e9
    pts = [np.array([[p[0], p[1]] for p in r["pts"]], dtype=np.float32)
           if r["n"] else np.zeros((0, 2), dtype=np.float32) for r in recs]
    return ts, pts


def _load_pointcloud_stream(
    session_dir: Path,
) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    """Load every point cloud emission as a separate frame.

    Returns ``(ts_s, kinds, clouds)`` parallel arrays. RTABMap republishes
    the FULL cumulative cloud per emission, so viewers should pick the
    latest emission per kind that has ``ts_s <= playhead_t`` and union the
    two kinds — NOT concatenate every emission (that would 30x-duplicate).
    """
    idx_path = session_dir / "basalt" / "pointcloud.jsonl"
    recs = _load_jsonl(idx_path)
    if not recs:
        return np.zeros(0), [], []
    ts = np.array([r["ts_ns"] for r in recs], dtype=np.float64) / 1e9
    kinds: list[str] = []
    clouds: list[np.ndarray] = []
    for r in recs:
        kinds.append(r.get("kind", "obstacle"))
        p = session_dir / "basalt" / r["path"]
        if p.exists():
            clouds.append(np.fromfile(p, dtype=np.float32).reshape(-1, 3))
        else:
            clouds.append(np.zeros((0, 3), dtype=np.float32))
    return ts, kinds, clouds


# ============================================================================
# Playhead (shared timeline)
# ============================================================================

class Playhead(QWidget):
    """Master clock. Emits ``timeChanged(t_s)`` at ~30 Hz when playing."""

    timeChanged = pyqtSignal(float)

    def __init__(self, t_start: float, t_end: float, parent=None) -> None:
        super().__init__(parent)
        self.t_start = float(t_start)
        self.t_end = float(max(t_end, t_start + 1e-3))
        self.t = self.t_start
        self.speed = 1.0
        self._playing = False
        self._tick_dt_ms = 33     # ~30 Hz UI tick

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(self._tick_dt_ms)
        self._timer.timeout.connect(self._on_tick)

    def play(self) -> None:
        if self._playing:
            return
        if self.t >= self.t_end - 1e-3:
            self.t = self.t_start
        self._playing = True
        self._timer.start()

    def pause(self) -> None:
        self._playing = False
        self._timer.stop()

    def toggle(self) -> None:
        self.pause() if self._playing else self.play()

    def is_playing(self) -> bool:
        return self._playing

    def seek(self, t_s: float) -> None:
        self.t = float(np.clip(t_s, self.t_start, self.t_end))
        self.timeChanged.emit(self.t)

    def step(self, dt_s: float) -> None:
        self.seek(self.t + dt_s)

    def set_speed(self, s: float) -> None:
        self.speed = float(s)

    def _on_tick(self) -> None:
        self.t += self.speed * (self._tick_dt_ms / 1000.0)
        if self.t >= self.t_end:
            self.t = self.t_end
            self.pause()
        self.timeChanged.emit(self.t)


# ============================================================================
# Tabs
# ============================================================================

class OverviewTab(QWidget):
    def __init__(self, session_dir: Path, meta: dict, calib: dict,
                 imu_n: int, frame_n: int, vio_n: int, slam_n: int,
                 parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)

        title = QLabel(f"SESSION  {session_dir.name}")
        title.setObjectName("HeaderTitle")
        sub = QLabel(str(session_dir))
        sub.setObjectName("HeaderSub")
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addSpacing(8)

        txt = QTextEdit(readOnly=True)
        txt.setStyleSheet(
            f"background:{theme.PANEL}; color:{theme.TEXT}; "
            f"border:1px solid {theme.PANEL_EDGE}; padding:8px;"
        )

        lines = ["=== META ===", json.dumps(meta, indent=2), "",
                 "=== COUNTS (actual files) ===",
                 f"  frames    : {frame_n}",
                 f"  imu       : {imu_n}",
                 f"  vio poses : {vio_n}",
                 f"  slam poses: {slam_n}"]
        dur = meta.get("duration_s", 0.0)
        if dur > 0:
            lines += ["", "=== RATES (avg) ===",
                      f"  frames : {frame_n/dur:6.2f} Hz",
                      f"  imu    : {imu_n/dur:6.2f} Hz",
                      f"  vio    : {vio_n/dur:6.2f} Hz",
                      f"  slam   : {slam_n/dur:6.2f} Hz"]
        lines += ["", "=== CALIBRATION ===", json.dumps(calib, indent=2)]
        txt.setPlainText("\n".join(lines))
        lay.addWidget(txt, 1)


class FrameTab(QWidget):
    """C0: stereo left + right + depth colormap; driven by Playhead.

    If ``features`` is provided, tracked corners are overlaid on the left
    image (green circles) — nearest record by timestamp.
    """

    def __init__(self, session_dir: Path, frames: list[dict],
                 t_offset: float,
                 feat_ts: np.ndarray | None = None,
                 feat_pts: list[np.ndarray] | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.session_dir = session_dir
        self.frames = frames
        self.ts = _frame_ts_array(frames) - t_offset
        self.feat_ts = (feat_ts - t_offset) if feat_ts is not None and len(feat_ts) else np.zeros(0)
        self.feat_pts = feat_pts or []
        self._last_idx = -1

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self.info = QLabel("(no frames)" if not frames else "")
        self.info.setObjectName("HeaderSub")
        lay.addWidget(self.info)

        img_row = QHBoxLayout()
        img_row.setSpacing(6)
        self.lbl_left = self._make_img_label("LEFT")
        self.lbl_right = self._make_img_label("RIGHT")
        self.lbl_depth = self._make_img_label("DEPTH")
        for grp in (self.lbl_left, self.lbl_right, self.lbl_depth):
            img_row.addWidget(grp["frame"], 1)
        lay.addLayout(img_row, 1)

        if frames:
            self.on_time(0.0)

    def on_time(self, t_s: float) -> None:
        if not self.frames:
            return
        idx = int(np.searchsorted(self.ts, t_s))
        if idx >= len(self.ts):
            idx = len(self.ts) - 1
        elif idx > 0 and (t_s - self.ts[idx - 1]) < (self.ts[idx] - t_s):
            idx -= 1
        if idx == self._last_idx:
            return
        self._last_idx = idx
        self._show(idx, t_s)

    def _make_img_label(self, title: str) -> dict:
        frame = QWidget()
        frame.setObjectName("Panel")
        frame.setStyleSheet(
            f"#Panel {{ background:{theme.PANEL}; "
            f"border:1px solid {theme.PANEL_EDGE}; border-radius:4px; }}"
        )
        v = QVBoxLayout(frame)
        v.setContentsMargins(4, 4, 4, 4)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("PanelTitle")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl = QLabel()
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl.setMinimumSize(320, 200)
        img_lbl.setStyleSheet("background:#000;")
        v.addWidget(title_lbl)
        v.addWidget(img_lbl, 1)
        return {"frame": frame, "title": title_lbl, "img": img_lbl}

    def _show(self, idx: int, t_query: float) -> None:
        rec = self.frames[idx]
        w, h = int(rec["width"]), int(rec["height"])
        base = self.session_dir / "input"
        left = cv2.imread(str(base / rec["left_path"]), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(base / rec["right_path"]), cv2.IMREAD_GRAYSCALE)
        depth = np.fromfile(base / rec["depth_path"], dtype="<u2").reshape(h, w)

        # Overlay tracked features on LEFT image (C7).
        # Color = depth at that pixel (Turbo: near=red, far=blue, none=gray).
        left_disp = left
        n_feat = 0
        if len(self.feat_ts):
            fi = int(np.searchsorted(self.feat_ts, t_query))
            if fi >= len(self.feat_ts):
                fi = len(self.feat_ts) - 1
            elif fi > 0 and (t_query - self.feat_ts[fi - 1]) < (self.feat_ts[fi] - t_query):
                fi -= 1
            pts = self.feat_pts[fi]
            n_feat = len(pts)
            if n_feat:
                left_disp = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
                # Depth lookup with a small 3x3 max-window to dodge invalid
                # pixels right on the corner.
                xs = np.clip(pts[:, 0].astype(np.int32), 1, w - 2)
                ys = np.clip(pts[:, 1].astype(np.int32), 1, h - 2)
                # gather a 3x3 neighborhood per point, take the closest valid
                depths_mm = np.zeros(n_feat, dtype=np.uint16)
                for k, (x, y) in enumerate(zip(xs, ys)):
                    patch = depth[y - 1:y + 2, x - 1:x + 2]
                    valid = patch[patch > 0]
                    if valid.size:
                        depths_mm[k] = int(valid.min())
                # Normalize against scene range (5cm..6m) for stable colors.
                d_min_mm, d_max_mm = 500, 6000
                valid_mask = depths_mm > 0
                norm = np.zeros(n_feat, dtype=np.uint8)
                if valid_mask.any():
                    d_clip = np.clip(depths_mm[valid_mask].astype(np.float32),
                                     d_min_mm, d_max_mm)
                    norm[valid_mask] = (
                        (d_clip - d_min_mm) / (d_max_mm - d_min_mm) * 255.0
                    ).astype(np.uint8)
                # Turbo colormap: 256 colors (BGR)
                lut = cv2.applyColorMap(
                    np.arange(256, dtype=np.uint8).reshape(-1, 1),
                    cv2.COLORMAP_TURBO,
                ).reshape(-1, 3)
                for (x_, y_), v, ok in zip(pts, norm, valid_mask):
                    if ok:
                        c = tuple(int(x) for x in lut[v])
                    else:
                        c = (128, 128, 128)  # gray = no depth
                    cv2.circle(left_disp, (int(x_), int(y_)), 3, c, -1,
                               cv2.LINE_AA)

        if left_disp.ndim == 2:
            self._set_gray(self.lbl_left["img"], left_disp)
        else:
            self._set_bgr(self.lbl_left["img"], left_disp)
        self._set_gray(self.lbl_right["img"], right)
        self._set_depth(self.lbl_depth["img"], depth)

        ts_frame = self.ts[idx]
        skew_ms = (ts_frame - t_query) * 1000.0
        self.info.setText(
            f"frame  seq={rec['seq']:>5d}   "
            f"t={ts_frame:7.3f}s ({skew_ms:+.0f}ms)   {w}x{h}   "
            f"depth: valid={int((depth > 0).sum())}/{w*h}  "
            f"min={int(depth[depth>0].min()) if (depth>0).any() else 0}mm  "
            f"max={int(depth.max())}mm   feats={n_feat}"
        )

    @staticmethod
    def _set_bgr(label: QLabel, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pm)

    @staticmethod
    def _set_gray(label: QLabel, img: np.ndarray | None) -> None:
        if img is None:
            label.clear(); return
        h, w = img.shape
        qimg = QImage(img.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        pm = QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pm)

    @staticmethod
    def _set_depth(label: QLabel, depth_u16: np.ndarray) -> None:
        valid = depth_u16 > 0
        if not valid.any():
            label.clear(); return
        vmax = float(depth_u16[valid].max())
        norm = np.zeros_like(depth_u16, dtype=np.uint8)
        norm[valid] = np.clip(
            (depth_u16[valid].astype(np.float32) / vmax * 255.0), 0, 255
        ).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        h, w = colored.shape[:2]
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pm)


class IMUTab(QWidget):
    """C1: 6-channel gyro + accel time series; vertical cursor follows time."""

    def __init__(self, imu: dict, t_offset: float, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        n = len(imu["ts_s"])
        info = QLabel(f"IMU samples: {n}" if n else "(no IMU data)")
        info.setObjectName("HeaderSub")
        lay.addWidget(info)
        self._cursors: list[pg.InfiniteLine] = []
        if n == 0:
            return

        pg.setConfigOption("background", theme.BG)
        pg.setConfigOption("foreground", theme.TEXT)

        ts = imu["ts_s"] - t_offset

        self.p_gyro = pg.PlotWidget(title="GYRO (rad/s)")
        self.p_gyro.showGrid(x=True, y=True, alpha=0.3)
        self.p_gyro.addLegend()
        for i, (name, color) in enumerate(
            (("x", theme.AXIS_N), ("y", theme.AXIS_E), ("z", theme.AXIS_U))
        ):
            self.p_gyro.plot(ts, imu["gyro"][:, i],
                             pen=pg.mkPen(color, width=1), name=name)

        self.p_acc = pg.PlotWidget(title="ACCEL (m/s²)")
        self.p_acc.showGrid(x=True, y=True, alpha=0.3)
        self.p_acc.addLegend()
        for i, (name, color) in enumerate(
            (("x", theme.AXIS_N), ("y", theme.AXIS_E), ("z", theme.AXIS_U))
        ):
            self.p_acc.plot(ts, imu["accel"][:, i],
                            pen=pg.mkPen(color, width=1), name=name)

        self.p_acc.setXLink(self.p_gyro)

        cursor_pen = pg.mkPen(theme.ACCENT, width=1, style=Qt.PenStyle.DashLine)
        for p in (self.p_gyro, self.p_acc):
            ln = pg.InfiniteLine(pos=0.0, angle=90, pen=cursor_pen, movable=False)
            p.addItem(ln, ignoreBounds=True)
            self._cursors.append(ln)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.p_gyro)
        split.addWidget(self.p_acc)
        lay.addWidget(split, 1)

    def on_time(self, t_s: float) -> None:
        for ln in self._cursors:
            ln.setPos(t_s)


class PoseTab(QWidget):
    """C2/C3: 3D trajectory + pos/quat timeseries with live markers."""

    def __init__(self, vio: dict, slam: dict, t_offset: float,
                 pcl_ts: np.ndarray | None = None,
                 pcl_kinds: list[str] | None = None,
                 pcl_clouds: list[np.ndarray] | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._vio_ts = vio["ts_s"] - t_offset
        self._vio_pos = vio["pos"]
        self._slam_ts = slam["ts_s"] - t_offset
        self._slam_pos = slam["pos"]

        # Point cloud stream rebased to shared timeline. We keep ALL emissions
        # in memory and pick the latest <= t per kind in on_time().
        if pcl_ts is not None and len(pcl_ts):
            self._pcl_ts = pcl_ts - t_offset
            self._pcl_kinds = list(pcl_kinds or [])
            self._pcl_clouds = list(pcl_clouds or [])
            total_pts = sum(int(c.shape[0]) for c in self._pcl_clouds)
        else:
            self._pcl_ts = np.zeros(0)
            self._pcl_kinds = []
            self._pcl_clouds = []
            total_pts = 0
        self._pcl_last_key: tuple = ()  # cache to skip redundant setData

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        info = QLabel(
            f"VIO poses: {len(vio['ts_s']):>5d}   "
            f"SLAM poses: {len(slam['ts_s']):>5d}   "
            f"(FLU world frame — raw)"
        )
        info.setObjectName("HeaderSub")
        lay.addWidget(info)

        # --- 3D trajectory ---
        gl_widget = gl.GLViewWidget()
        gl_widget.setBackgroundColor(theme.BG)
        gl_widget.setCameraPosition(distance=10, elevation=25, azimuth=-60)
        grid = gl.GLGridItem()
        grid.setColor(pg.mkColor(theme.GRID))
        grid.setSize(20, 20); grid.setSpacing(1, 1)
        gl_widget.addItem(grid)
        ax_len = 1.0
        for v, color in (((ax_len, 0, 0), theme.AXIS_N),
                         ((0, ax_len, 0), theme.AXIS_E),
                         ((0, 0, ax_len), theme.AXIS_U)):
            gl_widget.addItem(gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], v]),
                color=pg.glColor(color), width=2, antialias=True,
            ))
        if len(vio["pos"]) >= 2:
            gl_widget.addItem(gl.GLLinePlotItem(
                pos=vio["pos"].astype(np.float32),
                color=pg.glColor(theme.WARN), width=2, antialias=True,
            ))
        if len(slam["pos"]) >= 2:
            gl_widget.addItem(gl.GLLinePlotItem(
                pos=slam["pos"].astype(np.float32),
                color=pg.glColor(theme.GOOD), width=2, antialias=True,
            ))
        # RTABMap point cloud (obstacle + ground), animated by Playhead.
        # Start empty; on_time() will fill with the latest cumulative emission
        # whose timestamp <= current playhead time.
        self._pcl_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0.6, 0.6, 0.9, 0.35), size=2.0, pxMode=True,
        )
        gl_widget.addItem(self._pcl_scatter)
        # Live position markers
        self._mark_vio = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=pg.glColor(theme.WARN), size=14.0, pxMode=True,
        )
        self._mark_slam = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=pg.glColor(theme.GOOD), size=14.0, pxMode=True,
        )
        gl_widget.addItem(self._mark_vio)
        gl_widget.addItem(self._mark_slam)

        legend = QLabel(
            f'<span style="color:{theme.WARN}">●</span> VIO (Basalt)   '
            f'<span style="color:{theme.GOOD}">●</span> SLAM (RTABMap)   '
            f'<span style="color:#9999ff">·</span> point cloud ('
            f'{len(self._pcl_ts)} emissions, peak {total_pts:,} pts)'
        )
        legend.setStyleSheet(f"color:{theme.TEXT}; padding:4px;")

        lay.addWidget(legend)
        lay.addWidget(gl_widget, 1)

    @staticmethod
    def _interp_pos(ts: np.ndarray, pos: np.ndarray, t: float) -> np.ndarray:
        if len(ts) == 0:
            return np.zeros(3, dtype=np.float32)
        if t <= ts[0]:
            return pos[0].astype(np.float32)
        if t >= ts[-1]:
            return pos[-1].astype(np.float32)
        i = int(np.searchsorted(ts, t))
        t0, t1 = ts[i - 1], ts[i]
        a = (t - t0) / max(t1 - t0, 1e-9)
        return ((1 - a) * pos[i - 1] + a * pos[i]).astype(np.float32)

    def on_time(self, t_s: float) -> None:
        self._mark_vio.setData(
            pos=self._interp_pos(self._vio_ts, self._vio_pos, t_s)[None, :]
        )
        self._mark_slam.setData(
            pos=self._interp_pos(self._slam_ts, self._slam_pos, t_s)[None, :]
        )
        self._update_pcl(t_s)

    def _update_pcl(self, t_s: float) -> None:
        if not len(self._pcl_ts):
            return
        # Pick latest emission index per kind whose ts <= t_s.
        latest_idx: dict[str, int] = {}
        for i, ts in enumerate(self._pcl_ts):
            if ts > t_s:
                break
            latest_idx[self._pcl_kinds[i]] = i
        key = tuple(sorted(latest_idx.items()))
        if key == self._pcl_last_key:
            return
        self._pcl_last_key = key
        if not latest_idx:
            self._pcl_scatter.setData(pos=np.zeros((1, 3), dtype=np.float32))
            return
        arrs = [self._pcl_clouds[i] for i in latest_idx.values()]
        pts = np.concatenate(arrs, axis=0)
        if len(pts) > 200_000:
            sel = np.random.choice(len(pts), 200_000, replace=False)
            pts = pts[sel]
        self._pcl_scatter.setData(pos=pts.astype(np.float32))


class EventsTab(QWidget):
    """C5 (loop closures) + C6 (tracking-lost gaps) on a shared time axis.

    Top plot: loop closure jump magnitudes (pos in m, rot in deg) as stems.
    Bottom plot: tracking-ok timeline (1=ok, 0=lost) derived from track_events.
    """

    def __init__(self, loop_events: list[dict], track_events: list[dict],
                 t_offset: float, t_end: float, parent=None) -> None:
        super().__init__(parent)
        self._cursors: list[pg.InfiniteLine] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        info = QLabel(
            f"loop closures: {len(loop_events)}    "
            f"tracking events: {len(track_events)}"
        )
        info.setObjectName("HeaderSub")
        lay.addWidget(info)

        pg.setConfigOption("background", theme.BG)
        pg.setConfigOption("foreground", theme.TEXT)

        # --- loop closure stems ---
        self.p_loop = pg.PlotWidget(title="C5 · LOOP CLOSURES (odom_correction jump)")
        self.p_loop.showGrid(x=True, y=True, alpha=0.3)
        self.p_loop.addLegend()
        self.p_loop.setLabel("left", "jump")
        self.p_loop.setLabel("bottom", "time (s)")

        if loop_events:
            ts = np.array([e["ts_ns"] for e in loop_events], dtype=np.float64) / 1e9 - t_offset
            pos_j = np.array([e["pos_jump_m"] for e in loop_events])
            rot_j = np.array([e["rot_jump_deg"] for e in loop_events])
            # stems: vertical line per event
            for t, dp, dr in zip(ts, pos_j, rot_j):
                self.p_loop.plot([t, t], [0, dp],
                                 pen=pg.mkPen(theme.AXIS_N, width=2))
                self.p_loop.plot([t, t], [0, dr / 50.0],  # scale rot for overlay
                                 pen=pg.mkPen(theme.AXIS_E, width=2))
            # markers + legend handles
            self.p_loop.plot(ts, pos_j, pen=None,
                             symbol="o", symbolBrush=theme.AXIS_N,
                             symbolSize=8, name="pos jump (m)")
            self.p_loop.plot(ts, rot_j / 50.0, pen=None,
                             symbol="t", symbolBrush=theme.AXIS_E,
                             symbolSize=8, name="rot jump (deg ÷ 50)")
        self.p_loop.setXRange(0.0, max(t_end, 1e-3))

        # --- tracking-ok timeline ---
        self.p_track = pg.PlotWidget(title="C6 · TRACKING STATE (1=ok, 0=lost)")
        self.p_track.showGrid(x=True, y=True, alpha=0.3)
        self.p_track.setLabel("left", "tracking_ok")
        self.p_track.setLabel("bottom", "time (s)")
        self.p_track.setYRange(-0.2, 1.2)
        self.p_track.setXLink(self.p_loop)

        # build step signal from track_events (lost / recovered toggles)
        ts_track = [0.0]
        v_track = [1]
        ok = True
        for e in sorted(track_events, key=lambda r: r["ts_ns"]):
            t = e["ts_ns"] / 1e9 - t_offset
            if e["event"] == "tracking_lost" and ok:
                ts_track += [t, t]; v_track += [1, 0]; ok = False
            elif e["event"] == "tracking_recovered" and not ok:
                ts_track += [t, t]; v_track += [0, 1]; ok = True
        ts_track.append(max(t_end, 1e-3))
        v_track.append(v_track[-1])
        self.p_track.plot(np.array(ts_track), np.array(v_track),
                          pen=pg.mkPen(theme.GOOD, width=2))
        self.p_track.setXRange(0.0, max(t_end, 1e-3))

        cursor_pen = pg.mkPen(theme.ACCENT, width=1, style=Qt.PenStyle.DashLine)
        for p in (self.p_loop, self.p_track):
            ln = pg.InfiniteLine(pos=0.0, angle=90, pen=cursor_pen, movable=False)
            p.addItem(ln, ignoreBounds=True)
            self._cursors.append(ln)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.p_loop)
        split.addWidget(self.p_track)
        lay.addWidget(split, 1)

    def on_time(self, t_s: float) -> None:
        for ln in self._cursors:
            ln.setPos(t_s)


# ============================================================================
# Playback bar
# ============================================================================

class PlaybackBar(QFrame):
    """Bottom transport: ⏮ |◀ ▶/⏸ ▶| + scrub slider + speed + time label."""

    def __init__(self, playhead: Playhead, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.ph = playhead
        self._scrubbing = False

        self.setStyleSheet(
            f"#Panel {{ background:{theme.PANEL}; "
            f"border-top:1px solid {theme.PANEL_EDGE}; }}"
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)

        self.btn_home = QPushButton("⏮")
        self.btn_back = QPushButton("|◀")
        self.btn_play = QPushButton("▶")
        self.btn_fwd = QPushButton("▶|")
        for b in (self.btn_home, self.btn_back, self.btn_play, self.btn_fwd):
            b.setFixedWidth(36)
        self.btn_play.setStyleSheet(
            f"QPushButton {{ background:{theme.BTN_PRIMARY}; "
            f"color:{theme.TEXT}; font-weight:bold; }}"
        )
        lay.addWidget(self.btn_home)
        lay.addWidget(self.btn_back)
        lay.addWidget(self.btn_play)
        lay.addWidget(self.btn_fwd)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(10000)
        lay.addWidget(self.slider, 1)

        self.lbl_time = QLabel("0.00 / 0.00 s")
        self.lbl_time.setStyleSheet(f"color:{theme.TEXT}; min-width:120px;")
        lay.addWidget(self.lbl_time)

        self.cmb_speed = QComboBox()
        for s in ("0.25×", "0.5×", "1×", "2×", "4×", "8×"):
            self.cmb_speed.addItem(s)
        self.cmb_speed.setCurrentText("1×")
        self.cmb_speed.setFixedWidth(70)
        lay.addWidget(self.cmb_speed)

        self.btn_home.clicked.connect(lambda: self.ph.seek(self.ph.t_start))
        self.btn_back.clicked.connect(lambda: self.ph.step(-0.1))
        self.btn_fwd.clicked.connect(lambda: self.ph.step(+0.1))
        self.btn_play.clicked.connect(self._toggle_play)
        self.slider.sliderPressed.connect(self._start_scrub)
        self.slider.sliderReleased.connect(self._end_scrub)
        self.slider.valueChanged.connect(self._on_slider_value)
        self.cmb_speed.currentTextChanged.connect(self._on_speed)
        self.ph.timeChanged.connect(self._on_playhead_time)

        self._on_playhead_time(self.ph.t)

    def _toggle_play(self) -> None:
        self.ph.toggle()
        self.btn_play.setText("⏸" if self.ph.is_playing() else "▶")

    def _start_scrub(self) -> None:
        self._scrubbing = True

    def _end_scrub(self) -> None:
        self._scrubbing = False

    def _on_slider_value(self, v: int) -> None:
        if not self._scrubbing:
            return
        frac = v / float(self.slider.maximum())
        t = self.ph.t_start + frac * (self.ph.t_end - self.ph.t_start)
        self.ph.seek(t)

    def _on_speed(self, label: str) -> None:
        self.ph.set_speed(float(label.rstrip("×")))

    def _on_playhead_time(self, t: float) -> None:
        dur = self.ph.t_end - self.ph.t_start
        rel = t - self.ph.t_start
        frac = (rel / dur) if dur > 0 else 0.0
        if not self._scrubbing:
            self.slider.blockSignals(True)
            self.slider.setValue(int(round(frac * self.slider.maximum())))
            self.slider.blockSignals(False)
        self.lbl_time.setText(f"{rel:6.2f} / {dur:6.2f} s")
        if not self.ph.is_playing():
            self.btn_play.setText("▶")


# ============================================================================
# Main window
# ============================================================================

class SessionViewer(QMainWindow):
    def __init__(self, session_dir: Path) -> None:
        super().__init__()
        self.setWindowTitle(f"oak-d session viewer — {session_dir.name}")
        self.resize(1400, 900)
        self.setStyleSheet(theme.QSS)

        meta = {}
        if (session_dir / "meta.json").exists():
            meta = json.loads((session_dir / "meta.json").read_text())
        calib = {}
        if (session_dir / "calib.json").exists():
            calib = json.loads((session_dir / "calib.json").read_text())

        frames = _load_jsonl(session_dir / "input" / "frames.jsonl")
        imu = _stack_imu(_load_jsonl(session_dir / "input" / "imu.jsonl"))
        vio = _stack_poses(_load_jsonl(session_dir / "basalt" / "vio_pose.jsonl"))
        slam = _stack_poses(_load_jsonl(session_dir / "basalt" / "slam_pose.jsonl"))
        loop_events = _load_jsonl(session_dir / "basalt" / "loop_events.jsonl")
        track_events = _load_jsonl(session_dir / "basalt" / "track_events.jsonl")
        feat_ts, feat_pts = _load_features(session_dir / "basalt" / "features.jsonl")
        pcl_ts, pcl_kinds, pcl_clouds = _load_pointcloud_stream(session_dir)

        # Shared timeline rebased to 0
        starts, ends = [], []
        frame_ts = _frame_ts_array(frames)
        for arr in (frame_ts, imu["ts_s"], vio["ts_s"], slam["ts_s"]):
            if len(arr):
                starts.append(float(arr[0])); ends.append(float(arr[-1]))
        if starts:
            t_offset = min(starts)
            t_end = max(ends) - t_offset
        else:
            t_offset, t_end = 0.0, 1.0
        self.playhead = Playhead(0.0, t_end)

        self.tab_overview = OverviewTab(
            session_dir, meta, calib,
            imu_n=len(imu["ts_s"]), frame_n=len(frames),
            vio_n=len(vio["ts_s"]), slam_n=len(slam["ts_s"]),
        )
        self.tab_frame = FrameTab(session_dir, frames, t_offset,
                                  feat_ts=feat_ts, feat_pts=feat_pts)
        self.tab_imu = IMUTab(imu, t_offset)
        self.tab_pose = PoseTab(vio, slam, t_offset,
                                pcl_ts=pcl_ts, pcl_kinds=pcl_kinds,
                                pcl_clouds=pcl_clouds)
        self.tab_events = EventsTab(loop_events, track_events, t_offset, t_end)

        tabs = QTabWidget()
        tabs.addTab(self.tab_overview, "Overview")
        tabs.addTab(self.tab_frame, "C0 · Frame")
        tabs.addTab(self.tab_imu, "C1 · IMU")
        tabs.addTab(self.tab_pose, "C2/C3 · Pose 3D")
        tabs.addTab(self.tab_events, "C5/C6 · Events")

        bar = PlaybackBar(self.playhead)

        self.playhead.timeChanged.connect(self.tab_frame.on_time)
        self.playhead.timeChanged.connect(self.tab_imu.on_time)
        self.playhead.timeChanged.connect(self.tab_pose.on_time)
        self.playhead.timeChanged.connect(self.tab_events.on_time)

        root = QWidget()
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(tabs, 1)
        v.addWidget(bar)
        self.setCentralWidget(root)

        QShortcut(QKeySequence(Qt.Key.Key_Space), self,
                  activated=bar._toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self,
                  activated=lambda: self.playhead.step(-0.1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self,
                  activated=lambda: self.playhead.step(+0.1))
        QShortcut(QKeySequence(Qt.Key.Key_Home), self,
                  activated=lambda: self.playhead.seek(self.playhead.t_start))

        self.playhead.seek(0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", help="path to a recorded session folder")
    args = ap.parse_args()

    sd = Path(args.session_dir).resolve()
    if not sd.is_dir():
        print(f"not a directory: {sd}", file=sys.stderr)
        return 2

    app = QApplication(sys.argv)
    win = SessionViewer(sd)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
