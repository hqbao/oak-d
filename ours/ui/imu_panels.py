"""Interactive IMU panels for the in-app synced camera/IMU window.

Two honest, real-data widgets fed straight from an
:class:`~ours.lib.flow.messages.ImuCamPacket` (no parallel pipeline):

* :class:`GyroPlot` -- an **auto-scaling** scrolling 3-axis line chart of the raw
  gyro samples (rad/s shown as deg/s). pyqtgraph auto-ranges Y to whatever the
  signal actually does, so the trace never clips or flat-lines off a fixed span.
* :class:`Accel3DView` -- a **real interactive 3D** view (pyqtgraph OpenGL) of the
  raw accel vector in body axes. The user can orbit it with the mouse and snap to
  three canonical viewpoints (BACK / LEFT / TOP) with the buttons.

Both draw exactly what the packet carries; nothing is computed elsewhere.
pyqtgraph (and OpenGL) are only pulled when this module is imported -- i.e. when
the synced window is opened -- so the base UI stays lightweight.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6 import QtWidgets
from PyQt6.QtGui import QColor

from . import theme

_RAD2DEG = 180.0 / np.pi
_G = 9.80665

# Per-axis colours reused from the 3D scene so x/y/z read the same everywhere:
#   x -> red, y -> green, z -> cyan.
_AXIS_HEX = (theme.AXIS_N, theme.AXIS_E, theme.AXIS_U)
_GYRO_FLOOR_DPS = 1.0       # never let the auto-scale collapse below ±this
_GYRO_SHRINK = 0.06         # expand-fast / shrink-slow hysteresis factor


def _rgba(hexstr: str, alpha: float = 1.0):
    c = QColor(hexstr)
    return (c.redF(), c.greenF(), c.blueF(), alpha)


def _rgba_u8(hexstr: str, alpha: int = 255):
    c = QColor(hexstr)
    return (c.red(), c.green(), c.blue(), alpha)


# ---------------------------------------------------------------------------
# Gyro -- auto-scaling scrolling line chart
# ---------------------------------------------------------------------------

class GyroPlot(pg.PlotWidget):
    """Scrolling 3-axis gyro chart (deg/s) with automatic, stable Y scaling.

    Y auto-scales to the signal, but with a minimum span and expand-fast /
    shrink-slow hysteresis so a *stationary* IMU (a few deg/s of sensor noise)
    does NOT get blown up to full height and rescaled every frame -- which would
    make a still drone look like it is tumbling.
    """

    def __init__(self, capacity: int = 600, parent=None) -> None:
        super().__init__(parent)
        self._cap = int(capacity)
        self.setObjectName("GyroPlot")
        self.setBackground(theme.BG)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setLabel("left", "gyro", units="deg/s")
        self.setLabel("bottom", "samples (newest at right)")
        self.setMenuEnabled(False)
        # Auto-scale is the whole point, but we drive Y by hand (below) so it
        # stays stable; lock the X window and disable mouse so it can't drift.
        self.setMouseEnabled(x=False, y=False)
        self.setXRange(0, self._cap, padding=0)
        # Zero angular-rate reference: the single most important gridline.
        self.addLine(y=0, pen=pg.mkPen(theme.TEXT_DIM, width=1))
        legend = self.addLegend(offset=(-8, 8),
                                labelTextColor=theme.TEXT,
                                brush=pg.mkBrush(theme.PANEL),
                                pen=pg.mkPen(theme.PANEL_EDGE))
        legend.setObjectName("GyroLegend")

        self._x = np.arange(self._cap, dtype=np.float64)
        self._data = np.zeros((self._cap, 3), dtype=np.float64)
        self._n = 0
        self._span = _GYRO_FLOOR_DPS
        self._curves = [
            self.plot(pen=pg.mkPen(QColor(_AXIS_HEX[i]), width=1.6), name=name)
            for i, name in enumerate(("x", "y", "z"))
        ]
        self.setYRange(-self._span, self._span, padding=0)

    # -- public API --------------------------------------------------------
    def clear_history(self) -> None:
        self._data[:] = 0.0
        self._n = 0
        self._span = _GYRO_FLOOR_DPS
        self.setYRange(-self._span, self._span, padding=0)
        for c in self._curves:
            c.setData([], [])

    def add(self, gyro_rad) -> None:
        """Append every gyro sample in a packet (``(M,3)`` rad/s)."""
        if gyro_rad is None or np.size(gyro_rad) == 0:
            return
        rows = np.atleast_2d(np.asarray(gyro_rad, dtype=np.float64)) * _RAD2DEG
        m = rows.shape[0]
        if m >= self._cap:
            self._data[:] = rows[-self._cap:]
            self._n = self._cap
        else:
            self._data[:-m] = self._data[m:]          # scroll left
            self._data[-m:] = rows
            self._n = min(self._cap, self._n + m)
        self._redraw()

    @property
    def sample_count(self) -> int:
        return self._n

    def latest(self):
        return None if self._n == 0 else self._data[-1].copy()

    # -- internal ----------------------------------------------------------
    def _redraw(self) -> None:
        if self._n < 2:
            return
        view = self._data[-self._n:]
        x = self._x[:self._n]
        for i, c in enumerate(self._curves):
            c.setData(x, view[:, i])
        # Stable auto-scale: target span = peak with headroom, floored; grow
        # instantly, shrink gently so the trace never strobes when near-still.
        target = max(float(np.abs(view).max()) * 1.15, _GYRO_FLOOR_DPS)
        if target >= self._span:
            self._span = target
        else:
            self._span += (target - self._span) * _GYRO_SHRINK
        self.setYRange(-self._span, self._span, padding=0)


# ---------------------------------------------------------------------------
# Accel -- real interactive 3D vector view
# ---------------------------------------------------------------------------

# (azimuth_deg, elevation_deg) -- pyqtgraph GLViewWidget conventions, matched to
# the pose viewer's presets so "BACK/LEFT/TOP" mean the same thing app-wide.
_ACCEL_PRESETS: dict[str, tuple[float, float]] = {
    "BACK": (90.0, 0.0),
    "LEFT": (180.0, 0.0),
    "TOP": (-90.0, 89.9),
}
_PRESET_TIP = {
    "BACK": "camera behind the +X body axis (looking forward)",
    "LEFT": "camera on the +Y side (looking across)",
    "TOP": "camera straight down the +Z body axis",
}
_VIEW_DIST = 3.2


def _ring(radius: float, plane: str, n: int = 72):
    """A wire circle of ``radius`` in the given plane ('xy' | 'xz')."""
    t = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    c, s = np.cos(t), np.sin(t)
    z = np.zeros_like(t)
    if plane == "xy":
        return np.stack([c, s, z], axis=1) * radius
    return np.stack([c, z, s], axis=1) * radius          # xz


# Checkerboard floor placed UNDER the vector so "down" reads at a glance and the
# 3D depth is easy to judge against a patterned ground (vs. empty black).
_FLOOR_PX = 256
_FLOOR_SIZE = 3.0
_FLOOR_Z = -1.2


def _checker_texture(squares: int = 8, px: int = _FLOOR_PX) -> np.ndarray:
    """An ``(px, px, 4)`` uint8 checkerboard in two subtle panel tones."""
    c0 = np.array(_rgba_u8(theme.PANEL), dtype=np.uint8)
    c1 = np.array(_rgba_u8(theme.GRID), dtype=np.uint8)
    tile = px // squares
    img = np.empty((px, px, 4), dtype=np.uint8)
    for i in range(squares):
        for j in range(squares):
            img[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile] = (
                c0 if (i + j) % 2 == 0 else c1)
    return img


def _arrow_mesh(length: float, shaft_r: float = 0.03, head_r: float = 0.085,
                head_frac: float = 0.26, cols: int = 20):
    """A solid +Z arrow (cylinder shaft + cone head) of total ``length``.

    Rendered as a mesh instead of a GL line so the vector is unmistakably
    visible regardless of the platform's line-width support (macOS clamps it).
    """
    head_len = max(length * head_frac, 1e-4)
    shaft_len = max(length - head_len, 1e-4)
    shaft = gl.MeshData.cylinder(rows=1, cols=cols,
                                 radius=[shaft_r, shaft_r], length=shaft_len)
    head = gl.MeshData.cylinder(rows=1, cols=cols,
                                radius=[head_r, 0.0], length=head_len)
    v1, f1 = shaft.vertexes(), shaft.faces()
    v2 = head.vertexes().copy()
    v2[:, 2] += shaft_len
    verts = np.vstack([v1, v2]).astype(np.float32)
    faces = np.vstack([f1, head.faces() + len(v1)])
    return gl.MeshData(vertexes=verts, faces=faces)


class _OrbitGL(gl.GLViewWidget):
    """GLViewWidget that reports when the user manually orbits the camera."""

    def __init__(self, on_user_orbit, parent=None) -> None:
        super().__init__(parent)
        self._on_user_orbit = on_user_orbit

    def mouseMoveEvent(self, ev):                                  # noqa: N802
        super().mouseMoveEvent(ev)
        if self._on_user_orbit is not None:
            self._on_user_orbit()


class Accel3DView(QtWidgets.QWidget):
    """Interactive 3D accel vector (orbit with mouse, BACK/LEFT/TOP presets)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Accel3DView")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        caption = QtWidgets.QLabel(
            "ACCEL — BODY FRAME · arrow = specific force · rings = 1 G · floor = down")
        caption.setObjectName("PanelTitle")
        lay.addWidget(caption)

        self._gl = _OrbitGL(self._on_user_orbit)
        self._gl.setObjectName("Accel3DGL")
        self._gl.setBackgroundColor(QColor(theme.BG))
        self._gl.setCameraPosition(distance=_VIEW_DIST)
        lay.addWidget(self._gl, stretch=1)

        # Checkerboard ground placed below the origin: the gravity arrow points
        # down onto it, so "which way is down" and the 3D depth are obvious.
        self._floor = gl.GLImageItem(_checker_texture())
        self._floor.scale(_FLOOR_SIZE / _FLOOR_PX, _FLOOR_SIZE / _FLOOR_PX, 1.0)
        self._floor.translate(-_FLOOR_SIZE / 2, -_FLOOR_SIZE / 2, _FLOOR_Z)
        self._gl.addItem(self._floor)

        # 1 G reference rings (XY + XZ) so vector magnitude is readable from any
        # orbit -- gravity at rest is a unit vector that lands on these rings.
        for plane in ("xy", "xz"):
            self._gl.addItem(gl.GLLinePlotItem(
                pos=_ring(1.0, plane), color=_rgba(theme.TEXT_DIM, 0.6),
                width=1.0, antialias=True, mode="line_strip"))

        # Body-axis triad: x red, y green, z cyan (same code as the scene), with
        # tip labels so axes stay identifiable after the user rotates the view.
        for vec, hexc, name in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                   _AXIS_HEX, ("X", "Y", "Z")):
            pts = np.array([(0, 0, 0), vec], dtype=np.float32)
            self._gl.addItem(gl.GLLinePlotItem(
                pos=pts, color=_rgba(hexc), width=2.0, antialias=True))
            try:
                self._gl.addItem(gl.GLTextItem(
                    pos=np.array(vec, dtype=np.float32) * 1.12,
                    text=name, color=QColor(hexc)))
            except Exception:
                pass                       # GLTextItem missing on old pyqtgraph

        # The live accel vector (specific force), drawn as a solid arrow mesh so
        # it is clearly visible; 1 G == unit length. Rebuilt each frame.
        self._accel_rgba = _rgba(theme.IMU_ACCEL)
        self._arrow = gl.GLMeshItem(
            meshdata=_arrow_mesh(1.0), color=self._accel_rgba, smooth=True,
            shader="shaded", glOptions="opaque")
        self._gl.addItem(self._arrow)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(4)
        self._buttons: dict[str, QtWidgets.QPushButton] = {}
        for name in ("BACK", "LEFT", "TOP"):
            b = QtWidgets.QPushButton(name)
            b.setCheckable(True)
            b.setToolTip(_PRESET_TIP[name])
            b.clicked.connect(lambda _=False, n=name: self.set_view(n))
            controls.addWidget(b)
            self._buttons[name] = b
        controls.addStretch(1)
        self._mag = QtWidgets.QLabel("|a| = —")
        self._mag.setObjectName("ImuCamStatus")
        controls.addWidget(self._mag)
        lay.addLayout(controls)

        self.accel = np.zeros(3, dtype=np.float64)
        self.set_view("BACK")

    # -- public API --------------------------------------------------------
    def set_view(self, name: str) -> None:
        if name not in _ACCEL_PRESETS:
            return
        az, el = _ACCEL_PRESETS[name]
        self._gl.setCameraPosition(azimuth=az, elevation=el, distance=_VIEW_DIST)
        for key, btn in self._buttons.items():
            btn.setChecked(key == name)

    def set_accel(self, accel_rows) -> None:
        """Draw the average specific-force vector of a packet's accel samples."""
        if accel_rows is None or np.size(accel_rows) == 0:
            return
        a = np.atleast_2d(np.asarray(accel_rows, dtype=np.float64)).mean(axis=0)
        self.accel = a
        v = a / _G
        length = float(np.linalg.norm(v))
        if length > 1e-4:
            # Rebuild the arrow at the true length, then rotate +Z -> direction.
            d = v / length
            axis = np.cross((0.0, 0.0, 1.0), d)
            s = float(np.linalg.norm(axis))
            c = float(np.clip(d[2], -1.0, 1.0))
            angle = float(np.degrees(np.arctan2(s, c)))
            axis = (1.0, 0.0, 0.0) if s < 1e-8 else tuple(axis / s)
            self._arrow.setMeshData(meshdata=_arrow_mesh(length))
            self._arrow.resetTransform()
            self._arrow.rotate(angle, axis[0], axis[1], axis[2])
        self._mag.setText(
            f"|a| = {float(np.linalg.norm(a)):5.2f} m/s² (mean)   "
            f"({a[0]:+.1f}, {a[1]:+.1f}, {a[2]:+.1f})")

    # -- internal ----------------------------------------------------------
    def _on_user_orbit(self) -> None:
        # The operator dragged the view; no preset is "current" any more.
        for btn in self._buttons.values():
            btn.setChecked(False)
