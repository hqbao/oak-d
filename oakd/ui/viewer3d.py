"""3D scene built with pyqtgraph's OpenGL viewport.

Renders, in ENU display coordinates (positions converted from internal NED):
  * ground grid + axis triad at world origin
  * live drone position as a triad (forward=red, right=green, down=cyan reversed to up)
  * trajectory polyline (NVG green)
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6 import QtCore, QtGui
from PyQt6.QtGui import QColor

from .. import frames
from ..pose import Pose, PoseHistory
from . import theme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qcolor(hexstr: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    c = QColor(hexstr)
    return (c.redF(), c.greenF(), c.blueF(), alpha)


def _make_grid(size_m: float = 20.0, step_m: float = 1.0,
               color=_qcolor(theme.GRID, 0.55)) -> gl.GLGridItem:
    g = gl.GLGridItem()
    g.setSize(size_m, size_m, 0)
    g.setSpacing(step_m, step_m, 0)
    g.setColor(QColor(theme.GRID))
    return g


def _make_world_axes(length: float = 1.5) -> list[gl.GLLinePlotItem]:
    """Origin axes shown in ENU: E (red), N (green), U (cyan)."""
    items: list[gl.GLLinePlotItem] = []
    specs = [
        ((1, 0, 0), theme.AXIS_E),   # East (X in scene)
        ((0, 1, 0), theme.AXIS_N),   # North (Y in scene)
        ((0, 0, 1), theme.AXIS_U),   # Up (Z in scene)
    ]
    for direction, hexc in specs:
        pts = np.array([(0, 0, 0), tuple(length * d for d in direction)],
                       dtype=np.float32)
        line = gl.GLLinePlotItem(
            pos=pts, color=_qcolor(hexc, 1.0), width=2.5, antialias=True,
        )
        items.append(line)
    return items


# ---------------------------------------------------------------------------
# Drone marker — body axes triad
# ---------------------------------------------------------------------------

class _DroneTriad:
    """Three colored line segments that follow the drone's pose.

    Body axes (FRD) are drawn in scene coordinates after NED->ENU rotation:
      Forward (red), Right (green), Down (cyan, but rendered as -Z for clarity).
    """

    def __init__(self, length: float = 0.6) -> None:
        self.length = float(length)
        # initialise at origin pointing along world axes
        zero = np.zeros((2, 3), dtype=np.float32)
        self.fwd = gl.GLLinePlotItem(pos=zero, color=_qcolor(theme.AXIS_N), width=3.0,
                                     antialias=True)
        self.right = gl.GLLinePlotItem(pos=zero, color=_qcolor(theme.AXIS_E), width=3.0,
                                       antialias=True)
        self.down = gl.GLLinePlotItem(pos=zero, color=_qcolor(theme.AXIS_U), width=3.0,
                                      antialias=True)
        # small filled marker at the body origin
        self.dot = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=_qcolor(theme.GOOD, 1.0),
            size=12.0,
            pxMode=True,
        )

    def items(self) -> list:
        return [self.fwd, self.right, self.down, self.dot]

    def update(self, pose: Pose) -> None:
        # body origin in scene coords (ENU)
        p_enu = frames.ned_to_enu(pose.pos_ned).astype(np.float32)
        # rotate body axes into ENU
        R_enu = frames.rot_ned_to_enu(pose.R).astype(np.float32)
        x_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        z_b = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        fwd_dir = R_enu @ x_b
        right_dir = R_enu @ y_b
        down_dir = R_enu @ z_b

        L = self.length
        self.fwd.setData(pos=np.stack([p_enu, p_enu + L * fwd_dir]))
        self.right.setData(pos=np.stack([p_enu, p_enu + L * right_dir]))
        # show "down" axis flipped so it reads as "up" in the scene
        self.down.setData(pos=np.stack([p_enu, p_enu - L * down_dir]))
        self.dot.setData(pos=p_enu.reshape(1, 3))


# ---------------------------------------------------------------------------
# Main 3D viewport widget
# ---------------------------------------------------------------------------

# (azimuth_deg, elevation_deg, distance_m)  -- pyqtgraph GLViewWidget conventions
VIEW_PRESETS: dict[str, tuple[float, float, float]] = {
    "ISO":    (45.0,  28.0, 14.0),
    "TOP":    (-90.0,  89.9, 14.0),
    "FRONT":  (-90.0,   0.0, 14.0),
    "BACK":   ( 90.0,   0.0, 14.0),
    "LEFT":   (180.0,   0.0, 14.0),
    "RIGHT":  (  0.0,   0.0, 14.0),
}


class Viewer3D(gl.GLViewWidget):
    """OpenGL viewport with grid, axes, trajectory line and drone triad."""

    def __init__(self, history: PoseHistory, parent=None) -> None:
        super().__init__(parent)
        self.history = history
        self.setBackgroundColor(QColor(theme.BG))

        # ---- static scene -------------------------------------------------
        self.addItem(_make_grid(size_m=20.0, step_m=1.0))
        for ax in _make_world_axes(length=1.5):
            self.addItem(ax)

        # ---- trajectory polyline -----------------------------------------
        self._traj = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=_qcolor(theme.TRACE_PATH, 0.95),
            width=2.0,
            antialias=True,
            mode="line_strip",
        )
        self.addItem(self._traj)

        # ---- drone triad --------------------------------------------------
        self._drone = _DroneTriad(length=0.6)
        for it in self._drone.items():
            self.addItem(it)

        # ---- follow-cam state --------------------------------------------
        self._follow = False
        self.set_view("ISO")

        # ---- refresh timer (UI 60 Hz, decoupled from source rate) --------
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ---- public API ------------------------------------------------------

    def set_view(self, name: str) -> None:
        if name not in VIEW_PRESETS:
            return
        az, el, dist = VIEW_PRESETS[name]
        self.setCameraPosition(azimuth=az, elevation=el, distance=dist)
        self.opts["center"] = QtGui.QVector3D(0, 0, 0)
        self.update()

    def set_follow(self, on: bool) -> None:
        self._follow = bool(on)

    # ---- internal --------------------------------------------------------

    def _refresh(self) -> None:
        traj, latest = self.history.snapshot()
        if traj.shape[0] >= 2:
            # convert the whole trajectory NED -> ENU in one shot
            traj_enu = frames.ned_to_enu(traj.astype(np.float64)).astype(np.float32)
            self._traj.setData(pos=traj_enu)
        if latest is not None:
            self._drone.update(latest)
            if self._follow:
                p_enu = frames.ned_to_enu(latest.pos_ned)
                self.opts["center"] = QtGui.QVector3D(*p_enu.astype(float))
                self.update()
