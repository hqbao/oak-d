"""MapWindow: a standalone 3D viewer for the SLAM keyframe point cloud.

Shows the room reconstructed from every keyframe at once (``ours.tools.slam_map3d``
builds the cloud; this just renders it): a coloured point cloud plus the keyframe
camera positions, on the same grid/axes the pose viewer uses. Points come in the
camera-optical world frame and are rotated to the viewer's ENU display frame with
the SAME convention as :class:`~ours.ui.viewer3d.Viewer3D`, so a map and a
trajectory line up.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QMainWindow

from ..lib.misc import frames
from . import theme
from .viewer3d import _make_grid, _make_world_axes, _qcolor

# Camera optical (x right, y down, z forward) -> world NED; then NED->ENU is the
# viewer's display transform (identical to Viewer3D, so maps + paths align).
_M_OPT_TO_NED = np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])


def _to_display(pts_opt: np.ndarray) -> np.ndarray:
    """Optical-world ``(N,3)`` -> ENU display ``(N,3)`` float32."""
    if len(pts_opt) == 0:
        return np.zeros((0, 3), np.float32)
    ned = np.asarray(pts_opt, np.float64) @ _M_OPT_TO_NED.T
    return frames.ned_to_enu(ned).astype(np.float32)


class MapWindow(QMainWindow):
    def __init__(self, points: np.ndarray, colors: np.ndarray,
                 cams: np.ndarray, title: str = "SLAM map") -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1100, 800)

        view = gl.GLViewWidget()
        view.setBackgroundColor(QColor(theme.BG))
        self.setCentralWidget(view)

        view.addItem(_make_grid(size_m=20.0, step_m=1.0))
        for ax in _make_world_axes(length=1.0):
            view.addItem(ax)

        pts = _to_display(points)
        rgba = np.concatenate(
            [np.clip(colors, 0.0, 1.0),
             np.ones((len(colors), 1), np.float32)], axis=1).astype(np.float32) \
            if len(colors) else np.zeros((0, 4), np.float32)
        view.addItem(gl.GLScatterPlotItem(pos=pts, color=rgba, size=2.0,
                                          pxMode=True))

        # Keyframe camera positions: amber dots so the capture path is visible.
        cam_enu = _to_display(cams)
        if len(cam_enu):
            view.addItem(gl.GLScatterPlotItem(
                pos=cam_enu, color=_qcolor(theme.WARN, 0.95), size=9.0,
                pxMode=True))

        # Frame the cloud: centre the orbit on its centroid, back off by its size.
        if len(pts):
            centre = pts.mean(axis=0)
            extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            view.opts["center"] = self._vec(centre)
            view.setCameraPosition(distance=max(extent * 0.8, 2.0),
                                   azimuth=45, elevation=30)

    @staticmethod
    def _vec(p):
        from PyQt6.QtGui import QVector3D
        return QVector3D(float(p[0]), float(p[1]), float(p[2]))
