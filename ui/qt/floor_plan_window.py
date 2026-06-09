"""FloorPlanWindow: a LIGHT 2D top-down floor plan of the room (NO OpenGL).

A cheap, readable alternative to the 3D SLAM map window (the landmark point
cloud): it is heavy GL on this Mac AND hard to read (noisy marginal depth in
perspective). This window instead shows a 2D TOP-DOWN occupancy raster -- the
walls/vertical structure read as a top-down OUTLINE with the camera path drawn over
it -- so the room LAYOUT is obvious. It renders on a pyqtgraph 2D
:class:`~pyqtgraph.PlotWidget` (an ``ImageItem`` for the raster + a ``PlotDataItem``
for the path) -- explicitly **NO** :class:`~pyqtgraph.opengl.GLViewWidget` / no 3D
GL mesh -- so it is light (no per-frame shader work) and never stutters the UI.

The caller (an IPC source, :class:`~ui.modules.ipc_sources.IpcFloorPlanSource`)
passes an ALREADY-BUILT raster (``rgb`` + a
:class:`~ui.viz.floor_plan.FloorPlanExtent`) + the keyframe camera positions; this
just places the image in world metres and draws the path. The plan is REBUILT live:
the source re-bins the keyframes a few times a second OFF the GUI thread and calls
:meth:`submit` (thread-safe) with the fresh raster -- so the room fills in / re-snaps
in place. The view keeps an EQUAL aspect ratio (so the plan isn't stretched) with
pan/zoom.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QMainWindow

from . import theme
from ui.viz.floor_plan import FloorPlanExtent


class FloorPlanWindow(QMainWindow):
    #: Carries a freshly built floor-plan raster + camera path from a background
    #: source thread onto the GUI thread. :meth:`submit` (thread-safe) emits it;
    #: the signal is connected to :meth:`update` so the plot items are only touched
    #: on the GUI thread.
    plan_ready = pyqtSignal(object, object, object, object)

    def __init__(self, title: str = "Floor Plan (top-down)") -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1000, 900)

        # ---- 2D plot (NO GLViewWidget) -----------------------------------
        # A plain pyqtgraph PlotWidget: cheap raster blit + a polyline, no GL
        # shaders / depth buffer. Equal aspect so the plan reads in true metres.
        self._plot = pg.PlotWidget()
        self._plot.setBackground(QColor(theme.BG))
        self._plot.setAspectLocked(True)              # 1 m east == 1 m north
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setLabel("bottom", "x (m)")        # optical x == right
        self._plot.setLabel("left", "z (m)")          # optical z == forward
        self.setCentralWidget(self._plot)

        # ---- raster image item -------------------------------------------
        # The occupancy raster. ``axisOrder='row-major'`` so an ``(H,W,3)`` array
        # maps rows->z, cols->x directly (no transpose); ``setRect`` later places
        # it in world metres so pan/zoom read in metres and the path overlays it.
        self._img = pg.ImageItem(axisOrder="row-major")
        self._plot.addItem(self._img)

        # ---- camera path polyline + latest-pose marker -------------------
        # The keyframe (and current) camera positions projected onto the same
        # ground plane: a bright NVG-green polyline so the user sees where the
        # camera went through the room, plus a larger amber dot on the LATEST
        # pose (the "you are here").
        self._path = self._plot.plot(
            [], [], pen=pg.mkPen(QColor(theme.TRACE_PATH), width=2.0))
        self._head = pg.ScatterPlotItem(
            size=12.0, brush=pg.mkBrush(QColor(theme.WARN)),
            pen=pg.mkPen(None))
        self._plot.addItem(self._head)

        # Auto-range the view ONCE on the first non-empty plan so the orbit
        # doesn't jump every rebuild as the room grows.
        self._framed = False

        # Route background-thread plans through the signal so `update` (plot items)
        # runs on the GUI thread (Qt auto-queues a cross-thread signal emit).
        self.plan_ready.connect(self.update)

    # ------------------------------------------------------------------ #
    def submit(self, rgb, path_px, cams, extent) -> None:
        """Thread-safe ingest: hand a freshly built floor plan in from any thread.

        Emits :attr:`plan_ready`; Qt queues the connected :meth:`update` onto the
        GUI thread, so a background IPC/rebuild thread can call this directly.

        ``rgb`` ``(H,W,3)`` uint8 is the occupancy raster; ``extent`` the
        :class:`~ui.viz.floor_plan.FloorPlanExtent` placing it in world metres;
        ``cams`` ``(M,3)`` the keyframe camera positions (optical world). ``path_px``
        is accepted for symmetry with the builder but unused here -- the window
        draws the path itself in WORLD metres (so it stays correct under pan/zoom),
        computed from ``cams`` + ``extent``.
        """
        self.plan_ready.emit(rgb, path_px, cams, extent)

    # ------------------------------------------------------------------ #
    def update(self, rgb: np.ndarray | None,
               path_px: np.ndarray | None,
               cams: np.ndarray | None,
               extent: FloorPlanExtent | None) -> None:
        """Replace the rendered raster + camera path with fresh data.

        ``rgb`` ``(H,W,3)`` uint8 is the occupancy raster placed in world metres
        via ``extent``; ``cams`` ``(M,3)`` are the keyframe camera positions in the
        optical world frame, drawn as the path polyline (+ a marker on the latest).
        Safe to call repeatedly from the GUI thread.
        """
        if rgb is None or extent is None or rgb.size <= 3:
            return                                    # empty plan -> nothing to draw
        img = np.asarray(rgb)
        self._img.setImage(img, autoLevels=False)
        # Place the image in WORLD metres: the image's (0,0) corner sits at
        # (x_min, z_min) and it spans width*cell x height*cell metres. setRect takes
        # (x, y, w, h); our rows index z (y-axis) and cols index x (x-axis).
        x_min, x_max, z_min, z_max = extent.world_extent()
        self._img.setRect(pg.QtCore.QRectF(
            x_min, z_min, x_max - x_min, z_max - z_min))

        # Camera path in WORLD metres (optical x, z) so it overlays the raster
        # exactly under any pan/zoom. Draw the polyline + a marker on the latest.
        if cams is not None and len(cams):
            c = np.asarray(cams, dtype=np.float64).reshape(-1, 3)
            self._path.setData(c[:, 0], c[:, 2])      # x (right), z (forward)
            self._head.setData([float(c[-1, 0])], [float(c[-1, 2])])
        else:
            self._path.setData([], [])
            self._head.setData([], [])

        if not self._framed:
            self._plot.autoRange()
            self._framed = True
