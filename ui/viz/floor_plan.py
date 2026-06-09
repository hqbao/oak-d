"""UI-only top-down FLOOR-PLAN builder (pure numpy, no GL / no Qt / no display).

The 3D map viewer (the SLAM landmark point cloud) is heavy GL on this Mac AND
hard to read (noisy marginal depth seen in perspective). This module builds a
LIGHT, READABLE alternative: a 2D top-down raster of the room's WALLS, with the
camera path drawn over it -- so the room LAYOUT is obvious at a glance. Because
the result is a 2D raster it renders as a cheap pyqtgraph ``ImageItem`` (no
``GLViewWidget``), and -- crucially -- it can be written to a PNG with pure
numpy/cv2 with NO GL/display, so the build is VISUALLY VERIFIABLE offscreen (the
GL viewers were not).

It is a pure CONSUMER of the SAME VIO keyframe feed the SLAM landmark map uses
(denoised ``depth_m`` + each keyframe's own VIO pose ``[R | t]``): no new topic,
no data-path change, no new dependency.

Frame / projection plane (which axis is the ground, which is height)
--------------------------------------------------------------------
The keyframe poses + the back-projected world points live in the CAMERA-OPTICAL
world frame (OpenCV optical: ``+x`` right, ``+y`` DOWN, ``+z`` forward), the same
frame the SLAM-map builder uses. The viewer's optical->NED map is
``_M_OPT_TO_NED = [[0,0,1],[1,0,0],[0,1,0]]``, whose Down row picks the optical
``+y`` axis -- so optical ``+y`` is world-DOWN (the VERTICAL axis) and the GROUND
plane is the optical ``(x, z)`` plane. The floor plan therefore:

* PROJECTS each world point onto ``(x, z)`` (drops the vertical ``y``), and
* uses ``y`` (vertical extent within a cell) to tell a WALL (a tall column of
  points spanning floor->ceiling) from the FLOOR (points at ~one height).

We render the WALLS, not the occupied-region boundary (why the room shape emerges)
----------------------------------------------------------------------------------
An earlier version outlined the BOUNDARY of the whole occupied region. But the
camera sits ~centre of the room and rotates, so its limited-range + noisy depth
fills a roughly CIRCULAR disc (floor + radial noise) around it; closing that disc
with morphology and taking its gradient yields a CIRCLE -- the SENSING HORIZON,
not the walls. So that approach drew the horizon, never the room's true shape.

This version renders the WALL SURFACES directly. The key physical fact: top-down,
a vertical WALL is a ground cell hit by depth pixels across its WHOLE HEIGHT --
i.e. a TALL COLUMN of points in one cell -- whereas the FLOOR is points at ~one
height (a near-zero column). So we score each ground cell by its VERTICAL EXTENT
(``max_h - min_h`` of the points that land in it) and keep only the cells whose
extent clears a wall-height threshold AND that have enough ray support. Those
surviving cells ARE the walls; we draw them DIRECTLY (a thin wall-cell mark per
cell), NOT the outline of the whole region. Consequences that make the room read:

* if the camera sensed the 4 walls, they appear as the lines/corners forming the
  room (a square reads square); if only part was sensed, partial walls show --
  HONEST about what was actually captured, never a synthesised closed loop.
* the flat floor (zero vertical extent) is excluded, so it cannot fill a disc.
* there is NO MORPH_CLOSE bridging across open space: nothing rounds the interior
  into a disc whose boundary becomes a circle. We keep only a tiny optional
  MORPH_OPEN + a connected-component area filter to kill isolated speckle.

Pipeline (all pure numpy + a few cheap 2D cv2 ops on the small raster):

1. WALL SCORE: a single ``np.add.at`` scatter accumulates per-cell count, min(y)
   and max(y); the per-cell vertical EXTENT is ``max(y) - min(y)``. A cell is a
   wall cell when ``extent >= WALL_EXTENT_M`` AND ``count >= MIN_CELL_COUNT``.
2. SPECKLE CLEANUP (:func:`_clean_wall_cells`): an optional tiny MORPH_OPEN scrubs
   1-cell specks, then ``cv2.connectedComponentsWithStats`` drops components below
   ``MIN_COMPONENT_CELLS``. NO close, NO gradient -- the wall cells stay as the
   thin marks they already are.
3. RENDER: the wall cells are drawn directly as a bright extent-tinted mark over a
   faint raw-occupancy context wash; the camera path overlays on the same grid.

So a rebuild is far cheaper than a 3D map build (it can run at a higher rate). No
Qt, no GL, no device, no comms.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Tunable build constants (the source binds these into its build; each is
# commented with which way to turn it).
# --------------------------------------------------------------------------- #
#: Ground-plane grid cell size (m). Each occupancy cell is CELL_M x CELL_M on the
#: optical ``(x, z)`` floor. ~8 cm is fine enough to resolve a wall as a thin
#: outline yet coarse enough that a handful of stray depth points don't speckle
#: the raster. LOWER for a finer (sharper but noisier / larger) plan; RAISE for a
#: coarser, smoother, lighter plan.
CELL_M = 0.08
#: Depth-map subsample stride: bin every ``STRIDE``-th pixel in u and v. The plan
#: only needs the room SHAPE, not every pixel, so a stride of 4 (1/16 the points)
#: keeps the build cheap while the walls/outline survive. LOWER (toward 1) for a
#: denser, heavier plan; RAISE for a lighter, sparser one.
STRIDE = 4
#: Valid-depth band (m) for the floor plan. ``MIN`` matches the SLAM-map builder
#: (below it stereo is unreliable). ``MAX`` must be large enough that depth
#: actually LANDS ON THE WALLS: with a tight 2.5 m cap, depth in a room wider than
#: ~5 m never reaches the far walls -- it only fills the near floor + radial noise,
#: so the plan shows a near disc, never the room. A WALL needs depth ON it to score
#: any vertical extent, so the cap is opened to ~4.0 m. The trade-off the other way:
#: far stereo range error grows ~range^2 and SPRAYS points radially ("starburst"),
#: which the wall-EXTENT score + the MIN_CELL_COUNT support gate + the speckle
#: cleanup tolerate far better than the old region-outline did (radial spray is
#: mostly low-extent floor-ish noise, gated out). Empirically on the gold sessions
#: (lab_loop_30s, corridor_60s) 4.0 m reveals the most genuine wall/corner structure
#: with acceptable noise: 2.5 m leaves only the near floor disc (no far wall), while
#: 5.0 m starts re-adding radial fan without revealing new real wall. RAISE for more
#: reach in a larger room (more fan); LOWER for an even crisper but shorter-range plan.
MIN_DEPTH_M = 0.3
MAX_DEPTH_M = 4.0
#: Edge-reject threshold (m): drop "flying pixels" on a depth discontinuity (a
#: foreground/background edge back-projects to points floating BETWEEN the two
#: surfaces, which would smear the plan). A pixel is kept only if BOTH its
#: vertical and horizontal depth gradient are <= this. 0 disables the reject.
#: SAME idea as the shared geometry back-projection edge-reject.
EDGE_MAX_M = 0.1
#: Cap on the grid's larger side (cells). A runaway extent (a diverged pose
#: shooting a point far away) must not allocate a giant raster; clamp the longer
#: axis to this many cells (the build then drops out-of-grid points). At
#: CELL_M=0.08 this is ~80 m across -- far beyond any indoor room.
MAX_GRID_CELLS = 1024
#: Percentile the score raster is normalised against (instead of the raw max) so a
#: single very dense cell can't wash the whole plan out to one faint tone. The
#: top few percent of cells saturate to full brightness; the rest spread across
#: the ramp. RAISE toward 100 to use the true max (more contrast lost to outliers).
SCORE_CLIP_PCT = 99.0
#: Minimum points (RAY SUPPORT) a ground cell must collect to be drawn as a wall
#: cell (else it is treated as empty). This is the floor-plan analogue of
#: ``voxel_downsample``'s ``min_count``: raw stereo depth at far/grazing range
#: SPRAYS thin radial noise outward from each camera (few points per cell, spread
#: across many cells), whereas a real wall surface is hit by MANY rays across
#: keyframes so its cells are dense. Requiring this much support (alongside the
#: wall-extent gate) removes the worst of the radial "starburst" noise; the speckle
#: cleanup below then drops the rest. RAISE for a cleaner (sparser, more holes)
#: plan; LOWER to keep fainter structure (noisier).
MIN_CELL_COUNT = 3

# --------------------------------------------------------------------------- #
# Wall-cell tunables (the "wall = vertical extent" detector + a LIGHT 2D speckle
# cleanup). We render the WALL CELLS DIRECTLY (the thin marks where tall columns
# of points stand), NOT the outline of the occupied region -- so the room's true
# shape (square if the 4 walls were sensed) emerges instead of a sensing-horizon
# circle. The cleanup is deliberately minimal: a tiny optional OPEN + a connected-
# component area filter to kill isolated speckle, and explicitly NO MORPH_CLOSE
# (it bridges across open space and rounds the shape into a disc) and NO gradient
# outline. Each knob is exposed + commented with which way to turn it.
# --------------------------------------------------------------------------- #
#: WALL vertical-extent threshold (m): a ground cell is a WALL cell only if the
#: points in it span AT LEAST this much in height (``max_h - min_h``). This is the
#: core "wall = high vertical extent" detector: a vertical wall is hit by depth
#: pixels across its whole height -> a TALL column in one cell, whereas an ideal
#: FLOOR is points at ~one height (a near-zero column) and is excluded -- so on
#: CLEAN depth the bright cells mark WHERE THE WALLS ARE, not the swept floor disc.
#:
#: HONEST tuning note (gold OAK-D stereo): on the current gold sessions the fused
#: per-cell vertical span is large at NEARLY EVERY (x,z) cell (median ~3 m -- see
#: the functional check), because noisy/flying stereo points + real floor/ceiling/
#: clutter fill the column across the whole swept disc. So NO single ``max-min``
#: threshold here cleanly isolates walls: a low value lights the whole disc, a high
#: value selects the densest-NOISE blobs (not straight walls). ~2.0 m is chosen as a
#: middle ground that is at least SELECTIVE (drops the lowest-extent cells) without
#: collapsing to a few noise blobs; on cleaner per-pixel depth (the real VL53 ToF
#: target) this same gate WILL resolve walls. RAISE to demand taller columns (fewer,
#: only the tallest); LOWER toward 0 to keep more (the disc fills back in).
WALL_EXTENT_M = 2.0
#: cv2 MORPH_OPEN kernel size (square, odd, px==cells) applied to the wall-cell
#: mask. An OPEN (erode then dilate) deletes any wall blob THINNER than the kernel
#: -- isolated 1-cell specks of noise -- while leaving connected wall runs intact.
#: Kept SMALL (2) on purpose: a wall, top-down, is itself only ~1-2 cells thick, so
#: a large open would erode the very walls we want to show. RAISE only to scrub
#: heavier speckle (risks thinning real walls); 0 disables the open (rely on the
#: component filter alone).
MORPH_OPEN_PX = 2
#: Minimum connected-component area (cells) a run of wall cells must have to survive
#: (``cv2.connectedComponentsWithStats`` 8-connectivity). The residual far-range
#: noise is a scatter of tiny isolated blobs; a real wall is a longer connected run.
#: Dropping every component under this area vanishes the isolated noise specks but
#: keeps genuine wall segments. Kept MODEST (8 cells) because we want PARTIAL walls
#: to survive (honest about what was sensed), not only a fully-closed loop. RAISE
#: for a cleaner (sparser) plan that may drop short real segments; LOWER to keep
#: smaller fragments (noisier). ~8 cells (~0.05 m^2 at CELL_M=0.08) keeps short wall
#: runs, drops single/double-cell specks.
MIN_COMPONENT_CELLS = 8
#: Brightness [0,1] the faint RAW occupancy context is drawn at UNDER the bright
#: wall cells. The wall cells are the bright structure; the raw occupancy (every
#: cell with enough support, walls + floor) is kept very dim beneath them purely for
#: spatial context (so the room isn't just a skeleton on black -- you can see the
#: swept floor area faintly). 0 -> pure wall cells on background (maximally crisp,
#: no context); RAISE toward 1 to show more of the raw cloud (more context, more
#: fuzz). ~0.18 is a faint hint that doesn't compete with the wall cells.
RAW_CONTEXT_GAIN = 0.18


class FloorPlanExtent:
    """World<->pixel mapping for a built floor-plan raster (optical (x, z) plane).

    The raster rows index the optical ``z`` (forward) axis and columns index the
    optical ``x`` (right) axis; ``(x_min, z_min)`` is the world coordinate of the
    raster's ``(col=0, row=0)`` corner and ``cell_m`` the metres-per-cell. This is
    everything a window needs to place the ``ImageItem`` in world metres (so pan/
    zoom read in metres) and to map the camera path onto the SAME pixels.

    Stored as a tiny POD (no numpy state) so it is trivially picklable / loggable
    and the window can position the image with ``setRect``.
    """

    __slots__ = ("x_min", "z_min", "cell_m", "width", "height")

    def __init__(self, x_min: float, z_min: float, cell_m: float,
                 width: int, height: int) -> None:
        self.x_min = float(x_min)
        self.z_min = float(z_min)
        self.cell_m = float(cell_m)
        self.width = int(width)        # raster columns (along optical x)
        self.height = int(height)      # raster rows    (along optical z)

    # ------------------------------------------------------------------ #
    def world_xz_to_px(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray,
                                                                    np.ndarray]:
        """Optical ``(x, z)`` world metres -> fractional raster ``(col, row)``.

        Vectorised; the caller rounds/clips as needed. Used by the window to draw
        the camera path on the SAME pixel grid as the occupancy raster.
        """
        col = (np.asarray(x, np.float64) - self.x_min) / self.cell_m
        row = (np.asarray(z, np.float64) - self.z_min) / self.cell_m
        return col, row

    def world_extent(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, z_min, z_max)`` world bounds of the raster (metres)."""
        return (self.x_min, self.x_min + self.width * self.cell_m,
                self.z_min, self.z_min + self.height * self.cell_m)


# --------------------------------------------------------------------------- #
def keyframes_to_ground_points(depths, Rs, ts, K, *,
                               stride: int = STRIDE,
                               min_depth: float = MIN_DEPTH_M,
                               max_depth: float = MAX_DEPTH_M,
                               edge_max: float = EDGE_MAX_M):
    """Back-project keyframe depth maps to world points, gated + strided.

    Mirrors the SLAM-map builder's geometry: each keyframe's depth is
    back-projected with the pinhole to its camera frame and transformed to the
    camera-optical WORLD frame by the keyframe's OWN VIO pose ``Xw = R Xc + t``
    (one pose source per keyframe -> seam-free). The depth grid is subsampled by
    ``stride`` and gated to ``[min_depth, max_depth]``; when ``edge_max > 0`` a
    pixel on a depth discontinuity (a "flying pixel") is also dropped.

    * ``depths`` -- list of ``(H,W)`` metric depth maps.
    * ``Rs`` / ``ts`` -- per-keyframe ``(3,3)`` rotation / ``(3,)`` translation.
    * ``K`` -- ``(3,3)`` intrinsic for the full-res depth grid.

    Returns ``(N,3)`` float32 world points in the optical frame (empty when none
    valid). Pure numpy; each keyframe is back-projected VECTORISED (no per-pixel
    loop), then all keyframes are stacked.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    s = max(1, int(stride))
    parts: list[np.ndarray] = []
    for i in range(len(depths)):
        d = np.asarray(depths[i], dtype=np.float32)
        if d.ndim != 2:
            continue
        h, w = d.shape
        # Per-pixel validity over the FULL grid first, so the edge gradient is
        # computed on the native resolution (a discontinuity between adjacent
        # full-res pixels), THEN subsampled by stride -- matching the dense
        # geometry helper's edge reject.
        m = np.isfinite(d) & (d >= float(min_depth)) & (d <= float(max_depth))
        if edge_max > 0.0:
            # Drop flying pixels: a foreground/background edge interpolates to
            # points floating between the two surfaces. ``append`` keeps the diff
            # the same shape as the grid (the last row/col compares to itself).
            dv = np.abs(np.diff(d, axis=0, append=d[-1:]))
            dh = np.abs(np.diff(d, axis=1, append=d[:, -1:]))
            m &= (dv <= float(edge_max)) & (dh <= float(edge_max))
        keep = m[::s, ::s]
        if not np.any(keep):
            continue
        # Pixel coordinates of the kept, subsampled grid.
        us = np.arange(0, w, s, dtype=np.float64)
        vs = np.arange(0, h, s, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs)                      # (r0, c0)
        z = d[::s, ::s].astype(np.float64)
        uu, vv, z = uu[keep], vv[keep], z[keep]           # flat (M,)
        # Pinhole back-projection to the camera frame, then to the world by the
        # keyframe's OWN pose.
        cam = np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, z], axis=1)
        R = np.asarray(Rs[i], dtype=np.float64).reshape(3, 3)
        t = np.asarray(ts[i], dtype=np.float64).reshape(3)
        parts.append((cam @ R.T + t).astype(np.float32))
    if not parts:
        return np.zeros((0, 3), np.float32)
    return np.concatenate(parts, axis=0)


def _compose_plan(context: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    """Compose the final floor-plan RGB from a faint context + the bright wall cells.

    Two layers (pure numpy, no matplotlib):

    * ``context`` ``(H,W)`` in [0,1] -- the RAW occupancy (already attenuated by
      :data:`RAW_CONTEXT_GAIN`), drawn as a very dim dark-navy->blue wash so the
      room has spatial context (it is NOT the structure -- it stays faint).
    * ``wall_mask`` ``(H,W)`` boolean -- the WALL CELLS (cells whose points span a
      tall vertical column, after the tiny OPEN + connected-component speckle
      filter), drawn as bright cyan-white marks that OVERWRITE the context wherever
      a wall is. These thin marks ARE the walls (drawn directly, not an outline of
      the occupied region) -- the crisp reading the eye locks onto.

    Returns ``(H,W,3)`` uint8. Background (no context, no wall) is the deep navy
    the window's dark theme expects, so the plan reads as a "lit room" rather than
    holes in black.
    """
    c = np.clip(np.asarray(context, np.float64), 0.0, 1.0)
    # Faint context wash: a dim navy->blue ramp. Kept low-contrast on purpose so it
    # never competes with the wall cells -- it is only a spatial hint.
    r = 0.05 + 0.10 * c
    g = 0.08 + 0.18 * c
    b = 0.20 + 0.45 * c
    rgb = np.stack([r, g, b], axis=-1)
    # Bright wall cells: a cyan-white that OVERWRITES the context where a wall cell
    # is (a hard write, not a blend, so the wall marks stay sharp-edged).
    wm = np.asarray(wall_mask, dtype=bool)
    rgb[wm] = (0.85, 0.97, 1.0)
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _clean_wall_cells(wall_cells: np.ndarray, *,
                      open_px: int = MORPH_OPEN_PX,
                      min_component_cells: int = MIN_COMPONENT_CELLS
                      ) -> np.ndarray:
    """Wall-cell mask ``(H,W)`` -> the same cells with isolated speckle removed.

    ``wall_cells`` is the raw "wall = tall vertical column" mask (cells whose points
    span ``>= WALL_EXTENT_M`` in height AND have enough ray support -- see
    :func:`build_floor_plan`). These cells ARE the walls already; we DO NOT outline
    a region or close gaps -- doing so re-creates a sensing-horizon disc/circle.
    We only scrub isolated noise specks in two cheap 2D passes (microseconds on the
    small raster):

    1. cv2 MORPH_OPEN (erode then dilate, small ``open_px`` kernel): delete any wall
       blob THINNER than the kernel -- isolated 1-cell specks -- while leaving
       connected wall runs intact. Kept small (a wall is itself ~1-2 cells thick), so
       a big open would erode the very walls we want; ``open_px < 2`` skips it.
    2. cv2.connectedComponentsWithStats (8-connectivity): drop every component
       smaller than ``min_component_cells`` -- the residual far-range noise is a
       scatter of tiny blobs, a real wall is a longer connected run. NO close, NO
       gradient: the wall cells stay the thin marks they are.

    Returns a boolean ``(H,W)`` mask (the cleaned wall cells). Every step is a no-op
    when its knob is 0 / the mask is empty, so the pipeline degrades gracefully.
    """
    import cv2

    h, w = wall_cells.shape
    mask = np.ascontiguousarray(wall_cells, dtype=np.uint8)  # 0/1, cv2 wants uint8
    if not mask.any():
        return np.zeros((h, w), dtype=bool)

    # (1) Tiny MORPH_OPEN to scrub isolated specks (NO close -- closing would bridge
    # across the open room interior and round the walls into a disc).
    if open_px and open_px >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_px),) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if not mask.any():
        return np.zeros((h, w), dtype=bool)

    # (2) Connected-component area filter: drop the small isolated noise blobs, keep
    # the longer connected wall runs. Label 0 is the background; stats[:,AREA] is the
    # cell count per label. Modest threshold so PARTIAL walls survive (honest).
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    keep = np.zeros((h, w), dtype=np.uint8)
    min_area = max(1, int(min_component_cells))
    for lab in range(1, n_labels):                     # skip background (0)
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == lab] = 1
    return keep.astype(bool)


def build_floor_plan(points: np.ndarray, *,
                     cell_m: float = CELL_M,
                     score_clip_pct: float = SCORE_CLIP_PCT,
                     min_cell_count: int = MIN_CELL_COUNT,
                     max_grid_cells: int = MAX_GRID_CELLS,
                     wall_extent_m: float = WALL_EXTENT_M,
                     open_px: int = MORPH_OPEN_PX,
                     min_component_cells: int = MIN_COMPONENT_CELLS,
                     raw_context_gain: float = RAW_CONTEXT_GAIN):
    """Bin world points onto the ground plane -> a WALL-CELL raster (not an outline).

    ``points`` are ``(N,3)`` world points in the camera-optical frame (from
    :func:`keyframes_to_ground_points`). They are projected onto the GROUND plane
    by DROPPING the vertical optical ``+y`` (down) axis -- so the plan uses optical
    ``x`` (right, raster columns) and ``z`` (forward, raster rows).

    We render the WALLS directly, NOT the boundary of the occupied region (which is
    just the camera's circular sensing horizon -- see the module docstring). The
    per-cell scatter feeds two products:

    * WALL CELLS. A cell is a WALL cell when its points span ``>= wall_extent_m``
      VERTICALLY (``max_h - min_h`` -- the core "wall = high vertical extent"
      detector: a vertical wall is a tall column of points, the flat floor is a
      near-zero column and is excluded) AND it is hit by ``>= min_cell_count`` rays
      (drops thin radial noise). Those cells ARE the walls; they are drawn DIRECTLY
      (a thin mark per cell) after a LIGHT speckle cleanup by
      :func:`_clean_wall_cells` (a tiny MORPH_OPEN to drop 1-cell specks +
      connectedComponentsWithStats to drop isolated noise blobs -- NO close, NO
      gradient, so the room's true shape emerges instead of a disc/circle).
    * faint RAW occupancy CONTEXT -- the per-cell point count (support-gated),
      normalised + attenuated by ``raw_context_gain``, drawn dim UNDER the wall cells
      purely for spatial context (so the plan isn't a skeleton on black).

    :func:`_compose_plan` draws the dim context wash + overwrites it with the bright
    wall cells. Returns ``(rgb (H,W,3) uint8, extent FloorPlanExtent)``; with no
    points a 1x1 black raster + a degenerate extent is returned.

    Implementation: a single scatter accumulates per-cell count, MIN(y) and MAX(y)
    (via ``np.minimum.at`` / ``np.maximum.at``), so the per-cell vertical extent is
    the TRUE span ``max_y - min_y`` -- computed without any Python loop over cells.
    The cleanup is one tiny cv2 morphology + one connected-component pass on the
    small raster. Pure numpy + cv2, O(N) + O(cells).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return (np.zeros((1, 1, 3), np.uint8),
                FloorPlanExtent(0.0, 0.0, float(cell_m), 1, 1))

    x = pts[:, 0]        # optical right  -> raster columns
    y = pts[:, 1]        # optical DOWN   -> vertical (height) axis, NOT binned
    z = pts[:, 2]        # optical forward-> raster rows
    cell = float(cell_m)

    # Grid bounds from the point cloud's (x, z) extent, padded one cell so the
    # extreme points fall strictly inside the raster.
    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())
    width = int(np.floor((x_max - x_min) / cell)) + 1
    height = int(np.floor((z_max - z_min) / cell)) + 1
    width = max(1, min(width, int(max_grid_cells)))
    height = max(1, min(height, int(max_grid_cells)))

    # Cell index per point; clip so any point beyond the (clamped) grid lands on
    # the border cell rather than indexing out of bounds.
    col = np.clip(((x - x_min) / cell).astype(np.int64), 0, width - 1)
    row = np.clip(((z - z_min) / cell).astype(np.int64), 0, height - 1)
    flat = row * width + col                                  # row-major cell id
    ncells = width * height

    # Scatter-accumulate per cell: count, MIN(y) and MAX(y). One pass, no per-cell
    # loop. The per-cell vertical EXTENT is the TRUE span max(y) - min(y) -- a wall
    # spans floor->ceiling (large), the flat floor spans ~none. (min/max is a more
    # faithful "does this cell hold a tall column" test than a std proxy: a wall's
    # full height is exactly max-min, which a std would under-report.)
    count = np.zeros(ncells, np.float64)
    min_y = np.full(ncells, np.inf, np.float64)
    max_y = np.full(ncells, -np.inf, np.float64)
    np.add.at(count, flat, 1.0)
    np.minimum.at(min_y, flat, y)
    np.maximum.at(max_y, flat, y)
    nz = count > 0
    extent_m = np.zeros(ncells, np.float64)
    extent_m[nz] = max_y[nz] - min_y[nz]                     # true vertical span

    # WALL CELLS (binary): a cell is a wall cell when it has a real vertical column
    # AND enough ray support.
    #  * ``extent_m >= wall_extent_m`` is the core "wall = high vertical extent"
    #    detector: a flat-floor cell (points at ~one height) is excluded, so the
    #    bright structure marks WHERE THE WALLS ARE, not the swept floor disc.
    #  * ``count >= min_cell_count`` drops the thin radial stereo-noise floor (a real
    #    wall is hit by many rays across keyframes; noise sprays thin).
    enough = count >= float(min_cell_count)
    tall = extent_m >= float(wall_extent_m)
    wall_cells = (enough & tall).reshape(height, width)

    # WALL CELLS, cleaned: scrub isolated speckle ONLY (tiny MORPH_OPEN + connected-
    # component area filter). NO close / NO gradient -- the wall cells stay the thin
    # marks they are, so the room's true shape (square if sensed) reads instead of a
    # sensing-horizon circle.
    wall_mask = _clean_wall_cells(
        wall_cells, open_px=open_px, min_component_cells=min_component_cells)

    # FAINT raw-occupancy CONTEXT: the per-cell point count (the thin-noise floor
    # removed), normalised to a high percentile (so one dense cell can't wash it
    # out) and attenuated -- a dim spatial hint UNDER the wall cells.
    ctx = count.copy()
    ctx[~enough] = 0.0
    pos = ctx[ctx > 0]
    if pos.size:
        hi = float(np.percentile(pos, float(score_clip_pct)))
    else:
        hi = 1.0
    hi = hi if hi > 1e-9 else 1.0
    context = (np.clip(ctx / hi, 0.0, 1.0) * float(raw_context_gain)
               ).reshape(height, width)

    rgb = _compose_plan(context, wall_mask)
    extent = FloorPlanExtent(x_min, z_min, cell, width, height)
    return rgb, extent


def floor_plan_with_path(points: np.ndarray, cams: np.ndarray, *,
                         cell_m: float = CELL_M,
                         min_cell_count: int = MIN_CELL_COUNT,
                         wall_extent_m: float = WALL_EXTENT_M,
                         open_px: int = MORPH_OPEN_PX,
                         min_component_cells: int = MIN_COMPONENT_CELLS,
                         raw_context_gain: float = RAW_CONTEXT_GAIN):
    """Convenience: build the raster AND project the camera path onto its pixels.

    Returns ``(rgb (H,W,3) uint8, path_px (M,2) float32, extent)`` where
    ``path_px`` is the keyframe camera positions ``cams`` (``(M,3)`` optical-world)
    projected to fractional raster ``(col, row)`` on the SAME grid as the raster --
    so a caller (the offscreen PNG verifier, a test) can overlay the path without
    re-deriving the extent. The window draws the path itself in world metres via
    the returned ``extent`` (see :class:`FloorPlanExtent`); this helper is mainly
    for the headless PNG check + the unit tests. The wall-detector + cleanup knobs
    are forwarded to :func:`build_floor_plan` so a caller can tune the crispness.
    """
    rgb, extent = build_floor_plan(
        points, cell_m=cell_m, min_cell_count=min_cell_count,
        wall_extent_m=wall_extent_m, open_px=open_px,
        min_component_cells=min_component_cells,
        raw_context_gain=raw_context_gain)
    cams = np.asarray(cams, dtype=np.float64).reshape(-1, 3)
    if cams.shape[0] == 0:
        return rgb, np.zeros((0, 2), np.float32), extent
    col, row = extent.world_xz_to_px(cams[:, 0], cams[:, 2])
    path_px = np.stack([col, row], axis=1).astype(np.float32)
    return rgb, path_px, extent
