"""UI-only top-down FLOOR-PLAN builder (pure numpy, no GL / no Qt / no display).

The 3D map viewer (the SLAM landmark point cloud) is heavy GL on this Mac AND
hard to read (noisy marginal depth seen in perspective). This module builds a
LIGHT, READABLE alternative: a 2D top-down OCCUPANCY raster of the room -- the
walls/vertical structure read as a top-down OUTLINE, with the camera path drawn
over it -- so the room LAYOUT is obvious at a glance. Because the result is a 2D
raster it renders as a cheap pyqtgraph ``ImageItem`` (no ``GLViewWidget``), and --
crucially -- it can be written to a PNG with pure numpy/cv2 with NO GL/display, so
the build is VISUALLY VERIFIABLE offscreen (the GL viewers were not).

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

Wall emphasis + cleanup (why walls read as a CRISP outline)
-----------------------------------------------------------
Raw top-down occupancy is a fuzzy blue cloud: a vertical wall does read brighter
(it is hit by depth pixels across its whole height -> a tall column of points in
one ground cell), but far/grazing stereo SPRAYS thin radial "star-burst" streaks
that smear it. The key realisation is that, top-down, a WALL reads as the BOUNDARY
between occupied space and free space -- not as an interior bright blob -- so we
build a clean SOLID occupied-region mask and take its OUTLINE. Three stages:

1. OCCUPIED REGION: a cell joins the region only if it is hit by enough rays
   (``>= MIN_CELL_COUNT`` -- drops the thin radial noise) AND its points span a
   real vertical column (``extent_m >= FLOOR_EXTENT_M`` -- the explicit "wall =
   vertical extent" gate that drops the flat floor), so the region hugs the
   VERTICAL structure (walls / furniture), not the swept floor.
2. CLEANUP (pure 2D + cv2, :func:`_clean_wall_mask`): ``cv2.morphologyEx``
   MORPH_OPEN deletes the thin radial streaks, MORPH_CLOSE bridges depth-dropout
   gaps so the room is one connected area, and ``cv2.connectedComponentsWithStats``
   drops the small isolated star-burst islands (keeping the large room region).
3. OUTLINE: the cleaned region's morphological GRADIENT (``MORPH_GRADIENT``) is a
   crisp 1-2 cell boundary LINE = the top-down wall, drawn as a bright outline over
   a faint raw-occupancy context wash.

Everything here is pure numpy + a single ``np.add.at`` histogram + a few cheap 2D
cv2 ops on the small raster, so a rebuild is far cheaper than a 3D map build (it
can run at a higher rate). No Qt, no GL, no device, no comms.
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
#: (below it stereo is unreliable). ``MAX`` is deliberately TIGHTER than the
#: SLAM-map builder's 6.0 m: a dense per-pixel occupancy plan seen
#: top-down is dominated by the FAR depth, where stereo range error grows ~with
#: range^2 and SPRAYS points radially along each viewing ray -- a "starburst" fan
#: from every camera that smears the walls. Those fans are the single biggest
#: readability killer top-down, so the plan uses only the RELIABLE near band
#: (~2.5 m), which both gold sessions read far more clearly at (verified by the
#: saved PNGs). The sparse SLAM landmark map can afford 6 m because it keeps only
#: PnP-inlier landmarks; the dense plan cannot. RAISE for more reach in a big room
#: (at the cost of more radial fan); LOWER for an even crisper near outline.
MIN_DEPTH_M = 0.3
MAX_DEPTH_M = 2.5
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
#: Minimum points a ground cell must collect to be drawn (else it is treated as
#: empty). This is the floor-plan analogue of ``voxel_downsample``'s ``min_count``:
#: raw stereo depth at far/grazing range SPRAYS thin radial noise outward from each
#: camera (few points per cell, spread across many cells), whereas a real surface
#: is hit by MANY rays across keyframes so its cells are dense. Dropping cells
#: under this count removes the worst of the radial "starburst" noise; the
#: morphology + connected-component pass below then cleans the rest. RAISE for a
#: cleaner (sparser, more holes) plan; LOWER to keep fainter structure (noisier).
MIN_CELL_COUNT = 3

# --------------------------------------------------------------------------- #
# Wall-mask cleanup tunables (the 2D + cv2 pass that turns the fuzzy occupancy
# cloud into a CRISP wall outline). The key realisation: top-down, a "wall" reads
# as the BOUNDARY between occupied space and free space, NOT as an interior bright
# blob -- so we build a clean SOLID occupied-region mask (drop flat floor by
# vertical extent, drop sparse radial noise by count, morphology to scrub thin
# streaks + fill gaps, connected-component to drop islands) and then take its
# morphological-GRADIENT OUTLINE as the crisp wall line. All cheap ops on the
# small raster. The user explicitly accepts the over-clean risk, so each knob is
# exposed + commented with which way to turn it.
# --------------------------------------------------------------------------- #
#: Vertical-extent floor gate (m): a ground cell whose points span LESS than this
#: in height is treated as FLAT FLOOR and dropped from the occupied region (only
#: cells with a real vertical column -- walls / furniture / structure -- survive).
#: This is the explicit "wall = vertical extent" emphasis: it removes the flat
#: floor so the occupied region (and thus its outline) hugs the vertical structure
#: rather than the swept floor. RAISE to demand taller columns (keeps only clear
#: walls, may drop low structure); LOWER toward 0 to keep more (the floor creeps
#: back in). ~0.5 m cleanly separates a swept-floor cell from a wall column.
FLOOR_EXTENT_M = 0.5
#: cv2 MORPH_OPEN kernel size (square, odd, px==cells). An OPEN (erode then dilate)
#: deletes any occupied blob THINNER than the kernel -- i.e. the 1-2 cell wide
#: radial streaks fanning off each camera -- while leaving thicker, solid regions
#: intact. 3 removes single/double-cell streaks; RAISE (5) to scrub heavier noise
#: (risks eroding thin real structure); 0 disables the open.
MORPH_OPEN_PX = 3
#: cv2 MORPH_CLOSE kernel size (square, odd). A CLOSE (dilate then erode) bridges
#: small gaps WITHIN the occupied region (a few missing cells where depth dropped
#: out) so the room reads as ONE connected area with a continuous boundary, not a
#: dashed/holey one. Run AFTER the open so it doesn't re-grow the streaks the open
#: removed. RAISE to bridge bigger gaps (risks fusing across a doorway / to nearby
#: noise); 0 disables the close. ~5 closes the typical few-cell depth dropouts.
MORPH_CLOSE_PX = 5
#: Minimum connected-component area (cells) an occupied region must have to survive
#: (``cv2.connectedComponentsWithStats`` 8-connectivity). After the open, the
#: residual star-burst is a scatter of small isolated blobs; the real room is one
#: large connected region. Dropping every component under this area vanishes the
#: isolated noise islands but keeps the room. RAISE for an even cleaner plan (risks
#: dropping a small detached real structure); LOWER to keep smaller fragments
#: (noisier). ~40 cells (~0.26 m^2 at CELL_M=0.08) keeps real runs, drops specks.
MIN_COMPONENT_CELLS = 40
#: Draw the wall as the OUTLINE of the occupied region (its morphological gradient
#: = a crisp 1-2 cell boundary line) rather than as the FILLED region. True is the
#: floor-plan reading the eye expects (a thin wall line tracing the room); set
#: False to render the solid occupied region instead (useful for debugging the
#: cleanup, or if you prefer a filled footprint).
WALL_OUTLINE = True
#: Brightness [0,1] the faint RAW occupancy context is drawn at UNDER the crisp
#: wall outline. The cleaned wall outline is the bright line; the raw occupancy is
#: kept very dim beneath it purely for spatial context (so the room isn't just a
#: skeleton on black). 0 -> pure outline on background (maximally crisp, no
#: context); RAISE toward 1 to show more of the raw cloud (more context, more
#: fuzz). ~0.18 is a faint hint that doesn't compete with the outline.
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
    """Compose the final floor-plan RGB from a faint context + a crisp wall mask.

    Two layers (pure numpy, no matplotlib):

    * ``context`` ``(H,W)`` in [0,1] -- the RAW occupancy (already attenuated by
      :data:`RAW_CONTEXT_GAIN`), drawn as a very dim dark-navy->blue wash so the
      room has spatial context (it is NOT the structure -- it stays faint).
    * ``wall_mask`` ``(H,W)`` boolean -- the CLEANED wall cells (after threshold +
      morphology + connected-component filter), drawn as a bright cyan-white
      OUTLINE that overwrites the context wherever a wall is. This is the crisp
      reading the eye locks onto.

    Returns ``(H,W,3)`` uint8. Background (no context, no wall) is the deep navy
    the window's dark theme expects, so the plan reads as a "lit room" outline
    rather than holes in black.
    """
    c = np.clip(np.asarray(context, np.float64), 0.0, 1.0)
    # Faint context wash: a dim navy->blue ramp. Kept low-contrast on purpose so it
    # never competes with the wall outline -- it is only a spatial hint.
    r = 0.05 + 0.10 * c
    g = 0.08 + 0.18 * c
    b = 0.20 + 0.45 * c
    rgb = np.stack([r, g, b], axis=-1)
    # Crisp wall outline: a bright cyan-white that OVERWRITES the context where a
    # wall cell is (a hard write, not a blend, so the outline stays sharp-edged).
    wm = np.asarray(wall_mask, dtype=bool)
    rgb[wm] = (0.85, 0.97, 1.0)
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _clean_wall_mask(occupied: np.ndarray, *,
                     open_px: int = MORPH_OPEN_PX,
                     close_px: int = MORPH_CLOSE_PX,
                     min_component_cells: int = MIN_COMPONENT_CELLS,
                     outline: bool = WALL_OUTLINE) -> np.ndarray:
    """Binary occupied-region mask ``(H,W)`` -> a CRISP boolean WALL mask via cv2.

    ``occupied`` is the wall-candidate region (cells that are NOT flat floor and
    are hit by enough rays -- see :func:`build_floor_plan`). Top-down, a wall reads
    as the BOUNDARY of this occupied region, not its interior; this cleans the
    region and extracts that boundary in four cheap 2D passes (all on the small
    raster -- microseconds):

    1. cv2 MORPH_OPEN (erode then dilate, ``open_px`` kernel): delete any occupied
       blob thinner than the kernel -- the 1-2 cell radial streaks fanning off each
       camera -- while leaving thicker solid regions intact.
    2. cv2 MORPH_CLOSE (dilate then erode, ``close_px``): bridge small gaps WITHIN
       the region (depth dropouts) so the room is ONE connected area with a
       continuous boundary. Run AFTER the open so it doesn't re-grow the streaks.
    3. cv2.connectedComponentsWithStats (8-connectivity): drop every component
       smaller than ``min_component_cells`` -- the residual star-burst is a scatter
       of small isolated blobs; the real room is one large connected region.
    4. If ``outline``: the cleaned region's morphological GRADIENT
       (``MORPH_GRADIENT`` = dilate - erode) is a crisp 1-2 cell boundary LINE = the
       wall. Else return the filled region (debug / footprint).

    Returns a boolean ``(H,W)`` mask (the wall cells). Every step is a no-op when
    its knob is 0 / the mask is empty, so the pipeline degrades gracefully.
    """
    import cv2

    h, w = occupied.shape
    mask = np.ascontiguousarray(occupied, dtype=np.uint8)   # 0/1, cv2 wants uint8
    if not mask.any():
        return np.zeros((h, w), dtype=bool)

    # (1) MORPH_OPEN (kill thin radial streaks) then (2) MORPH_CLOSE (bridge gaps).
    if open_px and open_px >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_px),) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_px and close_px >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_px),) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if not mask.any():
        return np.zeros((h, w), dtype=bool)

    # (3) Connected-component area filter: drop the small isolated star-burst blobs,
    # keep the large connected room region. Label 0 is the background; stats[:,AREA]
    # is the cell count per label.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    keep = np.zeros((h, w), dtype=np.uint8)
    min_area = max(1, int(min_component_cells))
    for lab in range(1, n_labels):                     # skip background (0)
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == lab] = 1
    if not keep.any():
        return np.zeros((h, w), dtype=bool)

    # (4) Outline = morphological gradient (dilate - erode) of the cleaned region:
    # a crisp 1-2 cell boundary line = the top-down wall. Else the filled region.
    if outline:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        keep = cv2.morphologyEx(keep, cv2.MORPH_GRADIENT, k)
    return keep.astype(bool)


def build_floor_plan(points: np.ndarray, *,
                     cell_m: float = CELL_M,
                     score_clip_pct: float = SCORE_CLIP_PCT,
                     min_cell_count: int = MIN_CELL_COUNT,
                     max_grid_cells: int = MAX_GRID_CELLS,
                     floor_extent_m: float = FLOOR_EXTENT_M,
                     open_px: int = MORPH_OPEN_PX,
                     close_px: int = MORPH_CLOSE_PX,
                     min_component_cells: int = MIN_COMPONENT_CELLS,
                     outline: bool = WALL_OUTLINE,
                     raw_context_gain: float = RAW_CONTEXT_GAIN):
    """Bin world points onto the ground plane -> a CRISP wall-outline raster.

    ``points`` are ``(N,3)`` world points in the camera-optical frame (from
    :func:`keyframes_to_ground_points`). They are projected onto the GROUND plane
    by DROPPING the vertical optical ``+y`` (down) axis -- so the plan uses optical
    ``x`` (right, raster columns) and ``z`` (forward, raster rows).

    Pipeline (the per-cell scatter feeds two products):

    * OCCUPIED REGION -> WALL OUTLINE. A cell is part of the occupied region when it
      is hit by ``>= min_cell_count`` rays (drops thin radial noise) AND its points
      span ``>= floor_extent_m`` vertically (the explicit "wall = vertical extent"
      gate: a flat-floor cell -- points at ~one height -- is dropped, so the region
      hugs the VERTICAL structure, not the swept floor). That binary region is
      cleaned + reduced to a crisp boundary line by :func:`_clean_wall_mask` (cv2
      MORPH_OPEN to scrub thin streaks -> MORPH_CLOSE to bridge gaps ->
      connectedComponentsWithStats to drop isolated islands -> MORPH_GRADIENT to
      take the outline) -- the bright wall line.
    * faint RAW occupancy CONTEXT -- the per-cell point count, normalised and
      attenuated by ``raw_context_gain``, drawn dim UNDER the outline purely for
      spatial context (so the plan isn't a skeleton on black).

    :func:`_compose_plan` draws the dim context wash + overwrites it with the bright
    crisp wall outline. Returns ``(rgb (H,W,3) uint8, extent FloorPlanExtent)``; with
    no points a 1x1 black raster + a degenerate extent is returned.

    Implementation: a single ``np.add.at`` scatter accumulates per-cell count, sum
    of height and sum of height^2 -- so the per-cell vertical extent is computed
    without any Python loop over cells (``extent ~= sqrt(var) * 2``, a robust
    spread proxy). The cleanup is cheap 2D cv2 morphology + one connected-component
    pass on the small raster. Pure numpy + cv2, O(N) + O(cells).
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

    # Scatter-accumulate per cell: count, sum(y), sum(y^2). One pass, no per-cell
    # loop. The vertical EXTENT proxy is 2*std(y) = 2*sqrt(E[y^2]-E[y]^2).
    count = np.zeros(ncells, np.float64)
    sum_y = np.zeros(ncells, np.float64)
    sum_y2 = np.zeros(ncells, np.float64)
    np.add.at(count, flat, 1.0)
    np.add.at(sum_y, flat, y)
    np.add.at(sum_y2, flat, y * y)
    nz = count > 0
    mean_y = np.zeros(ncells, np.float64)
    var_y = np.zeros(ncells, np.float64)
    mean_y[nz] = sum_y[nz] / count[nz]
    var_y[nz] = np.maximum(sum_y2[nz] / count[nz] - mean_y[nz] ** 2, 0.0)
    extent_m = 2.0 * np.sqrt(var_y)                          # ~full vertical span

    # OCCUPIED REGION (binary): a cell belongs to the room's occupied region when it
    # is hit by enough rays AND has a real vertical column.
    #  * ``count >= min_cell_count`` drops the thin radial stereo-noise floor (real
    #    surfaces are hit by many rays across keyframes; noise sprays thin).
    #  * ``extent_m >= floor_extent_m`` is the explicit "wall = vertical extent"
    #    gate: a flat-floor cell (points at ~one height) is dropped, so the region
    #    hugs VERTICAL structure (walls / furniture), not the swept floor -- which
    #    keeps the extracted outline on the walls.
    enough = count >= float(min_cell_count)
    tall = extent_m >= float(floor_extent_m)
    occupied = (enough & tall).reshape(height, width)

    # CRISP wall outline: clean the region (cv2 MORPH_OPEN/CLOSE + connected-
    # component area filter) and take its boundary (MORPH_GRADIENT) -- all cheap 2D.
    wall_mask = _clean_wall_mask(
        occupied, open_px=open_px, close_px=close_px,
        min_component_cells=min_component_cells, outline=outline)

    # FAINT raw-occupancy CONTEXT: the per-cell point count (the thin-noise floor
    # removed), normalised to a high percentile (so one dense cell can't wash it
    # out) and attenuated -- a dim spatial hint UNDER the outline.
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
                         floor_extent_m: float = FLOOR_EXTENT_M,
                         open_px: int = MORPH_OPEN_PX,
                         close_px: int = MORPH_CLOSE_PX,
                         min_component_cells: int = MIN_COMPONENT_CELLS,
                         outline: bool = WALL_OUTLINE,
                         raw_context_gain: float = RAW_CONTEXT_GAIN):
    """Convenience: build the raster AND project the camera path onto its pixels.

    Returns ``(rgb (H,W,3) uint8, path_px (M,2) float32, extent)`` where
    ``path_px`` is the keyframe camera positions ``cams`` (``(M,3)`` optical-world)
    projected to fractional raster ``(col, row)`` on the SAME grid as the raster --
    so a caller (the offscreen PNG verifier, a test) can overlay the path without
    re-deriving the extent. The window draws the path itself in world metres via
    the returned ``extent`` (see :class:`FloorPlanExtent`); this helper is mainly
    for the headless PNG check + the unit tests. The wall-cleanup knobs are
    forwarded to :func:`build_floor_plan` so a caller can tune the crispness.
    """
    rgb, extent = build_floor_plan(
        points, cell_m=cell_m, min_cell_count=min_cell_count,
        floor_extent_m=floor_extent_m, open_px=open_px, close_px=close_px,
        min_component_cells=min_component_cells, outline=outline,
        raw_context_gain=raw_context_gain)
    cams = np.asarray(cams, dtype=np.float64).reshape(-1, 3)
    if cams.shape[0] == 0:
        return rgb, np.zeros((0, 2), np.float32), extent
    col, row = extent.world_xz_to_px(cams[:, 0], cams[:, 2])
    path_px = np.stack([col, row], axis=1).astype(np.float32)
    return rgb, path_px, extent
