"""Unit selftests for the pure-numpy+cv2 floor-plan builder (ui.viz.floor_plan).

No Qt, no GL, no IPC -- just the projection (back-project keyframe depth by pose)
+ the ground-plane WALL-CELL detection / cleanup on SYNTHETIC inputs with a known
answer. The plan renders the WALL CELLS DIRECTLY (cells whose points span a tall
vertical column), NOT the outline of the occupied region (which would be the
camera's circular sensing horizon, not the room's walls):

* ``test_backproject_single_pixel`` -- one depth pixel back-projects to the
  pinhole-predicted world point under identity AND a translated/rotated pose.
* ``test_ground_plane_projection`` -- the builder drops the optical-y (DOWN)
  axis: two points differing only in y land in the SAME ground cell.
* ``test_extent_and_cell_index`` -- a known point lands in the expected raster
  cell and the world<->pixel extent round-trips.
* ``test_wall_extent_gate`` -- a FLAT slab (small vertical extent = floor) is
  dropped while a TALL column (a wall) is kept as a wall cell, so the bright marks
  track vertical structure, not the swept floor.
* ``test_noise_island_dropped`` -- a small isolated noise island is removed by the
  morphology + connected-component filter while a longer wall run survives.
* ``test_walls_drawn_directly_not_outline`` -- two SEPARATE wall segments with open
  floor between them both stay lit (the walls are drawn directly); the open floor
  between them is NOT bridged/filled -- i.e. nothing closes the gap into a disc.
* ``test_clean_wall_cells_helper`` -- the cv2 cleanup helper directly: open drops a
  thin speck, the component filter drops a small blob, a wall run survives.
* ``test_camera_path_projection`` -- the camera path projects onto the same grid
  pixels the raster uses.
* ``test_empty_inputs`` -- no points -> a harmless 1x1 raster + degenerate extent.

Run: ``.venv/bin/python -m ui.tests.floor_plan_selftest``
"""
from __future__ import annotations

import numpy as np

from ui.viz import floor_plan


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


# A simple pinhole intrinsic for the synthetic depth maps (fx=fy=100, principal
# point at the image centre of a 64x48 grid).
_K = np.array([[100.0, 0.0, 32.0],
               [0.0, 100.0, 24.0],
               [0.0, 0.0, 1.0]], dtype=np.float64)


def test_backproject_single_pixel() -> None:
    """One valid depth pixel -> the pinhole world point, under I and a pose."""
    h, w = 48, 64
    depth = np.zeros((h, w), np.float32)
    # Put a single valid pixel exactly at the principal point (cx,cy)=(32,24) at
    # z=2.0 m. Back-projection there is (0,0,z) in the camera frame.
    depth[24, 32] = 2.0
    # No edge reject (a lone pixel against zero-depth neighbours would be culled);
    # we want to verify the geometry, not the reject (covered elsewhere).
    pts = floor_plan.keyframes_to_ground_points(
        [depth], [np.eye(3)], [np.zeros(3)], _K, stride=1, edge_max=0.0)
    _check(pts.shape == (1, 3), f"identity pose -> exactly one point ({pts.shape})")
    _check(np.allclose(pts[0], [0.0, 0.0, 2.0], atol=1e-4),
           f"principal-point pixel back-projects to (0,0,z) ({pts[0]})")

    # A pure translation t=(1,-2,3) must shift the world point by t (Xw=R Xc+t).
    t = np.array([1.0, -2.0, 3.0])
    pts_t = floor_plan.keyframes_to_ground_points(
        [depth], [np.eye(3)], [t], _K, stride=1, edge_max=0.0)
    _check(np.allclose(pts_t[0], [1.0, -2.0, 5.0], atol=1e-4),
           f"translated pose shifts the world point by t ({pts_t[0]})")

    # A 90-deg rotation about optical-y maps camera +z -> world +x. The camera
    # point is (0,0,2); Xw = R Xc should be (2,0,0).
    Ry = np.array([[0.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0],
                   [-1.0, 0.0, 0.0]])
    pts_r = floor_plan.keyframes_to_ground_points(
        [depth], [Ry], [np.zeros(3)], _K, stride=1, edge_max=0.0)
    _check(np.allclose(pts_r[0], [2.0, 0.0, 0.0], atol=1e-4),
           f"rotation maps camera +z to world +x ({pts_r[0]})")


def test_ground_plane_projection() -> None:
    """The builder drops optical-y (DOWN): same (x,z), different y -> same cell."""
    # Two points at the SAME (x,z)=(0.5, 0.5) but very different y (height): they
    # must accumulate in ONE ground cell (the vertical axis is dropped). Relax the
    # cleanup knobs (count=1, no component-area floor, no open) so the single
    # wall cell isn't scrubbed -- here we verify the PROJECTION only.
    pts = np.array([[0.5, -1.0, 0.5],
                    [0.5, 2.0, 0.5]], dtype=np.float64)
    rgb, extent = floor_plan.build_floor_plan(
        pts, cell_m=0.1, min_cell_count=1, min_component_cells=1, open_px=0)
    # The two points span x in [0.5,0.5], z in [0.5,0.5] -> a 1x1 grid, both in it.
    _check(extent.width == 1 and extent.height == 1,
           f"co-located (x,z) points -> a 1x1 ground grid ({extent.width}x"
           f"{extent.height})")
    _check(rgb.shape == (1, 1, 3),
           f"raster is 1x1x3 for one occupied cell ({rgb.shape})")
    # That single cell saw 2 points spanning 3 m of height (a tall column, well
    # over the wall-extent gate) -> it survives as a wall cell -> a bright colour.
    _check(int(rgb[0, 0].max()) > 200,
           f"the tall (wall) cell is lit bright, not background ({rgb[0,0]})")


def test_extent_and_cell_index() -> None:
    """A known point lands in the expected cell; the extent round-trips."""
    # Points spanning x in [0,1], z in [0,2] at cell 0.5 -> width=3 (0,0.5,1),
    # height=5 (0,..,2). Each corner is a TALL column (so it survives the wall
    # gate) at the (x,z) corners. A point at (1.0, *, 2.0) is the top-right cell.
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0],     # tall column at (0,0)
                    [1.0, 0.0, 2.0], [1.0, 2.0, 2.0]],    # tall column at (1,2)
                   dtype=np.float64)
    # Relax the cleanup (count=1, component floor=1, no open) so the two lone tall
    # corner cells survive -- here we verify the EXTENT / cell indexing only (the
    # gate + cleanup are tested separately).
    rgb, extent = floor_plan.build_floor_plan(
        pts, cell_m=0.5, min_cell_count=1, min_component_cells=1, open_px=0)
    _check(extent.width == 3 and extent.height == 5,
           f"grid is 3x5 cells for the (1m x 2m)/0.5 extent ({extent.width}x"
           f"{extent.height})")
    _check(abs(extent.x_min - 0.0) < 1e-9 and abs(extent.z_min - 0.0) < 1e-9,
           f"extent origin at the min corner ({extent.x_min},{extent.z_min})")
    # world_xz_to_px round-trip: (x=0.5, z=1.0) -> col 1, row 2.
    col, row = extent.world_xz_to_px(np.array([0.5]), np.array([1.0]))
    _check(abs(float(col[0]) - 1.0) < 1e-9 and abs(float(row[0]) - 2.0) < 1e-9,
           f"world->px maps (0.5,1.0) to (col 1, row 2) ({col[0]},{row[0]})")
    # The two tall corner columns must light their corner cells (row 0 col 0 and
    # the last row/col), and the empty interior must stay background-dark.
    _check(int(rgb[0, 0].max()) > 200 and int(rgb[4, 2].max()) > 200,
           "both tall corner columns light their cells")
    _check(int(rgb[2, 1].sum()) < int(rgb[0, 0].sum()),
           "an empty interior cell is darker than an occupied (wall) corner")


def test_wall_extent_gate() -> None:
    """A flat slab (floor) is gated out by vertical extent; a tall block survives.

    The core "wall = high vertical extent" detector: a cell whose points span less
    than ``WALL_EXTENT_M`` in height is flat floor and is NOT a wall cell, so the
    rendered marks track vertical structure (walls), not the swept floor.
    """
    rng = np.random.default_rng(0)
    # A solid FLOOR slab: a 1x1 m patch of many points at ~one height (extent ~0).
    n_floor = 4000
    fx = rng.uniform(0.0, 1.0, n_floor)
    fz = rng.uniform(0.0, 1.0, n_floor)
    fy = rng.normal(1.0, 0.01, n_floor)                # ~flat -> extent << gate
    floor = np.stack([fx, fy, fz], axis=1)
    # A solid TALL block at a far (x,z) region: a 1x1 m footprint of points each
    # spanning a 2 m vertical column (a wall/structure) -> extent >> gate.
    n_wall = 4000
    wx = rng.uniform(5.0, 6.0, n_wall)
    wz = rng.uniform(5.0, 6.0, n_wall)
    wy = rng.uniform(0.0, 3.0, n_wall)                 # 3 m column -> big extent
    wall = np.stack([wx, wy, wz], axis=1)
    pts = np.concatenate([floor, wall], axis=0)
    # Use an explicit gate (1.0 m) that clearly separates the flat floor (extent
    # ~0) from the 3 m wall column -- the test verifies the GATE MECHANISM, not the
    # production default (which is tuned for real data and may sit anywhere).
    rgb, extent = floor_plan.build_floor_plan(
        pts, cell_m=0.1, wall_extent_m=1.0)

    def region_lit(x0, z0, x1, z1):
        c0, r0 = extent.world_xz_to_px(np.array([x0]), np.array([z0]))
        c1, r1 = extent.world_xz_to_px(np.array([x1]), np.array([z1]))
        rs = slice(int(max(0, r0[0])), int(min(extent.height, r1[0] + 1)))
        cs = slice(int(max(0, c0[0])), int(min(extent.width, c1[0] + 1)))
        sub = rgb[rs, cs].reshape(-1, 3)
        return int(sub.max()) if sub.size else 0

    floor_lit = region_lit(0.0, 0.0, 1.0, 1.0)
    wall_lit = region_lit(5.0, 5.0, 6.0, 6.0)
    # The wall block reads as bright (near-white) wall cells; the gated-out floor is
    # at most the faint raw-occupancy context wash (well below the bright walls).
    _check(floor_lit < 120,
           f"the flat floor slab is gated out (only faint context, {floor_lit})")
    _check(wall_lit > 200,
           f"the tall block survives the extent gate (bright walls, {wall_lit})")


def test_noise_island_dropped() -> None:
    """A small isolated noise island is dropped; a longer wall run survives.

    The morphology OPEN + connected-component area filter remove the small isolated
    star-burst blobs while keeping a longer connected wall run.
    """
    rng = np.random.default_rng(1)
    # A LARGE solid tall region (a wall sheet) -> a big connected component.
    n_big = 8000
    bx = rng.uniform(0.0, 2.0, n_big)
    bz = rng.uniform(0.0, 0.4, n_big)                  # a 2.0 x 0.4 m wall footprint
    by = rng.uniform(0.0, 3.0, n_big)                  # 3 m tall column
    big = np.stack([bx, by, bz], axis=1)
    # A TINY isolated tall island far away (a single ~1-cell speck of noise).
    n_isle = 6
    isx = rng.normal(8.0, 0.01, n_isle)
    isz = rng.normal(8.0, 0.01, n_isle)
    isy = rng.uniform(0.0, 3.0, n_isle)                # tall, so the gate keeps it
    isle = np.stack([isx, isy, isz], axis=1)
    pts = np.concatenate([big, isle], axis=0)
    # Explicit gate (1.0 m) clearly clears the 3 m columns -- the test verifies the
    # speckle cleanup (open + component filter), not the production extent default.
    rgb, extent = floor_plan.build_floor_plan(
        pts, cell_m=0.1, wall_extent_m=1.0,
        min_component_cells=floor_plan.MIN_COMPONENT_CELLS)
    # The big region's centre must be lit; the isolated island must be dropped.
    bc, br = extent.world_xz_to_px(np.array([1.0]), np.array([0.2]))
    ic, ir = extent.world_xz_to_px(np.array([8.0]), np.array([8.0]))
    bc = int(np.clip(round(float(bc[0])), 0, extent.width - 1))
    br = int(np.clip(round(float(br[0])), 0, extent.height - 1))
    ic = int(np.clip(round(float(ic[0])), 0, extent.width - 1))
    ir = int(np.clip(round(float(ir[0])), 0, extent.height - 1))
    # The big wall run reads as bright (near-white) wall cells; the dropped island
    # is at most the faint raw-occupancy context wash (no bright wall cell).
    _check(int(rgb[br, bc].max()) > 200,
           f"the longer wall run survives (bright, {rgb[br,bc]})")
    _check(int(rgb[ir, ic].max()) < 120,
           f"the small isolated noise island is dropped (faint, {rgb[ir,ic]})")


def test_walls_drawn_directly_not_outline() -> None:
    """Walls are drawn DIRECTLY (both segments lit); open floor between is NOT filled.

    Two SEPARATE tall wall segments with a wide swept FLOOR strip between them: both
    walls must stay lit (they are the bright structure -- drawn directly, not the
    outline of one occupied region), and the open floor BETWEEN them must NOT be
    bridged/filled bright. This is exactly what stops the plan rounding into a disc:
    nothing closes across open space.
    """
    rng = np.random.default_rng(3)
    # Two tall wall sheets at x in [0,0.3] and x in [3.0,3.3], each spanning z in
    # [0,3] and a 3 m vertical column (clearly walls). Between them (x ~1.5) is a
    # FLAT floor strip (~one height) -- it must NOT light up bright.
    def sheet(x0, x1, n):
        return np.stack([rng.uniform(x0, x1, n), rng.uniform(0.0, 3.0, n),
                         rng.uniform(0.0, 3.0, n)], axis=1)
    wall_a = sheet(0.0, 0.3, 6000)
    wall_b = sheet(3.0, 3.3, 6000)
    n_floor = 8000
    floor = np.stack([rng.uniform(0.4, 2.9, n_floor),  # the open span between walls
                      rng.normal(1.0, 0.01, n_floor),  # flat -> not a wall
                      rng.uniform(0.0, 3.0, n_floor)], axis=1)
    pts = np.concatenate([wall_a, wall_b, floor], axis=0)
    # Explicit gate (1.0 m) clearly separates the flat floor strip (extent ~0) from
    # the 3 m wall columns -- the test verifies "walls drawn directly, no disc", not
    # the production extent default.
    rgb, extent = floor_plan.build_floor_plan(pts, cell_m=0.1, wall_extent_m=1.0)
    bright = rgb.max(axis=2) > 200

    def lit_at(x, z):
        c, r = extent.world_xz_to_px(np.array([x]), np.array([z]))
        c = int(np.clip(round(float(c[0])), 0, extent.width - 1))
        r = int(np.clip(round(float(r[0])), 0, extent.height - 1))
        return bool(bright[r, c])

    _check(lit_at(0.15, 1.5), "wall A is lit (drawn directly)")
    _check(lit_at(3.15, 1.5), "wall B is lit (drawn directly)")
    # The open floor between the two walls (x ~1.5) must NOT be bright -- a CLOSE
    # would have bridged it into a filled disc; we do not close.
    _check(not lit_at(1.5, 1.5),
           "the open floor between the walls is NOT filled bright (no disc)")
    # Sanity: the bright area is a small fraction of the raster (two thin walls, not
    # a filled blob between them).
    n_bright = int(bright.sum())
    _check(n_bright < 0.4 * (extent.width * extent.height),
           f"the lit walls are thin marks, not a filled region: {n_bright} of "
           f"{extent.width * extent.height} cells")


def test_clean_wall_cells_helper() -> None:
    """The cv2 cleanup helper directly: open drops specks, components drop blobs."""
    h, w = 40, 40
    occ = np.zeros((h, w), np.uint8)
    occ[10:30, 12:18] = 1                  # a solid 20x6 wall run (a real wall)
    occ[5, 5] = 1                          # a single-cell speck (noise)
    occ[35:37, 36:38] = 1                  # a tiny isolated 2x2 blob (noise island)
    cleaned = floor_plan._clean_wall_cells(
        occ, open_px=2, min_component_cells=8)
    _check(not cleaned[5, 5],
           "MORPH_OPEN erased the single-cell speck")
    _check(not cleaned[35:37, 36:38].any(),
           "the connected-component filter dropped the tiny isolated blob")
    _check(cleaned[20, 15],
           "the solid wall run survived the cleanup")
    # The wall run is drawn AS-IS (filled), NOT reduced to an outline: an interior
    # cell deep inside the 6-wide run stays lit (we do not take a gradient boundary).
    _check(cleaned[20, 14] and cleaned[20, 16],
           "the wall run's interior cells stay lit (walls drawn directly, not "
           "outlined)")


def test_camera_path_projection() -> None:
    """The camera path projects onto the same grid pixels the raster uses."""
    pts = np.array([[0.0, 0.0, 0.0],
                    [2.0, 0.0, 3.0]], dtype=np.float64)
    cams = np.array([[0.0, 0.5, 0.0],     # at the (x,z) origin corner
                     [2.0, 0.5, 3.0]],    # at the far corner
                    dtype=np.float64)
    rgb, path_px, extent = floor_plan.floor_plan_with_path(
        pts, cams, cell_m=0.5, min_cell_count=1)
    _check(path_px.shape == (2, 2), f"path has one (col,row) per cam ({path_px.shape})")
    # cam0 at world (0,0) -> pixel (0,0); cam1 at (2,3) -> (col 4, row 6).
    _check(np.allclose(path_px[0], [0.0, 0.0], atol=1e-6),
           f"first cam projects to the origin pixel ({path_px[0]})")
    _check(np.allclose(path_px[1], [4.0, 6.0], atol=1e-6),
           f"last cam projects to the far-corner pixel ({path_px[1]})")
    # The path pixels must lie within the raster (the window draws them on it).
    _check(0 <= path_px[:, 0].max() <= extent.width and
           0 <= path_px[:, 1].max() <= extent.height,
           "every path pixel is inside the raster extent")


def test_empty_inputs() -> None:
    """No points -> a harmless 1x1 raster + degenerate extent (no crash)."""
    rgb, extent = floor_plan.build_floor_plan(np.zeros((0, 3), np.float64))
    _check(rgb.shape == (1, 1, 3), f"empty -> 1x1x3 raster ({rgb.shape})")
    _check(extent.width == 1 and extent.height == 1, "empty -> 1x1 extent")
    # Empty keyframe list -> empty point cloud.
    pts = floor_plan.keyframes_to_ground_points([], [], [], _K)
    _check(pts.shape == (0, 3), f"no keyframes -> no points ({pts.shape})")


def main() -> int:
    print("test_backproject_single_pixel"); test_backproject_single_pixel()
    print("test_ground_plane_projection"); test_ground_plane_projection()
    print("test_extent_and_cell_index"); test_extent_and_cell_index()
    print("test_wall_extent_gate"); test_wall_extent_gate()
    print("test_noise_island_dropped"); test_noise_island_dropped()
    print("test_walls_drawn_directly_not_outline"); test_walls_drawn_directly_not_outline()
    print("test_clean_wall_cells_helper"); test_clean_wall_cells_helper()
    print("test_camera_path_projection"); test_camera_path_projection()
    print("test_empty_inputs"); test_empty_inputs()
    print("\nALL FLOOR_PLAN SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
