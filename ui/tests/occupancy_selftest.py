#!/usr/bin/env python3
"""Unit tests for the SLAM-map LOG-ODDS occupancy + RAY-CARVING fusion.

The ModalAI/VOXL-style SLAM Map (:class:`~ui.modules.ipc_sources.IpcSlamMapSource`)
builds a VOXEL OCCUPANCY map as a PROBABILISTIC LOG-ODDS GRID with FREE-SPACE RAY
CARVING (OctoMap/Voxblox-style). Each depth ray adds OCCUPIED evidence (``+L_OCC``)
at its hit voxel and FREE evidence (``+L_FREE``) to every voxel it passes THROUGH;
a cell is rendered OCCUPIED when ``log_odds >= L_OCC_THRESH``. The key win over the
old hit-count-only gate is that carving REMOVES wrongly-added voxels: a noise cell a
later ray sees through accumulates free evidence and drops below threshold, so the
map self-cleans. These tests drive that fusion with SYNTHETIC geometry (so it is
hand-checkable) and assert:

* the vectorised DDA visits a CORRECT, CONTIGUOUS voxel line C->P (axis-aligned and
  diagonal), from the origin voxel up to JUST BEFORE the hit voxel (the hit excluded),
* a single ray marks its IN-BETWEEN voxels FREE (log_odds < 0) and its END voxel
  OCCUPIED (log_odds > 0),
* CARVING REMOVES A VOXEL: a cell hit once (+L_OCC) then crossed by 2 rays (+2*L_FREE)
  drops BELOW L_OCC_THRESH and is no longer occupied (the core behaviour),
* the RENDER CONFIDENCE GATE separates the wall from the behind-wall spray: a
  HIGH-confidence voxel (re-hit many times -> log_odds >= L_DISPLAY) RENDERS, while a
  LOW-confidence-but-occupied voxel (log_odds in [L_OCC_THRESH, L_DISPLAY), e.g. the
  behind-wall noise carving can't reach) is in the internal occupied set but is NOT
  rendered -- and crucially the grid KEEPS that low evidence so carving still works,
* log-odds are CLAMPED to [L_MIN, L_MAX] (so a long dwell can't pin a cell un-carvably
  high, and free evidence can't run off to -inf),
* fusion is INCREMENTAL (a new keyframe only folds itself; never rebuilds),
* the emitted points are the correct VOXEL CENTRES (index * VOXEL_M + half-cell),
* we render ALL occupied cells (the map GROWS) and the high MAX_VOXELS safety cap,
  when it trips, thins FAIRLY (keeps new-area voxels, not just the long-dwelt start),
* feeding keyframes that cover NEW spatial areas GROWS the occupied count + extent,
* the re-emit signature changes on growth/shift but is stable on a repeat,
* a flat-`y` colour fallback + a green-by-height gradient stay in [0,1].

We build the source WITHOUT connecting (no ``start_cloud``): the fusion + build are
pure methods on the accumulated ``_kf_depth`` / ``_kf_R`` / ``_kf_t`` dicts, so we
populate those directly and call the methods. The depth grid is a plane at constant
depth so every valid pixel maps to one Z and the world hits land in a known band.

Run::

    python -m ui.tests.occupancy_selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.modules.ipc_sources import IpcSlamMapSource                  # noqa: E402


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def _make_source() -> IpcSlamMapSource:
    """An UN-connected source with a simple pinhole K (never starts a thread)."""
    # fx=fy=100, cx=cy at the image centre of a 64x64 grid.
    K = np.array([[100.0, 0.0, 32.0],
                  [0.0, 100.0, 32.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return IpcSlamMapSource("oak.vio.unit", "oak.slam.unit", K)


def _identity_kf(src: IpcSlamMapSource, seq: int, depth_val: float,
                 *, h: int = 64, w: int = 64) -> None:
    """Stash a synthetic keyframe: a constant-depth plane at the identity pose.

    With R=I, t=0 the world hit of pixel (u,v) is ((u-cx)/fx*z, (v-cy)/fy*z, z) --
    so a constant depth + a small image span land all hits in a tight world band
    (hence into a few voxels), which makes the log-odds bookkeeping checkable.
    """
    depth = np.full((h, w), float(depth_val), dtype=np.float32)
    src._kf_depth[seq] = depth                                  # noqa: SLF001
    src._kf_R[seq] = np.eye(3, dtype=np.float64)                # noqa: SLF001
    src._kf_t[seq] = np.zeros(3, dtype=np.float64)              # noqa: SLF001


def _occupied_cells(src: IpcSlamMapSource) -> set:
    """The set of integer cell keys currently OCCUPIED (log_odds >= L_OCC_THRESH)."""
    th = src.L_OCC_THRESH
    return {k for k, lo in src._log.items() if lo >= th}        # noqa: SLF001


def _rendered_cells(src: IpcSlamMapSource) -> set:
    """The set of cell keys ``_build`` would RENDER (log_odds >= L_DISPLAY).

    Recovers the integer voxel keys from the emitted voxel CENTRES so the test
    asserts the actual render output, not just the internal grid.
    """
    pts, _, _ = src._build()                                    # noqa: SLF001
    if pts.shape[0] == 0:
        return set()
    keys = np.round(pts / src.VOXEL_M - 0.5).astype(int)
    return {tuple(int(c) for c in k) for k in keys}


def _seed_to_display(src: IpcSlamMapSource, cell: tuple, *,
                     hits: int) -> None:
    """Push ``cell`` up by ``hits`` HIT updates (clamped), as repeated keyframes would.

    Lets a test build a cell to a chosen confidence (e.g. enough hits to clear
    L_DISPLAY for a 'wall', or just one/two for low-confidence 'behind-wall noise').
    """
    lmin, lmax = src.L_MIN, src.L_MAX
    for _ in range(hits):
        v = src._log.get(cell, 0.0) + src.L_OCC                 # noqa: SLF001
        src._log[cell] = max(lmin, min(lmax, v))                # noqa: SLF001


# --------------------------------------------------------------------------- #
# The DDA voxel traversal (the heart of the carving).
# --------------------------------------------------------------------------- #
def test_dda_axis_aligned_contiguous_line() -> None:
    print("\n  carving DDA: axis-aligned ray visits a contiguous voxel line, "
          "hit excluded")
    src = _make_source()
    # In VOXEL units: C at the centre of voxel (0,0,0), P at the centre of (5,0,0).
    # The ray should pass THROUGH voxels x=0..4 (origin up to just before the hit
    # voxel x=5), all at y=0,z=0; the hit voxel x=5 is EXCLUDED.
    C = np.array([0.5, 0.5, 0.5])
    P = np.array([[5.5, 0.5, 0.5]])
    free = src._carve_free_cells(C, P)                          # noqa: SLF001
    xs = sorted(int(c[0]) for c in free)
    _check(xs == [0, 1, 2, 3, 4],
           f"free voxels are the contiguous line x=0..4 (got {xs})")
    _check(all(int(c[1]) == 0 and int(c[2]) == 0 for c in free),
           "free voxels stay on the ray's y=0,z=0 line")
    _check(not any(tuple(c) == (5, 0, 0) for c in free),
           "the HIT voxel (5,0,0) is NOT carved as free (it's the endpoint)")


def test_dda_diagonal_is_6connected_no_gaps() -> None:
    print("\n  carving DDA: diagonal ray is a 6-connected, gap-free voxel line")
    src = _make_source()
    # A 45-degree ray in the xy plane from (0.5,0.5) to (3.5,3.5): amanatides-woo
    # produces a 6-CONNECTED line (each step advances exactly ONE axis), with NO
    # gaps -- consecutive cells differ by exactly one in a single coordinate.
    C = np.array([0.5, 0.5, 0.5])
    P = np.array([[3.5, 3.5, 0.5]])
    free = src._carve_free_cells(C, P)                          # noqa: SLF001
    # Order along the ray by t = progress; sort by (x+y) then x is enough here.
    order = sorted(free.tolist(), key=lambda c: (c[0] + c[1], c[0]))
    diffs = [np.abs(np.subtract(order[i + 1], order[i])).sum()
             for i in range(len(order) - 1)]
    _check(all(d == 1 for d in diffs),
           f"every DDA step moves exactly one voxel (6-connected; deltas {diffs})")
    _check((0, 0, 0) in [tuple(c) for c in order],
           "the line starts at the origin voxel (0,0,0)")
    _check((3, 3, 0) not in [tuple(c) for c in order],
           "the line stops before the hit voxel (3,3,0)")


# --------------------------------------------------------------------------- #
# One ray: free in between, occupied at the end.
# --------------------------------------------------------------------------- #
def test_single_ray_free_between_occupied_end() -> None:
    print("\n  carving: a single ray marks IN-BETWEEN voxels free, the END occupied")
    src = _make_source()
    # Drive ONE ray directly through the fuse's scatter step (bypass back-projection
    # so the geometry is exact): C at (0,0,0)-ish voxel, P at voxel (5,0,0). We mimic
    # what _fuse_keyframe_locked does for one ray: free-carve, then hit.
    C = np.array([0.5, 0.5, 0.5])
    P = np.array([[5.5, 0.5, 0.5]])
    free = src._carve_free_cells(C, P)                          # noqa: SLF001
    hit = (5, 0, 0)
    lmin, lmax = src.L_MIN, src.L_MAX
    for cell in free:
        k = (int(cell[0]), int(cell[1]), int(cell[2]))
        v = src._log.get(k, 0.0) + src.L_FREE                   # noqa: SLF001
        src._log[k] = max(lmin, min(lmax, v))                   # noqa: SLF001
    v = src._log.get(hit, 0.0) + src.L_OCC                      # noqa: SLF001
    src._log[hit] = max(lmin, min(lmax, v))                     # noqa: SLF001

    _check(all(src._log[(int(c[0]), int(c[1]), int(c[2]))] < 0.0  # noqa: SLF001
               for c in free),
           "every in-between voxel has log_odds < 0 (free)")
    _check(src._log[hit] > 0.0,                                 # noqa: SLF001
           f"the end voxel has log_odds > 0 (occupied; {src._log[hit]:.2f})")
    occ = _occupied_cells(src)
    _check(occ == {hit},
           f"only the END voxel is OCCUPIED (>= L_OCC_THRESH); got {occ}")


# --------------------------------------------------------------------------- #
# THE CORE: carving REMOVES an already-added voxel.
# --------------------------------------------------------------------------- #
def test_carving_removes_a_crossed_voxel() -> None:
    print("\n  carving: a voxel hit once then crossed by 2 rays drops BELOW threshold "
          "(REMOVED)")
    src = _make_source()
    cell = (7, 7, 7)
    lmin, lmax = src.L_MIN, src.L_MAX

    def _add(c, dl):
        v = src._log.get(c, 0.0) + dl                           # noqa: SLF001
        src._log[c] = max(lmin, min(lmax, v))                   # noqa: SLF001

    # One hit -> occupied (a wrongly-added stereo-noise voxel).
    _add(cell, src.L_OCC)
    _check(cell in _occupied_cells(src),
           f"after ONE hit the voxel is occupied ({src._log[cell]:.2f} "    # noqa: SLF001
           f">= {src.L_OCC_THRESH})")
    # Two later rays from new viewpoints CROSS it (see through it) -> 2 free updates.
    _add(cell, src.L_FREE)
    _add(cell, src.L_FREE)
    _check(cell not in _occupied_cells(src),
           f"after +2 free crossings the voxel is REMOVED "
           f"({src._log[cell]:.2f} < {src.L_OCC_THRESH})")    # noqa: SLF001
    # Sanity: the math is L_OCC + 2*L_FREE and it lands below the threshold.
    want = src.L_OCC + 2.0 * src.L_FREE
    _check(abs(src._log[cell] - want) < 1e-6 and want < src.L_OCC_THRESH,  # noqa: SLF001
           f"log_odds = L_OCC + 2*L_FREE = {want:.2f} < L_OCC_THRESH "
           f"={src.L_OCC_THRESH} (carving wins)")


def test_render_confidence_gate_separates_wall_from_spray() -> None:
    """The RENDER gate L_DISPLAY shows a HIGH-confidence wall but NOT a low-confidence
    behind-wall voxel -- while the grid KEEPS the low evidence so carving still works.

    This is the principled fix for the behind-the-wall noise: carving cannot reach the
    space behind the wall (rays stop at the surface), so we separate by CONFIDENCE.
    A consistently-observed wall is re-hit many times -> log_odds >= L_DISPLAY -> it
    renders. The sporadic behind-wall spray is hit only once or twice -> its log_odds
    stays in [L_OCC_THRESH, L_DISPLAY) -> it is in the INTERNAL occupied set but is
    filtered OUT of the render.
    """
    print("\n  render gate: a high-confidence wall renders; a low-confidence "
          "behind-wall voxel does NOT")
    src = _make_source()
    # Sanity on the tuning itself: the render gate must sit strictly ABOVE the internal
    # occupied threshold (a separate, higher gate), and high enough that 1-2 hits do
    # NOT clear it (so sporadic noise is filtered) while it stays under L_MAX (so a
    # re-observed surface can climb past it and render).
    _check(src.L_DISPLAY > src.L_OCC_THRESH,
           f"L_DISPLAY ({src.L_DISPLAY}) is a SEPARATE, higher gate than "
           f"L_OCC_THRESH ({src.L_OCC_THRESH})")
    _check(2.0 * src.L_OCC < src.L_DISPLAY < src.L_MAX,
           f"L_DISPLAY ({src.L_DISPLAY}) filters 1-2 hits (2*L_OCC="
           f"{2.0 * src.L_OCC:.2f}) yet is reachable (< L_MAX={src.L_MAX})")

    wall = (3, 0, 5)        # a consistently-observed surface: re-hit many times
    spray = (9, 0, 5)       # behind-wall stereo noise: hit only twice (low confidence)
    # The wall is hit enough times to clear L_DISPLAY; the spray only twice (so it sits
    # below L_DISPLAY but above L_OCC_THRESH -> internally occupied, NOT rendered).
    _seed_to_display(src, wall, hits=6)
    _seed_to_display(src, spray, hits=2)

    _check(src._log[wall] >= src.L_DISPLAY,                     # noqa: SLF001
           f"the re-hit wall cleared L_DISPLAY ({src._log[wall]:.2f} "  # noqa: SLF001
           f">= {src.L_DISPLAY})")
    _check(src.L_OCC_THRESH <= src._log[spray] < src.L_DISPLAY,  # noqa: SLF001
           f"the behind-wall spray is occupied but LOW-confidence "
           f"({src._log[spray]:.2f} in [{src.L_OCC_THRESH}, {src.L_DISPLAY}))")  # noqa: SLF001

    # Internal occupied set holds BOTH (the UPDATE math / grid is unchanged) ...
    occ = _occupied_cells(src)
    _check(wall in occ and spray in occ,
           "BOTH cells are in the INTERNAL occupied set (grid keeps the low "
           "evidence so carving still works)")
    # ... but the RENDER shows ONLY the high-confidence wall.
    rendered = _rendered_cells(src)
    _check(wall in rendered,
           "the HIGH-confidence wall RENDERS (log_odds >= L_DISPLAY)")
    _check(spray not in rendered,
           "the LOW-confidence behind-wall voxel does NOT render (filtered by "
           "L_DISPLAY) even though it is internally occupied")


def test_logodds_clamped_to_band() -> None:
    print("\n  carving: log_odds is clamped to [L_MIN, L_MAX] (un-carvable-high "
          "prevented)")
    src = _make_source()
    lmin, lmax = src.L_MIN, src.L_MAX

    def _add(c, dl):
        v = src._log.get(c, 0.0) + dl                           # noqa: SLF001
        src._log[c] = max(lmin, min(lmax, v))                   # noqa: SLF001

    cell = (1, 2, 3)
    # Hammer the cell with many hits -> must saturate at L_MAX (not run away), so a
    # later run of free crossings CAN still carve it back below threshold.
    for _ in range(50):
        _add(cell, src.L_OCC)
    _check(abs(src._log[cell] - lmax) < 1e-6,                   # noqa: SLF001
           f"a long dwell saturates at L_MAX={lmax} (got {src._log[cell]:.2f})")
    # Now carve it with free evidence -> must bottom out at L_MIN, not -inf.
    for _ in range(50):
        _add(cell, src.L_FREE)
    _check(abs(src._log[cell] - lmin) < 1e-6,                   # noqa: SLF001
           f"sustained free evidence bottoms out at L_MIN={lmin} "
           f"(got {src._log[cell]:.2f})")
    _check(cell not in _occupied_cells(src),
           "the clamped-low cell is no longer occupied (carved away even after dwell)")


def test_fuse_carves_through_a_planted_noise_voxel() -> None:
    """End-to-end through _fuse_keyframe_locked: a planted noise voxel in front of a
    real surface gets CARVED by the keyframe's own free rays.

    A constant-depth plane keyframe back-projects to a wall of hit voxels at z~=2 m.
    Every ray from C=(0,0,0) to that wall passes through the free space at z<2 m. We
    pre-seed a NOISE voxel sitting in that free corridor (occupied), fuse the
    keyframe, and assert the rays carve the noise voxel below threshold while the
    real wall voxels become occupied.
    """
    print("\n  carving: fusing a real keyframe carves a planted noise voxel in its "
          "free corridor")
    src = _make_source()
    # A small plane at 2.0 m (R=I, t=0) -> hits near z=2.0 -> hit voxels at iz~=20
    # (VOXEL_M=0.1). The free corridor is iz = 0..19 along each ray.
    _identity_kf(src, seq=0, depth_val=2.0, h=40, w=40)
    # Plant a NOISE voxel near the optical axis at iz=10 (mid free-corridor), pushed
    # occupied as if an earlier bad keyframe had added it.
    noise = (0, 0, 10)
    src._log[noise] = src.L_OCC * 3.0   # noqa: SLF001 strongly "occupied"
    src._log[noise] = min(src._log[noise], src.L_MAX)           # noqa: SLF001
    _check(noise in _occupied_cells(src), "noise voxel starts OCCUPIED (planted)")

    src._build()    # noqa: SLF001 folds keyframe 0: carves the corridor, hits the wall
    occ = _occupied_cells(src)
    # The real wall (cells at iz ~= 20) must now be occupied...
    wall = [c for c in occ if c[2] >= 18]
    _check(len(wall) > 0,
           f"the real surface (iz~=20) is occupied after fusing ({len(wall)} cells)")
    # ...and the noise voxel must have been carved (a single keyframe's rays cross it
    # once each, so its log-odds drops by L_FREE; a strongly-planted voxel may need
    # more than one keyframe, so assert it at least DROPPED, and fully verify removal
    # by fusing a few more identical keyframes from the same viewpoint).
    before = src.L_OCC * 3.0
    _check(src._log[noise] < before,                            # noqa: SLF001
           f"the noise voxel's log-odds DROPPED after one carve "
           f"({src._log[noise]:.2f} < {before:.2f})")     # noqa: SLF001
    for seq in range(1, 12):
        _identity_kf(src, seq=seq, depth_val=2.0, h=40, w=40)
        src._build()                                            # noqa: SLF001
    _check(noise not in _occupied_cells(src),
           f"after repeated crossings the noise voxel is REMOVED "
           f"({src._log[noise]:.2f} < {src.L_OCC_THRESH})")     # noqa: SLF001


# --------------------------------------------------------------------------- #
# Incremental fusion + voxel centres.
# --------------------------------------------------------------------------- #
def test_fusion_is_incremental_and_centres_correct() -> None:
    print("\n  occupancy: fusion is incremental; emitted points are voxel CENTRES")
    src = _make_source()
    # Fold the FIRST plane keyframe (seq 0). One hit only reaches L_OCC=0.85 < the
    # RENDER gate L_DISPLAY, so the wall is in the grid but not yet displayed -- the
    # cell IS occupied internally, proving fusion happened.
    _identity_kf(src, seq=0, depth_val=2.0)
    src._build()                                                # noqa: SLF001
    with src._lock:                                             # noqa: SLF001
        fused_1 = set(src._fused_seqs)
        grid_1 = dict(src._log)
    _check(len(_occupied_cells(src)) > 0,
           "first keyframe produced internally-occupied voxels (fusion happened)")
    _check(fused_1 == {0}, f"seq 0 marked fused (got {fused_1})")

    # Re-observe the SAME plane a few more times so the wall cells climb past the
    # RENDER gate L_DISPLAY and the build emits them -- then check the emitted points
    # are the cells' CENTRES: (idx + 0.5) * VOXEL_M.
    for seq in range(1, 4):
        _identity_kf(src, seq=seq, depth_val=2.0)
        pts1, cols1, _ = src._build()                           # noqa: SLF001
    _check(pts1.shape[0] > 0,
           "after a few re-observations the wall clears L_DISPLAY and RENDERS")
    _check(cols1.shape == pts1.shape, "colours align with the voxel centres")

    vm = src.VOXEL_M
    keys = np.round(pts1 / vm - 0.5).astype(int)
    recon = (keys.astype(np.float32) + 0.5) * np.float32(vm)
    _check(np.allclose(pts1, recon, atol=1e-5),
           "every emitted point is a voxel centre (idx+0.5)*VOXEL_M")

    # Fold ANOTHER identical keyframe: the SAME hit cells are re-observed (their
    # log-odds rises, saturating at L_MAX) and earlier seqs are NOT re-fused
    # (incremental).
    _identity_kf(src, seq=4, depth_val=2.0)
    src._build()                                                # noqa: SLF001
    with src._lock:                                             # noqa: SLF001
        fused_2 = set(src._fused_seqs)
        grid_2 = dict(src._log)
    _check(fused_2 == {0, 1, 2, 3, 4}, f"all seqs marked fused (got {fused_2})")
    # Re-observed hit cells climbed (or saturated) -- never dropped -- proving the
    # second keyframe ADDED evidence rather than rebuilding from scratch.
    re_hit = [k for k in grid_1 if grid_1[k] > 0 and k in grid_2]
    _check(re_hit and all(grid_2[k] >= grid_1[k] - 1e-6 for k in re_hit),
           "re-observed occupied cells gained (or saturated) log-odds (incremental)")


def test_max_voxels_safety_cap_is_fair() -> None:
    print("\n  occupancy: MAX_VOXELS safety cap thins FAIRLY (not by log-odds)")
    src = _make_source()
    src.MAX_VOXELS = 100
    # The bug was a "keep top-N by confidence" cap that permanently favoured the
    # long-dwelt START area and starved the newest (low-confidence) areas. Model
    # that: 50 "old/start" cells at near-max log-odds + 150 "new area" cells just over
    # the RENDER gate L_DISPLAY. ALL 200 are DISPLAYABLE; the 100-cap must trip and --
    # crucially -- the kept set must include NEW-area cells (a fair random subsample),
    # NOT only the 50 high-confidence start cells. (Both groups sit at/above L_DISPLAY
    # so they actually render; the fairness is about the SAFETY-cap subsample, not the
    # display gate.)
    grid = {}
    for i in range(50):
        grid[(i, 0, 5)] = src.L_MAX                 # old/start: re-observed often
    for i in range(150):
        grid[(1000 + i, 0, 5)] = src.L_DISPLAY + 1e-3   # new area: just displayable
    src._log = grid                                            # noqa: SLF001
    src._fused_seqs = {0}                                      # noqa: SLF001
    pts, cols, _ = src._build()                                # noqa: SLF001
    _check(pts.shape[0] == 100,
           f"render thinned to the MAX_VOXELS=100 safety cap (got {pts.shape[0]})")
    _check(cols.shape == pts.shape, "colours still align after the safety thin")
    vm = src.VOXEL_M
    kept_x = np.round(pts[:, 0] / vm - 0.5).astype(int)
    n_new = int(np.sum(kept_x >= 1000))
    _check(n_new > 0,
           f"the fair thin kept NEW-area voxels too (kept {n_new} new-area cells), "
           "not just the high-confidence start cells")
    # Under the cap (no thinning) ALL occupied cells render -- the map GROWS.
    src.MAX_VOXELS = 100_000
    pts2, _, _ = src._build()                                  # noqa: SLF001
    _check(pts2.shape[0] == len(grid),
           f"under the safety cap ALL occupied cells render ({pts2.shape[0]} vs "
           f"{len(grid)}) -- no top-N starvation")


def test_map_grows_with_new_areas() -> None:
    """Regression: keyframes covering NEW spatial areas GROW the occupied set.

    Feeding keyframes whose poses translate into fresh, non-overlapping world
    regions must GROW both the occupied count AND its world bounding box (extent) --
    the map keeps extending instead of freezing at the start area.
    """
    print("\n  occupancy: NEW-area keyframes GROW the occupied set + extent (no freeze)")
    src = _make_source()
    src.STRIDE = 2

    # Helper: stash a constant-depth plane keyframe translated by ``tx`` metres
    # along the world x-axis (R=I), so each new tx places hits in a fresh region.
    def _planar_kf(seq: int, tx: float, depth_val: float = 2.0,
                   h: int = 32, w: int = 32) -> None:
        depth = np.full((h, w), float(depth_val), dtype=np.float32)
        src._kf_depth[seq] = depth                              # noqa: SLF001
        src._kf_R[seq] = np.eye(3, dtype=np.float64)            # noqa: SLF001
        src._kf_t[seq] = np.array([tx, 0.0, 0.0], np.float64)   # noqa: SLF001

    counts, x_spans = [], []
    seq = 0
    # Walk the camera down +x in 1.5 m steps (> the plane's x-footprint so each
    # station's hits land in a fresh column). We DWELL a few keyframes per station
    # (re-observing the same plane) so its hit cells climb past the RENDER gate
    # L_DISPLAY (a single hit at L_OCC=0.85 is below L_DISPLAY -- the deliberate
    # confidence filter), so each station's wall becomes DISPLAYABLE; the count +
    # extent must both keep growing as we explore new stations.
    dwell = 3                                          # keyframes per station
    for station in range(6):
        tx = 1.5 * station
        for _ in range(dwell):
            _planar_kf(seq, tx)
            seq += 1
            pts, _, cams = src._build()                         # noqa: SLF001
        _check(pts.shape[0] > 0,
               f"station {station}: voxels DISPLAYABLE after dwell")
        counts.append(pts.shape[0])
        x_spans.append(float(pts[:, 0].max() - pts[:, 0].min()))
        _check(cams.shape[0] == seq,
               f"camera path tracks every keyframe ({cams.shape[0]} vs {seq})")

    # COUNT grows: each new station adds fresh occupied cells (monotone non-
    # decreasing -- the start area never starves the new ones, the heart of the fix).
    _check(all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))
           and counts[-1] > counts[0],
           f"occupied count GROWS as new areas are explored (counts {counts})")
    # EXTENT grows: the world bounding box widens as the camera moves down +x.
    _check(x_spans[-1] > x_spans[0] + 4.0,
           f"world x-extent GROWS as the camera moves (spans {x_spans})")


def test_reemit_gate_fires_on_growth_not_on_repeat() -> None:
    """Regression: the re-emit signature fires on growth/shift, not on a repeat."""
    print("\n  occupancy: re-emit signature changes on growth/shift, stable on repeat")
    sig = IpcSlamMapSource._occ_signature                       # noqa: SLF001
    vm = IpcSlamMapSource.VOXEL_M
    base = (np.array([[0, 0, 5], [1, 0, 5], [2, 0, 5]], np.float64) + 0.5) * vm
    base = base.astype(np.float32)
    grown = np.vstack([base, ((np.array([[9, 0, 5]], np.float64) + 0.5)
                              * vm).astype(np.float32)])
    # Same COUNT, different CELLS (a spatial shift the old count-only gate missed).
    shifted = (np.array([[0, 0, 5], [1, 0, 5], [7, 0, 5]], np.float64) + 0.5) * vm
    shifted = shifted.astype(np.float32)
    _check(sig(base) == sig(base.copy()),
           "identical set -> identical signature (no needless re-upload)")
    _check(sig(base) != sig(grown),
           "a GROWN set -> different signature (re-emit when the map extends)")
    _check(sig(base) != sig(shifted),
           "a SHIFTED set at the SAME count -> different signature "
           "(the count-only gate would have missed this)")
    _check(sig(np.zeros((0, 3), np.float32)) == (0, 0),
           "empty cloud -> the sentinel (0,0) signature")


def test_green_by_height_in_range() -> None:
    print("\n  occupancy: green-by-height colour stays in [0,1]; flat-y fallback")
    src = _make_source()
    # A height span -> a gradient; all channels in [0,1], green dominant.
    y = np.array([-3.0, -1.0, 0.0, 2.0], np.float32)            # optical down
    c = src._green_by_height(y)                                 # noqa: SLF001
    _check(c.shape == (4, 3) and c.dtype == np.float32,
           f"colour is (N,3) float32 ({c.shape}, {c.dtype})")
    _check(np.all(c >= 0.0) and np.all(c <= 1.0),
           "all colour channels in [0,1]")
    _check(np.all(c[:, 1] >= c[:, 0]),
           "green channel dominates red (the ModalAI green look)")
    # A flat-y input must not divide by zero (degenerate span -> mid gradient).
    flat = src._green_by_height(np.full(3, -1.0, np.float32))   # noqa: SLF001
    _check(np.all(np.isfinite(flat)) and np.all((flat >= 0) & (flat <= 1)),
           "flat-y colour is finite + in [0,1] (no /0 on a single-height grid)")
    _check(src._green_by_height(                                # noqa: SLF001
        np.zeros(0, np.float32)).shape == (0, 3),
        "empty height -> empty (0,3) colour")


def main() -> int:
    print("occupancy_selftest: log-odds occupancy + free-space ray carving")
    test_dda_axis_aligned_contiguous_line()
    test_dda_diagonal_is_6connected_no_gaps()
    test_single_ray_free_between_occupied_end()
    test_carving_removes_a_crossed_voxel()
    test_render_confidence_gate_separates_wall_from_spray()
    test_logodds_clamped_to_band()
    test_fuse_carves_through_a_planted_noise_voxel()
    test_fusion_is_incremental_and_centres_correct()
    test_max_voxels_safety_cap_is_fair()
    test_map_grows_with_new_areas()
    test_reemit_gate_fires_on_growth_not_on_repeat()
    test_green_by_height_in_range()
    print("\nALL OCCUPANCY SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
