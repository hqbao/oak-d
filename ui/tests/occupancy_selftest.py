#!/usr/bin/env python3
"""Unit tests for the SLAM-map occupancy fusion (no IPC, no Qt, no replay).

The ModalAI-style SLAM Map (:class:`~ui.modules.ipc_sources.IpcSlamMapSource`)
builds a VOXEL OCCUPANCY map by TEMPORAL OCCUPANCY FUSION: a PERSISTENT per-voxel
hit-count grid that increments once per (keyframe, cell), and a cell is rendered
OCCUPIED only once ``hit_count >= OCC_HITS``. These tests drive that fusion with
SYNTHETIC depth keyframes (so the geometry is hand-checkable) and assert:

* a cell hit by ``>= OCC_HITS`` keyframes SURVIVES while a once-hit "noise" cell is
  DROPPED (the temporal noise filter),
* the fusion is INCREMENTAL (folding a new keyframe only adds; never rebuilds),
* the emitted points are the correct VOXEL CENTRES (index * VOXEL_M + half-cell),
* the OCC_HITS sweep is MONOTONE non-increasing (a stricter gate never grows the
  map) and OCC_HITS=1 == every touched cell,
* the MAX_VOXELS cap keeps the most-re-observed cells,
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
    (hence into a few voxels), which makes the hit-count bookkeeping checkable.
    """
    depth = np.full((h, w), float(depth_val), dtype=np.float32)
    src._kf_depth[seq] = depth                                  # noqa: SLF001
    src._kf_R[seq] = np.eye(3, dtype=np.float64)                # noqa: SLF001
    src._kf_t[seq] = np.zeros(3, dtype=np.float64)              # noqa: SLF001


def test_threshold_and_centres() -> None:
    print("\n  occupancy: >= OCC_HITS survives, once-hit noise dropped")
    src = _make_source()
    src.OCC_HITS = 3
    # A persistent grid built directly (bypass back-projection so the cells are
    # exact + hand-checkable): one "real surface" cell hit by 4 keyframes, one
    # "noise" cell hit once.
    real = (5, -2, 30)
    noise = (40, 7, 12)
    src._hits = {real: 4, noise: 1}                             # noqa: SLF001
    src._fused_seqs = {0, 1, 2, 3}     # noqa: SLF001 pretend already fused
    points, colors, cams = src._build()                         # noqa: SLF001
    _check(points.shape == (1, 3),
           f"only the >= OCC_HITS=3 cell survives (got {points.shape[0]} voxels)")
    _check(colors.shape == points.shape,
           f"colours align with voxels ({colors.shape} vs {points.shape})")
    # The emitted point is the REAL cell's centre: index * VOXEL_M + half a cell.
    vm = src.VOXEL_M
    want = (np.asarray(real, np.float64) + 0.5) * vm
    _check(np.allclose(points[0], want, atol=1e-5),
           f"voxel centre = (idx+0.5)*VOXEL_M (got {points[0]}, want {want})")
    # The once-hit noise cell is NOT in the output.
    _check(not np.any(np.all(np.isclose(
        points, (np.asarray(noise, np.float64) + 0.5) * vm), axis=1)),
        "once-hit noise cell is rejected at OCC_HITS=3")


def test_fusion_increments_per_keyframe() -> None:
    print("\n  occupancy: fusion increments ONCE per (keyframe, cell), incremental")
    src = _make_source()
    src.OCC_HITS = 1
    # Two keyframes of the SAME constant-depth plane at the SAME identity pose ->
    # the SAME cells hit each time. Each keyframe should bump every touched cell by
    # exactly +1 (a cell hit by many rays in ONE keyframe still counts once).
    _identity_kf(src, seq=0, depth_val=2.0)
    src._build()                                                # noqa: SLF001
    with src._lock:                                             # noqa: SLF001
        grid_after_1 = dict(src._hits)
        fused_1 = set(src._fused_seqs)
    _check(len(grid_after_1) > 0, "first keyframe fused some cells")
    _check(all(c == 1 for c in grid_after_1.values()),
           "every touched cell has hit_count == 1 after ONE keyframe "
           "(multi-ray hits collapse to a single +1)")
    _check(fused_1 == {0}, f"seq 0 marked fused (got {fused_1})")

    # Fold a SECOND identical keyframe: the SAME cells now have hit_count 2, and no
    # earlier cell is re-fused (the first keyframe is not re-counted).
    _identity_kf(src, seq=1, depth_val=2.0)
    src._build()                                                # noqa: SLF001
    with src._lock:                                             # noqa: SLF001
        grid_after_2 = dict(src._hits)
        fused_2 = set(src._fused_seqs)
    _check(set(grid_after_2) == set(grid_after_1),
           "the second identical keyframe touches the SAME cells (same geometry)")
    _check(all(grid_after_2[k] == 2 for k in grid_after_1),
           "each cell is now hit_count == 2 (incremental, not rebuilt)")
    _check(fused_2 == {0, 1}, f"both seqs marked fused (got {fused_2})")


def test_occ_hits_sweep_monotone() -> None:
    print("\n  occupancy: OCC_HITS sweep is monotone non-increasing; 1 == touched")
    src = _make_source()
    # A grid with a spread of hit counts (some cells hit 1..5 times).
    rng = np.random.default_rng(0)
    grid = {}
    for i in range(500):
        cell = (int(rng.integers(-50, 50)), int(rng.integers(-50, 50)),
                int(rng.integers(1, 60)))
        grid[cell] = int(rng.integers(1, 6))      # hit_count in 1..5
    src._hits = grid                                            # noqa: SLF001
    src._fused_seqs = set(range(10))                            # noqa: SLF001
    counts = []
    for occ in (1, 2, 3, 5):
        src.OCC_HITS = occ
        src._last_emit_occ = -1                                 # noqa: SLF001
        pts, _, _ = src._build()                                # noqa: SLF001
        counts.append(pts.shape[0])
    _check(counts[0] == len(grid),
           f"OCC_HITS=1 keeps every touched cell ({counts[0]} vs {len(grid)})")
    _check(all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)),
           f"a stricter OCC_HITS never grows the map (counts {counts})")
    _check(counts[-1] < counts[0],
           f"the strictest gate DROPS the count vs OCC_HITS=1 ({counts})")


def test_max_voxels_cap() -> None:
    print("\n  occupancy: MAX_VOXELS cap keeps the most-re-observed voxels")
    src = _make_source()
    src.OCC_HITS = 1
    src.MAX_VOXELS = 10
    # 50 cells: 10 "strong" (hit_count 9) + 40 "weak" (hit_count 1). The cap must
    # keep the 10 strong ones (highest hit_count = most confident).
    grid = {}
    for i in range(10):
        grid[(i, 0, 5)] = 9
    for i in range(40):
        grid[(100 + i, 0, 5)] = 1
    src._hits = grid                                            # noqa: SLF001
    src._fused_seqs = {0}                                       # noqa: SLF001
    pts, cols, _ = src._build()                                 # noqa: SLF001
    _check(pts.shape[0] == 10, f"capped to MAX_VOXELS=10 (got {pts.shape[0]})")
    _check(cols.shape == pts.shape, "colours still align after the cap")
    # Every kept voxel must be one of the 10 strong cells (x in 0..9).
    vm = src.VOXEL_M
    kept_x = np.round(pts[:, 0] / vm - 0.5).astype(int)
    _check(set(int(x) for x in kept_x) <= set(range(10)),
           "the cap kept the high-hit-count (most-re-observed) cells, not noise")


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
    print("occupancy_selftest: temporal occupancy fusion")
    test_threshold_and_centres()
    test_fusion_increments_per_keyframe()
    test_occ_hits_sweep_monotone()
    test_max_voxels_cap()
    test_green_by_height_in_range()
    print("\nALL OCCUPANCY SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
