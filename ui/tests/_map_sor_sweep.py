#!/usr/bin/env python3
"""Functional sweep: render the SLAM-map voxels at several SOR aggressiveness levels.

After the L_DISPLAY render gate the WALL shows -- but a sparse spray of ISOLATED
noise voxels can still pass the gate OUTSIDE the wall (stereo specks that carving
could not reach and that happened to be re-hit a couple of times). The principled,
standard fix is the SAME radius-outlier filter PCL/Open3D apply to a point cloud
(:meth:`~ui.modules.ipc_sources.IpcSlamMapSource._spatial_outlier_filter`): a REAL
wall is a DENSE surface (every occupied voxel has many occupied neighbours), an
isolated speck has few -- so KEEP only voxels with ``>= MIN_NEIGHBORS`` neighbours in
the ``(2*NEIGHBOR_RADIUS+1)^3`` box. The dense walls survive; the lone specks drop.

This probe boots imu_camera(replay) + vio + slam over IPC, drives a REAL
:class:`~ui.modules.ipc_sources.IpcSlamMapSource` exactly as the live UI does (folding
each keyframe into the persistent log-odds grid), and -- once the whole corridor has
been traversed -- renders the DISPLAYED voxels at a SWEEP of MIN_NEIGHBORS values into
PNGs, BOTH:

* a TOP-DOWN view (looking down the world DOWN axis -- the floor plan), and
* a SIDE view looking ALONG the wall (so "outside the wall" is obvious: the wall is a
  tight band, the isolated noise is a spray around/outside it).

MIN_NEIGHBORS=0 renders the unfiltered (L_DISPLAY-only) set as the BEFORE baseline.
For each value it prints the displayed voxel COUNT and the per-build cost the SOR adds
(measured by timing ``_spatial_outlier_filter`` over the full displayed set); read the
PNGs to pick the value that removes the outside-wall spray while keeping the walls
solid + connected.

The voxels are rasterised with the SAME tiny pure-numpy orthographic projection +
stdlib ``zlib`` PNG writer as ``_map_display_sweep`` -- NO matplotlib / PIL / Qt / GL
(no new deps, fully off-screen, deterministic).

This is a developer/tester probe (not part of the assertion selftest).

Run::

    python -m ui.tests._map_sor_sweep --out /tmp/mapsor
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.main import _await_calib_bundle                              # noqa: E402
from ui.modules.ipc_sources import IpcSlamMapSource                  # noqa: E402
# Reuse the dependency-free PNG writer + orthographic rasteriser from the existing
# display-gate sweep so this probe adds no duplicate plumbing.
from ui.tests._map_display_sweep import _rasterise, _write_png       # noqa: E402


# --------------------------------------------------------------------------- #
def _displayed_cells(src: IpcSlamMapSource):
    """The (N,3) int voxel keys passing the L_DISPLAY gate (the SOR's INPUT set).

    Reads the persistent grid directly so the sweep does not perturb it; this is
    exactly the set ``_build`` hands to the spatial outlier filter.
    """
    with src._lock:                                     # noqa: SLF001
        occ = [k for k, lo in src._log.items()          # noqa: SLF001
               if lo >= src.L_DISPLAY]
    return (np.asarray(occ, np.int64) if occ
            else np.zeros((0, 3), np.int64))


def _voxels_at(src: IpcSlamMapSource, keys_in: np.ndarray, min_neighbors: int):
    """Apply the SOR at ``min_neighbors`` and return (centres, colours, build_ms).

    ``keys_in`` is the L_DISPLAY-gated set; we time JUST the filter (the per-build
    cost the SOR adds on top of the existing build) and return the surviving voxel
    CENTRES + green-by-height colours, matching ``_build``.
    """
    src.MIN_NEIGHBORS = int(min_neighbors)
    t0 = time.perf_counter()
    kept = src._spatial_outlier_filter(keys_in.copy())  # noqa: SLF001
    sor_ms = (time.perf_counter() - t0) * 1e3
    if kept.shape[0] == 0:
        return (np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.float32), sor_ms)
    pts = ((kept.astype(np.float32) + 0.5) * np.float32(src.VOXEL_M))
    colors = src._green_by_height(pts[:, 1])            # noqa: SLF001
    return pts, colors, sor_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="sessions/gold/corridor_60s")
    ap.add_argument("--max-frames", type=int, default=1199)
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--out", default="/tmp/mapsor")
    ap.add_argument("--min-neighbors", default="0,3,6,10",
                    help="comma-separated MIN_NEIGHBORS values to render (0 = off)")
    args = ap.parse_args()
    mn_values = [int(v) for v in args.min_neighbors.split(",")]
    os.makedirs(args.out, exist_ok=True)

    pid = os.getpid()
    cap_ep = f"oak.cap.s{pid & 0xFFF:x}"
    vio_ep = f"oak.vio.s{pid & 0xFFF:x}"
    slam_ep = f"oak.slm.s{pid & 0xFFF:x}"
    py = sys.executable
    env = dict(os.environ)
    lk = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}

    vio_proc = subprocess.Popen(
        [py, "-m", "vio.main", "--capture-endpoint", cap_ep,
         "--endpoint", vio_ep, "--kf-every", str(args.kf_every)], env=env, **lk)
    slam_proc = subprocess.Popen(
        [py, "-m", "slam.main", "--vio-endpoint", vio_ep,
         "--endpoint", slam_ep], env=env, **lk)
    time.sleep(0.3)
    cap_proc = subprocess.Popen(
        [py, "-m", "imu_camera.main", "--endpoint", cap_ep,
         "--session", args.session, "--max-frames", str(args.max_frames)],
        env=env, **lk)
    procs = (cap_proc, vio_proc, slam_proc)

    src = None
    try:
        bundle = _await_calib_bundle(vio_ep, timeout_s=20.0)
        _await_calib_bundle(slam_ep, timeout_s=20.0)
        W, H = int(bundle.width), int(bundle.height)
        print(f"  capture {W}x{H}, kf_every={args.kf_every}, "
              f"VOXEL_M={IpcSlamMapSource.VOXEL_M}, "
              f"L_DISPLAY={IpcSlamMapSource.L_DISPLAY}, "
              f"NEIGHBOR_RADIUS={IpcSlamMapSource.NEIGHBOR_RADIUS}, "
              f"default MIN_NEIGHBORS={IpcSlamMapSource.MIN_NEIGHBORS}\n")

        src = IpcSlamMapSource(vio_ep, slam_ep, bundle.K, width=W, height=H,
                               connect_timeout_s=20.0)
        src.start_cloud(lambda *a: None)                # accumulate; no live render
        if src.error:
            print(f"  source error: {src.error}")
            return 1

        cap_proc.wait(timeout=180.0)
        vio_proc.wait(timeout=180.0)
        slam_proc.wait(timeout=180.0)
        time.sleep(1.0)                                 # drain the last keyframes

        # Fold any not-yet-fused keyframes (force a final full fuse), then sweep.
        src.MIN_NEIGHBORS = 0                           # neutral fold (no prune)
        src._last_emit_sig = (-1, 0)                    # noqa: SLF001
        src._build()                                    # noqa: SLF001 fold remaining
        with src._lock:                                 # noqa: SLF001
            n_kf = len(src._kf_depth)
            n_cells = len(src._log)
        keys_in = _displayed_cells(src)
        print(f"  accumulated keyframes: {n_kf}, grid cells: {n_cells}, "
              f"L_DISPLAY-gated voxels (SOR input): {keys_in.shape[0]}\n")

        # The grid lives in the optical-world frame: +x right, +y DOWN, +z forward.
        # TOP-DOWN (floor plan) = the x-z plane (look down +y). SIDE-ALONG-WALL: the
        # corridor runs along z, so look along x -> the z-y plane shows the wall as a
        # tight band and the isolated noise as a spray outside it.
        print(f"  {'MIN_NEIGH':>10}  {'voxels':>8}  {'kept%':>6}  "
              f"{'SOR ms':>7}   PNGs")
        n_in = keys_in.shape[0]
        for mn in mn_values:
            pts, cols, sor_ms = _voxels_at(src, keys_in, mn)
            n = pts.shape[0]
            pct = (100.0 * n / n_in) if n_in else 0.0
            tag = f"mn{mn:02d}"
            top = os.path.join(args.out, f"top_{tag}.png")
            side = os.path.join(args.out, f"side_{tag}.png")
            # TOP-DOWN: horizontal = x, vertical = z (forward up the screen).
            _write_png(top, _rasterise(pts[:, 0], pts[:, 2], cols))
            # SIDE: horizontal = z (along corridor), vertical = -y (world UP).
            _write_png(side, _rasterise(pts[:, 2], -pts[:, 1], cols))
            print(f"  {mn:>10d}  {n:>8}  {pct:>5.1f}  {sor_ms:>7.1f}   "
                  f"{top}  {side}")
        print(f"\n  wrote PNGs to {args.out}/  (top_*.png + side_*.png per value)")
        print("  (MIN_NEIGHBORS=0 is the unfiltered BEFORE baseline; the SOR ms is "
              "the per-build cost the filter ADDS.)")
        return 0
    finally:
        if src is not None:
            try:
                src.stop()
            except Exception:                                      # noqa: BLE001
                pass
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:                                  # noqa: BLE001
                    pass
        for p in procs:
            try:
                p.wait(timeout=5.0)
            except Exception:                                      # noqa: BLE001
                try:
                    p.kill()
                except Exception:                                  # noqa: BLE001
                    pass
        for name, p in (("cap", cap_proc), ("vio", vio_proc),
                        ("slam", slam_proc)):
            try:
                _o, e = p.communicate(timeout=2.0)
            except Exception:                                      # noqa: BLE001
                e = b""
            if e and e.strip():
                print(f"\n  --- {name}.stderr (tail) ---\n"
                      f"{e.decode(errors='replace')[-600:]}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
