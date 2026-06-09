#!/usr/bin/env python3
"""Functional probe: the SLAM Map (3D room) GROWS as the camera explores.

This is the regression probe for the "frozen map" bug. It boots
imu_camera(replay) + vio + slam over IPC, drives a REAL
:class:`~ui.modules.ipc_sources.IpcSlamMapSource` exactly as the live UI does
(keyframes arrive incrementally; the source folds each into the persistent
hit-count grid and re-emits via its signature gate), and -- as the replay runs --
samples the source every few keyframes to log:

* the rendered VOXEL COUNT (occupied cells emitted to the viewer), and
* the WORLD-EXTENT (axis-aligned bounding box) of those voxels, and
* the per-tick rebuild time.

The bug was that BOTH the count and the extent were frozen at the START area
(the top-N-by-hit_count cap locked the displayed set there + the count-only
re-emit gate never fired again). With the fix (render ALL occupied cells, raise
OCC_HITS, fair safety cap, growth/shift re-emit gate) the count + extent must
GROW as the camera traverses the corridor, then PLATEAU (a bounded room).

This is a developer/tester probe (not part of the assertion selftest); it prints
numbers for the report.

Run::

    python -m ui.tests._map_growth_functional --max-frames 1199 --kf-every 5
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


def _extent(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Return (dx, dy, dz, diag) of the voxel bounding box (m); 0 if empty."""
    if pts.shape[0] == 0:
        return (0.0, 0.0, 0.0, 0.0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    d = (hi - lo).astype(float)
    diag = float(np.linalg.norm(d))
    return (float(d[0]), float(d[1]), float(d[2]), diag)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="sessions/gold/corridor_60s")
    ap.add_argument("--max-frames", type=int, default=1199)   # whole corridor
    ap.add_argument("--kf-every", type=int, default=5)
    args = ap.parse_args()

    pid = os.getpid()
    cap_ep = f"oak.cap.g{pid & 0xFFF:x}"
    vio_ep = f"oak.vio.g{pid & 0xFFF:x}"
    slam_ep = f"oak.slm.g{pid & 0xFFF:x}"
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
    # Capture every emit (count, extent, rebuild-ms) the source pushes live.
    samples: list[tuple[int, int, tuple, float]] = []
    emit_n = {"i": 0}

    try:
        bundle = _await_calib_bundle(vio_ep, timeout_s=20.0)
        _await_calib_bundle(slam_ep, timeout_s=20.0)
        W, H = int(bundle.width), int(bundle.height)
        print(f"  capture {W}x{H}, kf_every={args.kf_every}, "
              f"max_frames={args.max_frames}, VOXEL_M={IpcSlamMapSource.VOXEL_M}, "
              f"OCC_HITS={IpcSlamMapSource.OCC_HITS}, "
              f"MAX_VOXELS={IpcSlamMapSource.MAX_VOXELS}\n")

        src = IpcSlamMapSource(vio_ep, slam_ep, bundle.K, width=W, height=H,
                               connect_timeout_s=20.0)

        def on_cloud(p, c, cams):
            # The source calls this from its rebuild thread on every CHANGED emit;
            # record the live count + extent so we can show they GROW over time.
            emit_n["i"] += 1
            with src._lock:                          # noqa: SLF001
                n_kf = len(src._kf_depth)
            samples.append((n_kf, int(p.shape[0]), _extent(p), 0.0))

        src.start_cloud(on_cloud)
        if src.error:
            print(f"  source error: {src.error}")
            return 1

        # Drain the whole capped replay so the camera traverses the corridor.
        cap_proc.wait(timeout=180.0)
        vio_proc.wait(timeout=180.0)
        slam_proc.wait(timeout=180.0)
        time.sleep(1.0)                              # let the last keyframes drain

        with src._lock:                              # noqa: SLF001
            n_kf = len(src._kf_depth)
        print(f"  accumulated VIO keyframes: {n_kf}, live emits: {len(samples)}\n")

        # Time a clean full rebuild on the final grid (the per-tick render cost).
        src._last_emit_sig = (-1, 0)                 # noqa: SLF001 force a re-count
        t0 = time.perf_counter()
        final_pts, _, final_cams = src._build()      # noqa: SLF001
        rebuild_ms = (time.perf_counter() - t0) * 1e3

        # Report the LIVE count + extent trajectory: print every ~20 keyframes so
        # the GROWTH (then plateau) is visible -- the proof the map is not frozen.
        print("  rendered voxel count + world extent over time "
              "(sampled every ~20 keyframes):")
        print(f"    {'kf':>5}  {'voxels':>8}  "
              f"{'dx':>6} {'dy':>6} {'dz':>6} {'diag':>6}  (m)")
        last_kf = -999
        for (kf, n, ext, _ms) in samples:
            if kf - last_kf < 20 and kf != n_kf:
                continue
            last_kf = kf
            dx, dy, dz, diag = ext
            print(f"    {kf:>5}  {n:>8}  "
                  f"{dx:>6.2f} {dy:>6.2f} {dz:>6.2f} {diag:>6.2f}")

        # Hard evidence the bug is fixed: count + extent at the END are well above
        # the START (a frozen map would keep the first sample's values).
        if len(samples) >= 2:
            n0, e0 = samples[0][1], samples[0][2][3]
            n1, e1 = samples[-1][1], samples[-1][2][3]
            print(f"\n  GROWTH: voxels {n0} -> {n1}  "
                  f"({'GREW' if n1 > n0 else 'FROZEN'});  "
                  f"extent-diag {e0:.2f}m -> {e1:.2f}m  "
                  f"({'GREW' if e1 > e0 + 0.5 else 'FROZEN'})")
        print(f"  final: {final_pts.shape[0]} rendered voxels, "
              f"{final_cams.shape[0]} cams, full rebuild {rebuild_ms:.1f} ms  "
              f"(<= MAX_VOXELS safety cap? "
              f"{'yes' if final_pts.shape[0] <= IpcSlamMapSource.MAX_VOXELS else 'NO'})")
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
                      f"{e.decode(errors='replace')[-800:]}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
