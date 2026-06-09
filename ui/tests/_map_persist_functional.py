#!/usr/bin/env python3
"""Functional probe: log-odds CARVING vs NO-carving on a real moving-camera replay.

Boots imu_camera(replay) + vio + slam over IPC, drives a real
:class:`~ui.modules.ipc_sources.IpcSlamMapSource` so it accumulates the
per-keyframe depth + VIO poses, then builds the occupancy map TWICE on the SAME
accumulated keyframes:

* WITH free-space ray carving (the real log-odds fusion), and
* WITHOUT carving (``_carve_free_cells`` monkeypatched to return no free cells, so
  only the hit (occupied) evidence is folded -- a stand-in for the old
  hit-only map that never removed a wrongly-added voxel).

Carving must REMOVE noise voxels -> a LOWER, CLEANER occupied count than the
no-carving build. It also reports the per-keyframe fuse time (the carving cost,
which must stay off-thread-affordable) and renders the occupied points to a
TOP-DOWN PNG (pure numpy, no GL) so the filled-cone noise being carved away is
eyeball-able vs the no-carving blob.

This is a developer/tester probe (not part of the assertion selftest); it prints
numbers + writes PNGs for the report.

Run::

    python -m ui.tests._map_persist_functional --max-frames 1199 --kf-every 5
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


def _fold_grid(K, kf, *, carve: bool) -> tuple[dict, list[float]]:
    """Fold every accumulated keyframe into a FRESH log-odds grid.

    ``kf`` is the snapshot ``{seq: (depth, R, t)}``. When ``carve`` is False the
    free-space DDA is bypassed (``_carve_free_cells`` -> empty), so only the hit
    (occupied) evidence is added -- the stand-in for the old never-remove map.
    Returns the persistent grid + the per-keyframe fuse times (ms).
    """
    src = IpcSlamMapSource("x", "y", K)          # un-connected: no thread, no IPC
    if not carve:
        # No-carve baseline: every ray contributes ONLY its hit voxel (no free cells
        # along the ray), so nothing is ever removed -- exactly the old behaviour.
        src._carve_free_cells = lambda C, P: np.empty((0, 3), np.int64)  # noqa: SLF001
    times = []
    for seq in sorted(kf):
        depth, R, t = kf[seq]
        t0 = time.perf_counter()
        with src._lock:                          # noqa: SLF001 (probe)
            src._fuse_keyframe_locked(depth, R, t)   # noqa: SLF001
        times.append((time.perf_counter() - t0) * 1e3)
    return dict(src._log), times                 # noqa: SLF001


def _occupied_centres(grid: dict, vm: float, thresh: float) -> np.ndarray:
    """Occupied voxel CENTRES (N,3) from a log-odds grid (>= thresh)."""
    occ = [k for k, lo in grid.items() if lo >= thresh]
    if not occ:
        return np.zeros((0, 3), np.float32)
    keys = np.asarray(occ, np.int64)
    return ((keys.astype(np.float32) + 0.5) * np.float32(vm))


def _topdown_png(pts: np.ndarray, path: str, *, res: int = 600,
                 margin: float = 0.5) -> None:
    """Render occupied points top-down (x-z plane) to a grayscale PNG (numpy only).

    A density histogram of the occupied voxel centres projected onto the floor
    plane: a CLEAN map shows thin wall/structure lines; a noisy blob (filled cone)
    shows a dense filled patch. No GL, no matplotlib -- writes a minimal 8-bit PNG.
    """
    if pts.shape[0] == 0:
        img = np.zeros((res, res), np.uint8)
    else:
        x, z = pts[:, 0], pts[:, 2]
        lo = np.array([x.min(), z.min()]) - margin
        hi = np.array([x.max(), z.max()]) + margin
        span = np.maximum(hi - lo, 1e-3)
        ix = ((x - lo[0]) / span[0] * (res - 1)).astype(int)
        iz = ((z - lo[1]) / span[1] * (res - 1)).astype(int)
        hist = np.zeros((res, res), np.int64)
        np.add.at(hist, (iz, ix), 1)             # row = z, col = x
        # Log-scale the density so a single-voxel wall is visible next to a blob.
        img = (np.log1p(hist) / np.log1p(hist.max() if hist.max() else 1)
               * 255).astype(np.uint8)
    _write_png_gray(img, path)


def _write_png_gray(img: np.ndarray, path: str) -> None:
    """Write an 8-bit grayscale (H,W) numpy array as a PNG (stdlib zlib + struct)."""
    import struct
    import zlib
    h, w = img.shape
    raw = bytearray()
    for row in img:                              # filter byte 0 (none) per scanline
        raw.append(0)
        raw.extend(row.tobytes())

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)   # 8-bit grayscale
    idat = zlib.compress(bytes(raw), 9)
    with open(path, "wb") as f:
        f.write(sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat)
                + _chunk(b"IEND", b""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="sessions/gold/corridor_60s")
    ap.add_argument("--max-frames", type=int, default=1199)   # whole corridor
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--png-dir", default="/tmp")
    args = ap.parse_args()

    pid = os.getpid()
    cap_ep = f"oak.cap.p{pid & 0xFFF:x}"
    vio_ep = f"oak.vio.p{pid & 0xFFF:x}"
    slam_ep = f"oak.slm.p{pid & 0xFFF:x}"
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
              f"max_frames={args.max_frames}, VOXEL_M={IpcSlamMapSource.VOXEL_M}, "
              f"L_OCC={IpcSlamMapSource.L_OCC}, L_FREE={IpcSlamMapSource.L_FREE}, "
              f"L_OCC_THRESH={IpcSlamMapSource.L_OCC_THRESH}")

        # Drive a REAL source so it accumulates keyframes + corrected poses, but
        # give it a no-op sink (we fold them ourselves below, twice).
        src = IpcSlamMapSource(vio_ep, slam_ep, bundle.K, width=W, height=H,
                               connect_timeout_s=20.0)
        src.start_cloud(lambda p, c, cams: None)
        if src.error:
            print(f"  source error: {src.error}")
            return 1

        # Drain the whole capped replay so the camera traverses the corridor.
        cap_proc.wait(timeout=180.0)
        vio_proc.wait(timeout=180.0)
        slam_proc.wait(timeout=180.0)
        time.sleep(1.0)                          # let the last keyframes drain

        with src._lock:                          # noqa: SLF001 (probe)
            kf = {s: (src._kf_depth[s].copy(),   # noqa: SLF001
                      src._kf_R[s].copy(), src._kf_t[s].copy())
                  for s in src._kf_depth}         # noqa: SLF001
        n_kf = len(kf)
        print(f"  accumulated VIO keyframes: {n_kf}\n")
        if n_kf == 0:
            print("  no keyframes accumulated -- cannot compare")
            return 1

        vm = IpcSlamMapSource.VOXEL_M
        th = IpcSlamMapSource.L_OCC_THRESH

        # Fold the SAME keyframes twice: with carving (real) and without (hit-only).
        grid_carve, t_carve = _fold_grid(bundle.K, kf, carve=True)
        grid_nocarve, t_nocarve = _fold_grid(bundle.K, kf, carve=False)

        occ_carve = _occupied_centres(grid_carve, vm, th)
        occ_nocarve = _occupied_centres(grid_nocarve, vm, th)

        def _stats(t):
            a = np.asarray(t)
            return (a.mean(), np.percentile(a, 95), a.max()) if a.size else (0, 0, 0)

        m_c, p95_c, mx_c = _stats(t_carve)
        m_n, p95_n, mx_n = _stats(t_nocarve)

        print("  per-keyframe FUSE time (ms) -- the carving cost (off-GUI-thread):")
        print(f"    with carving : mean={m_c:6.1f}  p95={p95_c:6.1f}  "
              f"max={mx_c:6.1f}")
        print(f"    no carving   : mean={m_n:6.1f}  p95={p95_n:6.1f}  "
              f"max={mx_n:6.1f}\n")

        print("  OCCUPIED voxel count -- carving must REMOVE noise (lower, cleaner):")
        print(f"    no carving (hit-only)   : {occ_nocarve.shape[0]:>8} voxels")
        print(f"    with carving (log-odds) : {occ_carve.shape[0]:>8} voxels")
        if occ_nocarve.shape[0]:
            pct = 100.0 * (occ_nocarve.shape[0] - occ_carve.shape[0]) \
                / occ_nocarve.shape[0]
            print(f"    -> carving removed {occ_nocarve.shape[0] - occ_carve.shape[0]}"
                  f" voxels ({pct:.1f}% cleaner)  "
                  f"{'GOOD' if occ_carve.shape[0] < occ_nocarve.shape[0] else 'NO-OP'}")
        print(f"    persistent grid cells   : carve={len(grid_carve)}  "
              f"no-carve={len(grid_nocarve)}\n")

        # Top-down PNGs so the filled-cone noise (no-carve) vs cleaned (carve) is
        # eyeball-able without GL.
        png_c = os.path.join(args.png_dir, "slam_map_topdown_carve.png")
        png_n = os.path.join(args.png_dir, "slam_map_topdown_nocarve.png")
        _topdown_png(occ_carve, png_c)
        _topdown_png(occ_nocarve, png_n)
        print(f"  top-down PNGs written:\n    carve   : {png_c}\n"
              f"    no-carve: {png_n}")
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
            if e.strip():
                print(f"\n  --- {name}.stderr (tail) ---\n"
                      f"{e.decode(errors='replace')[-800:]}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
