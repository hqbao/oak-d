#!/usr/bin/env python3
"""Functional sweep: render the SLAM-map voxels at several L_DISPLAY render gates.

The behind-the-wall noise problem: the log-odds occupancy grid carves FREE space
correctly, so the WALL shows -- but carving CANNOT reach the sporadic stereo spray
BEHIND the wall (rays stop at the wall surface; nothing crosses the space behind it).
The wall is a consistently-observed surface (HIGH log-odds confidence); the spray is
hit once or twice (LOW confidence). The fix is a SEPARATE, higher RENDER gate
``L_DISPLAY``: keep the occupancy UPDATE math unchanged (carving stays correct) but
RENDER only voxels with ``log_odds >= L_DISPLAY``.

This probe boots imu_camera(replay) + vio + slam over IPC, drives a REAL
:class:`~ui.modules.ipc_sources.IpcSlamMapSource` exactly as the live UI does (folding
each keyframe into the persistent log-odds grid), and -- once the whole corridor has
been traversed -- renders the OCCUPIED voxels at a SWEEP of L_DISPLAY values into
PNGs, BOTH:

* a TOP-DOWN view (looking down the world DOWN axis -- the floor plan), and
* a SIDE view looking ALONG the wall (so "behind the wall" is obvious: the wall is a
  tight vertical line, the noise is a tail spilling behind it).

The voxels are rasterised with a tiny pure-numpy orthographic projection + a stdlib
``zlib`` PNG writer -- NO matplotlib / PIL / Qt / GL (no new deps, fully off-screen,
deterministic). For each L_DISPLAY it prints the displayed voxel COUNT; read the PNGs
to pick the gate where the wall stays crisp but the behind-wall tail is gone.

This is a developer/tester probe (not part of the assertion selftest).

Run::

    python -m ui.tests._map_display_sweep --out /tmp/mapsweep
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.main import _await_calib_bundle                              # noqa: E402
from ui.modules.ipc_sources import IpcSlamMapSource                  # noqa: E402


# --------------------------------------------------------------------------- #
# Minimal, dependency-free PNG writer (stdlib zlib only).
# --------------------------------------------------------------------------- #
def _write_png(path: str, rgb: np.ndarray) -> None:
    """Write an ``(H,W,3)`` uint8 RGB array to a PNG (stdlib zlib only)."""
    h, w, _ = rgb.shape
    rows = bytearray()
    for y in range(h):
        rows.append(0)                                  # filter type 0 (none)
        rows.extend(rgb[y].tobytes())
    comp = zlib.compress(bytes(rows), 9)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", comp))
        f.write(_chunk(b"IEND", b""))


def _rasterise(ax_h: np.ndarray, ax_v: np.ndarray, colors: np.ndarray, *,
               px: int = 700, pad: int = 24, pt: int = 2,
               bg=(16, 18, 22)) -> np.ndarray:
    """Orthographic scatter of 2-D points (``ax_h``,``ax_v``) into an RGB image.

    A square ``2*pt+1`` px is painted per point so sparse clouds stay visible. The
    vertical screen axis is flipped (image y grows DOWN) so "up" reads up. Equal
    world scale on both axes keeps the wall's aspect honest. Empty -> a blank frame.
    """
    img = np.empty((px, px, 3), np.uint8)
    img[:] = np.array(bg, np.uint8)
    n = ax_h.shape[0]
    if n == 0:
        return img
    lo_h, hi_h = float(ax_h.min()), float(ax_h.max())
    lo_v, hi_v = float(ax_v.min()), float(ax_v.max())
    span = max(hi_h - lo_h, hi_v - lo_v, 1e-6)          # equal scale both axes
    scale = (px - 2 * pad) / span
    cx = (px - (hi_h + lo_h) * scale) * 0.5             # centre the content
    cy = (px - (hi_v + lo_v) * scale) * 0.5
    sx = np.clip((ax_h * scale + cx).astype(int), pt, px - pt - 1)
    # Flip vertical: world-up -> image-up.
    sy = np.clip((px - 1 - (ax_v * scale + cy)).astype(int), pt, px - pt - 1)
    col = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    for dx in range(-pt, pt + 1):
        for dy in range(-pt, pt + 1):
            img[sy + dy, sx + dx] = col
    return img


# --------------------------------------------------------------------------- #
def _voxels_at(src: IpcSlamMapSource, l_display: float):
    """The (centres, green-by-height colours) for cells with log_odds >= l_display.

    Reads the persistent grid directly so the sweep does not perturb it; centres are
    (idx + 0.5) * VOXEL_M in the optical-world frame, matching ``_build``.
    """
    with src._lock:                                     # noqa: SLF001
        occ = [k for k, lo in src._log.items() if lo >= l_display]  # noqa: SLF001
    if not occ:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
    keys = np.asarray(occ, np.int64)
    pts = ((keys.astype(np.float32) + 0.5) * np.float32(src.VOXEL_M))
    colors = src._green_by_height(pts[:, 1])            # noqa: SLF001
    return pts, colors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="sessions/gold/corridor_60s")
    ap.add_argument("--max-frames", type=int, default=1199)
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--out", default="/tmp/mapsweep")
    ap.add_argument("--gates", default="0.5,1.5,2.0,2.5",
                    help="comma-separated L_DISPLAY values to render")
    args = ap.parse_args()
    gates = [float(g) for g in args.gates.split(",")]
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
              f"VOXEL_M={IpcSlamMapSource.VOXEL_M}, L_OCC={IpcSlamMapSource.L_OCC}, "
              f"L_FREE={IpcSlamMapSource.L_FREE}, L_MAX={IpcSlamMapSource.L_MAX}, "
              f"L_OCC_THRESH={IpcSlamMapSource.L_OCC_THRESH}, "
              f"default L_DISPLAY={IpcSlamMapSource.L_DISPLAY}\n")

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
        src._last_emit_sig = (-1, 0)                    # noqa: SLF001
        t0 = time.perf_counter()
        src._build()                                    # noqa: SLF001 fold remaining
        rebuild_ms = (time.perf_counter() - t0) * 1e3
        with src._lock:                                 # noqa: SLF001
            n_kf = len(src._kf_depth)
            n_cells = len(src._log)
        print(f"  accumulated keyframes: {n_kf}, grid cells: {n_cells}, "
              f"final fold {rebuild_ms:.1f} ms\n")

        # The grid lives in the optical-world frame: +x right, +y DOWN, +z forward.
        # TOP-DOWN (floor plan) = the x-z plane (look down +y). SIDE-ALONG-WALL: the
        # corridor runs along z, so look along x -> the z-y plane shows the wall as a
        # vertical (y) line and the behind-wall spray as a tail in z.
        print(f"  {'L_DISPLAY':>10}  {'voxels':>8}   PNGs")
        for g in gates:
            pts, cols = _voxels_at(src, g)
            n = pts.shape[0]
            tag = f"{g:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
            top = os.path.join(args.out, f"top_{tag}.png")
            side = os.path.join(args.out, f"side_{tag}.png")
            # TOP-DOWN: horizontal = x, vertical = z (forward up the screen).
            _write_png(top, _rasterise(pts[:, 0], pts[:, 2], cols))
            # SIDE: horizontal = z (along corridor), vertical = -y (world UP).
            _write_png(side, _rasterise(pts[:, 2], -pts[:, 1], cols))
            print(f"  {g:>10.1f}  {n:>8}   {top}  {side}")
        print(f"\n  wrote PNGs to {args.out}/  (top_*.png + side_*.png per gate)")
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
