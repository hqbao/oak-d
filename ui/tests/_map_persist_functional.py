#!/usr/bin/env python3
"""Functional probe: IpcSlamMapSource persistence-filter on a real replay.

Boots imu_camera(replay) + vio + slam over IPC on the corridor_60s session,
drives a real :class:`~ui.modules.ipc_sources.IpcSlamMapSource` so it accumulates
the per-keyframe gray/depth + VIO poses + tracks, then runs ``_build_cloud`` at
several ``PERSIST_KF`` thresholds -- reporting the output point count + rebuild
time at each. The cloud positions every landmark from the keyframe's own VIO pose
over the DENSE VIO keyframe set (not the sparse slam.map subset), so the map
populates. This is a developer/tester probe (not part of the assertion selftest);
it prints numbers for the report.

Run::

    python -m ui.tests._map_persist_functional --max-frames 240 --kf-every 8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.main import _await_calib_bundle                              # noqa: E402
from ui.modules.ipc_sources import IpcSlamMapSource                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="sessions/gold/corridor_60s")
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--kf-every", type=int, default=8)
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
              f"max_frames={args.max_frames}")

        # Drive a REAL source so it accumulates keyframes + corrected poses, but
        # give it a no-op cloud sink (we call _build_cloud ourselves below).
        src = IpcSlamMapSource(vio_ep, slam_ep, bundle.K, width=W, height=H,
                               connect_timeout_s=20.0)
        src.start_cloud(lambda p, c, cams: None)
        if src.error:
            print(f"  source error: {src.error}")
            return 1

        # Drain the whole capped replay so all keyframes land.
        cap_proc.wait(timeout=120.0)
        vio_proc.wait(timeout=120.0)
        slam_proc.wait(timeout=120.0)
        time.sleep(1.0)                          # let the last keyframes drain

        with src._lock:                          # noqa: SLF001 (probe)
            n_kf = len(src._kf_gray)
            # slam.map only places a SPARSE subset; the cloud no longer gates on it
            # (positions come from each keyframe's own VIO pose), so this is just a
            # diagnostic of how sparse SLAM's keyframe set is vs the VIO set.
            n_placed = sum(1 for s in src._kf_gray if s in src._kf_corr_pos)
        print(f"  accumulated VIO keyframes: {n_kf} "
              f"({n_placed} also placed by slam.map)\n")

        # Sweep PERSIST_KF; time only the build (input is identical each time).
        for persist in (3, 6, 10):
            src.PERSIST_KF = persist             # instance override of the const
            # Warm + timed run (numpy unique JIT-free, but be fair: 2 runs).
            src._build_cloud()                   # noqa: SLF001 warm
            t0 = time.perf_counter()
            pts, cols, cams = src._build_cloud() # noqa: SLF001
            dt = time.perf_counter() - t0
            print(f"  PERSIST_KF={persist:>2}: {pts.shape[0]:>7} points  "
                  f"({cams.shape[0]} cams)  rebuild={dt * 1e3:7.1f} ms")
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
                print(f"\n  --- {name}.stderr ---\n{e.decode(errors='replace')}",
                      file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
