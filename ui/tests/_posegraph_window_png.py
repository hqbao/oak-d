#!/usr/bin/env python3
"""End-to-end PNG proof for the "Pose Graph" before/after window (ALGORITHMS.md §4.3).

Boots the SPLIT 3-process stack (imu_camera replay + vio + slam LIVE) on a gold
session WITH loops, drives the REAL
:class:`~ui.modules.ipc_sources.IpcPoseGraphSource` (it joins VIO's raw
``pose.odom`` = BEFORE, SLAM's ``slam.map`` corrected poses = AFTER, and SLAM's
``slam.loop`` edges), renders the window's 2D top-down image with the REAL
:func:`~ui.viz.posegraph_render.render_pose_graph`, and writes it to a PNG. No
OpenGL is involved (the pose-graph view is pure 2D), so this runs headless.

Asserts a REAL loop-closure snapshot was assembled (>=2 anchored nodes, >=1 loop
edge, a non-trivial per-node correction so the before/after actually differs),
that the source's slider buffer accumulated it, and that the rendered PNG is
non-blank with the loop-edge chord + correction arrows drawn AND the before/after
toggle visibly changing pixels.

This is a PURE CONSUMER of existing topics (no new IPC field / data-path change),
so it needs no special flag on any process -- exactly like the slam.map / loop
windows. Run::

    .venv/bin/python -m ui.tests._posegraph_window_png
    .venv/bin/python -m ui.tests._posegraph_window_png --out /tmp/pose_graph.png
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.comms import IPCPubSub                                    # noqa: E402
from ui.modules import IpcPoseGraphSource                        # noqa: E402
from ui.viz.posegraph_render import (                            # noqa: E402
    render_pose_graph, _LOOP, _WARN)


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def _await_calib(endpoint: str, timeout_s: float):
    got = threading.Event()
    box = [None]

    def on(wm):
        box[0] = wm
        got.set()
    c = IPCPubSub(endpoint, role="client", connect_timeout_s=timeout_s)
    c.subscribe("calib.bundle", on)
    c.start()
    try:
        if not got.wait(timeout=timeout_s):
            raise TimeoutError(f"no calib.bundle from {endpoint!r}")
    finally:
        c.stop()
    return box[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--out", default="/tmp/pose_graph_lab_loop_30s.png")
    args = ap.parse_args()

    pid = os.getpid()
    cap_ep = f"oak.cap.pg{pid & 0xFFF:x}"
    vio_ep = f"oak.vio.pg{pid & 0xFFF:x}"
    slam_ep = f"oak.slm.pg{pid & 0xFFF:x}"
    py = sys.executable
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    lk = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}

    print("posegraph_window_png")
    print(f"  session={args.session} max-frames={args.max_frames}")

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

    snaps: list = []
    lock = threading.Lock()
    src = None
    try:
        bundle = _await_calib(slam_ep, timeout_s=25.0)
        W, H = int(bundle.width), int(bundle.height)
        print(f"  slam ready ({W}x{H})")

        # The REAL window source: pose.odom (before) + slam.map (after) + slam.loop.
        def on_snap(s) -> None:
            with lock:
                snaps.append(s)

        src = IpcPoseGraphSource(vio_ep, slam_ep, connect_timeout_s=25.0)
        src.start(on_snap)

        cap_proc.wait(timeout=180.0)
        vio_proc.wait(timeout=180.0)
        slam_proc.wait(timeout=180.0)
        time.sleep(1.0)                            # drain in-flight snapshots

        _check(src.error is None, f"source has no connect error ({src.error})")
        with lock:
            got = list(snaps)
        print(f"  captured {len(got)} pose-graph snapshot(s) (one per loop close)")
        _check(len(got) >= 1, "at least one loop-closure snapshot reached the source")
        _check(src.snapshot_count() >= 1,
               f"the slider buffer accumulated snapshots "
               f"(count={src.snapshot_count()})")

        # Prefer the richest snapshot (most nodes, then most loop edges).
        def score(s):
            return (int(s.n_kf), len(s.loop_edges))
        m = max(got, key=score)
        before = np.asarray(m.kf_before_xz, np.float64).reshape(-1, 2)
        after = np.asarray(m.kf_after_xz, np.float64).reshape(-1, 2)
        max_corr = (float(np.linalg.norm(after - before, axis=1).max())
                    if len(before) else 0.0)
        print(f"  chosen snapshot: nodes {m.n_kf}  loop edges {len(m.loop_edges)} "
              f"{list(m.loop_edges)}  loops {m.n_loops}  "
              f"max node correction {max_corr:.3f} m")
        _check(int(m.n_kf) >= 2, f"snapshot carries >=2 anchored nodes (n={m.n_kf})")
        _check(len(m.loop_edges) >= 1, "snapshot carries >=1 loop edge (chord)")
        _check(before.shape == after.shape and len(before) == int(m.n_kf),
               "before/after node arrays align (the per-node deltas exist)")
        _check(max_corr > 1e-3,
               f"PGO actually moved nodes (max correction {max_corr:.4f} m > 0)")
        _check(np.asarray(m.before_traj_xz).shape[0] >= 2
               and np.asarray(m.after_traj_xz).shape[0]
               == np.asarray(m.before_traj_xz).shape[0],
               "dense before/after trails present and aligned")

        img = render_pose_graph(m, 1100, 620, show_before=False)
        _check(img.shape == (620, 1100, 3) and img.dtype == np.uint8,
               f"rendered (620,1100,3) uint8 (got {img.shape} {img.dtype})")
        nonbg = int((img.reshape(-1, 3) != np.array([13, 17, 23])).any(1).sum())
        _check(nonbg > 1000, f"rendered canvas is non-blank ({nonbg} px drawn)")
        # before/after toggle actually changes pixels on the real snapshot.
        img_pre = render_pose_graph(m, 1100, 620, show_before=True)
        d = int((img != img_pre).any(2).sum())
        _check(d > 100, f"before/after toggle changes pixels ({d} px)")
        # the loop-edge chord (magenta) + correction arrows (amber) are drawn.
        flat = img.reshape(-1, 3).astype(int)
        loop_px = int((np.abs(flat - np.array(_LOOP)).sum(1) < 60).sum())
        warn_px = int((np.abs(flat - np.array(_WARN)).sum(1) < 60).sum())
        _check(loop_px > 5, f"loop-edge chord drawn in the PNG ({loop_px} px)")
        _check(warn_px > 5, f"correction-delta arrows drawn ({warn_px} px)")

        # Persist (cv2 wants BGR; the canvas is RGB).
        import cv2
        out = Path(args.out)
        cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"\n  wrote {out}  ({loop_px} loop / {warn_px} correction px)")
        print("POSE-GRAPH WINDOW PNG PASS")
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


if __name__ == "__main__":
    raise SystemExit(main())
