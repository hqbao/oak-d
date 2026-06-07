#!/usr/bin/env python3
"""Build a 3D point-cloud map of the room from a session's keyframes.

Every keyframe carries a pose + a metric depth map; back-project each depth into
its camera frame, lift it to the world by that keyframe's pose, and stack them all
into ONE cloud -- so the room is reconstructed from every viewpoint at once. This
is the honest "what did SLAM see" view: pure REAL outputs (keyframe poses + the
chip/SGM depth), no invented geometry.

It runs OFFLINE over a recorded session (no device): it replays the same flow
graph the live run uses, taps the ``keyframe`` messages off the bus, and -- with
``--slam`` -- the loop-closure corrections so the map uses drift-corrected poses.

    # interactive 3D viewer (point cloud + keyframe camera path)
    python ours/tools/slam_map3d.py --session sessions/gold/lab_loop_30s
    # loop-corrected poses (slower: runs SLAM), denser cloud
    python ours/tools/slam_map3d.py --session sessions/gold/loop_closure_45s --slam --stride 2
    # headless: write a coloured PLY to open in MeshLab / CloudCompare
    python ours/tools/slam_map3d.py --session sessions/gold/lab_loop_30s --export /tmp/room.ply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.app import build_replay                                  # noqa: E402
from ours.lib.io.reader import SessionReader                       # noqa: E402
from ours.lib.flow.pubsub import Bus                              # noqa: E402
from ours.lib.flow.messages import END                            # noqa: E402
from ours.lib.flow import topics                                  # noqa: E402
from ours.lib.misc.geometry import (keyframe_pointcloud,           # noqa: E402
                                    keyframe_landmark_cloud)


def collect_keyframes(session: str, *, kf_every: int, max_frames: int,
                      use_slam: bool):
    """Replay ``session`` and return ``(K, keyframes, corrections)``.

    ``keyframes`` are the real ``Keyframe`` messages (pose + gray + depth);
    ``corrections`` the ``loop.correction`` messages (only when ``use_slam``).
    """
    reader = SessionReader(Path(session))
    bus = Bus()
    (cam_flow, imu_flow), flows, _ui = build_replay(
        bus, reader, kf_every=kf_every, use_gyro=True, depth_fast=True,
        max_frames=max_frames, with_backend_slam=False,
        slam=use_slam, backend=False)
    kfs: list = []
    corr: list = []

    def _on_kf(m):
        if m is END:
            return
        kfs.append(m)
        if len(kfs) % 20 == 0:                  # live progress (replay is silent)
            print(f"[map3d] replaying… {len(kfs)} keyframes", flush=True)

    bus.subscribe(topics.KEYFRAME, _on_kf)
    bus.subscribe(topics.LOOP_CORRECTION,
                  lambda m: None if m is END else corr.append(m))
    n_frames = max_frames or "all"
    print(f"[map3d] replaying {Path(session).name} ({n_frames} frames)"
          f"{' + SLAM' if use_slam else ''}…", flush=True)
    odom = flows[0]
    for f in flows:
        f.start()
    imu_flow.start()
    cam_flow.start()
    cam_flow.join()
    odom.done.wait(timeout=180.0)
    imu_flow.stop()
    for f in flows:
        f.stop()
    return reader.K, kfs, corr


def _corrected_poses(keyframes, corrections):
    """Per-keyframe pose, replaced by the latest loop-closure correction if any."""
    latest: dict[int, np.ndarray] = {}
    for c in corrections:                       # last correction wins per seq
        latest.update(c.kf_poses)
    return [latest.get(int(kf.seq), kf.T_world_cam) for kf in keyframes]


def build_map(session: str, *, kf_every: int, max_frames: int, use_slam: bool,
              stride: int = 4, max_depth: float = 6.0, max_points: int = 0,
              dense: bool = False):
    K, kfs, corr = collect_keyframes(session, kf_every=kf_every,
                                     max_frames=max_frames, use_slam=use_slam)
    poses = _corrected_poses(kfs, corr)
    depths = [kf.depth_m for kf in kfs]
    grays = [kf.gray_left for kf in kfs]
    if dense:
        print(f"[map3d] fusing {len(kfs)} keyframes (DENSE depth)…", flush=True)
        points, colors = keyframe_pointcloud(poses, depths, grays, K,
                                             stride=stride, max_depth=max_depth)
    else:
        # Default: only the PnP-inlier landmarks (clean) — the dense depth the
        # solve rejected is noise (flying points before/behind real surfaces).
        print(f"[map3d] fusing {len(kfs)} keyframes (PnP-inlier landmarks)…",
              flush=True)
        points, colors = keyframe_landmark_cloud(
            poses, [kf.track_ids for kf in kfs], [kf.track_px for kf in kfs],
            depths, [kf.inlier_ids for kf in kfs], K, grays=grays,
            max_depth=max_depth)
    # Cap the cloud so the interactive GL viewer stays responsive (a full session
    # is millions of points; pyqtgraph's scatter bogs down well before that).
    # Deterministic random subsample keeps the room shape; raise --max-points or
    # use --export for the full cloud.
    if max_points and len(points) > max_points:
        idx = np.random.default_rng(0).choice(len(points), max_points,
                                              replace=False)
        points, colors = points[idx], colors[idx]
    cams = np.array([np.asarray(p)[:3, 3] for p in poses], dtype=np.float32) \
        if poses else np.zeros((0, 3), np.float32)
    n_loops = corr[-1].n_loops if corr else 0
    return {"points": points, "colors": colors, "cams": cams,
            "n_kf": len(kfs), "n_loops": n_loops}


def write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a coloured ASCII PLY (openable in MeshLab / CloudCompare)."""
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, rgb):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="3D point-cloud map from keyframes")
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--slam", action="store_true",
                    help="run loop-closure SLAM and use the drift-corrected poses "
                         "(slower; cleaner map on sessions that revisit a place)")
    ap.add_argument("--dense", action="store_true",
                    help="back-project the FULL depth map instead of only the "
                         "PnP-inlier landmarks (noisier; the default shows just the "
                         "clean trusted points)")
    ap.add_argument("--stride", type=int, default=4,
                    help="DENSE only: depth subsample (lower = more points) [4]")
    ap.add_argument("--max-depth", type=float, default=6.0,
                    help="drop points beyond this many metres (far depth is noisy)")
    ap.add_argument("--max-points", type=int, default=500_000,
                    help="cap the cloud for the interactive viewer so GL stays "
                         "responsive (0 = no cap; --export always writes the full "
                         "cloud) [500000]")
    ap.add_argument("--export", default="",
                    help="write a coloured PLY here and exit (headless, no GUI)")
    args = ap.parse_args()

    # --export gets the full cloud; the GUI gets the capped one (responsiveness).
    cap = 0 if args.export else args.max_points
    m = build_map(args.session, kf_every=args.kf_every, max_frames=args.max_frames,
                  use_slam=args.slam, stride=args.stride, max_depth=args.max_depth,
                  max_points=cap, dense=args.dense)
    print(f"[map3d] {Path(args.session).name}: {m['n_kf']} keyframes, "
          f"{m['n_loops']} loop(s) -> {len(m['points'])} points", flush=True)
    if len(m["points"]) == 0:
        print("[map3d] no points (no valid keyframe depth) — nothing to show",
              file=sys.stderr)
        return 1

    if args.export:
        write_ply(args.export, m["points"], m["colors"])
        print(f"[map3d] wrote {len(m['points'])} points -> {args.export}")
        return 0

    # Interactive 3D viewer (Qt). Imported lazily so --export stays headless.
    try:
        from PyQt6.QtWidgets import QApplication
        from ours.ui import theme
        from ours.ui.map_window import MapWindow
    except Exception as e:                                          # noqa: BLE001
        print(f"[map3d] Qt viewer unavailable ({e}). Use --export <file>.ply and "
              f"open it in MeshLab / CloudCompare instead.", file=sys.stderr)
        return 1
    print(f"[map3d] opening viewer ({len(m['points'])} points)… "
          f"close the window to exit", flush=True)
    theme.ensure_gl_format()
    app = QApplication(sys.argv)
    win = MapWindow(m["points"], m["colors"], m["cams"],
                    title=f"SLAM map · {Path(args.session).name}")
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
