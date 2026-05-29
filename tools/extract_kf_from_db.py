#!/usr/bin/env python3
"""C4: extract keyframe events from RTABMap SQLite database.

Reads <session>/basalt/rtabmap.db, emits:
  - <session>/basalt/kf_events.jsonl  -- one entry per Node (keyframe)
      {ts_ns, kf_id, weight, pos:[x,y,z], quat_wxyz:[w,x,y,z]}
  - <session>/basalt/kf_loops.jsonl   -- one entry per Link of loop type
      {ts_ns, from_kf, to_kf, type, type_name,
       transform_pos:[x,y,z], transform_quat_wxyz:[w,x,y,z]}

Pose blobs: 3x4 float32 row-major [R|t].
Stamps: seconds since device boot, matching ts_ns of session jsonl files.
Loop types kept (RTABMap convention):
  1 = GlobalClosure  2 = LocalSpaceClosure  3 = LocalTimeClosure
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

LOOP_TYPES = {1: "global", 2: "local_space", 3: "local_time"}


def _rt_blob_to_pose(blob: bytes) -> tuple[list[float], list[float]]:
    """Decode 48-byte 3x4 float32 [R|t] -> (pos, quat_wxyz)."""
    if blob is None or len(blob) != 48:
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    m = np.array(struct.unpack("<12f", blob), dtype=np.float64).reshape(3, 4)
    R = m[:, :3]
    t = m[:, 3]
    # Rotation matrix -> quaternion (w, x, y, z), Shepperd's method
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(t[0]), float(t[1]), float(t[2])], [
        float(w), float(x), float(y), float(z),
    ]


def extract(session_dir: Path) -> tuple[int, int]:
    db = session_dir / "basalt" / "rtabmap.db"
    if not db.exists():
        print(f"[err] {db} not found", file=sys.stderr)
        sys.exit(2)

    out_kf = session_dir / "basalt" / "kf_events.jsonl"
    out_lp = session_dir / "basalt" / "kf_loops.jsonl"

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        # Keyframes (Nodes), ordered by stamp ascending.
        kf_rows = con.execute(
            "SELECT id, weight, stamp, pose FROM Node ORDER BY stamp ASC"
        ).fetchall()
        # Index id -> pose for loop endpoint lookup.
        kf_pos: dict[int, tuple[list[float], list[float], float]] = {}
        with out_kf.open("w") as f:
            for kf_id, weight, stamp, pose_blob in kf_rows:
                pos, quat = _rt_blob_to_pose(pose_blob)
                ts_ns = int(round(float(stamp) * 1e9))
                kf_pos[int(kf_id)] = (pos, quat, float(stamp))
                f.write(json.dumps({
                    "ts_ns": ts_ns,
                    "kf_id": int(kf_id),
                    "weight": int(weight) if weight is not None else 0,
                    "pos": pos,
                    "quat_wxyz": quat,
                }) + "\n")

        # Loop closures (Links with type in LOOP_TYPES). Stamp = stamp of `to_id`
        # (the query keyframe that triggered the closure).
        lp_rows = con.execute(
            "SELECT from_id, to_id, type, transform FROM Link "
            "WHERE type IN (1,2,3) ORDER BY to_id ASC"
        ).fetchall()
        n_lp = 0
        with out_lp.open("w") as f:
            for from_id, to_id, t_type, tf_blob in lp_rows:
                pos, quat = _rt_blob_to_pose(tf_blob)
                ref = kf_pos.get(int(to_id)) or kf_pos.get(int(from_id))
                stamp_s = ref[2] if ref else 0.0
                f.write(json.dumps({
                    "ts_ns": int(round(stamp_s * 1e9)),
                    "from_kf": int(from_id),
                    "to_kf": int(to_id),
                    "type": int(t_type),
                    "type_name": LOOP_TYPES.get(int(t_type), str(t_type)),
                    "transform_pos": pos,
                    "transform_quat_wxyz": quat,
                }) + "\n")
                n_lp += 1
    finally:
        con.close()

    return len(kf_rows), n_lp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", type=Path, help="path to session directory")
    args = ap.parse_args()
    n_kf, n_lp = extract(args.session)
    print(f"[ok] {args.session.name}: {n_kf} keyframes, {n_lp} loop links")
    print(f"     -> {args.session}/basalt/kf_events.jsonl")
    print(f"     -> {args.session}/basalt/kf_loops.jsonl")


if __name__ == "__main__":
    main()
