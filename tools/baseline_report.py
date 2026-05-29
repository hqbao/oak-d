#!/usr/bin/env python3
"""Scan a folder of gold sessions and emit a Markdown baseline report.

For each subfolder under <gold_dir>, runs SLAM-vs-VIO comparison and
writes a table of ATE/RPE numbers plus session-level counts (loop
events, tracking events, durations). Output is Markdown to stdout.

Usage::

    .venv/bin/python tools/baseline_report.py sessions/gold/ > docs/GOLD_BASELINE.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.compare_sessions import (                # noqa: E402
    _load_jsonl, _load_pose_stream, compare,
)


def _summarize_session(session: Path, delta_s: float) -> dict:
    meta_path = session / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    counts = meta.get("counts", {})

    vio = _load_pose_stream(session, "vio")
    slam = _load_pose_stream(session, "slam")
    cmp: dict | None = None
    if len(vio["ts_s"]) >= 2 and len(slam["ts_s"]) >= 2:
        cmp = compare(slam, vio, label=session.name,
                      delta_s=delta_s, no_align=False, verbose=False)

    return {
        "name": session.name,
        "duration_s": meta.get("duration_s", 0.0),
        "counts": counts,
        "compare": cmp,
    }


def _fmt_row(s: dict) -> str:
    c = s["counts"]
    cmp = s["compare"] or {}
    ate = cmp.get("ate", {})
    rpe = cmp.get("rpe", {})
    return (
        f"| `{s['name']}` "
        f"| {s['duration_s']:5.1f}s "
        f"| {c.get('frames', 0):4d} "
        f"| {c.get('vio_poses', 0):4d} "
        f"| {c.get('slam_poses', 0):4d} "
        f"| {c.get('loop_events', 0):2d} "
        f"| {c.get('tracking_events', 0):2d} "
        f"| {ate.get('rmse_m', 0)*100:6.2f} "
        f"| {ate.get('max_m', 0)*100:6.2f} "
        f"| {rpe.get('trans_rmse_m', 0)*100:5.2f} "
        f"| {rpe.get('rot_rmse_deg', 0):5.2f} "
        f"|"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gold_dir", help="folder containing session subfolders")
    ap.add_argument("--delta-s", type=float, default=1.0,
                    help="RPE window in seconds (default 1.0)")
    args = ap.parse_args()

    gold = Path(args.gold_dir).resolve()
    if not gold.is_dir():
        print(f"not a directory: {gold}", file=sys.stderr)
        return 2

    sessions = sorted(p for p in gold.iterdir()
                      if p.is_dir() and (p / "meta.json").exists())
    if not sessions:
        print(f"no sessions found under {gold}", file=sys.stderr)
        return 1

    summaries = [_summarize_session(s, args.delta_s) for s in sessions]

    print(f"# Gold Baseline Report")
    print()
    print(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"**Source**: `{gold}`")
    print(f"**Pipeline**: BasaltVIO + RTABMapSLAM (depthai 3.6.1)")
    print(f"**RPE window**: {args.delta_s}s")
    print()
    print("ATE/RPE compare **SLAM (ref) vs VIO (test)** — they measure how "
          "much loop closure correction RTABMap adds on top of pure VIO. "
          "Higher numbers = more correction (i.e. VIO drifted more).")
    print()
    print("| Session | Dur | Frm | VIO | SLAM | Lp | Trk "
          "| ATE rmse cm | ATE max cm | RPE t cm | RPE r deg |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        print(_fmt_row(s))
    print()
    print("---")
    print()
    print("## How to regenerate")
    print()
    print("```bash")
    print(f".venv/bin/python tools/baseline_report.py {gold.relative_to(gold.parent.parent)} \\")
    print("    > docs/GOLD_BASELINE.md")
    print("```")
    print()
    print("## How to interpret")
    print()
    print("- **ATE rmse**: average position error between VIO trajectory and "
          "SLAM (loop-corrected) trajectory, after SE(3) alignment.")
    print("- **ATE max**: worst-case offset (usually right before a loop closure).")
    print("- **RPE t / r**: per-second drift in metres and degrees.")
    print("- **Lp** / **Trk**: number of loop closures + tracking events detected.")
    print()
    print("When `skyslam` is implemented, re-run on the same sessions and "
          "expect ATE numbers within ~20% of these (or better).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
