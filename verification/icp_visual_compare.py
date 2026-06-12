#!/usr/bin/env python3
"""Visual A/B of the dense-ICP factor: overlay estimated trajectories vs Basalt GT.

Runs the tight VIO at 54x42 (the ToF regime where the ICP factor is meant to
help) under a few opt-in configs and plots their Sim3-aligned trajectories on
the SAME axes as the ground truth, so the difference (or lack of it) is visible.

    .venv/bin/python verification/icp_visual_compare.py [--only clip ...]
                                                        [--max-frames N]
    -> writes /tmp/icp_visual_compare.png

Configs:
  OFF       baseline tight VIO (no velocity prior, no ICP)
  VEL       + --stabilize-velocity  (the real win -46%/-58%)
  ICP+VEL   + --depth-icp + --stabilize-velocity  (the new algorithm)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from verification.loose_vs_tight_bench import run_session  # noqa: E402

GOLD = Path(__file__).resolve().parents[1] / "sessions" / "gold"

CONFIGS = [
    ("OFF",     dict(stabilize_velocity=False, depth_icp=False), "tab:red"),
    ("VEL",     dict(stabilize_velocity=True,  depth_icp=False), "tab:orange"),
    ("ICP+VEL", dict(stabilize_velocity=True,  depth_icp=True),  "tab:green"),
]


def _best_axes(gt: np.ndarray) -> tuple[int, int]:
    """The two GT axes with the most spread -> the natural 2D viewing plane."""
    var = gt.var(axis=0)
    order = np.argsort(var)[::-1]
    a, b = sorted(order[:2])
    return int(a), int(b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*",
                    default=["lab_straight_20s", "push_straight_fast_15s"])
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="/tmp/icp_visual_compare.png")
    args = ap.parse_args()

    clips = [c for c in args.only if (GOLD / c).is_dir()]
    if not clips:
        print("no valid clips"); return 1

    fig, axes = plt.subplots(1, len(clips), figsize=(7 * len(clips), 6.5),
                             squeeze=False)
    axlabel = "XYZ"

    for col, clip in enumerate(clips):
        ax = axes[0][col]
        sd = GOLD / clip
        gt_plotted = False
        ai = bi = 0
        for tag, kw, color in CONFIGS:
            r = run_session(sd, backend="vio", resolution="tof54",
                            min_ba_views=1, max_frames=args.max_frames, **kw)
            if r is None or "est_aligned" not in r:
                print(f"  {clip:24s} {tag:8s} -> no result"); continue
            est, gt = r["est_aligned"], r["gt"]
            if not gt_plotted:
                ai, bi = _best_axes(gt)
                ax.plot(gt[:, ai], gt[:, bi], "k-", lw=2.6, label="ground truth",
                        zorder=1)
                ax.scatter(*gt[0, [ai, bi]], c="k", s=70, marker="o", zorder=5)
                gt_plotted = True
            ax.plot(est[:, ai], est[:, bi], "-", color=color, lw=1.6, alpha=0.85,
                    label=f"{tag}  (ATE {r['ate_cm']:.0f}cm, scale {r['scale']:.2f})",
                    zorder=2)
            print(f"  {clip:24s} {tag:8s} ATE={r['ate_cm']:6.1f}cm "
                  f"scale={r['scale']:.2f} maxstep={r['max_step_cm']:.1f}cm")
        ax.set_title(f"{clip}  @ 54x42 ToF", fontsize=11)
        ax.set_xlabel(f"{axlabel[ai]} (m)"); ax.set_ylabel(f"{axlabel[bi]} (m)")
        ax.axis("equal"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="best")

    fig.suptitle("Dense-ICP factor visual A/B  --  estimate vs ground truth "
                 "(Sim3-aligned), tight VIO @ 54x42", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
