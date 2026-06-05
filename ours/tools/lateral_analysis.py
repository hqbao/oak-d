#!/usr/bin/env python3
"""Decompose our per-frame motion into longitudinal vs lateral error.

Answers the device observation "during a fast forward push the path veers
sideways". After rigid-aligning our trajectory to Basalt, each frame's
displacement is split into the component ALONG Basalt's instantaneous direction
(longitudinal -- the real motion) and PERPENDICULAR to it (lateral -- the
sideways error). We report this over the fast-motion frames so we can see how
much sideways drift the vision translation injects, and where.

Usage::

    python tools/lateral_analysis.py --session sessions/gold/push_fwdback_20s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.tools.live_replay import replay  # noqa: E402
from ours.tools.vio_run import load_basalt_positions, umeyama  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="sessions/gold/push_fwdback_20s")
    ap.add_argument("--no-accel", action="store_true")
    ap.add_argument("--lock", action="store_true",
                    help="enable lock_translation_to_rotation (default joint PnP)")
    ap.add_argument("--fast-mm", type=float, default=20.0,
                    help="per-frame Basalt step (mm) above which a frame counts "
                         "as 'fast motion'")
    args = ap.parse_args()

    sd = Path(args.session)
    pos = replay(sd, no_accel=args.no_accel, lock_trans=args.lock)
    pos.pop("_meta", None)
    basalt = load_basalt_positions(sd)
    seqs = sorted(set(pos) & set(basalt))
    ours = np.array([pos[s] for s in seqs])
    ref = np.array([basalt[s] for s in seqs])

    # rigid align ours -> Basalt (no scale, so lateral error is in real metres)
    R, t, s = umeyama(ours, ref, with_scale=False)
    oursA = (s * (R @ ours.T).T + t)

    d_our = np.diff(oursA, axis=0)        # our per-frame displacement (aligned)
    d_ref = np.diff(ref, axis=0)          # Basalt per-frame displacement
    ref_step = np.linalg.norm(d_ref, axis=1)

    fast = ref_step > (args.fast_mm * 1e-3)
    print(f"=== {sd} (no_accel={args.no_accel}) ===")
    print(f"  frames {len(seqs)}  fast-motion frames {int(fast.sum())} "
          f"(>{args.fast_mm:.0f}mm/frame)")

    long_abs = 0.0
    lat_abs = 0.0
    lat_max = 0.0
    lat_max_i = -1
    for i in range(len(d_ref)):
        n = ref_step[i]
        if n < 1e-6:
            continue
        u = d_ref[i] / n                  # Basalt direction
        lon = float(d_our[i] @ u)         # our motion along it
        lat_vec = d_our[i] - lon * u      # perpendicular component
        lat = float(np.linalg.norm(lat_vec))
        if fast[i]:
            long_abs += abs(lon)
            lat_abs += lat
            if lat > lat_max:
                lat_max = lat
                lat_max_i = i

    print(f"  over fast frames:")
    print(f"    longitudinal (along Basalt) total : {long_abs*1000:7.1f} mm")
    print(f"    lateral (perp = sideways error)   : {lat_abs*1000:7.1f} mm")
    print(f"    lateral / longitudinal ratio      : {lat_abs/max(long_abs,1e-6):.2f}")
    print(f"    worst single-frame lateral        : {lat_max*1000:6.1f} mm "
          f"(seq {seqs[lat_max_i] if lat_max_i>=0 else -1})")

    # --- scale by speed regime: do we UNDER-shoot the fast pushes? ----------
    # For each frame compute our longitudinal step / Basalt step, bucketed by how
    # fast Basalt was moving. A scale < 1 in the fast bucket = "push fast, travel
    # short" (the reported symptom). Uses the rigid (no-scale) alignment so the
    # numbers are real metric ratios.
    print("  scale (our_longitudinal / Basalt_step) by Basalt speed bucket:")
    buckets = [(0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 1e9)]
    for lo, hi in buckets:
        m = (ref_step * 1000 >= lo) & (ref_step * 1000 < hi)
        if m.sum() < 3:
            continue
        our_lon = 0.0
        ref_sum = 0.0
        for i in np.where(m)[0]:
            n = ref_step[i]
            u = d_ref[i] / n
            our_lon += float(d_our[i] @ u)
            ref_sum += n
        print(f"    {lo:4.0f}-{hi:4.0f} mm/frame : {int(m.sum()):3d} frames  "
              f"scale {our_lon/max(ref_sum,1e-6):.2f}")

    # Basalt's own lateral (consecutive direction change) for a baseline: how
    # non-straight is the TRUE path? (so we don't blame ourselves for real turns)
    ref_lat = 0.0
    ref_lon = 0.0
    for i in range(1, len(d_ref)):
        n = ref_step[i]
        if n < 1e-6 or not fast[i]:
            continue
        prev = d_ref[i - 1]
        pn = np.linalg.norm(prev)
        if pn < 1e-6:
            continue
        u = prev / pn
        lon = float(d_ref[i] @ u)
        ref_lon += abs(lon)
        ref_lat += float(np.linalg.norm(d_ref[i] - lon * u))
    print(f"  Basalt path's own lateral/longitudinal (true curviness): "
          f"{ref_lat/max(ref_lon,1e-6):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
