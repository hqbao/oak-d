#!/usr/bin/env python3
"""Validate our sparse stereo depth against the OAK-D chip depth (the oracle).

This is the regression guard for ``ours/vio/stereo.py``, mirroring exactly what
``klt_selftest.py`` does for optical flow: it proves our from-scratch block
matcher agrees with the trusted reference (here the chip's SGBM depth stored as
``*_D.raw16``) before we let the VIO trust it. We do NOT use the chip depth in
the production path -- only here, to measure.

For each gold session it:
  1. loads a handful of frames (left + right + chip depth),
  2. picks the Shi-Tomasi corners the VIO would actually query,
  3. computes our depth at those pixels from left+right only,
  4. compares to the chip depth at the same pixels (where the chip is valid),
  5. reports match rate, median relative error, and inlier fractions + timing.

Usage::

    python ours/tools/stereo_selftest.py
    python ours/tools/stereo_selftest.py --session sessions/gold/corridor_60s
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib import (  # noqa: E402
    SessionReader, SGMConfig, SGMStereoMatcher, StereoConfig, StereoMatcher,
)
from ours.lib.frontend.corners import good_features_to_track  # noqa: E402
from ours.lib.stereo.stereo import HAVE_NUMBA  # noqa: E402


def _make_matcher(engine: str, reader: SessionReader):
    if engine == "sgm":
        # Rectify the raw right frame ourselves (the recorded right is unrectified
        # syncedRight), then dense semi-global match -- fully library-free.
        return SGMStereoMatcher.from_calib(reader.calib, SGMConfig())
    return StereoMatcher(reader.K, reader.calib.baseline_m, StereoConfig())


def score_session(session_dir: Path, n_frames: int, max_corners: int,
                  engine: str = "sgm", quiet: bool = False) -> dict | None:
    reader = SessionReader(session_dir)
    if len(reader) == 0:
        return None
    matcher = _make_matcher(engine, reader)

    # Sample frames evenly across the session.
    idxs = np.linspace(0, len(reader) - 1, min(n_frames, len(reader)))
    idxs = sorted(set(int(round(i)) for i in idxs))

    rel_errs: list[float] = []
    n_query = 0
    n_matched = 0
    n_compared = 0
    t_sum = 0.0

    for i in idxs:
        f = reader.load_frame(i, load_right=True)
        if f.gray_right is None:
            return None
        pts = good_features_to_track(f.gray_left, max_corners)
        if pts.shape[0] == 0:
            continue
        n_query += pts.shape[0]

        t0 = time.perf_counter()
        depth = matcher.depth_at(f.gray_left, f.gray_right, pts)
        t_sum += time.perf_counter() - t0

        us = np.round(pts[:, 0]).astype(int)
        vs = np.round(pts[:, 1]).astype(int)
        chip = f.depth_m[vs, us]

        matched = depth > 0
        n_matched += int(matched.sum())
        # Compare only where BOTH ours and the chip have a valid metric depth.
        both = matched & (chip > 0.1) & (chip < 12.0)
        n_compared += int(both.sum())
        if both.any():
            rel = np.abs(depth[both] - chip[both]) / chip[both]
            rel_errs.extend(rel.tolist())

    if n_compared == 0:
        return None
    rel = np.asarray(rel_errs)
    res = {
        "frames": len(idxs),
        "match_rate": n_matched / max(n_query, 1),
        "compared": n_compared,
        "median_rel": float(np.median(rel)),
        "mean_rel": float(rel.mean()),
        "within_5pct": float((rel < 0.05).mean()),
        "within_10pct": float((rel < 0.10).mean()),
        "ms_per_frame": 1e3 * t_sum / max(len(idxs), 1),
    }
    if not quiet:
        print(f"session       : {reader.dir.name}")
        print(f"baseline (m)  : {reader.calib.baseline_m:.4f}")
        print(f"frames sampled: {res['frames']}")
        print(f"match rate    : {res['match_rate']*100:.1f}% of corners")
        print(f"compared px   : {res['compared']} (ours & chip both valid)")
        print(f"median rel err: {res['median_rel']*100:.2f}%")
        print(f"within 5%/10% : {res['within_5pct']*100:.1f}% / "
              f"{res['within_10pct']*100:.1f}%")
        print(f"time          : {res['ms_per_frame']:.1f} ms/frame "
              f"({'numba' if HAVE_NUMBA else 'pure-numpy'})")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=None,
                    help="score a single session; default = all gold sessions")
    ap.add_argument("--frames", type=int, default=12,
                    help="frames sampled per session [12]")
    ap.add_argument("--corners", type=int, default=200,
                    help="max Shi-Tomasi corners per frame [200]")
    ap.add_argument("--engine", choices=["sgm", "sparse"], default="sgm",
                    help="matcher: dense SGM [default] or sparse block match")
    args = ap.parse_args()

    if args.session:
        res = score_session(Path(args.session), args.frames, args.corners,
                            args.engine)
        return 0 if res is not None else 1

    gold = Path("sessions/gold")
    rows = []
    for d in sorted(gold.iterdir()):
        if not (d / "input" / "frames.jsonl").exists():
            continue
        res = score_session(d, args.frames, args.corners, args.engine,
                            quiet=True)
        rows.append((d.name, res))
        print(f"  {d.name:20s} done")

    print()
    print(f"engine: {args.engine} ({'numba' if HAVE_NUMBA else 'pure-numpy'})")
    print(f"{'session':20s} {'match%':>7s} {'medRel%':>8s} {'<5%':>6s} "
          f"{'<10%':>6s} {'ms/fr':>7s}")
    print("-" * 60)
    for name, res in rows:
        if res is None:
            print(f"{name:20s} {'--':>7s} {'--':>8s} {'--':>6s} "
                  f"{'--':>6s} {'--':>7s}")
            continue
        print(f"{name:20s} {res['match_rate']*100:6.1f}% "
              f"{res['median_rel']*100:7.2f}% {res['within_5pct']*100:5.1f}% "
              f"{res['within_10pct']*100:5.1f}% {res['ms_per_frame']:6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
