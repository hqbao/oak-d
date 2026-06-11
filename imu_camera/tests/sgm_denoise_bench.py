#!/usr/bin/env python3
"""Density-preserving SGM denoise gate: per-frame latency + keypoint density +
noise proxy, measured frame-by-frame on the gold replay sessions at the live
preset.

This is the explicit verification gate for the SGM noise-reduction work. It is a
MEASUREMENT harness, not a parity test:

  * LATENCY  -- time ONLY the ``dense_depth_rectified_left`` call (the live VIO's
    per-frame SGM depth) over >= ``--frames`` frames; report mean + p95 ms.
  * DENSITY  -- count valid-depth points at the KLT-style tracked corners
    (Shi-Tomasi, same fixture as stereo_sgm_selftest). MUST NOT drop.
  * NOISE    -- count isolated/speckle disparity components (small 4-connected
    blobs) as a proxy for the "flying" fragments. Should drop materially.

Pass two configs via ``--mode {off,on}`` to compare BEFORE (filters off) and
AFTER (filters on, i.e. SGMConfig.live() defaults). The harness uses the SAME
matcher path the live front-end uses (``SGMStereoMatcher.from_calib`` +
``dense_depth_rectified_left``) so the numbers reflect the real per-frame cost.

Run::

    .venv/bin/python -m imu_camera.tests.sgm_denoise_bench --mode off
    .venv/bin/python -m imu_camera.tests.sgm_denoise_bench --mode on
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imu_camera.io.reader import SessionReader                       # noqa: E402
from sky.depth.stereo import (                       # noqa: E402
    SGMConfig, SGMStereoMatcher,
)
from imu_camera.tests.stereo_sgm_selftest import (                   # noqa: E402
    good_features_to_track,
)


def _count_speckle_components(disp: np.ndarray, max_size: int,
                              max_diff: float) -> int:
    """Number of valid 4-connected disparity blobs with <= ``max_size`` pixels.

    A pure proxy for "flying"/salt-pepper fragments: tiny components that survive
    the L/R check but are isolated. Vectorised-ish flood fill in NumPy (this is a
    MEASUREMENT, not the production filter -- speed here does not matter).
    """
    H, W = disp.shape
    valid = np.isfinite(disp)
    seen = np.zeros((H, W), dtype=bool)
    small = 0
    # Iterative stack flood fill; only over valid pixels.
    vs, us = np.nonzero(valid)
    for v0, u0 in zip(vs.tolist(), us.tolist()):
        if seen[v0, u0]:
            continue
        stack = [(v0, u0)]
        seen[v0, u0] = True
        count = 0
        while stack:
            v, u = stack.pop()
            d = disp[v, u]
            count += 1
            for dv, du in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nv, nu = v + dv, u + du
                if 0 <= nv < H and 0 <= nu < W and not seen[nv, nu] \
                        and valid[nv, nu] and abs(disp[nv, nu] - d) <= max_diff:
                    seen[nv, nu] = True
                    stack.append((nv, nu))
        if count <= max_size:
            small += 1
    return small


def _cfg_for(mode: str) -> SGMConfig:
    """Live preset with denoise OFF (before) or ON (after = current live())."""
    base = SGMConfig.live()
    if mode == "off":
        return replace(base, median_disp=0, speckle_window=0)
    return base  # 'on' == the shipped live() defaults


def bench(session: Path, mode: str, frames: int) -> dict:
    reader = SessionReader(session)
    cfg = _cfg_for(mode)
    matcher = SGMStereoMatcher.from_calib(reader.calib, cfg)
    n = min(frames, len(reader)) if frames > 0 else len(reader)

    # The noise proxy uses a fixed small-blob definition (independent of the
    # config under test) so 'before' and 'after' are measured on the SAME ruler.
    PROXY_MAX_SIZE = 20
    PROXY_MAX_DIFF = 1.0

    times_ms: list[float] = []
    kp_total = 0
    kp_with_depth = 0
    speckle_components = 0

    # Warm the numba kernels once (compile is not part of the per-frame budget).
    f0 = reader.load_frame(0, load_right=True)
    matcher.dense_depth_rectified_left(f0.gray_left, f0.gray_right)

    for i in range(n):
        f = reader.load_frame(i, load_right=True)
        t0 = time.perf_counter()
        gray_track, depth = matcher.dense_depth_rectified_left(
            f.gray_left, f.gray_right)
        times_ms.append((time.perf_counter() - t0) * 1e3)

        # Keypoint depth density at the corners the VIO would track.
        corners = good_features_to_track(gray_track, max_corners=120)
        if corners.shape[0]:
            xs = np.clip(corners[:, 0].astype(int), 0, depth.shape[1] - 1)
            ys = np.clip(corners[:, 1].astype(int), 0, depth.shape[0] - 1)
            d = depth[ys, xs]
            kp_total += corners.shape[0]
            kp_with_depth += int(np.count_nonzero(d > 0.0))

        # Noise proxy on the raw disparity (same resolution as published depth).
        disp = matcher.dense_disparity(f.gray_left, f.gray_right)
        speckle_components += _count_speckle_components(
            disp, PROXY_MAX_SIZE, PROXY_MAX_DIFF)

    t = np.array(times_ms)
    return {
        "session": session.name,
        "mode": mode,
        "frames": n,
        "mean_ms": float(t.mean()),
        "p95_ms": float(np.percentile(t, 95)),
        "max_ms": float(t.max()),
        "kp_total": kp_total,
        "kp_with_depth": kp_with_depth,
        "kp_density_pct": 100.0 * kp_with_depth / max(1, kp_total),
        "speckle_components": speckle_components,
        "speckle_per_frame": speckle_components / max(1, n),
    }


def _edge_cases() -> bool:
    """Degenerate-frame robustness of the denoise filter (no replay needed)."""
    from dataclasses import replace as _replace

    from sky.depth.stereo import (
        SGMConfig as _Cfg, _denoise_disparity)
    base = _Cfg.live()
    ok = True

    # off == byte-identical no-op
    rng = np.random.RandomState(0)
    d0 = rng.rand(50, 60).astype(np.float64) * 30.0
    d0[d0 < 5] = np.nan
    d = d0.copy()
    _denoise_disparity(d, _replace(base, median_disp=0, speckle_window=0))
    c = np.array_equal(np.nan_to_num(d, nan=-9), np.nan_to_num(d0, nan=-9))
    print(f"  [{'ok' if c else 'FAIL'}] off = byte-identical no-op")
    ok = ok and c

    # all-NaN frame (whitewall) -> no crash, stays NaN
    dn = np.full((40, 40), np.nan)
    _denoise_disparity(dn, base)
    c = bool(np.all(np.isnan(dn)))
    print(f"  [{'ok' if c else 'FAIL'}] all-NaN frame handled (no crash)")
    ok = ok and c

    # clean uniform plane fully preserved (no density loss, values unchanged)
    dv = np.full((40, 40), 12.5)
    _denoise_disparity(dv, base)
    c = (np.count_nonzero(np.isfinite(dv)) == dv.size
         and np.allclose(dv[np.isfinite(dv)], 12.5))
    print(f"  [{'ok' if c else 'FAIL'}] clean plane preserved + values unchanged")
    ok = ok and c

    # lone speckle removed, large surface survives
    ds = np.full((40, 40), np.nan)
    ds[20, 20] = 15.0
    ds[5:15, 5:15] = 8.0
    _denoise_disparity(ds, base)
    c = (np.isnan(ds[20, 20])
         and np.count_nonzero(np.isfinite(ds[5:15, 5:15])) >= 90)
    print(f"  [{'ok' if c else 'FAIL'}] lone speckle removed, surface survives")
    return ok and c


def gate(sessions: list[str], frames: int) -> bool:
    """Assert the three hard constraints, OFF vs ON, per session.

    (A) latency increase small, (B) keypoint density not dropped, plus a
    material noise reduction. Returns True iff every gate holds.
    """
    MAX_MEAN_INCREASE_MS = 3.0   # generous; observed ~0.3-0.5 ms
    MIN_NOISE_DROP_FRAC = 0.30   # speckle proxy must fall >= 30%
    ok = True
    print("\n=== edge-case robustness ===")
    ok = _edge_cases() and ok
    for s in sessions:
        off = bench(Path(s), "off", frames)
        on = bench(Path(s), "on", frames)
        d_mean = on["mean_ms"] - off["mean_ms"]
        d_dens = on["kp_density_pct"] - off["kp_density_pct"]
        noise_drop = 1.0 - on["speckle_per_frame"] / max(
            1e-9, off["speckle_per_frame"])
        lat_ok = d_mean <= MAX_MEAN_INCREASE_MS
        dens_ok = on["kp_with_depth"] >= off["kp_with_depth"]
        noise_ok = noise_drop >= MIN_NOISE_DROP_FRAC
        print(f"\n=== {off['session']} (n={off['frames']}) ===")
        print(f"  [{'ok' if lat_ok else 'FAIL'}] latency  mean "
              f"{off['mean_ms']:.2f} -> {on['mean_ms']:.2f} ms "
              f"(+{d_mean:.2f}; p95 {off['p95_ms']:.2f} -> {on['p95_ms']:.2f})")
        print(f"  [{'ok' if dens_ok else 'FAIL'}] kp density "
              f"{off['kp_density_pct']:.2f}% -> {on['kp_density_pct']:.2f}% "
              f"({d_dens:+.2f}%, {off['kp_with_depth']} -> {on['kp_with_depth']})")
        print(f"  [{'ok' if noise_ok else 'FAIL'}] noise proxy "
              f"{off['speckle_per_frame']:.0f} -> {on['speckle_per_frame']:.0f} "
              f"/frame ({100*noise_drop:.0f}% drop)")
        ok = ok and lat_ok and dens_ok and noise_ok
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["off", "on", "gate"], default="gate")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--sessions", nargs="*", default=[
        "sessions/gold/lab_loop_30s", "sessions/gold/corridor_60s"])
    args = ap.parse_args()

    if args.mode == "gate":
        ok = gate(args.sessions, args.frames)
        print("\n" + "=" * 60)
        print("PASS -- density-preserving denoise meets all gates."
              if ok else "FAIL -- a gate was violated (see lines above).")
        return 0 if ok else 1

    print(f"=== SGM denoise bench  mode={args.mode}  frames<={args.frames} ===")
    print(f"  config: {_cfg_for(args.mode)}")
    for s in args.sessions:
        r = bench(Path(s), args.mode, args.frames)
        print(f"\n[{r['session']}]  n={r['frames']}")
        print(f"  per-frame depth ms : mean={r['mean_ms']:.2f}  "
              f"p95={r['p95_ms']:.2f}  max={r['max_ms']:.2f}")
        print(f"  keypoint density   : {r['kp_with_depth']}/{r['kp_total']}  "
              f"= {r['kp_density_pct']:.2f}%")
        print(f"  speckle proxy      : {r['speckle_components']} small blobs "
              f"({r['speckle_per_frame']:.1f}/frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
