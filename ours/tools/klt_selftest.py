#!/usr/bin/env python3
"""Gold self-test for our pure-NumPy optical flow + corner detector.

This is the regression test that was missing while we kept "fixing" live lag
blind. It pins down three independent things with hard pass/fail thresholds, so
any future change to ``ours/vio/klt.py`` or ``ours/vio/corners.py`` is caught:

1. CORRECTNESS (synthetic, does NOT trust OpenCV) -- warp a deterministic
   textured image by a KNOWN sub-pixel translation, detect corners with our own
   detector, track them with our own KLT, and check the recovered flow matches
   the known translation to sub-pixel. A larger shift exercises the coarse-to-
   fine pyramid. This is ground truth, independent of any library.

2. AGREEMENT vs OpenCV -- on the same synthetic frames (and, if a gold session
   is present, on two real adjacent frames) compare our corners and our flow
   against ``cv2.goodFeaturesToTrack`` / ``cv2.calcOpticalFlowPyrLK``. They
   should land on the same features and the same displacements to a few tenths
   of a pixel.

3. TIMING -- report per-frame milliseconds for ours (full + the live preset)
   and cv2 against the 20 fps (50 ms) live budget. Timing is hardware dependent
   so it does NOT fail the test, but it is PRINTED so a performance regression
   (or a config that blows the budget) is visible at a glance, instead of being
   discovered only by feeling lag on the device.

Run:  .venv/bin/python ours/tools/klt_selftest.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.frontend.corners import good_features_to_track            # noqa: E402
from ours.lib.frontend.klt import _bilinear, calc_optical_flow_pyr_lk   # noqa: E402
from ours.lib.frontend.frontend import FrontendConfig                   # noqa: E402

try:
    import cv2
    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False

LIVE_BUDGET_MS = 1000.0 / 20.0   # 50 ms at 20 fps


def make_texture(h: int = 400, w: int = 640, seed: int = 0) -> np.ndarray:
    """Deterministic textured grayscale image with trackable 2D structure.

    Low-pass-filtered white noise: gives gradients in both directions (so
    Shi-Tomasi finds corners) while staying smooth enough for KLT's local
    gradient model to hold. Fully reproducible from ``seed``.
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    # a few separable box blurs ~ Gaussian, cheap and dependency-free
    k = np.ones(5, np.float32) / 5.0
    for _ in range(3):
        img = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, img)
        img = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, img)
    img -= img.min()
    img *= 255.0 / max(img.max(), 1e-6)
    return img.astype(np.uint8)


def translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift ``img`` by (dx, dy) px with bilinear sampling (our own sampler).

    A point at ``p`` in ``img`` appears at ``p + (dx, dy)`` in the result, so
    tracking ``img -> result`` must recover the flow ``(dx, dy)``.
    """
    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w]
    val, _ = _bilinear(img.astype(np.float32),
                       (xs - dx).astype(np.float32),
                       (ys - dy).astype(np.float32))
    return val.astype(np.float32)


def _interior(pts: np.ndarray, shape, margin: float) -> np.ndarray:
    h, w = shape
    return ((pts[:, 0] > margin) & (pts[:, 0] < w - margin)
            & (pts[:, 1] > margin) & (pts[:, 1] < h - margin))


def nn_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each point in ``a``, distance to the nearest point in ``b``."""
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), np.inf)
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return d.min(1)


def test_correctness() -> bool:
    """Known sub-pixel + pyramid-scale shifts recovered to sub-pixel."""
    print("== 1. correctness (synthetic, ground truth) ==")
    img0 = make_texture(seed=1)
    pts = good_features_to_track(img0, max_corners=300, quality_level=0.01,
                                 min_distance=12.0, block_size=7)
    ok = True
    for dx, dy, win, lvl, tol, name in [
        (1.7, -0.9, 21, 3, 0.10, "small sub-pixel"),
        (12.0, 7.0, 21, 3, 0.30, "large (pyramid)"),
    ]:
        img1 = translate(img0, dx, dy)
        nxt, st = calc_optical_flow_pyr_lk(img0, img1, pts,
                                           win_size=win, max_level=lvl)
        flow = nxt - pts
        # only score points that stayed well inside (border = warp-undefined)
        keep = st.astype(bool) & _interior(nxt, img0.shape, win)
        err = np.linalg.norm(flow[keep] - np.array([dx, dy]), axis=1)
        mean_err = float(err.mean()) if err.size else np.inf
        p95 = float(np.percentile(err, 95)) if err.size else np.inf
        passed = mean_err < tol and keep.sum() >= 0.5 * len(pts)
        ok &= passed
        print(f"  {name:16s} shift=({dx:+.1f},{dy:+.1f}) "
              f"tracked={keep.sum():3d}/{len(pts)} "
              f"mean_err={mean_err:.3f}px p95={p95:.3f}px "
              f"tol={tol:.2f} -> {'PASS' if passed else 'FAIL'}")
    return ok


def test_agreement_synthetic() -> bool:
    """Our corners + flow agree with OpenCV on the synthetic frames."""
    print("== 2a. agreement vs OpenCV (synthetic) ==")
    if not HAVE_CV2:
        print("  cv2 not available -- SKIP")
        return True
    img0 = make_texture(seed=2)
    dx, dy = 2.3, -1.4
    img1 = translate(img0, dx, dy)
    g0 = img0.astype(np.uint8)

    ours_c = good_features_to_track(g0, max_corners=300, quality_level=0.01,
                                    min_distance=12.0, block_size=7)
    cv_c = cv2.goodFeaturesToTrack(g0, maxCorners=300, qualityLevel=0.01,
                                   minDistance=12.0, blockSize=7)
    cv_c = cv_c.reshape(-1, 2).astype(np.float32)
    d = nn_dist(ours_c, cv_c)
    corner_ok = float(d.mean()) < 1.0 and float(np.percentile(d, 90)) < 2.0
    print(f"  corners: ours={len(ours_c)} cv2={len(cv_c)} "
          f"nn mean={d.mean():.2f}px p90={np.percentile(d,90):.2f}px "
          f"-> {'PASS' if corner_ok else 'FAIL'}")

    # track the SAME points with both, compare endpoints
    g1 = img1.astype(np.uint8)
    o_nxt, o_st = calc_optical_flow_pyr_lk(g0, g1, cv_c, win_size=21, max_level=3)
    c_nxt, c_st, _ = cv2.calcOpticalFlowPyrLK(
        g0, g1, cv_c, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    both = o_st.astype(bool) & c_st.reshape(-1).astype(bool)
    ep = np.linalg.norm(o_nxt[both] - c_nxt[both], axis=1)
    flow_ok = float(ep.mean()) < 0.5 and float(np.percentile(ep, 95)) < 1.0
    print(f"  flow:    common={both.sum()}/{len(cv_c)} "
          f"endpoint mean={ep.mean():.3f}px p95={np.percentile(ep,95):.3f}px "
          f"-> {'PASS' if flow_ok else 'FAIL'}")
    return corner_ok and flow_ok


def _load_gold_pair():
    """Two adjacent real frames from a gold session, or None if unavailable."""
    base = Path(__file__).resolve().parents[1] / "sessions" / "gold"
    for name in ("lab_loop_30s", "corridor_60s", "quick_motion_15s"):
        sess = base / name
        if not (sess / "meta.json").exists():
            continue
        try:
            from ours.lib.io.reader import SessionReader
            r = SessionReader(sess)
            if len(r) < 20:
                continue
            # frames 0-9 can be blank camera warmup -> use 15, 16
            return name, r.load_frame(15).gray_left, r.load_frame(16).gray_left
        except Exception:
            continue
    return None


def test_agreement_real() -> bool:
    """On real adjacent frames, ours agrees with OpenCV (if data + cv2 present)."""
    print("== 2b. agreement vs OpenCV (real gold frame) ==")
    if not HAVE_CV2:
        print("  cv2 not available -- SKIP")
        return True
    pair = _load_gold_pair()
    if pair is None:
        print("  no gold session present -- SKIP (synthetic tests still cover it)")
        return True
    name, g0, g1 = pair
    g0 = np.ascontiguousarray(g0)
    g1 = np.ascontiguousarray(g1)
    cv_c = cv2.goodFeaturesToTrack(g0, maxCorners=400, qualityLevel=0.01,
                                   minDistance=12.0, blockSize=7)
    cv_c = cv_c.reshape(-1, 2).astype(np.float32)
    ours_c = good_features_to_track(g0, max_corners=400, quality_level=0.01,
                                    min_distance=12.0, block_size=7)
    d = nn_dist(ours_c, cv_c)
    corner_ok = float(d.mean()) < 1.5 and float(np.percentile(d, 90)) < 4.0

    o_nxt, o_st = calc_optical_flow_pyr_lk(g0, g1, cv_c, win_size=21, max_level=3)
    c_nxt, c_st, _ = cv2.calcOpticalFlowPyrLK(
        g0, g1, cv_c, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    both = o_st.astype(bool) & c_st.reshape(-1).astype(bool)
    ep = np.linalg.norm(o_nxt[both] - c_nxt[both], axis=1)
    flow_ok = float(np.median(ep)) < 0.5 and float(np.percentile(ep, 90)) < 2.0
    print(f"  session={name}")
    print(f"  corners: ours={len(ours_c)} cv2={len(cv_c)} "
          f"nn mean={d.mean():.2f}px p90={np.percentile(d,90):.2f}px "
          f"-> {'PASS' if corner_ok else 'FAIL'}")
    print(f"  flow:    common={both.sum()}/{len(cv_c)} "
          f"endpoint median={np.median(ep):.3f}px p90={np.percentile(ep,90):.3f}px "
          f"-> {'PASS' if flow_ok else 'FAIL'}")
    return corner_ok and flow_ok


def test_backend_agreement() -> bool:
    """The Numba and pure-NumPy backends must give the same flow (faithful JIT).

    Numba only accelerates our own algorithm; it must not change the result. If
    numba is not installed both calls take the NumPy path and this is trivially
    true, which is the point -- the fallback is exercised either way.
    """
    print("== 1b. numba vs numpy backend agreement ==")
    from ours.lib.frontend.klt_numba import HAVE_NUMBA
    img0 = make_texture(seed=5)
    img1 = translate(img0, 2.1, -1.3)
    g0, g1 = img0.astype(np.uint8), img1.astype(np.uint8)
    pts = good_features_to_track(g0, max_corners=300, quality_level=0.01,
                                 min_distance=12.0, block_size=7)
    n_nb, s_nb = calc_optical_flow_pyr_lk(g0, g1, pts, use_numba=True)
    n_np, s_np = calc_optical_flow_pyr_lk(g0, g1, pts, use_numba=False)
    both = s_nb.astype(bool) & s_np.astype(bool)
    d = np.linalg.norm(n_nb[both] - n_np[both], axis=1)
    status_ok = bool((s_nb == s_np).all())
    diff_ok = (d.max() < 0.01) if d.size else True
    passed = status_ok and diff_ok
    print(f"  numba installed={HAVE_NUMBA}  status match={status_ok}  "
          f"endpoint diff max={d.max() if d.size else 0:.4f}px "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def _time_ms(fn, n=5) -> float:
    fn()  # warm
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e3


def report_timing() -> None:
    """Print per-frame timing vs the live budget (informational, never fails)."""
    print("== 3. timing vs live budget (informational) ==")
    img0 = make_texture(seed=3)
    img1 = translate(img0, 1.5, 1.0)
    g0, g1 = img0.astype(np.uint8), img1.astype(np.uint8)

    full = FrontendConfig()
    live = FrontendConfig.live_own()

    def run(cfg, detector_pts):
        det = lambda: good_features_to_track(
            g0, max_corners=cfg.max_corners, quality_level=cfg.quality_level,
            min_distance=cfg.min_distance, block_size=cfg.block_size)
        trk = lambda: (
            calc_optical_flow_pyr_lk(g0, g1, detector_pts,
                                     win_size=cfg.win_size, max_level=cfg.max_level),
            calc_optical_flow_pyr_lk(g1, g0, detector_pts,
                                     win_size=cfg.win_size, max_level=cfg.max_level))
        return _time_ms(det), _time_ms(trk)

    for label, cfg in [("own full ", full), ("own live ", live)]:
        pts = good_features_to_track(
            g0, max_corners=cfg.max_corners, quality_level=cfg.quality_level,
            min_distance=cfg.min_distance, block_size=cfg.block_size)
        t_det, t_trk = run(cfg, pts)
        verdict = "OK" if t_trk < LIVE_BUDGET_MS else "OVER BUDGET"
        print(f"  {label}(win={cfg.win_size} lvl={cfg.max_level} "
              f"corners={cfg.max_corners}): detect {t_det:5.1f}ms  "
              f"track(fwd+bwd) {t_trk:5.1f}ms  -> {verdict} "
              f"(budget {LIVE_BUDGET_MS:.0f}ms)")

    if HAVE_CV2:
        pts = cv2.goodFeaturesToTrack(g0, maxCorners=400, qualityLevel=0.01,
                                      minDistance=12.0, blockSize=7).reshape(-1, 2)
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        t_det = _time_ms(lambda: cv2.goodFeaturesToTrack(
            g0, maxCorners=400, qualityLevel=0.01, minDistance=12.0, blockSize=7))
        t_trk = _time_ms(lambda: (
            cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, **lk),
            cv2.calcOpticalFlowPyrLK(g1, g0, pts, None, **lk)))
        print(f"  cv2       (win=21 lvl=3 corners=400): detect {t_det:5.1f}ms  "
              f"track(fwd+bwd) {t_trk:5.1f}ms  (reference)")


def main() -> int:
    r1 = test_correctness()
    rb = test_backend_agreement()
    r2 = test_agreement_synthetic()
    r3 = test_agreement_real()
    report_timing()
    ok = r1 and rb and r2 and r3
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
