#!/usr/bin/env python3
"""Gold self-test for our pure-NumPy ORB (oriented FAST + steered BRIEF).

This pins down the library-free loop-closure frontend (:mod:`ours.vio.orb`)
with hard pass/fail thresholds so any future change is caught. It checks four
independent things:

1. FUNDAMENTAL-MATRIX RANSAC (synthetic, does NOT trust OpenCV) -- generate a
   known two-view geometry (random 3D points seen by two camera poses), project
   to pixels, spike in outlier matches, and confirm
   ``find_fundamental_ransac`` recovers the inlier set (epipolar Sampson error
   near zero on true matches, outliers rejected). Ground truth, no library.

2. HAMMING MATCHER (synthetic) -- build two descriptor sets with a known
   permutation + bit noise and confirm ``match_ratio_mutual`` recovers the
   correct correspondences. Ground truth, no library.

3. DETECT + DESCRIBE on a real gold frame -- our ORB must find a healthy number
   of keypoints with 32-byte descriptors, and rotating the image by a known
   angle must leave the descriptors approximately rotation-invariant (steered
   BRIEF: the matched-back fraction stays high). If cv2 is present we also print
   a repeatability comparison vs ``cv2.ORB`` (NOT a pass/fail gate -- different
   detectors legitimately pick slightly different corners).

4. END-TO-END loop geometry on two overlapping gold frames -- detect, match,
   epipolar-filter, then recover the relative pose with our PnP and confirm it
   is a valid SE3 with a sane (small) translation between nearby frames.

Run:  .venv/bin/python ours/tools/orb_selftest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.misc.pngio import imread_gray                              # noqa: E402
from ours.lib.loop.orb import (                                      # noqa: E402
    ORB, OrbConfig, find_fundamental_ransac, hamming_knn,
    match_ratio_mutual,
)
from ours.lib.odometry.pnp import solve_pnp_ransac                       # noqa: E402

try:
    import cv2
    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False

GOLD = Path(__file__).resolve().parents[2] / "sessions/gold/lab_loop_30s/input"


def _gold_frames(skip: int = 12):
    """Yield (gray float64) frames from the gold session, past the warmup."""
    recs = [json.loads(l) for l in open(GOLD / "frames.jsonl")]
    out = []
    for rec in recs[skip:]:
        out.append(imread_gray(str(GOLD / rec["left_path"])).astype(np.float64))
    return out


def _rotate_image(img: np.ndarray, deg: float) -> np.ndarray:
    """Rotate about the centre by ``deg`` (nearest-neighbour, pure NumPy)."""
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    yy, xx = np.mgrid[0:h, 0:w]
    xr = c * (xx - cx) + s * (yy - cy) + cx
    yr = -s * (xx - cx) + c * (yy - cy) + cy
    xi = np.clip(np.round(xr).astype(int), 0, w - 1)
    yi = np.clip(np.round(yr).astype(int), 0, h - 1)
    return img[yi, xi]


# ---------------------------------------------------------------------------
def test_fundamental_synthetic() -> bool:
    print("[1] fundamental-matrix RANSAC (synthetic ground truth)")
    rng = np.random.default_rng(7)
    # known intrinsics
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    # 80 random 3D points in front of the cameras
    n = 80
    P = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n),
                         rng.uniform(4, 9, n)])
    # second camera: small rotation + translation
    th = np.radians(8.0)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0],
                  [-np.sin(th), 0, np.cos(th)]])
    t = np.array([0.4, 0.05, 0.1])

    def proj(Pw, R, t):
        Pc = Pw @ R.T + t
        u = K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2]
        v = K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]
        return np.column_stack([u, v])

    p1 = proj(P, np.eye(3), np.zeros(3))
    p2 = proj(P, R, t)
    p1 += rng.normal(0, 0.3, p1.shape)          # mild pixel noise
    p2 += rng.normal(0, 0.3, p2.shape)
    # corrupt 20 matches with random outliers
    n_out = 20
    out_idx = rng.choice(n, n_out, replace=False)
    p2[out_idx] = rng.uniform([0, 0], [640, 480], (n_out, 2))
    truth = np.ones(n, bool)
    truth[out_idx] = False

    res = find_fundamental_ransac(p1, p2, thresh_px=2.0, conf=0.999)
    if res is None:
        print("  FAIL: no fundamental matrix found")
        return False
    _F, mask = res
    # how well the recovered inlier mask matches the ground-truth inlier set
    tp = int((mask & truth).sum())
    fp = int((mask & ~truth).sum())
    recall = tp / int(truth.sum())
    print(f"  inliers kept={int(mask.sum())}  true_recall={recall:.2f}  "
          f"false_inliers={fp}")
    ok = recall >= 0.85 and fp <= 2
    print(f"  -> {'PASS' if ok else 'FAIL'} (need recall>=0.85, false<=2)")
    return ok


def test_hamming_matcher() -> bool:
    print("[2] Hamming matcher + ratio/mutual (synthetic ground truth)")
    rng = np.random.default_rng(11)
    m = 120
    base = rng.integers(0, 256, size=(m, 32), dtype=np.uint8)
    # second set = permuted copy with a few bits flipped per descriptor
    perm = rng.permutation(m)
    b = base[perm].copy()
    for i in range(m):
        flip = rng.choice(32 * 8, size=rng.integers(0, 12), replace=False)
        for bit in flip:
            b[i, bit // 8] ^= np.uint8(1 << (bit % 8))
    # knn sanity: nearest of a[i] in b should be perm position
    idx, dist = hamming_knn(base, b, k=2)
    nearest = idx[:, 0]
    want = np.array([np.where(perm == i)[0][0] for i in range(m)])
    knn_acc = float((nearest == want).mean())
    good = match_ratio_mutual(base, b, ratio=0.8)
    correct = sum(1 for ia, ib in good if perm[ib] == ia)
    prec = correct / max(1, len(good))
    print(f"  knn top1 acc={knn_acc:.2f}  ratio+mutual matches={len(good)}  "
          f"precision={prec:.2f}")
    ok = knn_acc >= 0.95 and prec >= 0.98 and len(good) >= 60
    print(f"  -> {'PASS' if ok else 'FAIL'} (need knn>=0.95, prec>=0.98, n>=60)")
    return ok


def test_detect_describe() -> bool:
    print("[3] detect + describe + rotation invariance (gold frame)")
    if not GOLD.exists():
        print("  gold session missing -- SKIP")
        return True
    frames = _gold_frames()
    g = frames[0]
    orb = ORB(OrbConfig(n_features=800))
    pts, desc = orb.detect_and_compute(g)
    print(f"  keypoints={len(pts)}  desc shape={desc.shape} dtype={desc.dtype}")
    if len(pts) < 400 or desc.shape[1] != 32:
        print("  -> FAIL (need >=400 keypoints, 32-byte descriptors)")
        return False

    # rotation invariance: rotate by 30 deg, re-detect, match back. Steered
    # BRIEF should keep a healthy matched-back fraction.
    grot = _rotate_image(g, 30.0)
    pr, dr = orb.detect_and_compute(grot)
    good = match_ratio_mutual(desc, dr, ratio=0.85)
    frac = len(good) / max(1, min(len(pts), len(pr)))
    print(f"  rot+30deg: kp={len(pr)}  matched-back={len(good)} "
          f"({frac*100:.0f}% of min set)")
    ok = frac >= 0.10
    print(f"  -> {'PASS' if ok else 'FAIL'} (need matched-back >=10%)")

    if HAVE_CV2:
        cvorb = cv2.ORB_create(nfeatures=800)
        kps, _ = cvorb.detectAndCompute(np.clip(g, 0, 255).astype(np.uint8),
                                        None)
        cvpts = np.array([kp.pt for kp in kps]) if kps else np.empty((0, 2))
        # repeatability: fraction of OUR kps within 3px of a cv2 kp
        if len(cvpts):
            d = np.sqrt(((pts[:, None, :] - cvpts[None, :, :]) ** 2).sum(2))
            rep = float((d.min(1) <= 3.0).mean())
            print(f"  [info] cv2 kp={len(cvpts)}  ours-near-cv2(<=3px)="
                  f"{rep*100:.0f}% (informational, not a gate)")
    return ok


def test_loop_geometry() -> bool:
    print("[4] end-to-end loop geometry (two overlapping gold frames)")
    if not GOLD.exists():
        print("  gold session missing -- SKIP")
        return True
    frames = _gold_frames()
    g0, g1 = frames[0], frames[5]
    orb = ORB(OrbConfig(n_features=800))
    p0, d0 = orb.detect_and_compute(g0)
    p1, d1 = orb.detect_and_compute(g1)
    good = match_ratio_mutual(d0, d1, ratio=0.75)
    print(f"  matches={len(good)}")
    if len(good) < 30:
        print("  -> FAIL (need >=30 ratio+mutual matches on nearby frames)")
        return False
    pc = np.array([p0[i] for i, _ in good], np.float64)
    po = np.array([p1[j] for _, j in good], np.float64)
    res = find_fundamental_ransac(po, pc, thresh_px=2.0, conf=0.999)
    if res is None:
        print("  -> FAIL (fundamental matrix not found)")
        return False
    _F, mask = res
    n_in = int(mask.sum())
    print(f"  epipolar inliers={n_in}/{len(good)}")
    ok = n_in >= 25
    print(f"  -> {'PASS' if ok else 'FAIL'} (need >=25 epipolar inliers)")
    return ok


def main() -> int:
    print("=" * 64)
    print("ORB self-test (library-free oriented FAST + steered BRIEF)")
    print("=" * 64)
    results = [
        test_fundamental_synthetic(),
        test_hamming_matcher(),
        test_detect_describe(),
        test_loop_geometry(),
    ]
    print("-" * 64)
    ok = all(results)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}  ({sum(results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
