"""Loop closure: ORB appearance matching + geometric verification.

This is the *frontend* of the SLAM layer. Visual odometry tells us where the
camera is up to accumulated drift; loop closure recognises when the camera has
physically returned to a previously-seen place and produces a precise relative
pose constraint between the two keyframes, which the pose graph
(:mod:`oakd.vio.posegraph`) then uses to cancel the drift.

Honest pipeline note
--------------------
The ORB features here are *our own* real features detected on the real recorded
image -- this is genuinely our loop-closure frontend, not a fake overlay of some
black box's internals. The relative pose comes from a real RANSAC PnP on real
depth-backprojected 3D points, so a confirmed loop is a real geometric fact.

Two stages, cheap-to-expensive:
1. **Appearance gate** -- match ORB descriptors (Hamming + Lowe ratio test)
   against an older keyframe. Needs enough good matches to bother verifying.
2. **Geometric verification** -- backproject the matched ORB keypoints of the
   *old* keyframe with its depth into 3D, then ``solvePnPRansac`` onto the
   *current* keyframe's matched pixels. Enough inliers => a confirmed loop with
   a metric relative transform ``T_cur_old`` (old-cam coords -> cur-cam coords).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LoopConfig:
    orb_features: int = 800
    ratio: float = 0.75           # Lowe ratio test
    min_matches: int = 50         # appearance gate: good descriptor matches
    min_fmat_inliers: int = 30    # epipolar (fundamental) RANSAC inliers
    min_inliers: int = 30         # geometric gate: PnP RANSAC inliers
    min_loop_gap: int = 25        # ignore candidates < this many KFs back
    ransac_reproj_px: float = 2.0
    ransac_iters: int = 200
    ransac_conf: float = 0.999
    fmat_thresh_px: float = 2.0   # fundamental-matrix RANSAC threshold
    min_depth_m: float = 0.2
    max_depth_m: float = 8.0


class KeyframeAppearance:
    """ORB keypoints + descriptors + per-keypoint metric depth for one keyframe.

    Stored per persistent keyframe so loop detection never needs the full image
    again (only the compact descriptors + sparse depth), which keeps the map
    light enough for the live path too.
    """

    __slots__ = ("kps", "desc", "depth", "K")

    def __init__(self, gray: np.ndarray, depth_m: np.ndarray, K: np.ndarray,
                 orb: cv2.ORB, cfg: LoopConfig):
        # ORB requires CV_8UC1; the live path now feeds the float32 rectified
        # left image (bilinear remap can land slightly outside [0,255]), so
        # clip + cast defensively for every caller (live + offline).
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        kps, desc = orb.detectAndCompute(gray, None)
        self.K = K
        if desc is None or len(kps) == 0:
            self.kps = np.empty((0, 2), np.float32)
            self.desc = np.empty((0, 32), np.uint8)
            self.depth = np.empty((0,), np.float32)
            return
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        h, w = depth_m.shape
        d = np.zeros(len(pts), np.float32)
        for n, (u, v) in enumerate(pts):
            pu, pv = int(round(u)), int(round(v))
            if 0 <= pu < w and 0 <= pv < h:
                z = float(depth_m[pv, pu])
                if cfg.min_depth_m <= z <= cfg.max_depth_m:
                    d[n] = z
        self.kps = pts
        self.desc = desc
        self.depth = d


class LoopDetector:
    """Detects loops between a query keyframe and a set of older keyframes."""

    def __init__(self, K: np.ndarray, cfg: LoopConfig | None = None):
        import cv2  # ORB loop closure (ours-slam only); not a runtime dep of f2f/ba
        self.K = np.asarray(K, dtype=np.float64)
        self.cfg = cfg or LoopConfig()
        self.orb = cv2.ORB_create(nfeatures=self.cfg.orb_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def make_appearance(self, gray: np.ndarray,
                        depth_m: np.ndarray) -> KeyframeAppearance:
        return KeyframeAppearance(gray, depth_m, self.K, self.orb, self.cfg)

    def _good_matches(self, a: KeyframeAppearance, b: KeyframeAppearance):
        """Lowe-ratio matches from a.desc -> b.desc; returns list of (ia, ib).

        Symmetric (mutual) check: a pair is kept only if ``a[i] -> b[j]`` passes
        the ratio test AND ``b[j]``'s best match back in ``a`` is ``i``. The
        cross-check cheaply removes one-sided descriptor coincidences (a common
        source of perceptual-aliasing false loops) before the geometry stages.
        """
        if len(a.desc) < 2 or len(b.desc) < 2:
            return []
        knn = self.matcher.knnMatch(a.desc, b.desc, k=2)
        # Best a-index for each b-index (for the mutual check).
        rev = self.matcher.match(b.desc, a.desc)
        best_ba = {m.queryIdx: m.trainIdx for m in rev}
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.cfg.ratio * n.distance:
                if best_ba.get(m.trainIdx, -1) == m.queryIdx:   # mutual
                    good.append((m.queryIdx, m.trainIdx))
        return good

    def verify(self, cur: KeyframeAppearance, old: KeyframeAppearance):
        """Geometric verification cur<->old.

        Returns ``(T_cur_old, n_inliers, n_matches)`` where ``T_cur_old`` maps a
        point in the OLD camera frame to the CURRENT camera frame, or ``None`` if
        the loop is not geometrically confirmed.
        """
        import cv2  # ORB loop closure (ours-slam only); lazy so f2f/ba stay cv2-free
        good = self._good_matches(cur, old)
        if len(good) < self.cfg.min_matches:
            return None

        # Epipolar pre-filter: a true revisit obeys a single fundamental matrix,
        # so RANSAC on the 2D-2D matches removes appearance mismatches (different
        # places that merely look alike -- corridor perceptual aliasing) before
        # the more expensive PnP. Geometrically impossible matches are dropped.
        if len(good) < 8:                      # FM needs >= 8 points
            return None
        pc = np.array([cur.kps[ic] for ic, _ in good], np.float32)
        po = np.array([old.kps[io] for _, io in good], np.float32)
        try:
            F, fmask = cv2.findFundamentalMat(
                po, pc, cv2.FM_RANSAC, self.cfg.fmat_thresh_px,
                self.cfg.ransac_conf)
        except cv2.error:
            return None
        if F is None or fmask is None or F.shape != (3, 3):
            return None                        # degenerate / multiple solutions
        fmask = fmask.ravel().astype(bool)
        if int(fmask.sum()) < self.cfg.min_fmat_inliers:
            return None
        good = [g for g, keep in zip(good, fmask) if keep]

        obj, img = [], []
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        for ic, io in good:
            z = float(old.depth[io])
            if z <= 0.0:
                continue
            u, v = old.kps[io]
            obj.append([(u - cx) * z / fx, (v - cy) * z / fy, z])  # 3D in old cam
            img.append(cur.kps[ic])                                # 2D in cur img
        if len(obj) < self.cfg.min_inliers:
            return None
        obj = np.asarray(obj, np.float64)
        img = np.asarray(img, np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, self.K, None,
            iterationsCount=self.cfg.ransac_iters,
            reprojectionError=self.cfg.ransac_reproj_px,
            confidence=self.cfg.ransac_conf,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None or len(inliers) < self.cfg.min_inliers:
            return None
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)          # T_cur_old: X_cur = R X_old + t
        return T, int(len(inliers)), len(good)
