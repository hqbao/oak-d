"""Loop closure: ORB appearance matching + geometric verification.

This is the *frontend* of the SLAM layer. Visual odometry tells us where the
camera is up to accumulated drift; loop closure recognises when the camera has
physically returned to a previously-seen place and produces a precise relative
pose constraint between the two keyframes, which the pose graph
(:mod:`slam.mathlib.loop.posegraph`) then uses to cancel the drift.

Honest pipeline note
--------------------
The ORB features here are *our own* real features detected on the real recorded
image -- this is genuinely our loop-closure frontend, not a fake overlay of some
black box's internals. Detection, description, matching, the epipolar pre-filter
and the metric PnP are ALL our own library-free NumPy (:mod:`slam.mathlib.loop.orb` +
:mod:`sky.front.pnp`); there is no cv2 anywhere on this path. The relative pose
comes from a real RANSAC PnP on real depth-backprojected 3D points, so a
confirmed loop is a real geometric fact.

Two stages, cheap-to-expensive:
1. **Appearance gate** -- match ORB descriptors (Hamming + Lowe ratio test)
   against an older keyframe. Needs enough good matches to bother verifying.
2. **Geometric verification** -- backproject the matched ORB keypoints of the
   *old* keyframe with its depth into 3D, then RANSAC-PnP onto the *current*
   keyframe's matched pixels. Enough inliers => a confirmed loop with a metric
   relative transform ``T_cur_old`` (old-cam coords -> cur-cam coords).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sky.front.pnp import solve_pnp_ransac

from .orb import (ORB, OrbConfig, find_fundamental_ransac, match_ratio_mutual)


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


# Per-match verification-stage labels (uint8). A match's stage is the FURTHEST
# funnel gate it survived: every appearance match starts at APPEARANCE; one that
# also passed the fundamental-matrix RANSAC is promoted to EPIPOLAR; one that was
# additionally a PnP-RANSAC inlier is promoted to PNP. These are the exact colour
# bands the UI's loop-closure viz draws (grey -> yellow -> green).
STAGE_APPEARANCE = 0   # matched on descriptors only (dropped before/at epipolar)
STAGE_EPIPOLAR = 1     # also a fundamental-matrix (epipolar) inlier, not a PnP one
STAGE_PNP = 2          # also a PnP-RANSAC inlier -> confirmed-loop correspondence


@dataclass
class LoopMatchCapture:
    """The full match funnel for ONE ``verify_capture`` candidate (opt-in only).

    Returned ALONGSIDE the normal verify result so the UI's loop-closure window
    can show WHY a candidate fired or was rejected. Pixel pairs are in the two
    keyframes' OWN pixel coordinates (the rectified-left grid the ORB ran on), so
    the UI can draw a line per match across two side-by-side keyframe images.

    * ``cur_px`` / ``old_px`` -- ``(N, 2)`` float32, one row per ALL appearance
      matches, in the SAME order; ``cur_px[i]`` <-> ``old_px[i]`` is one match.
    * ``stage`` -- ``(N,)`` uint8 per-match label (:data:`STAGE_APPEARANCE` /
      :data:`STAGE_EPIPOLAR` / :data:`STAGE_PNP`).
    * ``n_appearance`` / ``n_fmat_inliers`` / ``n_pnp_inliers`` -- the funnel
      counts (the gate inputs at each stage).
    * ``rot_deg`` -- the loop's relative rotation magnitude vs odometry in degrees
      (NaN when the engine did not supply an odometry pair / the gate is off).
    * ``rot_gate_deg`` -- the rotation-gate threshold (``loop_max_odom_rot_deg``,
      0 = gate disabled), carried so the UI never needs the config.
    * ``accepted`` -- True iff this candidate became a confirmed loop edge (passed
      every gate INCLUDING the engine-side rotation gate).
    """

    cur_px: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    old_px: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    stage: np.ndarray = field(default_factory=lambda: np.empty((0,), np.uint8))
    n_appearance: int = 0
    n_fmat_inliers: int = 0
    n_pnp_inliers: int = 0
    rot_deg: float = float("nan")
    rot_gate_deg: float = 0.0
    accepted: bool = False


class KeyframeAppearance:
    """ORB keypoints + descriptors + per-keypoint metric depth for one keyframe.

    Stored per persistent keyframe so loop detection never needs the full image
    again (only the compact descriptors + sparse depth), which keeps the map
    light enough for the live path too.
    """

    __slots__ = ("kps", "desc", "depth", "K")

    def __init__(self, gray: np.ndarray, depth_m: np.ndarray, K: np.ndarray,
                 orb: ORB, cfg: LoopConfig):
        # Our ORB clips+casts to uint8 internally; the live path feeds the
        # float32 rectified left image, so this works for live + offline alike.
        kps, desc = orb.detect_and_compute(gray)
        self.K = K
        if desc is None or len(kps) == 0:
            self.kps = np.empty((0, 2), np.float32)
            self.desc = np.empty((0, 32), np.uint8)
            self.depth = np.empty((0,), np.float32)
            return
        pts = np.asarray(kps, dtype=np.float32)
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
        self.K = np.asarray(K, dtype=np.float64)
        self.cfg = cfg or LoopConfig()
        # Our own oriented-FAST + rotated-BRIEF (no cv2).
        self.orb = ORB(OrbConfig(n_features=self.cfg.orb_features))

    def make_appearance(self, gray: np.ndarray,
                        depth_m: np.ndarray) -> KeyframeAppearance:
        return KeyframeAppearance(gray, depth_m, self.K, self.orb, self.cfg)

    def _good_matches(self, a: KeyframeAppearance, b: KeyframeAppearance):
        """Lowe-ratio + mutual matches from a.desc -> b.desc; list of (ia, ib).

        The cross-check (mutual) cheaply removes one-sided descriptor
        coincidences (a common source of perceptual-aliasing false loops) before
        the geometry stages. Delegates to our own Hamming matcher.
        """
        return match_ratio_mutual(a.desc, b.desc, ratio=self.cfg.ratio)

    def verify(self, cur: KeyframeAppearance, old: KeyframeAppearance):
        """Geometric verification cur<->old.

        Returns ``(T_cur_old, n_inliers, n_matches)`` where ``T_cur_old`` maps a
        point in the OLD camera frame to the CURRENT camera frame, or ``None`` if
        the loop is not geometrically confirmed.

        This is the canonical, byte-frozen entry point. It calls the shared core
        with ``capture=None``, so the path here is identical to the original
        (no funnel arrays built, same early returns) -- the offline / oracle path
        stays bit-for-bit unchanged. For the live UI funnel use
        :meth:`verify_capture`.
        """
        return self._verify_core(cur, old, None)

    def verify_capture(self, cur: KeyframeAppearance, old: KeyframeAppearance):
        """Opt-in funnel capture: same math as :meth:`verify`, plus the funnel.

        Returns ``(result, capture)`` where ``result`` is EXACTLY what
        :meth:`verify` returns (``(T_cur_old, n_inliers, n_matches)`` or ``None``)
        and ``capture`` is a populated :class:`LoopMatchCapture` describing every
        appearance match, its survived stage, and the funnel counts -- the data
        the UI's loop-closure window draws. ``verify`` itself is unaffected (this
        is a sibling method, LIVE-only), so the deterministic path is unchanged.

        Even a candidate REJECTED at the appearance gate yields a capture with the
        matches it had (so the UI can show "appearance 14 -> rejected"); the
        rotation gate (``loop_max_odom_rot_deg``) is applied by the engine AFTER
        this call, so ``capture.accepted`` / ``rot_deg`` / ``rot_gate_deg`` are
        finalised there via :meth:`LoopMatchCapture` field updates.
        """
        cap = LoopMatchCapture(rot_gate_deg=0.0)
        result = self._verify_core(cur, old, cap)
        return result, cap

    def _verify_core(self, cur: KeyframeAppearance, old: KeyframeAppearance,
                     capture):
        """Shared verification core for :meth:`verify` / :meth:`verify_capture`.

        When ``capture`` is ``None`` the body runs the ORIGINAL verify path with
        zero extra work (the funnel branches are all guarded by ``capture is not
        None``), so the frozen offline result is preserved bit-for-bit. When a
        :class:`LoopMatchCapture` is passed it is populated AS the funnel runs:
        every appearance match is recorded at :data:`STAGE_APPEARANCE`, the ones
        that survive the fundamental-matrix RANSAC are promoted to
        :data:`STAGE_EPIPOLAR`, and the PnP-RANSAC inliers to :data:`STAGE_PNP`.
        """
        good = self._good_matches(cur, old)
        # --- capture: record EVERY appearance match (the funnel's widest stage).
        # `good_all` keeps the full appearance set (verify() prunes `good` in
        # place at the epipolar stage; the capture needs the dropped ones too).
        good_all = good
        if capture is not None:
            capture.n_appearance = len(good_all)
            if good_all:
                capture.cur_px = np.array(
                    [cur.kps[ic] for ic, _ in good_all], np.float32)
                capture.old_px = np.array(
                    [old.kps[io] for _, io in good_all], np.float32)
            capture.stage = np.zeros(len(good_all), np.uint8)  # all APPEARANCE

        if len(good) < self.cfg.min_matches:
            return None

        # Epipolar pre-filter: a true revisit obeys a single fundamental matrix,
        # so RANSAC on the 2D-2D matches removes appearance mismatches (different
        # places that merely look alike -- corridor perceptual aliasing) before
        # the more expensive PnP. Geometrically impossible matches are dropped.
        if len(good) < 8:                      # FM needs >= 8 points
            return None
        pc = np.array([cur.kps[ic] for ic, _ in good], np.float64)
        po = np.array([old.kps[io] for _, io in good], np.float64)
        res_f = find_fundamental_ransac(
            po, pc, self.cfg.fmat_thresh_px, self.cfg.ransac_conf)
        if res_f is None:
            return None                        # degenerate / too few inliers
        _F, fmask = res_f
        if capture is not None:
            # Promote each epipolar inlier to STAGE_EPIPOLAR. `fmask` is aligned
            # with `good` here, which (when capture is on) is still the full
            # appearance set, so the mask indexes `capture.stage` 1:1.
            keep = np.asarray(fmask, dtype=bool).reshape(-1)
            n = min(len(keep), len(capture.stage))
            capture.stage[:n][keep[:n]] = STAGE_EPIPOLAR
            capture.n_fmat_inliers = int(keep.sum())
        if int(fmask.sum()) < self.cfg.min_fmat_inliers:
            return None
        # Indices into the FULL appearance set that survived the epipolar gate, so
        # the capture can map a PnP inlier (indexed in the pruned `good`) back to
        # its row in `capture.stage`.
        epi_idx = [i for i, keep in enumerate(fmask) if keep]
        good = [g for g, keep in zip(good, fmask) if keep]

        obj, img = [], []
        # `pnp_rows` maps each PnP correspondence (in solve order) back to its row
        # in the FULL appearance set (= capture.stage index), so a PnP inlier can
        # be promoted to STAGE_PNP. Built only when capturing (else unused).
        pnp_rows: list[int] = []
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        for j, (ic, io) in enumerate(good):
            z = float(old.depth[io])
            if z <= 0.0:
                continue
            u, v = old.kps[io]
            obj.append([(u - cx) * z / fx, (v - cy) * z / fy, z])  # 3D in old cam
            img.append(cur.kps[ic])                                # 2D in cur img
            if capture is not None:
                pnp_rows.append(epi_idx[j])
        if len(obj) < self.cfg.min_inliers:
            return None
        obj = np.asarray(obj, np.float64)
        img = np.asarray(img, np.float64)
        # Our own RANSAC PnP (library-free) -> T_cur_old.
        ok, R, t, inliers = solve_pnp_ransac(
            obj, img, self.K,
            reproj_px=self.cfg.ransac_reproj_px,
            iters=self.cfg.ransac_iters,
            conf=self.cfg.ransac_conf,
            min_points=self.cfg.min_inliers,
        )
        if capture is not None and ok and inliers is not None:
            # Promote each PnP inlier (index into the PnP-correspondence list) to
            # STAGE_PNP via its row in the full appearance set.
            for k in np.asarray(inliers).reshape(-1):
                if 0 <= int(k) < len(pnp_rows):
                    capture.stage[pnp_rows[int(k)]] = STAGE_PNP
            capture.n_pnp_inliers = int(len(inliers))
        if not ok or inliers is None or len(inliers) < self.cfg.min_inliers:
            return None
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.reshape(3)             # T_cur_old: X_cur = R X_old + t
        return T, int(len(inliers)), len(good)
