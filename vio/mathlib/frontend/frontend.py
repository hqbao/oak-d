"""Sparse feature frontend: Shi-Tomasi corners + KLT optical-flow tracking.

This is the visual frontend for the from-scratch VIO. It mirrors what a
feature-based VIO (e.g. Basalt) does at the lowest level: maintain a set of
point tracks across frames using pyramidal Lucas-Kanade optical flow, and top
up with fresh corners when tracks die off.

It is deliberately library-honest: every track is a *real* corner tracked by
real KLT, not a synthetic point. Each track carries a persistent integer id so
downstream code can associate observations across time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .corners import good_features_to_track
from .klt import calc_optical_flow_pyr_lk


@dataclass
class FrontendConfig:
    max_corners: int = 400
    quality_level: float = 0.01
    min_distance: float = 12.0  # px between corners
    block_size: int = 7
    # Bucketed (per-cell grid) corner detection. Default False -> the original
    # global detect path (byte-identical at the 640 baseline). Set True ONLY at
    # low resolution (the resolution builder turns it on) to force even spatial
    # coverage so clustered corners don't make the PnP geometry degenerate.
    bucketed: bool = False
    # KLT
    win_size: int = 21
    max_level: int = 3
    # bidirectional (forward-backward) check threshold in px
    fb_threshold: float = 1.0
    # re-detect when tracked count drops below this fraction of max_corners
    redetect_ratio: float = 0.6

    @classmethod
    def live_own(cls) -> "FrontendConfig":
        """Lighter library-free preset for *live* use of our own KLT.

        The full-quality config (``win_size=21``, ``max_level=3``,
        ``max_corners=400``) costs ~120 ms/frame for our pure-NumPy
        forward-backward KLT -- ~2x over the 50 ms budget at 20 fps, so the live
        read loop falls behind, skips frames and loses tracking. A smaller
        window + pyramid + corner budget drops the per-frame cost to ~38-58 ms
        (at/under budget) while keeping ATE essentially unchanged offline
        (lab_loop 1.27->1.25%, quick_motion 2.14->2.59%). Used live when Numba
        is unavailable; offline scoring keeps the full-quality default.
        """
        return cls(win_size=13, max_level=2, max_corners=200)



@dataclass
class TrackState:
    """Current set of live tracks: pixel coords + persistent ids."""

    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    ids: np.ndarray = field(default_factory=lambda: np.empty((0,), np.int64))


class KLTFrontend:
    """Maintains KLT tracks across a grayscale image stream."""

    def __init__(self, cfg: FrontendConfig | None = None):
        self.cfg = cfg or FrontendConfig()
        self._prev_gray: np.ndarray | None = None
        self._state = TrackState()
        self._next_id = 0

    @property
    def tracks(self) -> TrackState:
        return self._state

    def _track(self, prev_gray: np.ndarray, gray: np.ndarray,
               prev_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward+backward KLT; returns (next_pts, status_bool).

        Uses our own library-free :func:`calc_optical_flow_pyr_lk`, returning the
        ``(N, 2) float32`` next points + ``(N,) bool`` status contract.
        """
        nxt, st = calc_optical_flow_pyr_lk(
            prev_gray, gray, prev_pts,
            win_size=self.cfg.win_size, max_level=self.cfg.max_level)
        back, st2 = calc_optical_flow_pyr_lk(
            gray, prev_gray, nxt,
            win_size=self.cfg.win_size, max_level=self.cfg.max_level)
        fb_err = np.linalg.norm(prev_pts - back, axis=1)
        st = st.reshape(-1).astype(bool)
        st2 = st2.reshape(-1).astype(bool)
        h, w = gray.shape
        in_bounds = (
            (nxt[:, 0] >= 0) & (nxt[:, 0] < w)
            & (nxt[:, 1] >= 0) & (nxt[:, 1] < h)
        )
        good = st & st2 & (fb_err < self.cfg.fb_threshold) & in_bounds
        return nxt, good

    def _detect(self, gray: np.ndarray, existing: np.ndarray) -> np.ndarray:
        """Detect new corners, keeping clear of neighbourhoods of existing points."""
        need = self.cfg.max_corners - existing.shape[0]
        if need <= 0:
            return np.empty((0, 2), np.float32)
        return good_features_to_track(
            gray,
            max_corners=need,
            quality_level=self.cfg.quality_level,
            min_distance=self.cfg.min_distance,
            block_size=self.cfg.block_size,
            exclude=existing,
            bucketed=self.cfg.bucketed,
        )

    def process(self, gray: np.ndarray) -> TrackState:
        """Advance the tracker by one frame; returns the live track set.

        On the first frame this only detects. On subsequent frames it tracks the
        previous points forward with KLT + forward-backward consistency, drops
        failed tracks, then tops up with fresh corners.
        """
        if self._prev_gray is None:
            pts = self._detect(gray, np.empty((0, 2), np.float32))
            ids = np.arange(self._next_id, self._next_id + len(pts), dtype=np.int64)
            self._next_id += len(pts)
            self._state = TrackState(points=pts, ids=ids)
            self._prev_gray = gray
            return self._state

        prev_pts = self._state.points
        prev_ids = self._state.ids
        if prev_pts.shape[0] > 0:
            nxt, good = self._track(self._prev_gray, gray, prev_pts)
            tracked_pts = nxt[good].astype(np.float32)
            tracked_ids = prev_ids[good]
        else:
            tracked_pts = np.empty((0, 2), np.float32)
            tracked_ids = np.empty((0,), np.int64)

        # Top up with fresh corners if we lost too many.
        if tracked_pts.shape[0] < self.cfg.redetect_ratio * self.cfg.max_corners:
            fresh = self._detect(gray, tracked_pts)
            fresh_ids = np.arange(
                self._next_id, self._next_id + len(fresh), dtype=np.int64
            )
            self._next_id += len(fresh)
            tracked_pts = np.vstack([tracked_pts, fresh]) if len(fresh) else tracked_pts
            tracked_ids = np.concatenate([tracked_ids, fresh_ids])

        self._state = TrackState(points=tracked_pts, ids=tracked_ids)
        self._prev_gray = gray
        return self._state
