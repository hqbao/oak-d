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

import cv2
import numpy as np

from .corners import good_features_to_track
from .klt import calc_optical_flow_pyr_lk


@dataclass
class FrontendConfig:
    max_corners: int = 400
    quality_level: float = 0.01
    min_distance: float = 12.0  # px between corners
    block_size: int = 7
    # KLT
    win_size: int = 21
    max_level: int = 3
    # bidirectional (forward-backward) check threshold in px
    fb_threshold: float = 1.0
    # re-detect when tracked count drops below this fraction of max_corners
    redetect_ratio: float = 0.6
    # use our own pure-NumPy implementations (klt.py pyramidal LK + corners.py
    # Shi-Tomasi) instead of cv2. Tracks/detects the same corners to sub-pixel
    # agreement with cv2; slower but fully library-free. When False the frontend
    # falls back to OpenCV for both tracking and detection (faster live display).
    use_own_klt: bool = True


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
        self._lk_params = dict(
            winSize=(self.cfg.win_size, self.cfg.win_size),
            maxLevel=self.cfg.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    @property
    def tracks(self) -> TrackState:
        return self._state

    def _track(self, prev_gray: np.ndarray, gray: np.ndarray,
               prev_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward+backward KLT; returns (next_pts, status_bool).

        Uses our own :func:`calc_optical_flow_pyr_lk` when ``use_own_klt`` is set
        (the default, library-free), otherwise ``cv2.calcOpticalFlowPyrLK``. Both
        return the same ``(N, 2) float32`` / ``(N,) status`` contract.
        """
        if self.cfg.use_own_klt:
            nxt, st = calc_optical_flow_pyr_lk(
                prev_gray, gray, prev_pts,
                win_size=self.cfg.win_size, max_level=self.cfg.max_level)
            back, st2 = calc_optical_flow_pyr_lk(
                gray, prev_gray, nxt,
                win_size=self.cfg.win_size, max_level=self.cfg.max_level)
        else:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **self._lk_params)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(
                gray, prev_gray, nxt, None, **self._lk_params)
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
        if self.cfg.use_own_klt:
            return good_features_to_track(
                gray,
                max_corners=need,
                quality_level=self.cfg.quality_level,
                min_distance=self.cfg.min_distance,
                block_size=self.cfg.block_size,
                exclude=existing,
            )
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        r = int(self.cfg.min_distance)
        for x, y in existing:
            cv2.circle(mask, (int(x), int(y)), r, 0, -1)
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=need,
            qualityLevel=self.cfg.quality_level,
            minDistance=self.cfg.min_distance,
            blockSize=self.cfg.block_size,
            mask=mask,
        )
        if corners is None:
            return np.empty((0, 2), np.float32)
        return corners.reshape(-1, 2).astype(np.float32)

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
