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

from .corners import good_features_to_track, _shi_tomasi_response
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
    # Frontend-internals visualisation capture (opt-in, --frontend-viz). Default
    # False -> :class:`KLTFrontend` is used verbatim and the returned tracks are
    # byte-identical to before (the oracle relies on this). When True the VIO
    # process builds a :class:`CaptureKLTFrontend` instead, which STASHES the
    # per-frame Shi-Tomasi response heatmap + accepted corners + KLT flow field
    # (prev/next px, fb-error, culled mask) on a side-car snapshot for the UI's
    # "Frontend Internals" view -- WITHOUT changing the tracks it returns. This
    # flag never alters the detection / tracking math; it only flips which
    # frontend class the pipeline instantiates (see vio.modules.pipeline).
    capture: bool = False

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

    def _track_full(self, prev_gray: np.ndarray, gray: np.ndarray,
                    prev_pts: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward+backward KLT; returns ``(next_pts, status_bool, fb_err)``.

        The single arithmetic kernel shared by the base :meth:`_track` and the
        capture subclass. Factored out so the byte-identical ``(nxt, good)`` the
        base path returns and the extra ``fb_err`` (per-point forward-backward
        error, used ONLY for the visualisation colouring) come from the EXACT
        same computation -- there is no parallel re-derivation that could drift.
        Uses our own library-free :func:`calc_optical_flow_pyr_lk`, returning the
        ``(N, 2) float32`` next points + ``(N,) bool`` status + ``(N,) float64``
        fb-error contract.
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
        return nxt, good, fb_err

    def _track(self, prev_gray: np.ndarray, gray: np.ndarray,
               prev_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward+backward KLT; returns (next_pts, status_bool).

        Public frontend contract -- the SAME ``(nxt, good)`` it always returned
        (the ``fb_err`` from :meth:`_track_full` is dropped here so this remains
        byte-identical to the pre-refactor code; the capture subclass keeps it).
        """
        nxt, good, _fb_err = self._track_full(prev_gray, gray, prev_pts)
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


# --------------------------------------------------------------------------- #
# Frontend-internals visualisation capture (opt-in, oracle-inert)
# --------------------------------------------------------------------------- #
#: Longest side (px) the response heatmap is downsampled to before it goes on the
#: wire. Producer-side quantisation (Option B): we do NOT ship the full-res float
#: response, nor recompute it UI-side (the FrontendConfig params are VIO-side
#: only). Block-MAX downsampling (not mean) preserves the corner peaks.
_VIZ_HEATMAP_MAX_SIDE = 240


@dataclass
class FrontendVizSnap:
    """One frame's frontend-internals snapshot for the "Frontend Internals" view.

    A picklable PLAIN-NUMPY side-car the :class:`CaptureKLTFrontend` stashes on
    itself each frame while returning a byte-identical :class:`TrackState`. It
    carries exactly what the UI renders -- nothing is re-derived UI-side:

    * the quantised Shi-Tomasi (lambda_min) response heatmap (``resp_q`` uint8,
      block-MAX downsampled to longest side <= :data:`_VIZ_HEATMAP_MAX_SIDE`),
      with ``resp_max`` (the pre-quantisation log1p peak) + the ORIGINAL response
      dimensions (``resp_h`` / ``resp_w``) so the UI can place corners on it;
    * the accepted ``corner_xy`` this frame + the detection geometry
      (``min_distance`` / ``quality_level`` / ``bucketed`` / grid) that explains
      WHY those pixels were kept;
    * the per-track KLT flow field: ids + prev/next pixel + forward-backward
      error + the culled (``~good``) mask + ``fb_threshold`` (the cull gate).

    ``sky.front`` is a LEAF (numpy/cv2/numba only): this class is pure-numpy and
    imports no process / comms / Qt code. The wire form lives in ``comms.wire``.
    """

    seq: int
    ts_ns: int
    # --- Shi-Tomasi response heatmap (quantised, producer-side) ----------- #
    resp_q: np.ndarray            # (Hq, Wq) uint8 -- log1p-scaled, block-MAX ds
    resp_max: float               # pre-quantisation log1p peak (colourbar scale)
    resp_h: int                   # original (full-res) response height
    resp_w: int                   # original (full-res) response width
    # --- accepted corners + detection geometry ---------------------------- #
    corner_xy: np.ndarray         # (C, 2) float32 accepted corner pixels (x, y)
    min_distance: float           # min spacing between corners (circle radius)
    quality_level: float          # response acceptance fraction
    bucketed: bool                # per-cell grid detection on?
    grid_rows: int                # detection grid rows (0 when not bucketed)
    grid_cols: int                # detection grid cols (0 when not bucketed)
    # --- KLT flow field --------------------------------------------------- #
    flow_id: np.ndarray           # (T,) int64 tracked-point ids (prev frame)
    flow_prev: np.ndarray         # (T, 2) float32 prev-frame pixel
    flow_next: np.ndarray         # (T, 2) float32 KLT next-frame pixel
    flow_fb_err: np.ndarray       # (T,) float32 forward-backward error (px)
    flow_culled: np.ndarray       # (T,) bool -- True where the track was dropped
    fb_threshold: float           # the cull gate (fb_err >= this -> culled)


def _quantise_response(resp: np.ndarray) -> tuple[np.ndarray, float]:
    """Log-scale + 8-bit-quantise + block-MAX downsample a response map.

    Returns ``(resp_q uint8 (Hq, Wq), resp_max float)``. ``resp`` is the raw
    Shi-Tomasi lambda_min map (large at corners). We:

    1. ``r = log1p(resp)`` -- compress the huge corner-vs-flat dynamic range so a
       linear 8-bit ramp still shows mid-strength corners (not just the peak).
    2. quantise ``q = clip(round(255 * r / r.max()), 0, 255)`` to uint8.
    3. block-MAX downsample so the longest side is <= :data:`_VIZ_HEATMAP_MAX_SIDE`
       -- MAX (not mean) so a 1-px corner peak survives the downsample instead of
       being averaged into the flat background (mean would smear the corners).

    ``resp_max`` is the pre-quantisation ``log1p`` peak, shipped so the UI's
    colourbar can label the scale honestly.
    """
    r = np.log1p(np.maximum(resp.astype(np.float64), 0.0))
    rmax = float(r.max())
    if rmax <= 0.0:
        # All-flat frame (e.g. a blank/cover): a zero map + zero peak. The UI
        # renders it as a uniform dark heatmap (honest "no corners here").
        h, w = resp.shape
        step = max(1, int(np.ceil(max(h, w) / _VIZ_HEATMAP_MAX_SIDE)))
        return (np.zeros(((h + step - 1) // step, (w + step - 1) // step),
                         np.uint8), 0.0)
    q = np.clip(np.round(255.0 * r / rmax), 0, 255).astype(np.uint8)
    h, w = q.shape
    step = max(1, int(np.ceil(max(h, w) / _VIZ_HEATMAP_MAX_SIDE)))
    if step == 1:
        return q, rmax
    # Block-MAX pool with a `step`-sized stride (pad with 0 so the last partial
    # block still pools; 0 is the heatmap floor so padding never invents a peak).
    hq = (h + step - 1) // step
    wq = (w + step - 1) // step
    ph, pw = hq * step - h, wq * step - w
    if ph or pw:
        q = np.pad(q, ((0, ph), (0, pw)), mode="constant", constant_values=0)
    pooled = q.reshape(hq, step, wq, step).max(axis=(1, 3)).astype(np.uint8)
    return pooled, rmax


class CaptureKLTFrontend(KLTFrontend):
    """A :class:`KLTFrontend` that STASHES per-frame visualisation intermediates.

    Used ONLY when ``FrontendConfig.capture`` is True (the opt-in --frontend-viz
    path). It returns a :class:`TrackState` byte-identical to :class:`KLTFrontend`
    -- the detection / tracking math is the base class's, untouched -- and, as a
    pure side effect, stashes a :class:`FrontendVizSnap` on ``self._frontend_viz_snap``
    that the publish step polls. THE correctness invariant is that the RETURNED
    tracks never change; everything captured here is either read-only (the
    response heatmap) or a by-product already computed for the tracking decision
    (the fb-error + culled mask from :meth:`_track_full`).
    """

    def __init__(self, cfg: FrontendConfig | None = None):
        super().__init__(cfg)
        #: Latest per-frame snapshot (or None before the first frame). The
        #: publish step reads + clears this; the frontend overwrites it each frame.
        self._frontend_viz_snap: FrontendVizSnap | None = None

    def _capture_detect_geometry(self, gray: np.ndarray) -> tuple:
        """Recompute the Shi-Tomasi response READ-ONLY for the heatmap.

        Runs the SAME :func:`sky.front.corners._shi_tomasi_response` that
        :func:`good_features_to_track` uses internally, with ``mask=None`` (the
        full-frame response, so the heatmap shows the response everywhere -- not
        only where fresh detection was allowed). This is a pure read: it never
        touches the tracked points the base ``process`` returns. Returns
        ``(resp_q, resp_max, resp_h, resp_w, grid_rows, grid_cols)``.
        """
        img = gray.astype(np.float32)
        resp = _shi_tomasi_response(img, self.cfg.block_size, None)
        resp_q, resp_max = _quantise_response(resp)
        grid_rows = 5 if self.cfg.bucketed else 0   # good_features_to_track default
        grid_cols = 6 if self.cfg.bucketed else 0
        return (resp_q, resp_max, int(resp.shape[0]), int(resp.shape[1]),
                grid_rows, grid_cols)

    def process(self, gray: np.ndarray) -> TrackState:
        """Advance the tracker by one frame, capturing the viz snapshot.

        Mirrors :meth:`KLTFrontend.process` EXACTLY for the returned-track math;
        the only additions are (a) computing the read-only response heatmap and
        (b) keeping the fb-error + culled mask from :meth:`_track_full`. The
        ``TrackState`` returned is byte-identical to the base class.
        """
        # Read-only heatmap (never feeds the tracks): compute once per frame.
        (resp_q, resp_max, resp_h, resp_w,
         grid_rows, grid_cols) = self._capture_detect_geometry(gray)

        # --- per-frame flow capture (filled below, empty on the first frame) -
        flow_id = np.empty((0,), np.int64)
        flow_prev = np.empty((0, 2), np.float32)
        flow_next = np.empty((0, 2), np.float32)
        flow_fb = np.empty((0,), np.float32)
        flow_culled = np.empty((0,), bool)

        if self._prev_gray is None:
            # First frame: detect only (identical to the base path).
            pts = self._detect(gray, np.empty((0, 2), np.float32))
            ids = np.arange(self._next_id, self._next_id + len(pts), dtype=np.int64)
            self._next_id += len(pts)
            self._state = TrackState(points=pts, ids=ids)
            self._prev_gray = gray
        else:
            prev_pts = self._state.points
            prev_ids = self._state.ids
            if prev_pts.shape[0] > 0:
                # _track_full gives the SAME (nxt, good) as the base _track plus
                # the fb_err it computed for the cull decision -- no re-derivation.
                nxt, good, fb_err = self._track_full(
                    self._prev_gray, gray, prev_pts)
                tracked_pts = nxt[good].astype(np.float32)
                tracked_ids = prev_ids[good]
                # Capture the flow field over ALL prev points (kept + culled).
                flow_id = prev_ids.astype(np.int64)
                flow_prev = prev_pts.astype(np.float32)
                flow_next = nxt.astype(np.float32)
                flow_fb = fb_err.astype(np.float32)
                flow_culled = ~good
            else:
                tracked_pts = np.empty((0, 2), np.float32)
                tracked_ids = np.empty((0,), np.int64)

            if tracked_pts.shape[0] < self.cfg.redetect_ratio * self.cfg.max_corners:
                fresh = self._detect(gray, tracked_pts)
                fresh_ids = np.arange(
                    self._next_id, self._next_id + len(fresh), dtype=np.int64)
                self._next_id += len(fresh)
                tracked_pts = (np.vstack([tracked_pts, fresh]) if len(fresh)
                               else tracked_pts)
                tracked_ids = np.concatenate([tracked_ids, fresh_ids])

            self._state = TrackState(points=tracked_pts, ids=tracked_ids)
            self._prev_gray = gray

        # The accepted corners shown on the heatmap = this frame's live tracks
        # (the SAME points the motion estimate consumes -- not a parallel detect).
        self._frontend_viz_snap = FrontendVizSnap(
            seq=0, ts_ns=0,
            resp_q=resp_q, resp_max=resp_max, resp_h=resp_h, resp_w=resp_w,
            corner_xy=np.asarray(self._state.points, np.float32).reshape(-1, 2),
            min_distance=float(self.cfg.min_distance),
            quality_level=float(self.cfg.quality_level),
            bucketed=bool(self.cfg.bucketed),
            grid_rows=grid_rows, grid_cols=grid_cols,
            flow_id=flow_id, flow_prev=flow_prev, flow_next=flow_next,
            flow_fb_err=flow_fb, flow_culled=flow_culled,
            fb_threshold=float(self.cfg.fb_threshold))
        return self._state

    def take_viz_snap(self) -> FrontendVizSnap | None:
        """Pop the latest snapshot (the publish step consumes it once per frame)."""
        snap, self._frontend_viz_snap = self._frontend_viz_snap, None
        return snap
