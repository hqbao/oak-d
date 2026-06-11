"""Stereo checkerboard capture collector (Phase 3 -- the calibration math core).

A PURE, hardware-agnostic, unit-testable state machine that drives the on-screen
"point at the board, hold, move, repeat" wizard. Modelled directly on
:class:`sky.sensors.calib_collect.SixFaceCollector`: the UI feeds raw samples
(here, synced left+right grayscale frames) one at a time and polls a status
snapshot to render the wizard; the capture *logic* lives here so it is tested
offline, not buried in an untested Qt callback.

WHY DIVERSITY (and not "grab 15 frames")
----------------------------------------
A good intrinsic/extrinsic fit needs the board sampled across the image and across
poses; 20 near-identical frames add no information and bias the solve toward one
region. So a view is ACCEPTED only when the board is detected in BOTH cameras AND
the left view is sufficiently DIFFERENT from every already-accepted view. The
diversity metric (documented on :class:`StereoCheckerboardCollector`) covers three
axes an operator naturally varies:

  * board CENTROID position in the image (translate the board around the frame),
  * board apparent SIZE (move nearer / further), and
  * board SKEW / foreshortening (tilt the board so it is not fronto-parallel).

Each axis is normalised so a single scalar "novelty" distance gates acceptance, and
a per-axis "coverage" tally tells the operator which axis they still need to vary.

cv2 POLICY
----------
This module's dataclass logic holds NO cv2 state; the only OpenCV touch is the call
into :func:`ui.mathlib.calib.detect.detect_corners`, which lazy-imports cv2 itself.
Importing this module (or the package) therefore does not load OpenCV.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import detect_corners, reconcile_lr

# -- diversity tuning ------------------------------------------------------- #
# A new view must differ from EVERY accepted view by at least this combined
# (centroid + size + skew) novelty distance. The three sub-metrics below are each
# scaled to ~O(1) for a "clearly different" pose, so a threshold near 0.15 means
# "noticeably moved on at least one axis". Deliberately not huge: the operator
# should be able to fill the target with natural, varied poses, not acrobatics.
_MIN_NOVELTY = 0.15
# Centroid distance is measured in image fractions (0..~1 across the frame), so a
# 0.15 shift is ~15% of the image width/height -- a clear translation.
# Size is the board's bounding-box diagonal as a fraction of the image diagonal;
# its delta is taken relative to itself so moving 30% nearer/further reads as ~0.3.
# Skew is the fronto-parallel departure (see _view_descriptor); its delta is
# absolute since it already lives in [0, ~1).


@dataclass(frozen=True)
class _ViewDescriptor:
    """Compact, cv2-free fingerprint of one detected board pose (left camera).

    All fields are image-normalised so the novelty distance is resolution-agnostic.
    """

    cx: float        # board centroid x, as a fraction of image width  (0..1)
    cy: float        # board centroid y, as a fraction of image height (0..1)
    size: float      # board bounding-box diagonal / image diagonal     (0..~1)
    skew: float      # fronto-parallel departure: 0 = square-on, larger = tilted


def _view_descriptor(corners: np.ndarray, image_size: tuple[int, int]
                     ) -> _ViewDescriptor:
    """Reduce ``(N,2)`` corners to a normalised ``_ViewDescriptor``.

    ``image_size`` is ``(width, height)`` (cv2 convention). ``skew`` is the spread
    of the corner-to-centroid *distances* divided by their mean: a fronto-parallel
    board has near-uniform radii (low spread), while a tilted/foreshortened board
    has a wide spread, so this is a cheap, monotone proxy for board obliquity that
    needs no homography decomposition.
    """
    w, h = image_size
    diag = float(np.hypot(w, h))

    centroid = corners.mean(axis=0)                       # (2,) in pixels
    cx = float(centroid[0]) / w
    cy = float(centroid[1]) / h

    # Apparent size: bounding-box diagonal relative to the image diagonal.
    mins = corners.min(axis=0)
    maxs = corners.max(axis=0)
    bbox_diag = float(np.hypot(*(maxs - mins)))
    size = bbox_diag / diag

    # Skew: coefficient of variation of the corner radii about the centroid.
    radii = np.linalg.norm(corners - centroid, axis=1)
    mean_r = float(radii.mean())
    skew = float(radii.std() / mean_r) if mean_r > 1e-9 else 0.0

    return _ViewDescriptor(cx=cx, cy=cy, size=size, skew=skew)


def _novelty(a: _ViewDescriptor, b: _ViewDescriptor) -> float:
    """Combined pose-novelty distance between two view descriptors (>= 0).

    Sum of three normalised contributions:
      * centroid Euclidean shift in image fractions,
      * relative apparent-size change (delta / mean size), and
      * absolute skew change.
    Larger = the poses sample the calibration space more differently.
    """
    d_centroid = float(np.hypot(a.cx - b.cx, a.cy - b.cy))
    mean_size = 0.5 * (a.size + b.size)
    d_size = abs(a.size - b.size) / mean_size if mean_size > 1e-9 else 0.0
    d_skew = abs(a.skew - b.skew)
    return d_centroid + d_size + d_skew


@dataclass(frozen=True)
class AcceptedView:
    """One accepted stereo view: the L and R subpixel corners + its descriptor."""

    corners_left: np.ndarray   # (N,2) float32
    corners_right: np.ndarray  # (N,2) float32
    descriptor: _ViewDescriptor


@dataclass(frozen=True)
class FrameStatus:
    """Snapshot the UI polls each feed() to render the capture wizard.

    Mirrors :class:`sky.sensors.calib_collect.SixFaceStatus`: a small, plain,
    serialisable record describing what just happened and overall progress.
    """

    found_left: bool
    found_right: bool
    accepted: bool                  # was THIS frame accepted as a new view?
    accepted_count: int             # accepted views so far
    n_target: int
    reason: str                     # why-rejected / why-accepted, for the operator
    novelty: float                  # this frame's min novelty vs accepted views
    complete: bool                  # count reached AND tilt coverage sufficient
    coverage: "CoverageStatus"      # per-axis spread of the accepted set
    skew_ok: bool                   # accepted views span enough tilt (skew) range
    count_ok: bool                  # accepted_count >= n_target (count alone)


@dataclass(frozen=True)
class CoverageStatus:
    """How widely the accepted views sample each diversity axis (0..1-ish).

    Each value is the spread (range, or std for skew) of the accepted descriptors
    along one axis -- a quick "are your poses varied enough?" indicator the wizard
    can render as three progress bars. Empty until views are accepted.
    """

    centroid_x: float   # range of accepted centroid-x (image fractions)
    centroid_y: float   # range of accepted centroid-y (image fractions)
    size: float         # range of accepted apparent sizes
    skew: float         # std of accepted skews
    skew_range: float   # max-min of accepted skews (drives the tilt-coverage gate)


# -- coverage tuning -------------------------------------------------------- #
# Reaching ``n_target`` views is necessary but NOT sufficient: an operator can slide
# a FRONTO-PARALLEL board around the frame and satisfy the (centroid+size+skew) sum
# novelty on translation alone, leaving every view square-on -- which makes focal
# length weakly observable (no foreshortening to disambiguate fx/fy from depth). So
# completion ALSO requires genuine tilt coverage: the accepted views' skew (board
# obliquity proxy) must span at least this range. Empirically the skew CoV proxy
# reads ~0.39 for these boards; a fronto-parallel-only sweep spans ~5e-4, while a
# set with real ±10-18 deg tilts spans >=1.1e-2 (stable under 0.5 px noise). A
# 5e-3 threshold sits cleanly between the two with ~2x margin on both sides.
_MIN_SKEW_SPREAD = 5e-3


@dataclass
class CollectorConfig:
    """Capture targets + diversity thresholds (tweakable per board / camera)."""

    n_target: int = 15            # how many diverse views to collect
    min_novelty: float = _MIN_NOVELTY
    # Minimum required spread (range) of accepted-view skews: forces the operator to
    # include genuinely tilted boards, not just translated fronto-parallel ones.
    min_skew_spread: float = _MIN_SKEW_SPREAD


class StereoCheckerboardCollector:
    """Collect ``n_target`` DIVERSE stereo checkerboard views for the solve.

    Usage (mirrors the IMU collectors)::

        coll = StereoCheckerboardCollector(pattern_cols, pattern_rows, image_size)
        status = coll.feed(gray_left, gray_right)   # per synced frame
        ... render status ...
        if status.complete:
            result = solve_stereo(coll.views, ...)
    """

    def __init__(
        self,
        pattern_cols: int,
        pattern_rows: int,
        image_size: tuple[int, int],
        cfg: CollectorConfig | None = None,
    ) -> None:
        """``image_size`` is ``(width, height)`` (cv2 convention)."""
        self.pattern_cols = int(pattern_cols)
        self.pattern_rows = int(pattern_rows)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.cfg = cfg or CollectorConfig()
        self._accepted: list[AcceptedView] = []

    # -- introspection ----------------------------------------------------- #
    @property
    def accepted_count(self) -> int:
        return len(self._accepted)

    @property
    def count_ok(self) -> bool:
        """Have we accepted at least ``n_target`` diverse views? (count alone)."""
        return len(self._accepted) >= self.cfg.n_target

    @property
    def skew_ok(self) -> bool:
        """Do the accepted views span enough TILT (skew) range to fix focal length?

        Guards against an all-fronto-parallel dataset: translating a square-on board
        around the frame can satisfy the per-frame novelty gate yet leave fx/fy
        weakly observable. Requires the accepted skews' range to exceed
        ``cfg.min_skew_spread``.
        """
        return self._coverage().skew_range >= self.cfg.min_skew_spread

    @property
    def complete(self) -> bool:
        """Capture is done only when the count target AND tilt coverage are met."""
        return self.count_ok and self.skew_ok

    @property
    def views(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Accepted views as ``(corners_left, corners_right)`` pairs for the solve."""
        return [(v.corners_left, v.corners_right) for v in self._accepted]

    def reset(self) -> None:
        self._accepted.clear()

    # -- diversity --------------------------------------------------------- #
    def _min_novelty(self, desc: _ViewDescriptor) -> float:
        """Smallest novelty between ``desc`` and every accepted view.

        The first view is always maximally novel (no reference yet) -> +inf.
        """
        if not self._accepted:
            return float("inf")
        return min(_novelty(desc, v.descriptor) for v in self._accepted)

    def _coverage(self) -> CoverageStatus:
        """Per-axis spread of the accepted descriptors (operator coverage hint)."""
        if not self._accepted:
            return CoverageStatus(0.0, 0.0, 0.0, 0.0, 0.0)
        cx = np.array([v.descriptor.cx for v in self._accepted])
        cy = np.array([v.descriptor.cy for v in self._accepted])
        sz = np.array([v.descriptor.size for v in self._accepted])
        sk = np.array([v.descriptor.skew for v in self._accepted])
        return CoverageStatus(
            centroid_x=float(cx.max() - cx.min()),
            centroid_y=float(cy.max() - cy.min()),
            size=float(sz.max() - sz.min()),
            skew=float(sk.std()),
            skew_range=float(sk.max() - sk.min()),
        )

    # -- drive ------------------------------------------------------------- #
    def feed(self, gray_left: np.ndarray, gray_right: np.ndarray) -> FrameStatus:
        """Process one synced stereo frame; return a status snapshot.

        A view is ACCEPTED iff the board is detected in BOTH cameras and its left
        descriptor is at least ``cfg.min_novelty`` away from every accepted view
        (the anti-duplicate / diversity gate). Detection failure in either camera,
        or insufficient novelty, rejects the frame with an explanatory ``reason``.
        Capture reports ``complete`` (and stops accepting) only once ``n_target``
        views are in hand AND they span enough tilt (skew) coverage; until then,
        reaching the count with an all-fronto-parallel set keeps accepting so the
        operator can add the tilted views the solve needs.
        """
        if self.complete:
            return self._status(False, False, False, 0.0,
                                 "Target reached -- capture complete.")

        corners_left = detect_corners(
            gray_left, self.pattern_cols, self.pattern_rows)
        corners_right = detect_corners(
            gray_right, self.pattern_cols, self.pattern_rows)
        found_l = corners_left is not None
        found_r = corners_right is not None

        # Both cameras must see the full board: a corner set missing in either
        # camera cannot contribute a stereo correspondence to the solve.
        if not (found_l and found_r):
            if not found_l and not found_r:
                reason = "Board not found in EITHER camera."
            elif not found_l:
                reason = "Board not found in LEFT camera."
            else:
                reason = "Board not found in RIGHT camera."
            return self._status(found_l, found_r, False, 0.0, reason)

        # Diversity gate: reject views too similar to one already accepted
        # (anti-duplicate -- stops 20 near-identical frames from being dumped in).
        desc = _view_descriptor(corners_left, self.image_size)
        novelty = self._min_novelty(desc)
        if novelty < self.cfg.min_novelty:
            return self._status(
                found_l, found_r, False, novelty,
                f"Too similar to an accepted view (novelty {novelty:.3f} < "
                f"{self.cfg.min_novelty:.3f}). Move / tilt the board more.")

        # Accept. RECONCILE the L<->R corner ORDER before storing: detect_corners runs
        # independently per camera, so the right array may be 180-degree-reversed
        # relative to the left (the cols x rows corner-order ambiguity). Left unchanged
        # (the reference / object-point order), right flipped to match if needed -- so
        # every accepted view's (L,R) corners name the SAME board points when they reach
        # the solve. Without this a real OAK-D capture diverges to a ~1 m baseline.
        corners_left, corners_right = reconcile_lr(
            corners_left, corners_right, self.pattern_cols, self.pattern_rows)
        self._accepted.append(AcceptedView(
            corners_left=corners_left,
            corners_right=corners_right,
            descriptor=desc,
        ))
        if self.complete:
            reason = f"Captured view {self.accepted_count}/{self.cfg.n_target} -- complete."
        elif self.count_ok and not self.skew_ok:
            # Count target met but every view is too fronto-parallel: tell the
            # operator exactly which axis is still short so they tilt the board.
            reason = (
                f"Captured view {self.accepted_count}/{self.cfg.n_target}, but "
                f"boards are too flat-on (tilt span {self._coverage().skew_range:.3f} "
                f"< {self.cfg.min_skew_spread:.3f}). Tilt the board more.")
        else:
            reason = f"Captured view {self.accepted_count}/{self.cfg.n_target}."
        return self._status(found_l, found_r, True, novelty, reason)

    def _status(self, found_l: bool, found_r: bool, accepted: bool,
                novelty: float, reason: str) -> FrameStatus:
        return FrameStatus(
            found_left=found_l,
            found_right=found_r,
            accepted=accepted,
            accepted_count=self.accepted_count,
            n_target=self.cfg.n_target,
            reason=reason,
            novelty=novelty,
            complete=self.complete,
            coverage=self._coverage(),
            skew_ok=self.skew_ok,
            count_ok=self.count_ok,
        )


# Re-exported for callers/tests that want to inspect a frame's descriptor without
# reaching into the private helper (keeps the public surface explicit).
__all__ = [
    "AcceptedView",
    "CollectorConfig",
    "CoverageStatus",
    "FrameStatus",
    "StereoCheckerboardCollector",
]
