#!/usr/bin/env python3
"""SGM cost-volume explorer -- a learning tool that opens up the dense matcher.

WHAT THIS TEACHES
-----------------
Our dense stereo depth is Hirschmueller Semi-Global Matching
(``sky.depth.stereo.sgm_disparity``). For every pixel it builds two cost-vs-disparity
curves and picks the disparity that minimises the second one:

    C(d)  -- the RAW per-pixel matching cost: the census-Hamming distance between
             the left census at p and the right census at p-d. This is the
             evidence BEFORE any smoothness prior.
    S(d)  -- the N-path SGM-AGGREGATED cost: C plus the running cost of agreeing
             with neighbouring pixels along 4 or 8 scan directions. This is the
             evidence AFTER the global smoothness prior.

The depth that the matcher reports is ``argmin_d S(d)`` (winner-take-all), refined
to sub-pixel by a parabola through the three samples around the minimum, and
REJECTED entirely when a second, non-adjacent disparity is nearly as cheap (the
uniqueness gate) -- that is the "ambiguous, don't trust it" test.

THE 'AHA' (why textureless surfaces give noisy depth)
-----------------------------------------------------
* A TEXTURED pixel (a corner, high local gradient) has a SHARP single valley in
  C(d): one disparity matches far better than all others. The match is obvious;
  depth is reliable.
* A TEXTURELESS pixel (a blank wall / ceiling, low gradient) has a FLAT or
  MULTI-VALLEY C(d): many disparities match almost equally well, because the
  patch looks the same as its neighbours along the row. The raw winner is
  basically noise -- this is exactly why a blank wall came out noisy in the room
  map. Only AFTER the SGM aggregation does S(d) develop a clearer single minimum
  (the surrounding textured pixels "vote" the flat region toward a consistent
  disparity) -- that is the whole point of the global smoothness prior, made
  visible.

WHY THIS TOOL IS OFFLINE / STANDALONE
-------------------------------------
The cost volumes only exist if you have the raw rectified LEFT and RIGHT images,
and the live UI publishes only left+depth (never the right frame). So this tool
loads ONE frame from a recorded gold session, re-rectifies the recorded right
frame via the pipeline's ``RightRectifier`` + calibration, and runs the SGM with
the OPT-IN volume-capture hook (``SGMStereoMatcher.dense_disparity_capture``).
The production depth path is untouched; this only *keeps* the intermediate
volumes the normal path discards.

USAGE
-----
Interactive (needs a display)::

    .venv/bin/python -m depth.tools.sgm_cost_explorer --session sessions/gold/corridor_60s

    Click any pixel on the LEFT image or the DEPTH heatmap -> its C(d) and S(d)
    curves are plotted on the right, with the WTA minimum, the sub-pixel offset,
    and the uniqueness second-best band marked. Two preset markers (a textured
    corner [T] and a flat region [F]) are placed automatically so the contrast is
    one click away.

Headless render (no display needed -- the verifiable teaching evidence)::

    .venv/bin/python -m depth.tools.sgm_cost_explorer --session sessions/gold/corridor_60s \\
        --render /tmp/sgm_cost.png

    Auto-picks a textured corner + a flat region and writes a 2x2 PNG of their
    C(d)/S(d) curves so the textured-vs-textureless contrast is captured without a
    GUI. Dependency-free plot (numpy -> cv2 image), no matplotlib.

Dependencies: numpy + cv2 + pyqtgraph only (pyqtgraph/PyQt6 needed only for the
interactive window; ``--render`` needs just numpy + cv2).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Run as a module (-m) or as a script: make the repo root importable either way.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402  (approved dep; only used as an array/PNG backend here)

from depth.io.reader import SessionReader                              # noqa: E402
from sky.depth.stereo import (                             # noqa: E402
    SGMConfig, SGMStereoMatcher,
)


# --------------------------------------------------------------------------- #
# Compute core -- shared by the headless render and the interactive window.
# --------------------------------------------------------------------------- #
@dataclass
class CostFrame:
    """One captured frame: the rectified pair, depth, and the C / S cost volumes.

    Everything the explorer needs is computed once here (the SGM is the expensive
    part); the per-pixel curve lookups below are then trivial array slices.
    """

    left_rect: np.ndarray   # (H, W) float32 rectified-left (what was matched)
    right_rect: np.ndarray  # (H, W) float32 rectified-right
    disp: np.ndarray        # (H, W) float64 sub-pixel disparity (NaN = rejected)
    depth_m: np.ndarray     # (H, W) float32 metric depth (0 = invalid)
    C: np.ndarray           # (H, W, ndisp) int32 raw census-Hamming cost
    S: np.ndarray           # (H, W, ndisp) int32 N-path aggregated cost
    cfg: SGMConfig          # the config used (carries dmin / ndisp / uniqueness)
    gradient: np.ndarray    # (H, W) float32 local gradient energy (texturedness)

    @property
    def disparities(self) -> np.ndarray:
        """The disparity axis ``d = dmin + k`` for the curve x-values."""
        k = np.arange(self.cfg.num_disparities)
        return self.cfg.min_disparity + k


@dataclass
class PixelCurve:
    """The C(d) / S(d) analysis for one clicked pixel (everything the plot draws).

    All of this mirrors what ``_wta_lr`` does internally: the winner is
    ``argmin S``, the sub-pixel offset is a parabola through the three samples
    around it, and the uniqueness gate compares the best non-adjacent second
    minimum to ``best * (1 + uniqueness)``.
    """

    v: int
    u: int
    d: np.ndarray            # (ndisp,) disparity axis
    C: np.ndarray            # (ndisp,) raw cost
    S: np.ndarray            # (ndisp,) aggregated cost
    wta_k: int               # argmin S (integer disparity index)
    wta_disp: float          # dmin + wta_k
    subpixel_disp: float     # parabola-refined disparity (the reported value)
    second_k: int            # best non-adjacent S index (uniqueness 2nd-best)
    uniqueness_band: float   # S threshold = S[wta_k] * (1 + uniqueness)
    accepted: bool           # would the matcher KEEP this pixel?
    reject_reason: str       # "" if accepted, else why it was dropped
    gradient: float          # local gradient energy (how textured the pixel is)


def compute_cost_frame(session: str | Path, index: int,
                       cfg: SGMConfig | None = None) -> CostFrame:
    """Load one gold frame, re-rectify the right, run SGM with volume capture.

    ``cfg`` defaults to the full offline preset forced to ``downscale=1`` (the
    accuracy reference, and the only resolution at which the per-pixel curves are
    meaningful -- a downscaled volume is a box-averaged, upsampled approximation).
    """
    sr = SessionReader(session)
    frame = sr.load_frame(index, load_right=True)
    if frame.gray_right is None:
        raise RuntimeError(
            f"frame {index} of {session} has no recorded right image "
            "(the explorer needs left+right to build the cost volume)")

    if cfg is None:
        # Full default preset, downscale forced to 1: the volumes only make sense
        # on the real computed grid. This is the offline/accuracy reference path.
        cfg = SGMConfig(downscale=1)
    else:
        cfg = SGMConfig(**{**cfg.__dict__, "downscale": 1})

    matcher = SGMStereoMatcher.from_calib(sr.calib, cfg=cfg)
    left_rect, right_rect, disp, C, S = matcher.dense_disparity_capture(
        frame.gray_left, frame.gray_right)

    # Disparity -> metric depth, exactly as dense_depth does (Z = fx*B/d, clamped).
    depth_m = np.zeros(disp.shape, dtype=np.float32)
    good = np.isfinite(disp) & (disp > 1e-6)
    z = matcher.fx * matcher.baseline_m / disp[good]
    z[(z < cfg.min_depth) | (z > cfg.max_depth)] = 0.0
    depth_m[good] = z

    gradient = _gradient_energy(left_rect)
    return CostFrame(left_rect=left_rect.astype(np.float32),
                     right_rect=right_rect.astype(np.float32),
                     disp=disp, depth_m=depth_m, C=C, S=S, cfg=cfg,
                     gradient=gradient)


def _gradient_energy(img: np.ndarray) -> np.ndarray:
    """Local gradient energy = "how textured is this pixel's neighbourhood".

    A box-filtered Sobel gradient magnitude. High = a corner/edge (textured, a
    sharp C valley); near-zero = a blank wall (textureless, a flat C). Used to
    auto-pick the textured/textureless preset pixels.
    """
    g = img.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.boxFilter(mag, -1, (15, 15))


def analyse_pixel(cf: CostFrame, v: int, u: int) -> PixelCurve:
    """Extract + analyse the C(d)/S(d) curves at one pixel (the WTA/uniqueness math).

    This reproduces the decisions ``_wta_lr`` makes so the plot can show exactly
    why the pixel was kept or rejected -- nothing here feeds back into depth, it
    only explains the captured volumes.
    """
    cfg = cf.cfg
    d = cf.disparities
    C = cf.C[v, u].astype(np.float64)
    S = cf.S[v, u].astype(np.float64)
    ndisp = cfg.num_disparities

    # Winner-take-all on the AGGREGATED cost (what the matcher minimises).
    wta_k = int(np.argmin(S))
    wta_disp = float(cfg.min_disparity + wta_k)

    # Sub-pixel parabola through S[wta_k-1], S[wta_k], S[wta_k+1] (matches _wta_lr).
    subpixel_disp = wta_disp
    if cfg.subpixel and 0 < wta_k < ndisp - 1:
        cm, c0, cp = S[wta_k - 1], S[wta_k], S[wta_k + 1]
        den = cm - 2.0 * c0 + cp
        if den > 1e-9:
            subpixel_disp += 0.5 * (cm - cp) / den

    # Uniqueness: best NON-ADJACENT second minimum (the +/-1 neighbours of the
    # winner are excluded, they belong to the same valley).
    mask = np.ones(ndisp, dtype=bool)
    lo, hi = max(0, wta_k - 1), min(ndisp - 1, wta_k + 1)
    mask[lo:hi + 1] = False
    second_k = int(np.argmin(np.where(mask, S, np.inf))) if mask.any() else wta_k
    best = S[wta_k]
    uniqueness_band = best * (1.0 + cfg.uniqueness)

    # Acceptance (the rejection reasons the matcher would apply, in its order).
    accepted = True
    reason = ""
    if not np.isfinite(cf.disp[v, u]):
        accepted = False
        # Pinpoint which gate dropped it for the teaching label.
        if cfg.uniqueness > 0.0 and S[second_k] < uniqueness_band:
            reason = "uniqueness: a 2nd disparity is nearly as cheap (ambiguous)"
        else:
            reason = "rejected (L/R consistency or denoise)"

    return PixelCurve(
        v=v, u=u, d=d, C=C, S=S, wta_k=wta_k, wta_disp=wta_disp,
        subpixel_disp=subpixel_disp, second_k=second_k,
        uniqueness_band=uniqueness_band, accepted=accepted,
        reject_reason=reason, gradient=float(cf.gradient[v, u]))


def pick_preset_pixels(cf: CostFrame, border: int = 40) -> tuple[tuple[int, int],
                                                                 tuple[int, int]]:
    """Auto-pick a (textured, textureless) pixel pair for the preset markers.

    * textured   = the highest-gradient pixel that still got a valid disparity (a
      corner -- the sharp-C case),
    * textureless = the lowest-gradient pixel that still got a valid disparity (a
      flat wall/ceiling -- the flat/multi-valley-C case).

    Both are restricted to a valid, interior region so the curves are honest (not
    a border artefact). They are returned as ``(v, u)`` pixel coordinates.
    """
    H, W = cf.left_rect.shape
    valid = np.isfinite(cf.disp)
    interior = np.zeros_like(valid)
    # Keep the left margin clear of the full disparity range too: a pixel at small
    # u has its far-disparity right columns (u - d) out of range, so C saturates
    # to max_cost there -- a BORDER artefact, not a genuine textureless curve. We
    # want the textureless pixel's whole curve to be real, so require the deepest
    # disparity to land in-bounds (u >= dmin + ndisp + border).
    u_lo = border + cf.cfg.min_disparity + cf.cfg.num_disparities
    interior[border:H - border, u_lo:W - border] = True
    ok = valid & interior
    if not ok.any():
        # Degenerate frame: fall back to the image centre for both.
        c = (H // 2, W // 2)
        return c, c

    energy = cf.gradient
    tex = np.unravel_index(int(np.argmax(np.where(ok, energy, -np.inf))), ok.shape)
    flat = np.unravel_index(int(np.argmin(np.where(ok, energy, np.inf))), ok.shape)
    return (int(tex[0]), int(tex[1])), (int(flat[0]), int(flat[1]))


# --------------------------------------------------------------------------- #
# Headless render -- numpy -> cv2 PNG of the textured-vs-textureless curves.
# No matplotlib / pyqtgraph needed; this is the verifiable teaching evidence.
# --------------------------------------------------------------------------- #
# Theme (matches ui/qt/theme.py so the tool looks part of the suite). BGR for cv2.
_BG = (23, 27, 13)         # #0d1117 dark
_PANEL = (34, 27, 22)      # #161b22
_GRID = (61, 50, 42)       # #2a323d
_TEXT = (243, 237, 230)    # #e6edf3
_C_COLOR = (255, 225, 92)  # raw C(d): HUD cyan #5ce1ff in BGR (the "before" curve)
_S_COLOR = (92, 255, 124)  # aggregated S(d): NVG green #7cff5c in BGR ("after")
_WTA = (48, 59, 255)       # winner marker: master-warning red
_BAND = (0, 176, 255)      # uniqueness band: caution amber


def _plot_curve(panel: np.ndarray, x: np.ndarray, y: np.ndarray,
                color: tuple[int, int, int], y_lo: float, y_hi: float,
                pad: int) -> None:
    """Draw a polyline of ``y`` over ``x`` into ``panel`` (a single cv2 plot axis)."""
    h, w = panel.shape[:2]
    pw, ph = w - 2 * pad, h - 2 * pad
    xs = pad + (x - x[0]) / max(1e-9, (x[-1] - x[0])) * pw
    span = max(1e-9, y_hi - y_lo)
    ys = pad + ph - (y - y_lo) / span * ph     # invert: low cost near the bottom
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(panel, [pts], False, color, 2, cv2.LINE_AA)


def _render_axis(curve: PixelCurve, which: str, title: str,
                 size: tuple[int, int] = (460, 360)) -> np.ndarray:
    """Render one curve panel (``which`` = 'C' or 'S') to a BGR image.

    Draws the curve, the WTA minimum (red vertical), the sub-pixel disparity
    (dashed), and -- on the S panel -- the uniqueness second-best band (amber).
    """
    h, w = size
    panel = np.full((h, w, 3), _PANEL, dtype=np.uint8)
    pad = 44
    x = curve.d.astype(np.float64)
    y = (curve.C if which == "C" else curve.S).astype(np.float64)
    color = _C_COLOR if which == "C" else _S_COLOR

    y_lo, y_hi = float(y.min()), float(y.max())
    if y_hi - y_lo < 1.0:
        y_hi = y_lo + 1.0

    # Grid + frame.
    pw, ph = w - 2 * pad, h - 2 * pad
    for gx in np.linspace(pad, pad + pw, 5):
        cv2.line(panel, (int(gx), pad), (int(gx), pad + ph), _GRID, 1)
    for gy in np.linspace(pad, pad + ph, 5):
        cv2.line(panel, (pad, int(gy)), (pad + pw, int(gy)), _GRID, 1)
    cv2.rectangle(panel, (pad, pad), (pad + pw, pad + ph), _GRID, 1)

    # The uniqueness band (S panel only): everything BELOW this line is "nearly as
    # cheap as the winner" -- a 2nd minimum dipping under it means an ambiguous
    # match the matcher rejects.
    if which == "S" and curve.S[curve.wta_k] > 0:
        band_y = pad + ph - (curve.uniqueness_band - y_lo) / (y_hi - y_lo) * ph
        band_y = int(np.clip(band_y, pad, pad + ph))
        cv2.line(panel, (pad, band_y), (pad + pw, band_y), _BAND, 1, cv2.LINE_AA)
        cv2.putText(panel, "uniqueness band", (pad + 6, band_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _BAND, 1, cv2.LINE_AA)

    _plot_curve(panel, x, y, color, y_lo, y_hi, pad)

    # WTA minimum (the integer winner) -- a red vertical line.
    wx = pad + (curve.wta_disp - x[0]) / max(1e-9, (x[-1] - x[0])) * pw
    cv2.line(panel, (int(wx), pad), (int(wx), pad + ph), _WTA, 1, cv2.LINE_AA)
    cv2.circle(panel, (int(wx), int(pad + ph - (y[curve.wta_k] - y_lo)
                                    / (y_hi - y_lo) * ph)), 4, _WTA, -1)

    # Second-best (uniqueness 2nd minimum) -- small amber dot on this panel.
    s2x = pad + (curve.d[curve.second_k] - x[0]) / max(1e-9, (x[-1] - x[0])) * pw
    s2y = pad + ph - (y[curve.second_k] - y_lo) / (y_hi - y_lo) * ph
    cv2.circle(panel, (int(s2x), int(s2y)), 4, _BAND, 1, cv2.LINE_AA)

    # Labels. The curve-name (coloured) sits on the top line; the longer
    # pixel/verdict subtitle goes on a second line so nothing overlaps.
    yl = "C(d) raw census-Hamming" if which == "C" else "S(d) SGM-aggregated"
    cv2.putText(panel, yl, (pad, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA)
    cv2.putText(panel, title, (pad, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                _TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel, "disparity d (px)", (w // 2 - 60, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, _TEXT, 1, cv2.LINE_AA)
    sub = f"WTA d={curve.wta_disp:.0f}  subpix={curve.subpixel_disp:.2f}px"
    cv2.putText(panel, sub, (pad, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                _WTA, 1, cv2.LINE_AA)
    return panel


def render_contrast_png(cf: CostFrame, out_path: str | Path,
                        tex: tuple[int, int], flat: tuple[int, int]) -> str:
    """Write a 2x2 PNG: textured row (C | S) over textureless row (C | S).

    The single image that makes the teaching point: top row = a TEXTURED corner
    (sharp single C valley); bottom row = a TEXTURELESS flat region (flat /
    multi-valley C that S then sharpens). Returns the absolute path written.
    """
    tex_curve = analyse_pixel(cf, tex[0], tex[1])
    flat_curve = analyse_pixel(cf, flat[0], flat[1])

    def _row(curve: PixelCurve, label: str) -> np.ndarray:
        c = _render_axis(curve, "C",
                         f"{label}  px=({curve.u},{curve.v})  grad={curve.gradient:.0f}")
        s = _render_axis(curve, "S",
                         f"{label}  {'KEPT' if curve.accepted else 'REJECTED'}"
                         + (f": {curve.reject_reason}" if curve.reject_reason else ""))
        return np.hstack([c, s])

    top = _row(tex_curve, "TEXTURED (corner)")
    bot = _row(flat_curve, "TEXTURELESS (flat wall)")
    grid = np.vstack([top, bot])

    # A header banner explaining the read.
    banner_h = 56
    banner = np.full((banner_h, grid.shape[1], 3), _BG, dtype=np.uint8)
    cv2.putText(banner, "SGM cost-volume explorer: why textureless surfaces give noisy depth",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _TEXT, 1, cv2.LINE_AA)
    cv2.putText(banner, "C(d)=raw census-Hamming (cyan)   S(d)=SGM-aggregated (green)   "
                        "red=WTA min   amber=uniqueness 2nd-best",
                (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _TEXT, 1, cv2.LINE_AA)
    out = np.vstack([banner, grid])

    out_path = str(Path(out_path).resolve())
    cv2.imwrite(out_path, out)
    return out_path


# --------------------------------------------------------------------------- #
# Interactive window -- pyqtgraph: image + depth, click -> plot C(d)/S(d).
# Imported lazily so --render needs no Qt / display.
# --------------------------------------------------------------------------- #
def run_interactive(cf: CostFrame, session: str) -> int:
    """Open the click-to-inspect window (returns the Qt exit code)."""
    import pyqtgraph as pg                                  # noqa: PLC0415 (lazy)
    from pyqtgraph.Qt import QtWidgets, QtCore              # noqa: PLC0415

    from ui.viz.depth_render import colorize_depth          # noqa: PLC0415

    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")

    tex, flat = pick_preset_pixels(cf)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QWidget()
    win.setWindowTitle(f"SGM cost-volume explorer  --  {session}")
    win.resize(1280, 720)
    win.setStyleSheet("background-color:#0d1117; color:#e6edf3;")
    root = QtWidgets.QHBoxLayout(win)

    # ---- Left column: rectified-left + depth heatmap, both clickable -------- #
    left_col = QtWidgets.QVBoxLayout()
    glw = pg.GraphicsLayoutWidget()
    glw.setBackground("#0d1117")
    left_col.addWidget(glw, 1)

    # The rectified LEFT image (the grid the matcher worked on).
    vb_img = glw.addViewBox(row=0, col=0, lockAspect=True, enableMouse=False)
    img_item = pg.ImageItem(cf.left_rect)
    img_item.setLookupTable(_gray_lut())
    vb_img.addItem(img_item)
    vb_img.invertY(True)
    vb_img.autoRange(padding=0)
    glw.addLabel("rectified LEFT  (click a pixel)", row=1, col=0, color="#c9b97f")

    # The depth heatmap (our SGM output for this frame).
    vb_dep = glw.addViewBox(row=2, col=0, lockAspect=True, enableMouse=False)
    depth_rgb = cv2.cvtColor(colorize_depth(cf.depth_m), cv2.COLOR_BGR2RGB)
    dep_item = pg.ImageItem(depth_rgb)
    vb_dep.addItem(dep_item)
    vb_dep.invertY(True)
    vb_dep.autoRange(padding=0)
    glw.addLabel("DEPTH heatmap (near=bright, black=invalid)", row=3, col=0,
                 color="#c9b97f")

    # Marker that follows the clicked pixel on both panels.
    mark_img = pg.ScatterPlotItem(size=12, pen=pg.mkPen("#ff3b30", width=2),
                                  brush=None)
    mark_dep = pg.ScatterPlotItem(size=12, pen=pg.mkPen("#ff3b30", width=2),
                                  brush=None)
    vb_img.addItem(mark_img)
    vb_dep.addItem(mark_dep)
    # Preset markers (textured [T] / textureless [F]) so the contrast is one click.
    for (pv, pu), txt, col in [(tex, "T", "#7cff5c"), (flat, "F", "#ffb000")]:
        for vb in (vb_img, vb_dep):
            vb.addItem(pg.ScatterPlotItem([pu], [pv], size=14,
                                          pen=pg.mkPen(col, width=2), brush=None))
            t = pg.TextItem(txt, color=col)
            t.setPos(pu + 4, pv)
            vb.addItem(t)
    root.addLayout(left_col, 3)

    # ---- Right column: the C(d) / S(d) curve plots ------------------------- #
    plot_w = pg.GraphicsLayoutWidget()
    plot_w.setBackground("#0d1117")
    p_c = plot_w.addPlot(row=0, col=0, title="C(d) -- RAW census-Hamming cost")
    p_s = plot_w.addPlot(row=1, col=0, title="S(d) -- SGM-AGGREGATED cost")
    for p in (p_c, p_s):
        p.showGrid(x=True, y=True, alpha=0.25)
        p.setLabel("bottom", "disparity d (px)")
        p.setLabel("left", "cost")
    info = plot_w.addLabel("click the LEFT image or DEPTH map to inspect a pixel",
                           row=2, col=0, color="#8b949e")
    root.addWidget(plot_w, 4)

    def _redraw(v: int, u: int) -> None:
        curve = analyse_pixel(cf, v, u)
        mark_img.setData([u], [v])
        mark_dep.setData([u], [v])
        p_c.clear()
        p_s.clear()
        p_c.plot(curve.d, curve.C, pen=pg.mkPen("#5ce1ff", width=2))
        p_s.plot(curve.d, curve.S, pen=pg.mkPen("#7cff5c", width=2))
        # WTA minimum (red) + sub-pixel (dashed white) on the S plot.
        p_s.addItem(pg.InfiniteLine(curve.wta_disp, angle=90,
                                    pen=pg.mkPen("#ff3b30", width=1)))
        p_s.addItem(pg.InfiniteLine(curve.subpixel_disp, angle=90,
                                    pen=pg.mkPen("#e6edf3", width=1,
                                                 style=QtCore.Qt.PenStyle.DashLine)))
        p_c.addItem(pg.InfiniteLine(curve.wta_disp, angle=90,
                                    pen=pg.mkPen("#ff3b30", width=1)))
        # Uniqueness band (amber horizontal) + the 2nd-best dot on S.
        p_s.addItem(pg.InfiniteLine(curve.uniqueness_band, angle=0,
                                    pen=pg.mkPen("#ffb000", width=1,
                                                 style=QtCore.Qt.PenStyle.DashLine)))
        p_s.addItem(pg.ScatterPlotItem([curve.d[curve.second_k]],
                                       [curve.S[curve.second_k]], size=10,
                                       pen=pg.mkPen("#ffb000", width=2), brush=None))
        verdict = "KEPT" if curve.accepted else f"REJECTED ({curve.reject_reason})"
        info.setText(
            f"px=({u},{v})  grad={curve.gradient:.0f}  "
            f"WTA d={curve.wta_disp:.0f}  subpix={curve.subpixel_disp:.2f}px  "
            f"depth={cf.depth_m[v, u]:.2f}m  ->  {verdict}",
            color=("#7cff5c" if curve.accepted else "#ff3b30"))

    def _on_click(vb, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pt = vb.mapSceneToView(ev.scenePos())
        u, v = int(round(pt.x())), int(round(pt.y()))
        H, W = cf.left_rect.shape
        if 0 <= v < H and 0 <= u < W:
            _redraw(v, u)
            ev.accept()

    vb_img.scene().sigMouseClicked.connect(lambda ev: _on_click(vb_img, ev))
    vb_dep.scene().sigMouseClicked.connect(lambda ev: _on_click(vb_dep, ev))

    _redraw(*tex)        # start on the textured preset so the sharp valley shows
    win.show()
    return app.exec()


def _gray_lut() -> np.ndarray:
    """A plain 0..255 grayscale lookup table for the rectified-left ImageItem."""
    ramp = np.arange(256, dtype=np.uint8)
    return np.stack([ramp, ramp, ramp], axis=1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="SGM cost-volume explorer (learning tool): click a pixel -> "
                    "plot its raw C(d) and aggregated S(d) matching-cost curves.")
    ap.add_argument("--session", default="sessions/gold/corridor_60s",
                    help="recorded gold session directory")
    ap.add_argument("--frame", type=int, default=40,
                    help="frame index within the session")
    ap.add_argument("--render", metavar="PNG", default=None,
                    help="headless: write the textured-vs-textureless 2x2 curve "
                         "PNG to this path and exit (no display / Qt needed)")
    args = ap.parse_args(argv)

    cf = compute_cost_frame(args.session, args.frame)
    valid_pct = 100.0 * np.isfinite(cf.disp).mean()
    print(f"loaded {args.session} frame {args.frame}: "
          f"{cf.left_rect.shape[1]}x{cf.left_rect.shape[0]}, "
          f"ndisp={cf.cfg.num_disparities}, valid disparity {valid_pct:.1f}%")

    if args.render:
        tex, flat = pick_preset_pixels(cf)
        tc = analyse_pixel(cf, *tex)
        fc = analyse_pixel(cf, *flat)
        out = render_contrast_png(cf, args.render, tex, flat)
        # Report the measured contrast so the PNG can be judged from the console too.
        print(f"textured    px=({tex[1]},{tex[0]}) grad={tc.gradient:.0f}  "
              f"C valleys(<=min+1)={_count_valleys(tc.C)}  "
              f"C argmin d={tc.wta_disp if tc.C.argmin()==tc.wta_k else cf.cfg.min_disparity + int(tc.C.argmin())}  "
              f"S argmin d={tc.wta_disp:.0f}")
        print(f"textureless px=({flat[1]},{flat[0]}) grad={fc.gradient:.0f}  "
              f"C valleys(<=min+1)={_count_valleys(fc.C)}  "
              f"C argmin d={cf.cfg.min_disparity + int(fc.C.argmin())}  "
              f"S argmin d={fc.wta_disp:.0f}")
        print(f"wrote {out}")
        return 0

    return run_interactive(cf, args.session)


def _count_valleys(curve: np.ndarray) -> int:
    """How many disparity bins are within 1 cost unit of the minimum.

    1 = a single sharp valley (textured); many = a flat / multi-valley curve
    (textureless, ambiguous) -- the quantitative version of the teaching point.
    """
    return int((curve <= curve.min() + 1).sum())


if __name__ == "__main__":
    raise SystemExit(main())
