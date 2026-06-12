"""2D renderer for the "Frontend Internals" visualiser (two linked views).

Draws ONE :class:`~ui.comms.messages.FrameFrontend` snapshot as a single RGB
image (cv2 backend, numpy out -- no OpenGL, so it is light + PNG-verifiable,
mirrors :mod:`ui.viz.ba_render`). Two stacked panels explain HOW the frontend
finds + tracks features:

* **TOP -- Shi-Tomasi response heatmap.** The quantised lambda_min response
  (``resp_q``, log-scaled producer-side) coloured with the INFERNO colormap and
  upscaled to the original response size, with the accepted corners (``corner_xy``)
  marked + their ``min_distance`` spacing circles, the bucket grid (when
  ``bucketed``), and a colourbar labelled with the pre-quantisation ``resp_max``.
  Answers "why THIS pixel is a corner, not that bright edge."

* **BOTTOM -- KLT flow field.** One arrow per track from its prev pixel to its KLT
  next pixel, coloured green -> red by ``fb_err / fb_threshold`` (the cull gate),
  with the culled points (``flow_culled``) drawn as red X's. Answers "how tracking
  follows + culls bad / occluded points."

All data is REAL: the heatmap, corners, and flow are the VIO frontend's own
per-frame output, published on ``frame.frontend`` (see
``vio.modules.publish_frontend_viz``). Nothing is invented or re-derived UI-side.

``render_frontend(None)`` returns a "waiting" placeholder.

cv2 is only a drawing backend here -- importing this module is what pulls it, not
the base UI (mirrors :mod:`ui.viz.ba_render`).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Palette (RGB -- the canvas is blitted as Format_RGB888 by the window).
_BG = (13, 17, 23)             # theme.BG  #0d1117
_PANE_BG = (22, 27, 34)
_GRID = (42, 50, 61)           # theme.GRID #2a323d
_TEXT = (230, 237, 243)        # theme.TEXT
_TEXT_DIM = (139, 148, 158)    # theme.TEXT_DIM
_GOOD = (124, 255, 92)         # theme.GOOD -- small fb-error arrow / accepted corner
_WARN = (255, 176, 0)          # theme.WARN -- mid fb-error
_BAD = (255, 59, 48)           # theme.BAD  -- large fb-error / culled X
_CORNER = (124, 255, 92)       # accepted-corner marker (green)
_CIRCLE = (90, 110, 90)        # min_distance spacing circle (dim green)
_GRIDLINE = (74, 82, 54)       # bucket grid (olive, theme.PANEL_EDGE-ish)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _ramp_color(t: float) -> tuple:
    """green (t=0) -> amber -> red (t>=1) two-segment linear interpolation."""
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:                                # green -> amber
        f = t / 0.5
        a, b = _GOOD, _WARN
    else:                                      # amber -> red
        f = (t - 0.5) / 0.5
        a, b = _WARN, _BAD
    return tuple(int(round(a[i] + (b[i] - a[i]) * f)) for i in range(3))


def _heatmap_rgb(resp_q: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Colour the quantised response with INFERNO + upscale to ``(out_h, out_w)``.

    ``resp_q`` is the producer-side log-scaled, block-MAX-downsampled uint8 map.
    INFERNO reads "hot = strong corner" (black -> purple -> orange -> yellow).
    Nearest-neighbour upscale keeps the corner peaks crisp (a smooth upscale would
    blur them); cv2 colormaps are BGR so we flip to RGB for the canvas.
    """
    if resp_q.size == 0:
        return np.full((out_h, out_w, 3), _PANE_BG, np.uint8)
    cm_bgr = cv2.applyColorMap(resp_q, cv2.COLORMAP_INFERNO)
    cm_rgb = cv2.cvtColor(cm_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(cm_rgb, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def _draw_colorbar(canvas: np.ndarray, x0: int, y0: int, h: int,
                   resp_max: float) -> None:
    """A vertical INFERNO colourbar labelled 0 .. resp_max (the log1p peak)."""
    w = 14
    ramp = np.linspace(255, 0, h).astype(np.uint8).reshape(h, 1)
    bar_bgr = cv2.applyColorMap(np.repeat(ramp, w, axis=1), cv2.COLORMAP_INFERNO)
    bar = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2RGB)
    canvas[y0:y0 + h, x0:x0 + w] = bar
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), _TEXT_DIM, 1)
    cv2.putText(canvas, f"{resp_max:.1f}", (x0 - 2, y0 - 4), _FONT, 0.34,
                _TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, "0", (x0 + 2, y0 + h + 12), _FONT, 0.34,
                _TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, "lambda_min", (x0 - 18, y0 + h + 26), _FONT, 0.32,
                _TEXT_DIM, 1, cv2.LINE_AA)


def _fit_panel(src_h: int, src_w: int, avail_h: int, avail_w: int) -> float:
    """Scale that fits a ``src_h x src_w`` image into the available band."""
    if src_h <= 0 or src_w <= 0:
        return 1.0
    return min(avail_w / src_w, avail_h / src_h)


def _render_heatmap_panel(canvas, msg, x0, y0, panel_w, panel_h) -> None:
    """Top panel: response heatmap + corners + spacing circles + grid + colourbar."""
    resp_q = np.asarray(msg.resp_q, np.uint8)
    full_h, full_w = int(msg.resp_h), int(msg.resp_w)
    # Reserve a strip on the right for the colourbar.
    bar_pad = 56
    avail_w = panel_w - bar_pad
    avail_h = panel_h - 8
    scale = _fit_panel(full_h, full_w, avail_h, avail_w)
    draw_w = max(1, int(round(full_w * scale)))
    draw_h = max(1, int(round(full_h * scale)))

    heat = _heatmap_rgb(resp_q, draw_h, draw_w)
    ix, iy = x0, y0 + 4
    canvas[iy:iy + draw_h, ix:ix + draw_w] = heat

    # --- bucket grid (only when the detector bucketed) ------------------- #
    if bool(msg.bucketed) and int(msg.grid_rows) > 0 and int(msg.grid_cols) > 0:
        gr, gc = int(msg.grid_rows), int(msg.grid_cols)
        for r in range(1, gr):
            yy = iy + int(round(r * draw_h / gr))
            cv2.line(canvas, (ix, yy), (ix + draw_w, yy), _GRIDLINE, 1,
                     cv2.LINE_AA)
        for c in range(1, gc):
            xx = ix + int(round(c * draw_w / gc))
            cv2.line(canvas, (xx, iy), (xx, iy + draw_h), _GRIDLINE, 1,
                     cv2.LINE_AA)

    # --- accepted corners + min_distance spacing circles ----------------- #
    corners = np.asarray(msg.corner_xy, np.float32).reshape(-1, 2)
    md_px = float(msg.min_distance) * scale
    for cxy in corners:
        cx = ix + int(round(float(cxy[0]) * scale))
        cy = iy + int(round(float(cxy[1]) * scale))
        if not (ix <= cx < ix + draw_w and iy <= cy < iy + draw_h):
            continue
        if md_px >= 2.0:
            cv2.circle(canvas, (cx, cy), int(round(md_px)), _CIRCLE, 1,
                       cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 2, _CORNER, -1, cv2.LINE_AA)

    # --- colourbar ------------------------------------------------------- #
    _draw_colorbar(canvas, x0 + panel_w - bar_pad + 16, iy + 6,
                   max(40, draw_h - 28), float(msg.resp_max))

    # --- caption --------------------------------------------------------- #
    cap = (f"Shi-Tomasi response (INFERNO)  -  {len(corners)} accepted corners  -  "
           f"min_dist {float(msg.min_distance):.0f}px  -  q {float(msg.quality_level):.3f}"
           + ("  -  bucketed grid" if bool(msg.bucketed) else ""))
    cv2.putText(canvas, cap, (x0 + 2, y0 + panel_h - 2), _FONT, 0.40,
                _TEXT_DIM, 1, cv2.LINE_AA)


def _render_flow_panel(canvas, msg, x0, y0, panel_w, panel_h) -> None:
    """Bottom panel: KLT flow arrows coloured by fb-error + culled red X's."""
    full_h, full_w = int(msg.resp_h), int(msg.resp_w)
    avail_w = panel_w - 8
    avail_h = panel_h - 8
    scale = _fit_panel(full_h, full_w, avail_h, avail_w)
    draw_w = max(1, int(round(full_w * scale)))
    draw_h = max(1, int(round(full_h * scale)))
    ix, iy = x0, y0 + 4
    # Dark image plane so the flow arrows read.
    cv2.rectangle(canvas, (ix, iy), (ix + draw_w, iy + draw_h), _PANE_BG,
                  cv2.FILLED)
    cv2.rectangle(canvas, (ix, iy), (ix + draw_w, iy + draw_h), _GRID, 1)

    prev = np.asarray(msg.flow_prev, np.float32).reshape(-1, 2)
    nxt = np.asarray(msg.flow_next, np.float32).reshape(-1, 2)
    fb = np.asarray(msg.flow_fb_err, np.float32).reshape(-1)
    culled = np.asarray(msg.flow_culled, bool).reshape(-1)
    thr = max(float(msg.fb_threshold), 1e-6)

    n_kept = 0
    n_cull = 0
    for i in range(len(prev)):
        p0 = (ix + int(round(float(prev[i, 0]) * scale)),
              iy + int(round(float(prev[i, 1]) * scale)))
        if bool(culled[i]):
            n_cull += 1
            # Culled track: a red X at the prev location (occluded / slipped).
            d = 3
            cv2.line(canvas, (p0[0] - d, p0[1] - d), (p0[0] + d, p0[1] + d),
                     _BAD, 1, cv2.LINE_AA)
            cv2.line(canvas, (p0[0] - d, p0[1] + d), (p0[0] + d, p0[1] - d),
                     _BAD, 1, cv2.LINE_AA)
            continue
        n_kept += 1
        p1 = (ix + int(round(float(nxt[i, 0]) * scale)),
              iy + int(round(float(nxt[i, 1]) * scale)))
        col = _ramp_color(float(fb[i]) / thr)
        cv2.arrowedLine(canvas, p0, p1, col, 1, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(canvas, p0, 1, col, -1, cv2.LINE_AA)

    cap = (f"KLT flow  -  {n_kept} tracked (arrow prev->next, green->red by "
           f"fb-error / {thr:.1f}px)  -  {n_cull} culled (red X)")
    cv2.putText(canvas, cap, (x0 + 2, y0 + panel_h - 2), _FONT, 0.40,
                _TEXT_DIM, 1, cv2.LINE_AA)


def render_frontend(msg: Any, width: int = 1100, height: int = 860) -> np.ndarray:
    """Render one ``frame.frontend`` snapshot (or a waiting screen) -> (H,W,3) u8.

    ``msg`` is a :class:`~ui.comms.messages.FrameFrontend` (or ``None`` -> a
    "waiting" screen). The output is a single RGB image with the heatmap panel on
    top and the flow-field panel below.
    """
    canvas = np.full((height, width, 3), _BG, np.uint8)

    # --- header band ----------------------------------------------------- #
    cv2.rectangle(canvas, (0, 0), (width - 1, 34), _PANE_BG, cv2.FILLED)
    cv2.putText(canvas, "FRONTEND INTERNALS", (12, 23), _FONT, 0.6, _TEXT, 1,
                cv2.LINE_AA)
    cv2.putText(canvas, "how the frontend finds + tracks features  "
                "(response heatmap  -  KLT flow)", (228, 23), _FONT, 0.42,
                _TEXT_DIM, 1, cv2.LINE_AA)

    if msg is None:
        m = "waiting for a frontend frame ..."
        tw = cv2.getTextSize(m, _FONT, 0.7, 1)[0][0]
        cv2.putText(canvas, m, ((width - tw) // 2, height // 2), _FONT, 0.7,
                    _TEXT_DIM, 1, cv2.LINE_AA)
        return canvas

    top = 40
    gap = 26
    panel_h = (height - top - gap - 8) // 2
    panel_w = width - 16
    _render_heatmap_panel(canvas, msg, 8, top, panel_w, panel_h)
    _render_flow_panel(canvas, msg, 8, top + panel_h + gap, panel_w, panel_h)
    return canvas
