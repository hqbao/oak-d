"""2D top-down renderer for the "BA Window" visualiser.

Draws ONE windowed-BA solve snapshot as a flat top-down (world X -> right, world
Z -> up) RGB image (cv2 backend, numpy out -- no OpenGL, so it is light +
PNG-verifiable, mirrors :mod:`ui.viz.loop_render`):

* every in-window KEYFRAME pose as a small triangle, its heading taken from the
  camera-in-world quaternion (the optical +Z axis projected onto the X-Z plane).
  The NEWEST keyframe (the live frame) is highlighted; the OLDEST keyframe (the
  BA gauge anchor, held fixed) is marked with a ring;
* every shared 3D LANDMARK as a scatter dot;
* one observation RAY per (keyframe, landmark) observation, COLOUR-CODED by its
  reprojection error in pixels: green (sub-pixel, the BA minimised it) through
  amber to red (a large residual). This is "minimise reprojection error" made
  visible -- the rays that are still long/red are the ones BA could not satisfy;
* a before/after toggle: ``show_pre`` swaps the POST-solve poses/landmarks for
  the PRE-solve ones and additionally draws the POST state faintly behind, so the
  correction the solve applied is visible as the shift from ghost to solid.

All data is REAL: the poses + landmarks + per-observation reprojection error are
the VIO windowed-BA solve's own output, published on ``ba.window`` (see
``vio.modules.publish_ba_window``). Nothing is invented.

cv2 is only a drawing backend here -- importing this module is what pulls it, not
the base UI (mirrors :mod:`ui.viz.loop_render` / :mod:`ui.viz.gyrofuse_render`).
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
_GOOD = (124, 255, 92)         # theme.GOOD -- newest keyframe / sub-px ray
_WARN = (255, 176, 0)          # theme.WARN -- mid reprojection error
_BAD = (255, 59, 48)           # theme.BAD  -- large reprojection error
_KF = (120, 180, 255)          # keyframe triangle (cool blue)
_KF_OLD = (200, 160, 90)       # oldest keyframe ring (the gauge anchor)
_LM = (150, 200, 170)          # landmark scatter
_GHOST = (70, 80, 95)          # faint pre/post ghost (the "before/after" shadow)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Reprojection error (px) that maps to fully-red. Below ~0 px is green; the
#: gradient is clamped to this ceiling so a single huge residual does not wash
#: every other ray to the same red.
_REPROJ_RED_PX = 4.0


def _ray_color(reproj_px: float) -> tuple:
    """green (sub-px) -> amber -> red over ``[0, _REPROJ_RED_PX]`` px.

    A two-segment linear interpolation green->amber->red so a sub-pixel ray reads
    clearly green and a multi-pixel residual reads clearly red.
    """
    t = float(np.clip(reproj_px / _REPROJ_RED_PX, 0.0, 1.0))
    if t < 0.5:                                # green -> amber
        f = t / 0.5
        a, b = _GOOD, _WARN
    else:                                      # amber -> red
        f = (t - 0.5) / 0.5
        a, b = _WARN, _BAD
    return tuple(int(round(a[i] + (b[i] - a[i]) * f)) for i in range(3))


def _heading_xz(quat: np.ndarray) -> tuple:
    """Camera optical +Z axis projected onto the world X-Z plane, unit length.

    ``quat`` is ``(qw, qx, qy, qz)`` of the camera-in-world rotation. The optical
    axis in the camera frame is ``[0, 0, 1]``; rotating it by the quaternion and
    dropping the world-Y component gives the top-down heading the triangle points
    along. Falls back to ``(0, 1)`` (up) for a degenerate near-zero projection.
    """
    qw, qx, qy, qz = (float(quat[0]), float(quat[1]),
                      float(quat[2]), float(quat[3]))
    # Third column of R(quat) = R @ [0,0,1] (the optical axis in world frame).
    wx = 2.0 * (qx * qz + qw * qy)
    wz = 1.0 - 2.0 * (qx * qx + qy * qy)
    n = float(np.hypot(wx, wz))
    if n < 1e-9:
        return (0.0, 1.0)
    return (wx / n, wz / n)


def _fit_transform(pts_xz: np.ndarray, width: int, height: int,
                   top: int, bot: int, margin: int = 40):
    """Return ``(scale, ox, oy)`` mapping world (x, z) -> canvas (px, py).

    Fits all points into the drawing band with equal X/Z scale (no aspect
    distortion) so the top-down geometry is faithful. World +Z maps UP (canvas y
    decreases upward). Returns a centred identity-ish transform when there are no
    points (the "waiting" / empty case).
    """
    if len(pts_xz) == 0:
        return 60.0, width / 2.0, (top + bot) / 2.0
    xs, zs = pts_xz[:, 0], pts_xz[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    z0, z1 = float(zs.min()), float(zs.max())
    span_x = max(x1 - x0, 1e-3)
    span_z = max(z1 - z0, 1e-3)
    avail_w = width - 2 * margin
    avail_h = (bot - top) - 2 * margin
    scale = min(avail_w / span_x, avail_h / span_z)
    cx_world = 0.5 * (x0 + x1)
    cz_world = 0.5 * (z0 + z1)
    ox = width / 2.0 - cx_world * scale
    oy = (top + bot) / 2.0 + cz_world * scale     # +Z up => add (y grows down)
    return scale, ox, oy


def _w2c(x: float, z: float, xf) -> tuple:
    """World (x, z) -> integer canvas pixel using ``(scale, ox, oy)`` (Z up)."""
    scale, ox, oy = xf
    return (int(round(ox + x * scale)), int(round(oy - z * scale)))


def _draw_triangle(canvas, cx, cy, heading, color, size, thickness):
    """Draw a heading triangle at ``(cx, cy)`` pointing along ``heading=(hx,hz)``.

    Heading is in world X-Z; world +Z is canvas-up, so the canvas direction is
    ``(hx, -hz)``. The triangle's tip is the heading, the base is behind it.
    """
    hx, hz = heading
    dx, dy = hx, -hz                              # world Z-up -> canvas y-down
    px, py = -dy, dx                              # perpendicular (canvas)
    tip = (int(round(cx + dx * size)), int(round(cy + dy * size)))
    bl = (int(round(cx - dx * size * 0.6 + px * size * 0.6)),
          int(round(cy - dy * size * 0.6 + py * size * 0.6)))
    br = (int(round(cx - dx * size * 0.6 - px * size * 0.6)),
          int(round(cy - dy * size * 0.6 - py * size * 0.6)))
    pts = np.array([tip, bl, br], np.int32)
    if thickness < 0:
        cv2.fillConvexPoly(canvas, pts, color, cv2.LINE_AA)
    else:
        cv2.polylines(canvas, [pts], True, color, thickness, cv2.LINE_AA)


def render_ba_window(msg: Any, width: int = 1100, height: int = 620,
                     show_pre: bool = False) -> np.ndarray:
    """Render one ``ba.window`` snapshot (or a waiting screen) to ``(H, W, 3)`` u8.

    ``msg`` is a :class:`~ui.comms.messages.BaWindow` (or ``None`` -> a "waiting"
    screen). ``show_pre`` toggles the before/after view: when True the SOLID
    geometry is the PRE-solve state and the POST-solve state is drawn faintly
    behind (the correction is the shift); when False (default) the SOLID geometry
    is the POST-solve refined state.
    """
    canvas = np.full((height, width, 3), _BG, np.uint8)

    # --- header band ----------------------------------------------------- #
    cv2.rectangle(canvas, (0, 0), (width - 1, 34), _PANE_BG, cv2.FILLED)
    cv2.putText(canvas, "BA WINDOW", (12, 23), _FONT, 0.62, _TEXT, 1, cv2.LINE_AA)
    mode = "before (pre-solve, post ghosted)" if show_pre else "after (refined)"
    cv2.putText(canvas, f"top-down X-Z  -  window keyframes + landmarks  -  {mode}",
                (150, 23), _FONT, 0.44, _TEXT_DIM, 1, cv2.LINE_AA)

    if msg is None or int(getattr(msg, "n_kf", 0)) == 0:
        m = "waiting for a BA window ..."
        tw = cv2.getTextSize(m, _FONT, 0.7, 1)[0][0]
        cv2.putText(canvas, m, ((width - tw) // 2, height // 2), _FONT, 0.7,
                    _TEXT_DIM, 1, cv2.LINE_AA)
        return canvas

    top, bot = 44, height - 44

    kf_pos = np.asarray(msg.kf_pos, np.float64).reshape(-1, 3)
    kf_quat = np.asarray(msg.kf_quat, np.float64).reshape(-1, 4)
    lm_xyz = np.asarray(msg.lm_xyz, np.float64).reshape(-1, 3)
    kf_pos_pre = np.asarray(msg.kf_pos_pre, np.float64).reshape(-1, 3)
    kf_quat_pre = np.asarray(msg.kf_quat_pre, np.float64).reshape(-1, 4)
    lm_xyz_pre = np.asarray(msg.lm_xyz_pre, np.float64).reshape(-1, 3)

    # SOLID = the toggled state; GHOST = the other state (drawn faintly first).
    if show_pre:
        solid_kf, solid_q, solid_lm = kf_pos_pre, kf_quat_pre, lm_xyz_pre
        ghost_kf, ghost_lm = kf_pos, lm_xyz
    else:
        solid_kf, solid_q, solid_lm = kf_pos, kf_quat, lm_xyz
        ghost_kf, ghost_lm = kf_pos_pre, lm_xyz_pre

    # Fit transform from the UNION of both states' X-Z so the toggle does not
    # rescale the view (the shift stays legible).
    all_xz = []
    for arr in (solid_kf, ghost_kf, solid_lm, ghost_lm):
        if len(arr):
            all_xz.append(arr[:, [0, 2]])
    pts_xz = np.vstack(all_xz) if all_xz else np.zeros((0, 2))
    xf = _fit_transform(pts_xz, width, height, top, bot)

    # --- ghost landmarks + keyframes (the "before/after" shadow) --------- #
    for p in ghost_lm:
        cv2.circle(canvas, _w2c(p[0], p[2], xf), 1, _GHOST, -1, cv2.LINE_AA)
    for p in ghost_kf:
        cv2.circle(canvas, _w2c(p[0], p[2], xf), 3, _GHOST, 1, cv2.LINE_AA)

    # --- observation rays (KF -> landmark, coloured by reprojection error) - #
    obs_kf = np.asarray(msg.obs_kf, np.int32).reshape(-1)
    obs_lm = np.asarray(msg.obs_lm, np.int32).reshape(-1)
    obs_re = np.asarray(msg.obs_reproj_px, np.float32).reshape(-1)
    n_kf = len(solid_kf)
    n_lm = len(solid_lm)
    # Draw worst (red) rays last so they sit on top and stay legible.
    order = np.argsort(obs_re) if len(obs_re) else np.array([], int)
    for i in order:
        ci, lj = int(obs_kf[i]), int(obs_lm[i])
        if not (0 <= ci < n_kf and 0 <= lj < n_lm):
            continue
        p0 = _w2c(solid_kf[ci, 0], solid_kf[ci, 2], xf)
        p1 = _w2c(solid_lm[lj, 0], solid_lm[lj, 2], xf)
        cv2.line(canvas, p0, p1, _ray_color(float(obs_re[i])), 1, cv2.LINE_AA)

    # --- landmarks ------------------------------------------------------- #
    for p in solid_lm:
        cv2.circle(canvas, _w2c(p[0], p[2], xf), 2, _LM, -1, cv2.LINE_AA)

    # --- keyframe triangles (oldest = gauge ring, newest = highlighted) --- #
    for i, p in enumerate(solid_kf):
        cx, cy = _w2c(p[0], p[2], xf)
        heading = _heading_xz(solid_q[i])
        is_new = (i == n_kf - 1)
        is_old = (i == 0)
        color = _GOOD if is_new else _KF
        _draw_triangle(canvas, cx, cy, heading, color,
                       size=11 if is_new else 8, thickness=-1)
        if is_old:                               # gauge anchor ring
            cv2.circle(canvas, (cx, cy), 12, _KF_OLD, 1, cv2.LINE_AA)
            cv2.putText(canvas, "gauge", (cx + 10, cy - 10), _FONT, 0.38,
                        _KF_OLD, 1, cv2.LINE_AA)
        if is_new:
            cv2.putText(canvas, "newest", (cx + 12, cy + 4), _FONT, 0.4,
                        _GOOD, 1, cv2.LINE_AA)

    # --- legend ---------------------------------------------------------- #
    ly = bot + 26
    cv2.putText(canvas, "ray colour = reprojection error:", (12, ly), _FONT, 0.42,
                _TEXT_DIM, 1, cv2.LINE_AA)
    x = 12 + cv2.getTextSize("ray colour = reprojection error:", _FONT,
                             0.42, 1)[0][0] + 14
    for label, col in (("0 px", _GOOD), ("2 px", _WARN),
                       (f">={_REPROJ_RED_PX:.0f} px", _BAD)):
        cv2.line(canvas, (x, ly - 4), (x + 20, ly - 4), col, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 24, ly), _FONT, 0.42, _TEXT_DIM, 1,
                    cv2.LINE_AA)
        x += 30 + cv2.getTextSize(label, _FONT, 0.42, 1)[0][0]
    return canvas
