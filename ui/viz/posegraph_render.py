"""2D top-down renderer for the "Pose Graph" before/after visualiser (ALGORITHMS.md §4.3).

Draws the SLAM pose-graph "aha" as a flat top-down (world X -> right, world Z ->
up) RGB image (cv2 backend, numpy out -- no OpenGL, so it is light +
PNG-verifiable, mirrors :mod:`ui.viz.ba_render` / :mod:`ui.viz.loop_render`):

* the keyframe POSES as graph NODES (dots); ODOMETRY edges chain consecutive
  keyframes (thin lines) and the confirmed LOOP edge(s) connect the two revisited
  keyframes (a bright chord -- "keyframe 117 is back at keyframe 3");
* a BEFORE/AFTER toggle (``show_before``) swaps the node + trajectory positions:
  BEFORE = the raw/drifted VIO estimate (the loop does NOT close -- the two
  revisit keyframes sit far apart); AFTER = the pose-graph-optimised estimate (the
  loop CLOSES and the drift correction is redistributed SMOOTHLY along the whole
  path). The OTHER state is drawn faintly behind as a ghost so the shift reads;
* per-keyframe CORRECTION-DELTA arrows (AFTER view) from each node's before
  position to its after position -- so "the correction spreads along the whole
  trajectory" is literal: the arrows grow from ~0 near the gauge to large near the
  loop and taper smoothly, instead of one big jump dumped at the closure.

All data is REAL UI-side derived state (no new IPC topic): the BEFORE nodes/trail
are VIO's ``pose.odom`` (raw, drifted), the AFTER nodes are SLAM's ``slam.map``
(pose-graph-corrected ``kf_positions``), the per-node delta is their difference
(the same rubber-sheet ``ui.main`` already computes for the corrected-VIO line),
and the loop edges are SLAM's ``slam.loop`` ``(cur_seq, old_seq)``. Nothing is
invented. See :class:`~ui.modules.ipc_sources.IpcPoseGraphSource`.

cv2 is only a drawing backend here -- importing this module is what pulls it, not
the base UI (mirrors :mod:`ui.viz.ba_render` / :mod:`ui.viz.loop_render`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# Palette (RGB -- the canvas is blitted as Format_RGB888 by the window).
_BG = (13, 17, 23)             # theme.BG  #0d1117
_PANE_BG = (22, 27, 34)
_TEXT = (230, 237, 243)        # theme.TEXT
_TEXT_DIM = (139, 148, 158)    # theme.TEXT_DIM
_GOOD = (124, 255, 92)         # theme.GOOD -- AFTER trajectory / "closed" loop
_WARN = (255, 176, 0)          # theme.WARN -- correction-delta arrows
_BAD = (255, 59, 48)           # theme.BAD  -- the open loop gap (BEFORE)
_RAW = (120, 180, 255)         # BEFORE (raw/drifted) trajectory (cool blue)
_NODE = (200, 210, 225)        # keyframe node dot (solid state)
_ODOM = (90, 105, 130)         # odometry edges (the kf chain)
_LOOP = (255, 92, 200)         # the loop-closure edge chord (magenta)
_GHOST = (70, 80, 95)          # faint other-state ghost (the before/after shadow)
_ANCHOR = (200, 160, 90)       # the gauge / anchor node ring

_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class PoseGraphSnapshot:
    """One pose-graph before/after snapshot (UI-side derived, NOT an IPC message).

    All positions are world X-Z in the camera-optical frame (the same frame
    ``slam.map`` / ``pose.odom`` publish), 2-D top-down. The arrays are aligned:
    ``kf_seqs[i]`` is the keyframe whose pre-PGO node is ``kf_before_xz[i]`` and
    whose post-PGO node is ``kf_after_xz[i]``.

    * ``kf_seqs`` -- ``(N,)`` int64 keyframe source seqs, ascending (graph order).
    * ``kf_before_xz`` / ``kf_after_xz`` -- ``(N, 2)`` float64 node positions
      BEFORE (raw drifted VIO) / AFTER (pose-graph optimised). The per-node
      correction delta is ``after - before``.
    * ``before_traj_xz`` / ``after_traj_xz`` -- ``(M, 2)`` float64 dense
      trajectory polylines (raw VIO / rubber-sheet-corrected VIO) for context.
    * ``loop_edges`` -- list of ``(cur_idx, old_idx)`` index pairs INTO ``kf_seqs``
      for each CONFIRMED loop edge (the chord drawn cur<->old).
    * ``n_loops`` -- confirmed loop count (header readout).
    """

    kf_seqs: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    kf_before_xz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), np.float64))
    kf_after_xz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), np.float64))
    before_traj_xz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), np.float64))
    after_traj_xz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), np.float64))
    loop_edges: tuple = ()
    n_loops: int = 0

    @property
    def n_kf(self) -> int:
        return int(len(self.kf_seqs))


def _fit_transform(pts_xz: np.ndarray, width: int, height: int,
                   top: int, bot: int, margin: int = 44):
    """Return ``(scale, ox, oy)`` mapping world (x, z) -> canvas (px, py).

    Fits all points into the drawing band with equal X/Z scale (no aspect
    distortion) so the top-down geometry is faithful. World +Z maps UP (canvas y
    decreases upward). Returns a centred default when there are no points (the
    "waiting" / empty case). Mirrors :func:`ui.viz.ba_render._fit_transform`.
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


def _polyline(canvas, pts_xz, xf, color, thickness) -> None:
    """Draw a world-X-Z polyline (>=2 points) as an anti-aliased line strip."""
    if len(pts_xz) < 2:
        return
    px = np.array([_w2c(p[0], p[1], xf) for p in pts_xz], np.int32)
    cv2.polylines(canvas, [px], False, color, thickness, cv2.LINE_AA)


def _arrow(canvas, p0, p1, color, thickness=1) -> None:
    """A short anti-aliased arrow p0->p1; head scaled to the shaft length."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = float(np.hypot(dx, dy))
    if length < 1.5:                              # too short to read -> skip head
        return
    tip = 0.32 if length < 30 else 9.0 / length   # fraction-of-shaft head length
    cv2.arrowedLine(canvas, p0, p1, color, thickness, cv2.LINE_AA,
                    tipLength=float(min(tip, 0.5)))


def render_pose_graph(msg: Any, width: int = 1100, height: int = 620,
                      show_before: bool = False) -> np.ndarray:
    """Render one :class:`PoseGraphSnapshot` (or a waiting screen) to ``(H,W,3)`` u8.

    ``msg`` is a :class:`PoseGraphSnapshot` (or ``None`` -> a "waiting" screen).
    ``show_before`` toggles the before/after view: when True the SOLID geometry is
    the BEFORE (raw/drifted) state and the AFTER state is ghosted behind; when
    False (default) the SOLID geometry is the AFTER (pose-graph-corrected) state
    with the BEFORE ghosted and the per-node correction arrows drawn.
    """
    canvas = np.full((height, width, 3), _BG, np.uint8)

    # --- header band ----------------------------------------------------- #
    cv2.rectangle(canvas, (0, 0), (width - 1, 34), _PANE_BG, cv2.FILLED)
    cv2.putText(canvas, "POSE GRAPH", (12, 23), _FONT, 0.62, _TEXT, 1, cv2.LINE_AA)
    mode = ("before (raw / drifted -- loop open)" if show_before
            else "after (pose-graph optimised -- loop closed)")
    cv2.putText(canvas,
                f"top-down X-Z  -  keyframe nodes + odometry/loop edges  -  {mode}",
                (152, 23), _FONT, 0.44, _TEXT_DIM, 1, cv2.LINE_AA)

    if msg is None or int(getattr(msg, "n_kf", 0)) < 2:
        m = ("waiting for a loop closure ..." if msg is not None
             else "waiting for the pose graph ...")
        tw = cv2.getTextSize(m, _FONT, 0.7, 1)[0][0]
        cv2.putText(canvas, m, ((width - tw) // 2, height // 2), _FONT, 0.7,
                    _TEXT_DIM, 1, cv2.LINE_AA)
        return canvas

    top, bot = 44, height - 56

    kf_before = np.asarray(msg.kf_before_xz, np.float64).reshape(-1, 2)
    kf_after = np.asarray(msg.kf_after_xz, np.float64).reshape(-1, 2)
    tr_before = np.asarray(msg.before_traj_xz, np.float64).reshape(-1, 2)
    tr_after = np.asarray(msg.after_traj_xz, np.float64).reshape(-1, 2)
    n_kf = len(kf_after)

    # SOLID = the toggled state; GHOST = the other state (drawn faintly first).
    if show_before:
        solid_kf, ghost_kf = kf_before, kf_after
        solid_tr, ghost_tr = tr_before, tr_after
        solid_col = _RAW
    else:
        solid_kf, ghost_kf = kf_after, kf_before
        solid_tr, ghost_tr = tr_after, tr_before
        solid_col = _GOOD

    # Fit from the UNION of both states so the toggle does not rescale the view
    # (the shift -- the whole point -- stays legible). Mirrors ba_render.
    all_xz = [a for a in (kf_before, kf_after, tr_before, tr_after) if len(a)]
    pts_xz = np.vstack(all_xz) if all_xz else np.zeros((0, 2))
    xf = _fit_transform(pts_xz, width, height, top, bot)

    # --- ghost (the other state's trajectory + nodes) -------------------- #
    _polyline(canvas, ghost_tr, xf, _GHOST, 1)
    for p in ghost_kf:
        cv2.circle(canvas, _w2c(p[0], p[1], xf), 3, _GHOST, 1, cv2.LINE_AA)

    # --- solid dense trajectory (context) -------------------------------- #
    _polyline(canvas, solid_tr, xf, solid_col, 1)

    # --- odometry edges (the keyframe chain, on the SOLID nodes) --------- #
    _polyline(canvas, solid_kf, xf, _ODOM, 2)

    # --- correction-delta arrows (AFTER view: before-node -> after-node) -- #
    # This is the headline: each arrow shows how far PGO moved that node. The
    # arrows taper smoothly from ~0 (gauge) to large (near the loop) -- the
    # "spread along the whole trajectory" made literal. Only meaningful in the
    # AFTER view (BEFORE -> AFTER); skipped in the BEFORE view (the ghost shows it).
    if not show_before and len(kf_before) == n_kf:
        for i in range(n_kf):
            p0 = _w2c(kf_before[i, 0], kf_before[i, 1], xf)
            p1 = _w2c(kf_after[i, 0], kf_after[i, 1], xf)
            _arrow(canvas, p0, p1, _WARN, 1)

    # --- loop-closure edge(s) -------------------------------------------- #
    # The chord that says "keyframe cur is back at keyframe old". In the AFTER
    # view its endpoints have been pulled together (loop closed -> short, green);
    # in the BEFORE view they are far apart (loop open -> long, red).
    edge_col = _BAD if show_before else _GOOD
    for cur_idx, old_idx in msg.loop_edges:
        if not (0 <= cur_idx < n_kf and 0 <= old_idx < n_kf):
            continue
        pc = _w2c(solid_kf[cur_idx, 0], solid_kf[cur_idx, 1], xf)
        po = _w2c(solid_kf[old_idx, 0], solid_kf[old_idx, 1], xf)
        cv2.line(canvas, pc, po, _LOOP, 2, cv2.LINE_AA)
        # Mark the two revisited nodes so the eye lands on the closure.
        for p in (pc, po):
            cv2.circle(canvas, p, 7, edge_col, 2, cv2.LINE_AA)

    # --- keyframe node dots (solid; anchor ringed) ----------------------- #
    for i, p in enumerate(solid_kf):
        cx, cy = _w2c(p[0], p[1], xf)
        cv2.circle(canvas, (cx, cy), 3, _NODE, -1, cv2.LINE_AA)
    # Anchor (gauge) node = the first keyframe (id 0, pinned by PGO); ring it.
    if n_kf:
        ax, ay = _w2c(solid_kf[0, 0], solid_kf[0, 1], xf)
        cv2.circle(canvas, (ax, ay), 9, _ANCHOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, "anchor", (ax + 10, ay - 8), _FONT, 0.38,
                    _ANCHOR, 1, cv2.LINE_AA)

    # --- legend ---------------------------------------------------------- #
    ly = bot + 34
    x = 12
    items = [
        ("raw/drifted (before)", _RAW),
        ("PGO (after)", _GOOD),
        ("loop edge", _LOOP),
    ]
    if not show_before:
        items.append(("correction", _WARN))
    for label, col in items:
        cv2.line(canvas, (x, ly - 4), (x + 20, ly - 4), col, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 24, ly), _FONT, 0.42, _TEXT_DIM, 1,
                    cv2.LINE_AA)
        x += 30 + cv2.getTextSize(label, _FONT, 0.42, 1)[0][0] + 14
    return canvas
