"""Pure renderer for the gyro-fusion strip chart (ALGORITHMS.md #5).

A scrolling, window-free strip chart that explains WHY the gyro-fused VIO stays
straight where pure-vision (``pose.vo``, grey) drifts during fast yaw. It draws
exactly what a :class:`~ui.comms.messages.FrameGyroFuse` carries -- a REAL
odometry output -- and nothing computed in a parallel pipeline (honest viz).

The mental model the chart makes obvious, no verbal explanation needed:

    We are estimating how much the camera turned this frame. Two sources:
      * VISION ("the eyes")      -- accurate when slow, but UNDER-rotates during
                                    fast yaw because KLT loses features.
      * GYRO ("the inner ear")   -- always accurate over one ~50 ms frame.
    Fusion trusts vision normally; when the two DISAGREE past the gate (fast
    yaw, the eyes lost it) it hands rotation over to the gyro. That is why the
    fused VIO stays straight while pure-vision VO drifts.

Layout (newest sample at the right):

* LIVE STATUS BANNER (top) -- driven by the LATEST frame: GREEN "TRUSTING
  VISION" while the eyes & gyro agree, AMBER "GYRO TAKING OVER" the instant the
  disagreement crosses the gate (with the current gyro weight quantified).
* TOP lane (deg / frame) -- two traces over time, "VISION (eyes)" (grey, drifts)
  vs "GYRO (inner ear)" (cyan, trusted), the area between them shaded as the
  per-frame disagreement, the gate line annotated "above: trust gyro", and the
  frame ranges where the gyro took over tinted amber so the eye instantly catches
  WHEN fusion mattered.
* BOTTOM lane (0..1) -- ``gain`` (how much VISION is trusted: 1 = all vision,
  0 = all gyro) and ``t_trust`` (translation trust), so you see the correction
  collapse toward the gyro exactly as the disagreement spikes.
* TAKEAWAY (footer) -- one plain-language line tying it together.

The renderer is window-free (returns an ``RGB`` ``uint8`` image) so the same
drawing code can feed any backend; the in-app Qt window
(:mod:`ui.qt.gyrofuse_window`) blits it to a label. cv2 is only a drawing backend
here -- importing this module is what pulls it, not the base UI.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# Colours are RGB (the Qt blit uses Format_RGB888), matched to the app theme:
#   vision (drift) = grey, gyro (truth) = HUD cyan, gain = NVG green,
#   t_trust = amber, gate lines = caution amber / master red,
#   banner = NVG green (agree) / caution amber (handover).
_BG = (13, 17, 23)            # theme.BG  #0d1117
_GRID = (42, 50, 61)          # theme.GRID #2a323d
_TEXT = (230, 237, 243)       # theme.TEXT
_TEXT_DIM = (139, 148, 158)   # theme.TEXT_DIM
_VISION = (150, 160, 170)     # grey -- pure-vision rotation (drifts)
_GYRO = (92, 225, 255)        # cyan -- gyro rotation (near ground-truth)
_DISAGREE = (255, 120, 90)    # shaded area between the two traces (warm)
_GAIN = (124, 255, 92)        # NVG green -- vision-trust gain
_TTRUST = (255, 176, 0)       # amber -- translation trust
_GATE = (255, 176, 0)         # amber -- gate ("gyro starts taking over")
_FULL_GYRO = (255, 59, 48)    # red  -- gate+span ("full gyro")
_GOOD = (124, 255, 92)        # theme.GOOD -- banner "trusting vision"
_WARN = (255, 176, 0)         # theme.WARN -- banner "gyro taking over"
_TAKEOVER_TINT = (255, 176, 0)  # amber wash over takeover frame-ranges

_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class GyroFuseSample:
    """One frame's gyro-fusion record (mirrors the wire fields the chart needs)."""

    vision_rot_deg: float
    gyro_rot_deg: float
    disagree_deg: float
    gain: float
    t_trust: float
    gate_deg: float
    span_deg: float


class GyroFuseChart:
    """Scrolling two-lane gyro-fusion strip chart fed per-frame samples."""

    #: never let the deg/frame auto-scale collapse below this (keeps a still
    #: camera's ~0 deg/frame from being blown up to full height).
    _DEG_FLOOR = 2.0
    #: expand-fast / shrink-slow hysteresis so the trace does not strobe.
    _SHRINK = 0.06
    #: vertical bands of the canvas (px from top): banner, lanes, takeaway.
    _BANNER_H = 40
    _TAKEAWAY_H = 22

    def __init__(self, width: int = 720, height: int = 460,
                 capacity: int = 600) -> None:
        self.width = int(width)
        self.height = int(height)
        self._hist: deque[GyroFuseSample] = deque(maxlen=int(capacity))
        self._deg_span = self._DEG_FLOOR

    def clear(self) -> None:
        self._hist.clear()
        self._deg_span = self._DEG_FLOOR

    def add(self, s: GyroFuseSample) -> None:
        self._hist.append(s)

    @property
    def sample_count(self) -> int:
        return len(self._hist)

    # -- rendering ---------------------------------------------------------- #
    def render(self) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), _BG, dtype=np.uint8)
        # Vertical budget: banner on top, then two lanes, then a takeaway footer.
        # Each lane keeps a small header band for its title + legend.
        lanes_y0 = self._BANNER_H + 26
        lanes_y1 = self.height - self._TAKEAWAY_H
        lane_h = lanes_y1 - lanes_y0
        top_y0 = lanes_y0
        top_y1 = lanes_y0 + int(lane_h * 0.60)
        bot_y0 = lanes_y0 + int(lane_h * 0.62) + 20
        bot_y1 = lanes_y1 - 4

        self._draw_banner(canvas)
        self._draw_deg_lane(canvas, top_y0, top_y1)
        self._draw_unit_lane(canvas, bot_y0, bot_y1)
        self._draw_takeaway(canvas)
        return canvas

    # -- live status banner (latest frame) --------------------------------- #
    def _draw_banner(self, canvas) -> None:
        """Prominent top banner: GREEN agree / AMBER handover, from the latest frame."""
        x0, y0, x1, y1 = 6, 4, self.width - 6, self._BANNER_H
        if not self._hist:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), _GRID, 1)
            cv2.putText(canvas, "WAITING FOR GYRO-FUSED FRAMES...", (16, y0 + 26),
                        _FONT, 0.56, _TEXT_DIM, 1, cv2.LINE_AA)
            return

        last = self._hist[-1]
        taking_over = last.disagree_deg >= last.gate_deg
        accent = _WARN if taking_over else _GOOD
        if taking_over:
            head = "GYRO TAKING OVER"
            tail = "vision under-rotated (fast yaw)"
        else:
            head = "TRUSTING VISION"
            tail = "eyes & gyro agree"
        gyro_weight = float(np.clip(1.0 - last.gain, 0.0, 1.0))

        # Filled chip (dim wash of the accent) with a bright left rule + border,
        # so the state reads at a glance without shouting over the traces.
        wash = tuple(int(_BG[i] + (accent[i] - _BG[i]) * 0.20) for i in range(3))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), wash, cv2.FILLED)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), accent, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x0 + 5, y1), accent, cv2.FILLED)

        cv2.putText(canvas, head, (20, y0 + 25), _FONT, 0.74, accent, 2,
                    cv2.LINE_AA)
        head_w = cv2.getTextSize(head, _FONT, 0.74, 2)[0][0]
        cv2.putText(canvas, f"  -  {tail}", (24 + head_w, y0 + 24), _FONT, 0.46,
                    _TEXT, 1, cv2.LINE_AA)

        # Right-justified quantified gyro weight (the handover, made a number).
        wtxt = f"gyro weight {gyro_weight:.0%}"
        wtw = cv2.getTextSize(wtxt, _FONT, 0.50, 1)[0][0]
        cv2.putText(canvas, wtxt, (x1 - wtw - 14, y0 + 25), _FONT, 0.50, accent,
                    1, cv2.LINE_AA)

    # -- one-line plain-language takeaway (footer) ------------------------- #
    def _draw_takeaway(self, canvas) -> None:
        y = self.height - 7
        cv2.putText(
            canvas,
            "The lines split when you turn fast -- there the gyro saves the "
            "heading (VIO stays straight; pure-vision VO drifts).",
            (10, y), _FONT, 0.40, _TEXT_DIM, 1, cv2.LINE_AA)

    # -- top lane: vision vs gyro rotation (deg/frame) ---------------------- #
    def _draw_deg_lane(self, canvas, y0: int, y1: int) -> None:
        h = max(y1 - y0, 1)
        n = len(self._hist)
        # Auto-scale the deg axis to cover the data + the full-gyro line, floored,
        # with expand-fast/shrink-slow hysteresis (stable, never strobes).
        peak = self._DEG_FLOOR
        gate = span = 0.0
        if n:
            arr = self._hist
            peak = max(
                max(s.vision_rot_deg for s in arr),
                max(s.gyro_rot_deg for s in arr),
                max((s.gate_deg + s.span_deg) for s in arr),
            )
            last = arr[-1]
            gate, span = last.gate_deg, last.span_deg
        target = max(peak * 1.15, self._DEG_FLOOR)
        if target >= self._deg_span:
            self._deg_span = target
        else:
            self._deg_span += (target - self._deg_span) * self._SHRINK
        span_deg = max(self._deg_span, 1e-6)

        def y_of(v: float) -> int:
            return int(np.clip(y1 - (v / span_deg) * h, y0, y1))

        x_left, x_right = 44, self.width - 9

        # Tint the frame-ranges where the gyro took over (disagree >= gate), so
        # the eye instantly catches WHEN fusion mattered. Drawn FIRST, under the
        # grid + traces, as a subtle amber wash.
        if n >= 2:
            self._shade_takeover_spans(canvas, y0, y1, x_left, x_right)

        # Gridlines at 25 / 50 / 75 % of the span.
        for frac in (0.25, 0.5, 0.75):
            yy = int(y1 - frac * h)
            cv2.line(canvas, (x_left, yy), (x_right + 1, yy), _GRID, 1)
            cv2.putText(canvas, f"{frac * span_deg:4.1f}", (4, yy + 4),
                        _FONT, 0.34, _TEXT_DIM, 1, cv2.LINE_AA)

        # Gate reference lines (only once we know the thresholds).
        if gate > 0.0:
            yg = y_of(gate)
            cv2.line(canvas, (x_left, yg), (x_right + 1, yg), _GATE, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"gate {gate:.1f} deg   ^ above: trust gyro",
                        (48, yg - 4), _FONT, 0.38, _GATE, 1, cv2.LINE_AA)
            yf = y_of(gate + span)
            cv2.line(canvas, (x_left, yf), (x_right + 1, yf), _FULL_GYRO, 1,
                     cv2.LINE_AA)
            cv2.putText(canvas, f"full gyro {gate + span:.1f} deg",
                        (48, yf - 4), _FONT, 0.38, _FULL_GYRO, 1, cv2.LINE_AA)

        # Title + plain-language legend.
        cv2.putText(canvas, "HOW MUCH THE CAMERA TURNED  (deg/frame)", (8, y0 - 8),
                    _FONT, 0.42, _TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "VISION (eyes)", (self.width - 280, y0 - 8), _FONT,
                    0.40, _VISION, 1, cv2.LINE_AA)
        cv2.putText(canvas, "GYRO (inner ear)", (self.width - 165, y0 - 8), _FONT,
                    0.40, _GYRO, 1, cv2.LINE_AA)

        if n < 2:
            return
        x = np.linspace(x_left, x_right, n)
        vis = np.array([s.vision_rot_deg for s in self._hist])
        gyr = np.array([s.gyro_rot_deg for s in self._hist])
        vis_y = np.clip(y1 - (vis / span_deg) * h, y0, y1)
        gyr_y = np.clip(y1 - (gyr / span_deg) * h, y0, y1)

        # Shade the disagreement: the band between the two traces, per column.
        band = np.concatenate([
            np.stack([x, vis_y], axis=1),
            np.stack([x[::-1], gyr_y[::-1]], axis=1),
        ]).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [band], _DISAGREE)
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0, dst=canvas)

        # The two traces on top (gyro slightly thicker -- it is the trusted one).
        cv2.polylines(canvas, [np.stack([x, vis_y], axis=1).astype(np.int32)],
                      False, _VISION, 1, cv2.LINE_AA)
        cv2.polylines(canvas, [np.stack([x, gyr_y], axis=1).astype(np.int32)],
                      False, _GYRO, 2, cv2.LINE_AA)

    def _shade_takeover_spans(self, canvas, y0: int, y1: int,
                              x_left: int, x_right: int) -> None:
        """Amber-wash the column ranges where disagree_deg >= gate_deg."""
        n = len(self._hist)
        x = np.linspace(x_left, x_right, n)
        over = np.array([s.disagree_deg >= s.gate_deg for s in self._hist])
        if not over.any():
            return
        overlay = canvas.copy()
        # Contiguous runs of "taking over" -> one filled rectangle each, half a
        # column padded on each side so single-frame spikes are still visible.
        edges = np.diff(over.astype(np.int8))
        starts = list(np.where(edges == 1)[0] + 1)
        ends = list(np.where(edges == -1)[0])
        if over[0]:
            starts.insert(0, 0)
        if over[-1]:
            ends.append(n - 1)
        step = (x_right - x_left) / max(n - 1, 1)
        pad = max(step * 0.5, 1.0)
        for s_i, e_i in zip(starts, ends):
            xa = int(max(x_left, x[s_i] - pad))
            xb = int(min(x_right, x[e_i] + pad))
            cv2.rectangle(overlay, (xa, y0), (xb, y1), _TAKEOVER_TINT, cv2.FILLED)
        cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0.0, dst=canvas)

    # -- bottom lane: gain + t_trust (0..1) -------------------------------- #
    def _draw_unit_lane(self, canvas, y0: int, y1: int) -> None:
        h = max(y1 - y0, 1)
        x_left, x_right = 44, self.width - 9
        # 0 / 0.5 / 1 gridlines.
        for frac in (0.0, 0.5, 1.0):
            yy = int(y1 - frac * h)
            cv2.line(canvas, (x_left, yy), (x_right + 1, yy), _GRID, 1)
            cv2.putText(canvas, f"{frac:3.1f}", (12, yy + 4), _FONT, 0.34,
                        _TEXT_DIM, 1, cv2.LINE_AA)

        cv2.putText(canvas, "HOW THE FUSION SPLITS TRUST  (0..1)", (8, y0 - 8),
                    _FONT, 0.42, _TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "gain = vision trust (1 all vision / 0 all gyro)",
                    (8, y0 + 12), _FONT, 0.34, _GAIN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "t_trust = translation trust",
                    (self.width - 215, y0 + 12), _FONT, 0.34, _TTRUST, 1,
                    cv2.LINE_AA)

        n = len(self._hist)
        if n < 2:
            return
        x = np.linspace(x_left, x_right, n)
        gain = np.clip(np.array([s.gain for s in self._hist]), 0.0, 1.0)
        ttr = np.clip(np.array([s.t_trust for s in self._hist]), 0.0, 1.0)
        gain_y = y1 - gain * h
        ttr_y = y1 - ttr * h
        cv2.polylines(canvas, [np.stack([x, gain_y], axis=1).astype(np.int32)],
                      False, _GAIN, 2, cv2.LINE_AA)
        cv2.polylines(canvas, [np.stack([x, ttr_y], axis=1).astype(np.int32)],
                      False, _TTRUST, 1, cv2.LINE_AA)
