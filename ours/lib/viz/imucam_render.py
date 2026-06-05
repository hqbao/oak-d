"""Pure renderers for the split camera/IMU front-end's synchronised output.

Every panel draws exactly what an :class:`~ours.lib.flow.messages.ImuCamPacket`
carries -- the stereo pair, the raw gyro samples, the raw accel samples -- and
nothing computed in a parallel pipeline (honest visualisation):

    [ left cam | right cam |  gyro line chart  |  accel 3D vector ]

The functions are window-free (they return ``uint8`` BGR images) so the same
drawing code feeds both the cv2 dev tool (``ours.tools.imucam_view``) and the
in-app Qt window (``ours.ui.imucam_window``); there is a single, honest source of
truth for what the synced front-end looks like. cv2 is only a drawing backend
here -- importing this module is what pulls it, not the base UI.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

_PANEL_H = 360
_RAD2DEG = 180.0 / np.pi
_G = 9.80665

# Per-axis colours (BGR): x = right (red), y = down (green), z = forward (blue).
_AX_X = (80, 80, 255)
_AX_Y = (80, 255, 80)
_AX_Z = (255, 160, 90)
_ISO_C = 0.8660254037844387  # cos(30 deg)


def _gray_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(gray), cv2.COLOR_GRAY2BGR)


def _label(img: np.ndarray, text: str, y: int = 22) -> np.ndarray:
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (60, 255, 255), 1, cv2.LINE_AA)
    return img


def _fit(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    w = int(round(img.shape[1] * h / img.shape[0]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def render_cameras(left: np.ndarray, right: np.ndarray | None,
                   h: int = _PANEL_H) -> np.ndarray:
    """Side-by-side stereo panel (left | right)."""
    lp = _label(_fit(_gray_bgr(left), h), "left")
    if right is None:
        return lp
    rp = _label(_fit(_gray_bgr(right), h), "right")
    return np.hstack([lp, rp])


class GyroChart:
    """Scrolling 3-axis gyro line chart fed raw samples (rad/s -> deg/s)."""

    def __init__(self, width: int = 360, height: int = _PANEL_H,
                 capacity: int = 600, span_dps: float = 150.0) -> None:
        self.width = int(width)
        self.height = int(height)
        self.span = float(span_dps)
        self._hist: deque[np.ndarray] = deque(maxlen=capacity)

    def clear(self) -> None:
        self._hist.clear()

    def add(self, gyro_rad: np.ndarray) -> None:
        """Append every gyro sample in a packet (``(M,3)`` rad/s)."""
        if gyro_rad.size == 0:
            return
        for s in np.atleast_2d(gyro_rad):
            self._hist.append(np.asarray(s, dtype=np.float64) * _RAD2DEG)

    def render(self) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), 24, dtype=np.uint8)
        mid = self.height // 2
        cv2.line(canvas, (0, mid), (self.width, mid), (60, 60, 60), 1)
        for frac in (0.25, 0.75):
            y = int(self.height * frac)
            cv2.line(canvas, (0, y), (self.width, y), (40, 40, 40), 1)
        _label(canvas, "gyro deg/s  x y z")
        n = len(self._hist)
        if n >= 2:
            data = np.array(self._hist)                       # (n, 3)
            x = np.linspace(0, self.width - 1, n)

            def y_of(v):
                return np.clip(
                    mid - (v / self.span) * (self.height * 0.5),
                    0, self.height - 1)

            for axis, color in enumerate((_AX_X, _AX_Y, _AX_Z)):
                ys = y_of(data[:, axis])
                pts = np.stack([x, ys], axis=1).astype(np.int32)
                cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)
            latest = data[-1]
            cv2.putText(
                canvas,
                f"{latest[0]:+6.1f} {latest[1]:+6.1f} {latest[2]:+6.1f}",
                (8, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)
        return canvas


def _iso(v) -> tuple[float, float]:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return (x - z) * _ISO_C, (x + z) * 0.5 + y


def _arrow3d(canvas, center, vec3, length, color, label=None, thick=2):
    sx, sy = _iso(np.asarray(vec3, float))
    tip = (int(center[0] + sx * length), int(center[1] + sy * length))
    cv2.arrowedLine(canvas, center, tip, color, thick, cv2.LINE_AA,
                    tipLength=0.18)
    if label:
        cv2.putText(canvas, label, (tip[0] + 3, tip[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def render_accel3d(accel_rows: np.ndarray, w: int = 360,
                   h: int = _PANEL_H) -> np.ndarray:
    """Average the packet's accel samples and draw the specific-force vector."""
    canvas = np.full((h, w, 3), 24, dtype=np.uint8)
    _label(canvas, "accel m/s^2 (3D)")
    center = (w // 2, h // 2 + 20)
    ref = 26.0
    # Optical reference axes.
    _arrow3d(canvas, center, (1, 0, 0), ref, _AX_X, "x", 1)
    _arrow3d(canvas, center, (0, 1, 0), ref, _AX_Y, "y", 1)
    _arrow3d(canvas, center, (0, 0, 1), ref, _AX_Z, "z", 1)
    if accel_rows.size:
        a = np.atleast_2d(accel_rows).mean(axis=0)
        mag = float(np.linalg.norm(a))
        scale = (ref * 2.4) / _G
        _arrow3d(canvas, center, a, scale, (0, 215, 255), None, 2)
        cv2.putText(canvas, f"|a|={mag:5.2f}", (8, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"a=({a[0]:+.1f},{a[1]:+.1f},{a[2]:+.1f})",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def compose(packet, chart: GyroChart) -> np.ndarray:
    """Build the full visualisation row from one packet (updates ``chart``)."""
    chart.add(packet.gyro)
    cams = render_cameras(packet.gray_left, packet.gray_right)
    gyro_panel = chart.render()
    accel_panel = render_accel3d(packet.accel)
    row = np.hstack([cams, gyro_panel, accel_panel])
    cv2.putText(row, f"seq={packet.seq}  imu={packet.imu_ts.size}",
                (8, row.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
    return row
