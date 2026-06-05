#!/usr/bin/env python3
"""Visualise the split camera/IMU front-end's synchronised output.

This is the eyeball companion to the split-acquisition flows: it runs the REAL
:class:`~ours.flows.capture.cam_reader.CamReaderFlow` and
:class:`~ours.flows.capture.imu_reader.ImuReaderFlow` over a recorded session and
draws, for every :class:`~ours.lib.flow.messages.ImuCamPacket` they publish, the
four things that packet actually contains -- nothing computed in a parallel
pipeline (honest visualisation):

    [ left cam | right cam |  gyro line chart  |  accel 3D vector ]

* **Cameras** -- the stereo pair carried by the packet.
* **Gyro line chart** -- every raw gyro sample in the packet's interval, scrolled
  as three lines (wx red, wy green, wz blue) in deg/s. A direct measurement, so
  this is honest raw signal (no dead-reckoning drift).
* **Accel 3D** -- the packet's accelerometer samples averaged into one
  specific-force vector (m/s^2) drawn in an isometric box with the optical axes
  and a vertical reference; at rest it points along gravity with |a| ~ 9.8.

cv2 here is only a dev-tool display dependency (windowing), like the other
``tools/*`` viewers -- nothing here is in a production path.

Usage::

    python -m ours.tools.imucam_view
    python -m ours.tools.imucam_view --session sessions/gold/lab_loop_30s
    python -m ours.tools.imucam_view --fps 30 --max-frames 200

Keys: SPACE pause/resume, ``r`` clear the gyro chart, ``q`` / ESC quit.
"""
from __future__ import annotations

import argparse
import queue
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.flows.capture.cam_reader import CamReaderFlow              # noqa: E402
from ours.flows.capture.cam_sources import ReplayCamSource          # noqa: E402
from ours.flows.capture.imu_reader import ImuReaderFlow             # noqa: E402
from ours.flows.capture.imu_sources import ReplayImuSource          # noqa: E402
from ours.lib.flow import Bus, Flow, topics                         # noqa: E402
from ours.lib.io.reader import SessionReader                        # noqa: E402

_PANEL_H = 360
_RAD2DEG = 180.0 / np.pi
_G = 9.80665

# Per-axis colours (BGR): x = right (red), y = down (green), z = forward (blue).
_AX_X = (80, 80, 255)
_AX_Y = (80, 255, 80)
_AX_Z = (255, 160, 90)
_ISO_C = 0.8660254037844387  # cos(30 deg)


# --------------------------------------------------------------------------- #
# Pure renderers (no window) -- unit-tested headless by imucam_view_selftest.
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Interactive driver.
# --------------------------------------------------------------------------- #
class _SinkFlow(Flow):
    """Drop each ImuCamPacket into a queue for the main (display) thread."""

    def __init__(self, bus: Bus, out: "queue.Queue") -> None:
        super().__init__("viz-sink", bus)
        self._out = out
        self.on(topics.IMUCAM_SAMPLE, [self._task()])

    def _task(self):
        out = self._out

        class _T:
            name = "enqueue"

            def run(self, ctx, msg):
                out.put(msg)
                return None

        return _T()

    def on_end(self) -> None:
        self._out.put(None)             # sentinel: stream finished


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_straight_20s")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    reader = SessionReader(Path(args.session))
    bus = Bus()
    pkt_q: "queue.Queue" = queue.Queue(maxsize=8)

    imu_flow = ImuReaderFlow(bus, ReplayImuSource(reader, realtime=True))
    cam_flow = CamReaderFlow(
        bus, ReplayCamSource(reader, max_frames=args.max_frames),
        fps=args.fps, realtime=True)
    sink = _SinkFlow(bus, pkt_q)
    sink.expected_ends = 1

    sink.start()
    imu_flow.start()
    cam_flow.start()

    chart = GyroChart()
    win = "imucam — split camera/IMU sync (q quit, SPACE pause, r clear)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    paused = False
    print(f"[imucam_view] session={reader.dir}  fps={args.fps}")
    try:
        while True:
            if not paused:
                try:
                    packet = pkt_q.get(timeout=2.0)
                except queue.Empty:
                    if not cam_flow.is_alive():
                        break
                    continue
                if packet is None:                        # END sentinel
                    break
                frame = compose(packet, chart)
                cv2.imshow(win, frame)
            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                chart.clear()
    finally:
        cam_flow.stop()
        imu_flow.stop()
        sink.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
