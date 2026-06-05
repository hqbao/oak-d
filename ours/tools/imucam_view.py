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
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.flows.cam_reader import CamReaderFlow                   # noqa: E402
from ours.flows.cam_reader.sources import ReplayCamSource         # noqa: E402
from ours.flows.imu_reader import ImuReaderFlow                  # noqa: E402
from ours.flows.imu_reader.sources import ReplayImuSource        # noqa: E402
from ours.lib.flow import Bus, Flow, topics                       # noqa: E402
from ours.lib.io.reader import SessionReader                        # noqa: E402
# Honest renderers shared with the in-app Qt window (ours.ui.imucam_window) so
# both viewers draw the synced front-end identically; re-exported for the
# headless renderer self-test (imucam_view_selftest imports them from here).
from ours.lib.viz.imucam_render import (                            # noqa: E402,F401
    GyroChart, compose, render_accel3d, render_cameras,
)


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
