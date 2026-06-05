#!/usr/bin/env python3
"""Headless self-test for the in-app synced camera/IMU window.

Drives :class:`ours.ui.imucam_window.ImuCamWindow` off a recorded session (no
device) under the offscreen Qt platform and asserts the REAL split flows feed it
and it actually renders frames into the widget. This puts the in-app live viewer
on the offline sweep -- the live OAK-D path is the only part left to the bench.

Run::

    QT_QPA_PLATFORM=offscreen python -m ours.tools.imucam_window_selftest
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt6.QtWidgets import QApplication                            # noqa: E402

from ours.flows.cam_reader.sources import ReplayCamSource           # noqa: E402
from ours.flows.imu_reader.sources import ReplayImuSource           # noqa: E402
from ours.lib.io.reader import SessionReader                        # noqa: E402
from ours.ui.imucam_window import ImuCamWindow                      # noqa: E402

_SESSION = "sessions/gold/lab_loop_30s"
_MAX_FRAMES = 24


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def _replay_factory(reader: SessionReader):
    def _make():
        return (ReplayCamSource(reader, max_frames=_MAX_FRAMES),
                ReplayImuSource(reader, realtime=False))
    return _make


def main() -> int:
    print("imucam_window_selftest")
    reader = SessionReader(Path(_SESSION))
    app = QApplication.instance() or QApplication([])

    win = ImuCamWindow(_replay_factory(reader), fps=120)
    win.show()                                       # showEvent -> start()
    _check(win._running, "window started the split flows on show")

    # Pump the Qt loop until the widget has rendered frames (or timeout).
    seqs: list[int] = []
    deadline = time.time() + 15.0
    while time.time() < deadline:
        app.processEvents()
        pix = win._view.pixmap()
        txt = win._status.text()
        if pix is not None and not pix.isNull() and txt.startswith("seq="):
            seqs.append(int(txt.split("seq=")[1].split()[0]))
            if len(seqs) >= 5 or win._ended:
                break
        time.sleep(0.005)

    pix = win._view.pixmap()
    _check(pix is not None and not pix.isNull(), "widget shows a rendered pixmap")
    _check(pix.width() > 360 * 3 and pix.height() >= 1,
           "pixmap spans the cameras + gyro + accel row")
    _check(len(seqs) >= 1, f"status reported rendered packets (saw seq {seqs[:5]})")
    _check(seqs == sorted(seqs), "rendered packets arrive in frame order")
    _check("depthai" not in sys.modules,
           "offline replay path never imported depthai (stays lazy)")

    win.close()                                      # closeEvent -> stop()
    _check(not win._running, "window stopped the flows on close")

    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
