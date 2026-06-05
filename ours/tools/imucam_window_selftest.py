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


class _DeadCamSource(ReplayCamSource):
    """A camera source that fails to open -- mimics a missing/busy OAK-D.

    Reproduces the live failure where ``LiveCamSource.open()`` raises
    ``X_LINK_DEVICE_NOT_FOUND``; the window must surface it, not hang.
    """

    def open(self) -> None:
        raise RuntimeError(
            "Failed to find device (3.1.2), error message: X_LINK_DEVICE_NOT_FOUND")


def _dead_factory(reader: SessionReader):
    def _make():
        return (_DeadCamSource(reader, max_frames=_MAX_FRAMES),
                ReplayImuSource(reader, realtime=False))
    return _make


def _run_until(app, predicate, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline and not predicate():
        app.processEvents()
        time.sleep(0.005)


def test_happy_path(app, reader: SessionReader) -> None:
    print(" replay (happy path)")
    win = ImuCamWindow(_replay_factory(reader), fps=120)
    win.show()                                       # showEvent -> start()
    _check(win._running, "window started the split flows on show")

    seqs: list[int] = []

    def _got_frames() -> bool:
        pix = win._view.pixmap()
        txt = win._status.text()
        if pix is not None and not pix.isNull() and txt.startswith("seq="):
            seqs.append(int(txt.split("seq=")[1].split()[0]))
        return len(seqs) >= 5 or win._ended

    _run_until(app, _got_frames, 15.0)

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


def test_device_not_found(app, reader: SessionReader) -> None:
    print(" device-not-found (must fail clean, not hang)")
    win = ImuCamWindow(_dead_factory(reader), fps=120)
    win._startup_timeout_s = 3.0
    win.show()

    _run_until(app, lambda: win._failed, 8.0)

    _check(win._failed, "window flagged failure instead of hanging")
    _check(not win._first_seen, "no frame was rendered on the dead device")
    _check("X_LINK_DEVICE_NOT_FOUND" in win._view.text(),
           f"widget shows the device error (got: {win._view.text()!r})")
    _check(not win._timer.isActive(), "render timer stopped after failure")
    _check(not win._running, "failed window released the graph (running=False)")

    win.close()
    _check(not win._running, "window stopped the flows on close")


def test_retry_after_replug(app, reader: SessionReader) -> None:
    print(" retry after replug (fail -> device appears -> reopen streams)")
    # A factory that fails the first time (device absent) and works after.
    state = {"opened": 0}

    def _make():
        state["opened"] += 1
        if state["opened"] == 1:
            return (_DeadCamSource(reader, max_frames=_MAX_FRAMES),
                    ReplayImuSource(reader, realtime=False))
        return (ReplayCamSource(reader, max_frames=_MAX_FRAMES),
                ReplayImuSource(reader, realtime=False))

    win = ImuCamWindow(_make, fps=120)
    win._startup_timeout_s = 3.0
    win.show()                                       # 1st open -> fails
    _run_until(app, lambda: win._failed, 8.0)
    _check(win._failed, "first open failed (device absent)")

    win.ensure_started()                             # reopen -> retries
    _check(win._running and not win._failed, "reopen restarted the stream")

    seqs: list[int] = []

    def _got() -> bool:
        txt = win._status.text()
        if txt.startswith("seq="):
            seqs.append(int(txt.split("seq=")[1].split()[0]))
        return len(seqs) >= 3 or win._ended

    _run_until(app, _got, 12.0)
    _check(len(seqs) >= 1, f"streamed after replug (saw seq {seqs[:3]})")

    win.close()
    _check(not win._running, "window stopped the flows on close")


def main() -> int:
    print("imucam_window_selftest")
    reader = SessionReader(Path(_SESSION))
    app = QApplication.instance() or QApplication([])

    test_happy_path(app, reader)
    test_device_not_found(app, reader)
    test_retry_after_replug(app, reader)

    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
