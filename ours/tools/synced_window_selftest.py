#!/usr/bin/env python3
"""Headless self-test for the in-app synced (image | depth | IMU) window.

Drives :class:`ours.ui.synced_window.SyncedViewWindow` off a recorded session
(no device) under the offscreen Qt platform and asserts the polished window
renders all three honest panels -- camera image, colorised depth (+ valid %),
and the reused gyro/accel IMU widgets -- plus the footer/state wiring. This puts
the redesigned triplet view on the offline sweep; the live OAK-D path
(:class:`LiveTripletWorker`) is the only part left to the bench.

Under offscreen Qt the OpenGL context for the 3D accel view fails harmlessly, so
this checks DATA wiring (pixmaps set, sample counts, label text, state colours),
not GL pixels.

Run::

    QT_QPA_PLATFORM=offscreen python -m ours.tools.synced_window_selftest
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np                                                  # noqa: E402

from PyQt6.QtWidgets import QApplication                            # noqa: E402

from ours.ui import theme                                          # noqa: E402
from ours.ui.synced_window import (                                # noqa: E402
    ReplayTripletWorker, SyncedViewWindow, TripletSample, TripletWorker,
)

_SESSION = "sessions/gold/lab_loop_30s"
_MAX_FRAMES = 24


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def _run_until(app, predicate, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline and not predicate():
        app.processEvents()
        time.sleep(0.005)


def _replay_factory():
    return lambda: ReplayTripletWorker(_SESSION, fps=120,
                                       max_frames=_MAX_FRAMES)


class _DeadWorker(TripletWorker):
    """A worker that fails to open -- mimics a missing/busy OAK-D."""

    mode = "LIVE"

    def _drive(self, bus, sink):
        raise RuntimeError("X_LINK_DEVICE_NOT_FOUND")


def test_happy_path(app) -> None:
    print(" replay (happy path)")
    win = SyncedViewWindow(_replay_factory(), fps=120)
    win.resize(1500, 600)
    win.show()                                       # showEvent -> start()
    _check(win._running, "window started the replay worker on show")
    _check(win._mode_pill.text() == "REPLAY", "mode pill shows REPLAY")

    seqs: list[int] = []
    widths: list[int] = []

    def _got_frames() -> bool:
        if win._first_seen and win._status.text().startswith("SEQ"):
            seqs.append(int(win._status.text().split("SEQ")[1].split()[0]))
            widths.append(win.width())
        return len(seqs) >= 10 or win._ended

    _run_until(app, _got_frames, 15.0)

    img = win._image.pixmap()
    dep = win._depth.pixmap()
    _check(img is not None and not img.isNull(), "image panel shows a pixmap")
    _check(dep is not None and not dep.isNull(), "depth panel shows a pixmap")
    _check(win._gyro.sample_count > 0,
           f"gyro chart fed real samples (n={win._gyro.sample_count})")
    _check(float(np.linalg.norm(win._accel.accel)) > 1e-6,
           "accel 3D view received a non-zero vector")
    _check(win._valid.text().startswith("valid"),
           f"depth valid% rendered ({win._valid.text()!r})")
    _check(win._status.text().startswith("SEQ") and "imu_n" in win._status.text(),
           f"footer formatted ({win._status.text()!r})")
    _check("|a|" in win._imu_readout.text() and "tilt" in win._imu_readout.text(),
           f"accel readout shows tilt + |a| ({win._imu_readout.text()!r})")
    _check("RAW" in win._imu_title.text(),
           f"IMU title flags RAW with no calibration ({win._imu_title.text()!r})")
    _check(len(seqs) >= 1, f"reported rendered frames (saw seq {seqs[:5]})")
    _check(min(widths) == max(widths),
           f"window width stayed stable (no pixmap feedback growth) {set(widths)}")
    _check(abs(win._accel._view_dist - 4.6 / 0.7) < 1e-6,
           f"accel zoomed 0.7x (view_dist={win._accel._view_dist:.2f})")
    sizes = win._gyro.parent().sizes() if hasattr(win._gyro.parent(), "sizes") \
        else None
    if sizes and len(sizes) == 2 and min(sizes) > 0:
        _check(abs(sizes[0] - sizes[1]) <= max(sizes) * 0.1,
               f"gyro|accel split is symmetric ({sizes})")
    win.close()
    _check(not win._running, "window stopped the worker on close")


def test_calibrated_imu(app) -> None:
    print(" calibrated IMU -> title CALIBRATED + bias applied")
    from ours.lib.imu.accel_calib import AccelCalibration
    from ours.lib.imu.imu_calib import ImuCalibration

    bias = np.array([0.05, -0.03, 0.02])
    calib = ImuCalibration(gyro_bias=bias, accel=None)
    _check(not calib.is_identity, "injected calibration is non-identity")

    # Sanity: apply() subtracts the bias (raw -> calibrated differs by bias).
    raw = np.array([[0.10, 0.20, 0.30]])
    g_cal, _ = calib.apply(raw, np.empty((0, 3)))
    _check(np.allclose(g_cal, raw - bias), "gyro calibration subtracts the bias")

    win = SyncedViewWindow(
        lambda: ReplayTripletWorker(_SESSION, fps=120, max_frames=_MAX_FRAMES,
                                    calibration=calib), fps=120)
    win.show()

    def _calibrated() -> bool:
        return win._first_seen and "CALIBRATED" in win._imu_title.text()

    _run_until(app, _calibrated, 15.0)
    _check("CALIBRATED" in win._imu_title.text(),
           f"IMU title shows CALIBRATED ({win._imu_title.text()!r})")
    win.close()


def test_state_logic() -> None:
    print(" state-colour + readout logic (unit)")
    # |a| sanity bands.
    _check(SyncedViewWindow._mag_color(9.81) == theme.GOOD, "|a|~1G -> GOOD")
    _check(SyncedViewWindow._mag_color(9.0) == theme.WARN, "|a| off ~0.8 -> WARN")
    _check(SyncedViewWindow._mag_color(5.0) == theme.BAD, "|a| far off -> BAD")
    # depth valid% bands.
    _check(SyncedViewWindow._valid_color(70.0) == theme.GOOD, "valid 70% -> GOOD")
    _check(SyncedViewWindow._valid_color(40.0) == theme.WARN, "valid 40% -> WARN")
    _check(SyncedViewWindow._valid_color(10.0) == theme.BAD, "valid 10% -> BAD")


def test_no_imu_frame(app) -> None:
    print(" no-IMU-this-frame state (honest)")
    win = SyncedViewWindow(_replay_factory(), fps=120)
    win.show()
    _run_until(app, lambda: win._first_seen, 15.0)
    # Feed a synthetic frame with NO IMU samples and assert the readout/footer
    # honestly say so rather than holding a stale vector.
    empty = TripletSample(
        gray_left=np.zeros((40, 60), np.uint8),
        depth_m=np.zeros((40, 60), np.float32),
        gyro_rows=np.empty((0, 3)), accel_rows=np.empty((0, 3)),
        seq=999, t_s=1.0)
    _check(empty.imu_n == 0, "empty sample reports imu_n == 0")
    win._update_imu_readout(empty)
    win._update_footer(empty)
    _check("no IMU" in win._imu_readout.text(),
           f"readout flags no IMU ({win._imu_readout.text()!r})")
    _check("imu_n 0" in win._status.text(), "footer shows imu_n 0")
    win.close()


def test_device_absent(app) -> None:
    print(" device absent -> clean fail (no hang)")
    win = SyncedViewWindow(lambda: _DeadWorker(), fps=120)
    win._startup_timeout_s = 6.0
    win.show()
    _run_until(app, lambda: win._failed, 8.0)
    _check(win._failed, "window surfaced the open failure")
    _check(not win._running, "failed window released the worker")
    win.close()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    reader_ok = Path(_SESSION).exists()
    _check(reader_ok, f"gold session present: {_SESSION}")
    test_happy_path(app)
    test_state_logic()
    test_calibrated_imu(app)
    test_no_imu_frame(app)
    test_device_absent(app)
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    print("synced_window_selftest")
    raise SystemExit(main())
