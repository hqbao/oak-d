"""Offline construct + logic test for the calibration UI (no device, no display).

Runs under ``QT_QPA_PLATFORM=offscreen`` so it needs neither an OAK-D nor a real
display. It checks three things that must not regress on a product UI:

1. :class:`~ours.ui.mainwindow.MainWindow` builds with the feature menu bar and
   the View / Calibration / Visualize menus carry the expected actions.
2. :class:`~ours.ui.calib_dialogs.GyroCalibDialog` recovers a planted gyro bias
   from a synthetic still stream and enables SAVE.
3. :class:`~ours.ui.calib_dialogs.AccelCalibDialog` walks all six faces from a
   synthetic stream, solves, and enables SAVE with a small residual.

The dialogs are driven by injecting samples into ``_feed_sample`` and ticking
``_drain`` directly -- the same seam the real IMU stream uses -- so the tested
collector logic is exercised end to end through the Qt widgets without a device.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from PyQt6.QtWidgets import QApplication

from ..lib.imu.accel_calib import G_STANDARD
from ..ui.calib_dialogs import AccelCalibDialog, GyroCalibDialog
from ..ui.mainwindow import MainWindow
from ..ui.source import FakePoseSource


class _FakeStream:
    """Stand-in for ImuStream: the test feeds samples by hand, not a thread."""

    def __init__(self) -> None:
        self.error: str | None = None
        self.device_id = "SELFTEST-DEV"
        self.started = False

    def start(self, callback) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def is_running(self) -> bool:
        return self.started


def _feed(dialog, gyro, accel, n, t0, dt=0.005):
    """Inject ``n`` samples then drain them through the dialog's UI tick."""
    t = t0
    for _ in range(n):
        dialog._feed_sample(np.asarray(gyro, float), np.asarray(accel, float), t)
        t += dt
    dialog._drain()
    return t


def _menu_actions(menubar, title):
    for act in menubar.actions():
        if act.text() == title and act.menu() is not None:
            return [a.text() for a in act.menu().actions() if a.text()]
    return None


def test_mainwindow_menus(app):
    from ..lib.misc.pose import PoseHistory
    src = FakePoseSource()
    win = MainWindow(PoseHistory(capacity=100), src, source_name="fake")
    mb = win.menuBar()

    view = _menu_actions(mb, "View")
    cal = _menu_actions(mb, "Calibration")
    vis = _menu_actions(mb, "Visualize")
    assert view is not None, "View menu missing"
    assert cal is not None, "Calibration menu missing"
    assert vis is not None, "Visualize menu missing"

    for name in ("Iso", "Top", "Front", "Back", "Left", "Right"):
        assert name in view, f"View menu missing preset {name}: {view}"
    assert "Follow Camera" in view
    assert "Clear Trail" in view
    assert any("Gyroscope" in a for a in cal), cal
    assert any("Accelerometer" in a for a in cal), cal
    assert any("triplet" in a.lower() for a in vis), vis
    assert any("stereo" in a.lower() for a in vis), vis

    # The fake source has no clear_slam_map -> no Clear Keyframes entry.
    assert not any("Keyframe" in a for a in view), view
    win.close()
    print("[ui] MainWindow menu bar OK "
          f"(View={len(view)}, Calibration={len(cal)}, Visualize={len(vis)})")


def test_gyro_dialog(app):
    bias = np.array([0.012, -0.004, 0.008])
    dlg = GyroCalibDialog(stream=_FakeStream())
    dlg._on_start()
    assert not dlg._save_btn.isEnabled()

    # Still: gyro ~= bias, accel ~= level gravity (+Z up). Small jitter.
    rng = np.random.default_rng(0)
    t = 0.0
    for _ in range(12):
        g = bias + rng.normal(0, 0.002, 3)
        a = np.array([0.0, 0.0, G_STANDARD]) + rng.normal(0, 0.02, 3)
        t = _feed(dlg, g, a, 30, t)
        if dlg._bias is not None:
            break

    assert dlg._bias is not None, "gyro dialog never reached ready"
    assert dlg._save_btn.isEnabled(), "SAVE not enabled after bias captured"
    err = float(np.linalg.norm(dlg._bias - bias))
    assert err < 0.01, f"recovered bias off by {err:.4f}: {dlg._bias}"
    assert dlg._device_id == "SELFTEST-DEV"
    dlg.close()
    print(f"[ui] GyroCalibDialog OK (bias err={err:.5f} rad/s, n={dlg._coll.n})")


def test_accel_dialog(app):
    g = G_STANDARD
    # Synthetic per-face specific force (perfect sensor, +/- each axis).
    faces = [
        np.array([+g, 0, 0]), np.array([-g, 0, 0]),
        np.array([0, +g, 0]), np.array([0, -g, 0]),
        np.array([0, 0, +g]), np.array([0, 0, -g]),
    ]
    dlg = AccelCalibDialog(stream=_FakeStream())
    dlg._on_start()
    rng = np.random.default_rng(1)
    t = 0.0
    for fa in faces:
        # Motion between faces so the "must move first" latch clears.
        t = _feed(dlg, np.array([0.5, -0.4, 0.3]), fa, 5, t)
        # Then hold still on the face.
        for _ in range(6):
            acc = fa + rng.normal(0, 0.01, 3)
            t = _feed(dlg, np.zeros(3), acc, 25, t)
            if dlg._coll.complete:
                break

    assert dlg._coll.complete, (
        f"accel dialog only captured {dlg._coll.captured_faces}")
    cal = dlg._coll.calibration
    assert cal is not None, "calibration not solved on completion"
    assert dlg._save_btn.isEnabled(), "SAVE not enabled after 6 faces"
    assert cal.residual_g < 0.05, f"residual too high: {cal.residual_g:.4f}"
    # A near-perfect sensor -> T ~ identity, bias ~ 0.
    assert float(np.linalg.norm(cal.bias)) < 0.1, f"bias drift {cal.bias}"
    dlg.close()
    print(f"[ui] AccelCalibDialog OK (residual={cal.residual_g:.4f} m/s^2, "
          f"6/6 faces)")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    test_mainwindow_menus(app)
    test_gyro_dialog(app)
    test_accel_dialog(app)
    print("\nALL UI CALIB SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
