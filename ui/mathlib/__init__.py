"""``ui.mathlib`` -- the minimal math the UI process owns.

The UI is a sink: it renders trajectories + imagery fed over IPC and never runs
odometry / BA / SLAM, so it needs almost no math. The one exception is the
in-window **Calibration** menu:

* the gyro / accel calib dialogs (:mod:`ui.qt.calib_dialogs`) drive the tested
  stillness-gate / six-face collector state machines and persist the result --
  that IMU-calibration math is the SHARED :mod:`sky.sensors` library now
  (:mod:`sky.sensors.calib_collect` / :mod:`sky.sensors.calib_store` /
  :mod:`sky.sensors.accel_calib`), deduped out of the old ``ui.mathlib.imu``.
* the STEREO-CAMERA calibration wizard's device-free math lives here:

  * :mod:`ui.mathlib.calib` -- the printable checkerboard target, the corner
    detector + L<->R reconcile, the diversity-gated stereo collector, the stereo
    solve, and the calib.json writer.
"""
