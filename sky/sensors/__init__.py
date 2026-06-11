"""``sky.sensors`` -- the shared IMU (gyro/accel) calibration math + on-disk store.

This is the ONE canonical copy of the inertial-sensor calibration code. It used
to be vendored in two projects: ``imu_camera/mathlib/imu/`` (the acquisition
process loads a per-device calibration before publishing IMU) and ``ui/mathlib/
imu/`` (the in-window Calibration menu's gyro/accel dialogs run the collectors
and persist the result). The two copies were byte-identical except for
project-name strings in the docstrings, so consolidating to a single import here
removes the duplication outright.

* :mod:`sky.sensors.accel_calib` -- the six-position accelerometer model
  (:class:`~sky.sensors.accel_calib.AccelCalibration` + the least-squares solve).
* :mod:`sky.sensors.calib_collect` -- the stillness gate + six-face collector
  state machines (gyro-bias + accel-pose capture) the dialogs drive.
* :mod:`sky.sensors.calib_store` -- the per-device JSON store under the repo
  ``.cache`` dir (``save_/load_gyro_bias`` + ``save_/load_accel_calib``).
* :mod:`sky.sensors.imu_calib` -- :class:`~sky.sensors.imu_calib.ImuCalibration`,
  the (gyro_bias + accel) bundle a consumer applies to raw IMU.

It imports only ``numpy`` (plus the stdlib ``json``/``pathlib``/``time`` the
store needs) -- no process / comms / io / ui module -- so it stays a leaf and
movable (maps onto the C ``libskysensors`` layer in ``docs/C_PORT_PLAN.md``).

NOTE: this is the IMU (inertial) calibration. The STEREO-CAMERA calibration
(intrinsics/extrinsics wizard + per-device store) is a separate, single-copy
system that lives in ``imu_camera/mathlib/device/camera_calib_store.py`` and the
``ui/qt`` camera-calib dialogs -- it is deliberately NOT consolidated here.
"""
