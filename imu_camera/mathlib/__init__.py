"""``imu_camera.mathlib`` -- the math this project OWNS.

* :mod:`~imu_camera.mathlib.device` -- the shared live OAK-D + boot calibration
  (depthai-backed; imported lazily so the offline/replay path never pulls it).
* :mod:`~imu_camera.mathlib.imu` -- IMU calibration, preintegration, the
  timestamped sample buffer, packet decode.

The from-scratch SGM dense-stereo matcher + rectifiers used to be vendored here
at ``imu_camera/mathlib/stereo``; it now lives in the shared
:mod:`sky.depth.stereo` (one canonical copy, imported by both this project and
``depth``).
"""
