"""``imu_camera.mathlib.imu`` -- IMU calibration, preintegration, sample buffers.

Pure-numpy inertial math the acquisition pipeline composes:

* :mod:`sky.sensors.imu_calib` -- the per-device gyro-bias + accel correction
  (:class:`~sky.sensors.imu_calib.ImuCalibration`); the gyro/accel calibration
  math itself now lives in the shared :mod:`sky.sensors` library (the six-face
  collector :mod:`sky.sensors.calib_collect`, the on-disk
  :mod:`sky.sensors.calib_store`, and the accel model
  :mod:`sky.sensors.accel_calib`).
* :mod:`~imu_camera.mathlib.imu.imu` -- SO(3) preintegration + gyro integration.
* :mod:`~imu_camera.mathlib.imu.timed_buffer` -- the thread-safe timestamped IMU
  buffer the ``imu_cam`` module drains per camera trigger.
* :mod:`~imu_camera.mathlib.imu.decode` -- depthai IMU packet decode (live only).
* :mod:`~imu_camera.mathlib.imu.inertial_filter` -- the inertial translation
  filter.
"""
