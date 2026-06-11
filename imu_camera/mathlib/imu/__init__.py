"""``imu_camera.mathlib.imu`` -- the DEVICE-coupled IMU layer (depthai decode).

The device-free inertial math the acquisition pipeline composes has all been
RELOCATED into the shared :mod:`sky` leaf library; what stays here is only the
depthai-specific packet decode that genuinely needs the driver:

* :mod:`~imu_camera.mathlib.imu.decode` -- depthai IMU packet decode (live only);
  the one device-coupled module, so it STAYS per-project.

The shared inertial math now lives under :mod:`sky` (cross-referenced so the
acquisition flow still reads end-to-end):

* :mod:`sky.sensors.imu_calib` -- the per-device gyro-bias + accel correction
  (:class:`~sky.sensors.imu_calib.ImuCalibration`); the gyro/accel calibration
  math lives in :mod:`sky.sensors` (six-face collector
  :mod:`sky.sensors.calib_collect`, on-disk :mod:`sky.sensors.calib_store`, accel
  model :mod:`sky.sensors.accel_calib`) -- S3.
* :mod:`sky.imu.imu` -- the shared (loose) SO(3) preintegration + gyro
  integration the acquisition pipeline references (S4).
* :mod:`sky.imu.timed_buffer` -- the thread-safe timestamped IMU buffer the
  ``imu_cam`` module drains per camera trigger (relocated R6).
* :mod:`sky.imu.inertial_filter` -- the inertial translation filter
  (relocated R6).
"""
