"""``imu_camera`` -- the capture PROJECT (Phase 1 PROVEN TEMPLATE of the split).

Owns the OAK-D (or a recorded session) and publishes the synced camera / IMU /
depth streams over the canonical :mod:`imu_camera.comms` contract. It is the
first of the five split projects (imu_camera, depth, vio, slam, ui) and the
template the others copy.

Layers
------
* :mod:`imu_camera.comms` -- the FROZEN vendored comms contract (bit-identical
  across all five projects); this project only consumes its public API.
* :mod:`imu_camera.device` -- the device layer it owns: the shared live OAK-D
  acquisition + boot-time calibration, the per-device camera-calib store, and the
  depthai IMU-packet decode (:mod:`~imu_camera.device.imu_decode`).
* :mod:`imu_camera.resolution_build` / :mod:`imu_camera.warmup` -- the
  resolution-driven config builder and the JIT warmup for the inline SGM (the SGM
  stereo it runs now lives in the shared :mod:`sky.depth.stereo`).
* :mod:`imu_camera.io` -- recorded-session reading (the replay data source).
* :mod:`imu_camera.modules` -- the threaded acquisition pipeline (``cam`` +
  ``imu_cam`` with its inline depth steps), wired by
  :class:`~imu_camera.modules.pipeline.ImuCamWorker`.
* :mod:`imu_camera.main` -- the capture process: builds the pipeline on a local
  pub/sub and bridges it to an :class:`~imu_camera.comms.IPCPubSub` server on the
  ``oak.capture`` endpoint.
"""
