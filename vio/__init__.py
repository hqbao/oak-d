"""``vio`` -- the visual-inertial odometry PROJECT (Phase 3 of the split).

Subscribes to the ``imu_camera`` capture process over IPC (``imucam.sample`` +
``frame.depth`` + the retained ``calib.bundle``), runs the same RGB-D visual
odometry (+ gyro prior) and sliding-window bundle adjustment the pre-split
in-process graph ran, and republishes ``pose.odom`` / ``pose.vo`` /
``pose.refined`` / ``keyframe`` / ``frame.tracks`` / ``frame.inliers`` on its own
IPC endpoint for SLAM / UI / tools.

Built by replicating the PROVEN ``imu_camera`` template:

* :mod:`vio.comms` -- the FROZEN vendored comms contract, COPIED bit-identically
  from ``imu_camera.comms`` (a CI ``diff -r`` gate enforces byte-parity); this
  project only consumes its public API.
* :mod:`vio.engine` -- the swappable in-process / subprocess runners that drive
  the heavy keyframe solve (windowed BA + tight VIO window); the algorithm itself
  lives in the shared :mod:`sky` library (``sky.backend.windowed`` /
  ``sky.vio.window``).
* :mod:`vio.resolution_build` / :mod:`vio.warmup` -- the resolution-driven
  frontend/odometry config builders and the JIT warmup, the math-coupled glue VIO
  owns at the project root.
* :mod:`vio.modules` -- the odometry + backend reactive modules (was
  ``ours.flows.{odometry,backend}``), wired by
  :class:`~vio.modules.pipeline.OdometryModule` /
  :class:`~vio.modules.pipeline.BackendModule`.
* :mod:`vio.main` -- the VIO process: a calib client + a data client onto the
  capture endpoint, the local odometry / backend graph, and an
  :class:`~vio.comms.IPCPublisher` mirroring its outputs onto the ``oak.vio``
  endpoint.
"""
