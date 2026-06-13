"""``slam`` -- the loop-closure SLAM PROJECT (Phase 4 of the split).

Subscribes to the ``vio`` process over IPC (``keyframe`` + the retained
``calib.bundle``), runs the same ORB loop closure + SE(3) pose-graph optimisation
the pre-split in-process graph ran, and republishes ``loop.correction`` /
``slam.map`` on its own IPC endpoint for the UI / tools. It owns the SLAM map
(ORB feature index + pose graph); the VIO map (windowed BA) lives in the VIO
process, and the two maps are independent by design. The correction stream is
one-way: SLAM never closes the loop back into VIO.

Built by replicating the PROVEN ``imu_camera`` / ``vio`` template:

* :mod:`slam.comms` -- the FROZEN vendored comms contract, COPIED bit-identically
  from ``imu_camera.comms`` (a ``diff -r`` gate enforces byte-parity); this
  project only consumes its public API.
* :mod:`slam.engine` -- the swappable in-process / subprocess runners that drive
  the heavy keyframe solve (ORB loop closure + SE(3) pose-graph optimisation); the
  algorithm itself lives in the shared :mod:`sky.slam` library.
* :mod:`slam.resolution_build` -- the resolution-driven, math-coupled config
  builder SLAM owns at the project root.
* :mod:`slam.modules` -- the loop-closure pipeline (was ``ours.flows.slam``),
  now PROCEDURAL: the plain function :func:`~slam.modules.pipeline.process_keyframe`
  driven by the plain worker thread :class:`~slam.modules.pipeline.SlamWorker`
  (legacy alias :data:`~slam.modules.pipeline.SlamModule`).
* :mod:`slam.main` -- the SLAM process: a calib + keyframe client onto the VIO
  endpoint, the local SLAM graph, and an :class:`~slam.comms.IPCPublisher`
  mirroring ``loop.correction`` / ``slam.map`` onto the ``oak.slam`` endpoint.
"""
