"""``ours.proc`` -- per-process entry points for the 4-process live architecture.

Each module exposes a ``main()`` so it can be spawned as a standalone Python
process:

* :mod:`ours.proc.capture` -- owns the OAK-D (or a replay session) and publishes
  the cam / IMU / depth / calibration streams on its IpcBus endpoint.
* :mod:`ours.proc.vio` -- subscribes to capture, runs odometry + windowed BA,
  republishes pose / keyframe / refined-map streams.
* :mod:`ours.proc.slam` -- subscribes to VIO's keyframes, runs loop closure +
  pose-graph, republishes loop corrections + the slam map.
* :mod:`ours.proc.ui` -- Qt UI (tabs: VIO + SLAM); subscribes to all of the above.
* :mod:`ours.proc.launcher` -- spawns the 3 background procs + runs UI in fg.

See ``docs/PROC4_ARCHITECTURE.md`` for the full design.

OFFLINE replay (``ours.app.run_replay``) does NOT use this package; the
single-process codepath stays unchanged and remains the byte-identical reference.
"""
