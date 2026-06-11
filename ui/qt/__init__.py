"""``ui.qt`` -- the PyQt6 GUI (military-dark dashboard + 3D viewers).

The actual PyQt6 application widgets: ``viewer3d`` (the single 5-trajectory 3D
view), ``map_window`` (SLAM point cloud), the keypoint / triplet / cam-IMU
windows, the calib wizards, ``theme``, the telemetry ``panels`` / ``imu_panels``,
and the ``PoseSource`` bridges (``source`` / ``live_source``) that drive them.
Also ``mainwindow`` (the single-process top-level window).

Ported verbatim from ``ours.ui`` (import-rewrites only): ``ours.lib.misc`` /
``ours.lib.config`` -> ``ui.comms.lib.*``; ``ours.lib.viz`` -> ``ui.viz``;
the IMU-calib math (``ours.lib.imu``) is now the shared :mod:`sky.sensors`
library; ``ours.flows.ui`` -> ``ui.modules``; the in-proc ``Bus`` ->
``ui.comms.LocalPubSub`` and ``Flow`` -> ``ui.comms.Module``.

Device-free by contract: this is a sink GUI. The 4-process ``ui.main`` builds its
single viewer + toolbar + menus inline and feeds them over IPC; capture owns the
OAK-D, so nothing here opens a device. The single-process in-process front-end
drivers (``ours.app.build_live`` / ``build_replay`` + the live device sources)
are NOT vendored into this project, so the historical in-process worker paths
(``LiveTripletWorker`` / ``ReplayTripletWorker`` / ``LiveKeypointWorker`` /
``ReplayKeypointWorker`` / ``FlowPoseSource`` / ``ImuCamWindow``'s live source)
surface a clear error instead of opening a device. proc4 never reaches them: it
injects the IPC adapters from :mod:`ui.modules.ipc_sources` instead.

NOT the same as ``ui.modules`` -- that is the set of NO-Qt bus SINKS (+ the IPC
source adapters) that FEED these views. The dependency is one-way: ``ui.qt``
imports ``ui.modules`` sinks; ``ui.modules.ipc_sources`` imports ``ui.qt`` worker
base classes (the only crossing edge), never the reverse for the sinks.
"""
