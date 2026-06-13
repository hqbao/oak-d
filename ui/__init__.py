"""``ui`` -- the visualiser PROJECT (Phase 5 of the split).

Subscribes (over IPC) to the ``vio`` and ``slam`` processes -- and on-demand to
``capture`` -- and renders the whole pipeline in a SINGLE PyQt6 ``QMainWindow``:
ONE :class:`~ui.qt.viewer3d.Viewer3D` drawing 5 toggleable trajectory lines
(VO / VIO / VIO-BA / SLAM-corrected VIO / SLAM) behind a live marker, a Controls
toolbar (per-line toggles + Clear Trail + Restart), and View / Visualize /
Calibration menus. It imports **no depthai**: everything it shows is fed over IPC,
so the UI is device-agnostic by contract (capture owns the OAK-D).

Built by replicating the PROVEN ``imu_camera`` / ``vio`` / ``slam`` template:

* :mod:`ui.comms` -- the FROZEN vendored comms contract, COPIED bit-identically
  from ``imu_camera.comms`` (a CI ``diff -r`` gate enforces byte-parity); this
  project only consumes its public API (``IPCPubSub`` / ``LocalPubSub`` /
  ``Module`` / ``Step`` / the bridge + wire + misc helpers).
* :mod:`ui.qt` -- the PyQt6 GUI widgets (viewer3d, panels, the triplet / keypoint
  / calib windows, theme), ported verbatim from ``ours.ui`` with import-rewrites
  only. Qt is imported LAZILY (only inside :func:`ui.main.run_ui`), so
  ``import ui`` / ``import ui.main`` stays Qt-FREE.
* :mod:`ui.modules` -- the NO-Qt reactive bus sinks (ported from
  ``ours.flows.ui``; Flow -> Module, Task -> Step) + the IPC source adapters
  (ported from ``ours.proc.ui_ipc_sources``) that bridge the cross-process
  capture / VIO topics onto the local bus those sinks read.
* :mod:`ui.viz` -- the cv2 visualisation helpers (depth colourise, imucam render,
  keypoint overlay), ported from ``ours.lib.viz``.
* :mod:`ui.calib` -- the MINIMAL calibration math the UI owns: the printable
  checkerboard target generator + the thin I/O wrapper around the shared corner
  detector that the Calibration menu's dialogs drive (cv2-free / Qt-free).
* :mod:`ui.main` -- the UI process: the IPC subscribers (``IpcPoseSource`` +
  ``SlamMapTracker``), the single 5-line viewer + toolbar + menus, and the
  RESTART_EXIT_CODE=42 launcher handshake. Lazy Qt imports in ``run_ui``.
"""
