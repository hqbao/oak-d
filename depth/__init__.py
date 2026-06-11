"""``depth`` -- the stereo-depth PROJECT (the two depth steps + standalone harness).

depth runs the from-scratch SGM dense-stereo matcher (now the shared
:mod:`sky.depth.stereo`) through the two depth steps (``compute_depth`` ->
``publish_depth``). The stereo math used to be vendored here and copied
byte-identically into ``imu_camera`` under a ``diff -r`` lock-step gate; it has
since been consolidated into the single shared :mod:`sky.depth.stereo`, so that
gate is retired -- there is one copy, edited in one place.

This package is a STANDALONE, independently-runnable source tree (it is NOT
spawned by the launcher -- depth runs inline in imu_camera in the live
topology): :mod:`depth.main` is the harness that proves depth runs as its own
project. It SUBSCRIBES to raw ``cam.sync`` (left/right) over IPC, computes metric
depth with the SGM matcher, and PUBLISHES ``frame.depth`` (rectified-left +
metric depth) on its own endpoint.

Layers
------
* :mod:`depth.comms` -- the FROZEN vendored comms contract (bit-identical across
  all five split projects); this project only consumes its public API.
* :mod:`sky.depth.stereo` -- the shared SGM stereo math (one canonical copy,
  imported by both this project and imu_camera).
* :mod:`depth.io` -- recorded-session reading, used ONLY to read the full
  :class:`~depth.io.reader.StereoCalib` the matcher's rectifiers need (the wire
  ``calib.bundle`` carries only the rectified-left intrinsic, not the per-camera
  calibration).
* :mod:`depth.modules` -- the ``compute_depth`` + ``publish_depth`` steps.
* :mod:`depth.main` -- the standalone depth process: subscribes to ``cam.sync``,
  runs SGM, publishes ``frame.depth`` on an :class:`~depth.comms.IPCPubSub`
  server.
"""
