"""``sky.depth`` -- the shared from-scratch dense/sparse stereo depth math.

This is the ONE canonical copy of the semi-global block matcher (SGM) + the
rectifiers. It used to be vendored byte-identically in two places --
``imu_camera/mathlib/stereo`` and ``depth/mathlib/stereo`` (a ``diff -r`` gate
kept them in lock-step) -- because ``depth`` runs INLINE on the capture process's
``imu_cam`` thread today. Consolidating to a single import here retires that gate:
there is now nothing to keep in lock-step.

* :mod:`sky.depth.stereo` -- :class:`~sky.depth.stereo.SGMStereoMatcher` +
  ``SGMConfig`` (semi-global block matching with built-in left/right
  rectification, numba-accelerated) and the sparse
  :class:`~sky.depth.stereo.StereoMatcher` used by the stereo self-test.

The stereo math takes a stereo-calibration object (each project's
``io.reader.StereoCalib``); it bakes in no project name and pulls in no
process/comms/io module, so it stays movable (maps onto the C ``libskydepth``
layer -- ``depth`` is the C port's first real process).
"""
