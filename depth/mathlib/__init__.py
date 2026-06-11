"""``depth.mathlib`` -- this project's per-process glue (no stereo math here).

The SGM stereo math is no longer owned here: it has been consolidated into the
shared :mod:`sky.depth.stereo` (one canonical copy, imported by both ``depth``
and ``imu_camera``). That retired the old ``diff -r depth/mathlib/stereo
imu_camera/mathlib/stereo`` lock-step gate -- there is now a single copy, so
there is nothing left to keep in lock-step.

* :mod:`sky.depth.stereo` -- :class:`~sky.depth.stereo.SGMStereoMatcher` +
  ``SGMConfig`` (semi-global block matching with built-in left/right
  rectification) and the sparse :class:`~sky.depth.stereo.StereoMatcher` used by
  the self-test.
"""
