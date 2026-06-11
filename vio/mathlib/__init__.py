"""``vio.mathlib`` -- the math the VIO project OWNS.

Ported VERBATIM from ``ours.lib.{frontend,odometry,backend,engine,imu}`` (only the
cross-package import roots + the doc cross-references were re-rooted at
``vio.mathlib`` / ``vio.comms``; no algorithm changed, so the numerical output is
byte-identical to the reference oracle -- proved by
:mod:`vio.tests.vio_ba_selftest`).

* :mod:`sky.front` -- the from-scratch KLT optical-flow tracker + Shi-Tomasi
  corner detector (numba-accelerated; the ONLY numba kernel VIO warms). Relocated
  into the shared :mod:`sky` leaf library (single-copy; VIO is the only consumer).
* :mod:`~sky.front.odometry` -- frame-to-frame RGB-D visual odometry (PnP +
  optional gyro fusion).
* :mod:`~vio.mathlib.backend` -- the sliding-window bundle adjustment + the
  tight-coupled visual-inertial window optimiser.
* :mod:`~vio.mathlib.engine` -- the swappable in-process / subprocess runners for
  the heavy keyframe optimisers (VIO carries its OWN engine copy).
* :mod:`~vio.mathlib.imu` -- the SO(3) helpers + IMU preintegration the
  odometry / backend / pnp math depends on (numpy-only, self-contained).

The ARCHITECTURE RULE lives here too: the math-coupled config builders
(:mod:`~vio.mathlib.resolution_build`) and the JIT warmup
(:mod:`~vio.mathlib.warmup`) live in ``mathlib`` -- NOT in the vendored, generic,
bit-identical :mod:`vio.comms` -- because they import VIO's own math.
"""
