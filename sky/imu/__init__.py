"""``sky.imu`` -- the shared LOOSE IMU preintegration (gyro prior + Forster preint).

This is the ONE canonical copy of the *loose* inertial preintegration math. It
used to be vendored byte-identically in ``imu_camera/mathlib/imu/imu.py`` and
``slam/mathlib/imu/imu.py``; the two copies were byte-for-byte identical, so
consolidating to a single import here removes the duplication outright.

* :mod:`sky.imu.imu` -- :func:`~sky.imu.imu.preintegrate_imu` (on-manifold
  rotation/velocity/position preintegration, Forster et al. TRO 2017) +
  :class:`~sky.imu.imu.GyroPreintegrator` / :func:`~sky.imu.imu.integrate_gyro_camera`
  (the cheap gyro-only rotation prior used to seed PnP) +
  :func:`~sky.imu.imu.gravity_aligned_R0` (level the first frame from accel).

It imports only :mod:`sky.math` (SO(3) exp / right-Jacobian / skew) and ``numpy``
-- no process / comms / io module -- so it stays a leaf and movable (maps onto
the C ``libskyimu`` layer in ``docs/C_PORT_PLAN.md``).

NOTE -- variant deferral: ``vio/mathlib/imu/imu.py`` is a SUPERSET of this loose
copy (it adds preintegration-covariance + bias-Jacobian machinery for the
tight-coupled VIO window optimiser, and is the live Phase-4 research surface).
That divergent variant is deliberately NOT consolidated here yet; ``vio`` keeps
its own ``imu.py`` until Phase 4 freezes (see ``docs/CONSOLIDATION_PLAN.md``).
"""
