"""``sky.front`` -- the shared visual front-end geometry (PnP today; KLT/corners later).

This is the ONE canonical copy of the from-scratch, library-free pose solvers the
front-end needs. It used to be vendored byte-identically in two places --
``vio/mathlib/odometry/pnp.py`` (consumed by the frame-to-frame RGB-D odometry)
and ``slam/mathlib/odometry/pnp.py`` (forced-vendored for the loop-closure metric
verification). The two copies were byte-for-byte identical, so consolidating to a
single import here removes the duplication outright.

* :mod:`sky.front.pnp` -- :func:`~sky.front.pnp.solve_pnp_ransac`, a pure-NumPy
  drop-in for the subset of ``cv2.solvePnPRansac`` the from-scratch VIO/SLAM need
  (RANSAC over minimal-point DLT hypotheses + robust Gauss-Newton refinement).

It imports only :mod:`sky.math` (the SO(3) exp helper) and ``numpy`` -- no
process / comms / io module -- so it stays a leaf and movable (maps onto the C
``libskyfront`` layer in ``docs/C_PORT_PLAN.md``).
"""
