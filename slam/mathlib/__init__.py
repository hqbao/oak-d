"""``slam.mathlib`` -- the math the SLAM project OWNS.

Ported VERBATIM from ``ours.lib.{loop,engine}`` plus the FORCED dependencies of
the loop-closure import graph (only the cross-package import roots + the doc
cross-references were re-rooted at ``slam.mathlib`` / ``slam.comms``; no algorithm
changed, so the numerical output is byte-identical to the reference oracle --
proved by :mod:`slam.tests.loop_closure_selftest`).

* :mod:`~slam.mathlib.loop` -- the loop-closure frontend + backend SLAM owns:
  the from-scratch ORB detector/descriptor + Hamming matcher + fundamental-matrix
  RANSAC (:mod:`~slam.mathlib.loop.orb`), the appearance + geometric loop detector
  (:mod:`~slam.mathlib.loop.loopclosure`), the SE(3) pose-graph optimiser
  (:mod:`~slam.mathlib.loop.posegraph`), and the persistent-keyframe SLAM map
  orchestrator (:mod:`~slam.mathlib.loop.slam`: ``SlamMap`` / ``SlamConfig``).
* :mod:`~slam.mathlib.engine` -- the swappable in-process / subprocess runners for
  the heavy keyframe optimiser (SLAM carries its OWN engine copy).

FORCED-VENDOR dependencies (resolved from the loop import graph; vendored at the
minimal surface):

* :mod:`~slam.mathlib.backend` -- the SE(3) ``se3_exp`` / ``skew`` Lie-group
  helpers, forced by :mod:`~slam.mathlib.loop.posegraph`.

The PnP RANSAC that :mod:`~slam.mathlib.loop.loopclosure`'s metric geometric
verification needs is the shared :func:`sky.front.pnp.solve_pnp_ransac` (one
canonical copy, deduped out of the old per-project ``mathlib/odometry/pnp.py``).

The ARCHITECTURE RULE lives here too: the math-coupled config builder
(:mod:`~slam.mathlib.resolution_build`) lives in ``mathlib`` -- NOT in the
vendored, generic, bit-identical :mod:`slam.comms` -- because it imports SLAM's
own math (:class:`~slam.mathlib.loop.loopclosure.LoopConfig`). SLAM has NO numba
JIT kernel (its ORB frontend is pure NumPy), so -- unlike VIO -- there is no
``warmup`` module to pre-compile.
"""
