"""``sky.math`` -- the in-tree pure-math kernel shared across the oak-d projects.

This is the math sub-package of the shared :mod:`sky` library (it was the
standalone ``skymath`` package, re-homed under ``sky`` so the project has ONE
common library rather than several siblings).

This package holds ONLY the small Lie-group / linear-algebra *primitives* that
were previously copy-pasted across ``imu_camera/``, ``vio/`` and ``slam/``
``mathlib/``. The algorithms that *call* them (SGM, KLT, PnP, bundle adjustment,
the IMU preintegration body, the pose-graph solver, ...) stay where they live;
only the primitives moved here, so every project now imports one canonical copy.

Design rules (the Lie-group kernel of the ``sky.*`` consolidation; maps onto the
C ``libskymath`` layer in ``docs/C_PORT_PLAN.md``):

* **Kernel only.** No project glue, no I/O, no comms, no cv2 / numba / heavy
  deps -- a bare ``import sky.math`` pulls in nothing but ``numpy``.
* **Byte-identical numerics.** Every function reproduces, bit-for-bit, the
  behaviour of the local ``def`` it replaced; the byte-parity oracle stays
  ``gap = 0``. Where the old copies had *genuine* numerical drift (different
  near-singularity handling), the variants are preserved under DISTINCT names
  rather than silently unified -- see ``so3`` / ``se3`` for the notes.

The original wire-contract quaternion helpers in ``*/comms/lib/misc/`` and the
Basalt reference tooling in ``baseline/`` are deliberately NOT routed through
this package: the comms copies are the byte-identical wire contract and must
stay vendored per project.
"""
from __future__ import annotations

from .se3 import (
    se3_adjoint,
    se3_exp,
    se3_exp_unit,
    se3_from_Rp,
    se3_inv,
    se3_log,
    se3_log_robust,
)
from .so3 import (
    skew,
    so3_exp,
    so3_exp_unit,
    so3_log,
    so3_log_robust,
    so3_right_jacobian,
)

__all__ = [
    # SO(3)
    "skew",
    "so3_exp",
    "so3_exp_unit",
    "so3_log",
    "so3_log_robust",
    "so3_right_jacobian",
    # SE(3)
    "se3_from_Rp",
    "se3_inv",
    "se3_adjoint",
    "se3_exp",
    "se3_exp_unit",
    "se3_log",
    "se3_log_robust",
]
