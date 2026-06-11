"""``sky`` -- the ONE shared algorithm library for the oak-d projects.

The five processes (``imu_camera`` -> ``depth`` -> ``vio`` -> ``slam`` -> ``ui``)
used to copy-paste their algorithm code into per-project ``mathlib/`` trees. This
package is the single home those algorithms are being consolidated into, one
domain at a time, each step gated on the byte-parity oracle staying ``gap = 0``
(see ``docs/CONSOLIDATION_PLAN.md``). It is the Python precursor to the C
``libsky*`` layering in ``docs/C_PORT_PLAN.md``.

Sub-packages
------------
* :mod:`sky.math`  -- Lie-group / linear-algebra primitives (SO(3)/SE(3)); was
  the standalone ``skymath`` package, re-homed here.
* :mod:`sky.depth` -- the from-scratch SGM dense-stereo matcher + rectifiers
  (one canonical copy, shared by ``imu_camera`` and ``depth``).

Movability rule (enforced -- see :func:`assert_import_clean`)
-----------------------------------------------------------
``sky.*`` MUST stay a leaf library: it may import only ``numpy`` and -- where the
moved algorithm already used them -- ``cv2`` / ``numba``. It must NEVER import any
of the *processes* (``imu_camera``, ``depth``, ``vio``, ``slam``, ``ui``,
``launcher``, ``verification``, ``baseline``) nor their ``comms`` / ``io``
layers. Keeping the dependency arrow pointing only INTO ``sky`` is what makes the
package movable into its own repo and portable to C.
"""
from __future__ import annotations

import sys

#: Top-level modules a clean ``sky.*`` import is allowed to pull in. ``numpy`` is
#: the kernel dependency; ``cv2`` / ``numba`` are permitted only because the
#: already-moved algorithm code (SGM stereo) genuinely uses them. Anything else
#: from the third-party world is fine too -- the rule we actually police is the
#: NEGATIVE one below (no oak-d process / comms / io may be reachable).
ALLOWED_THIRD_PARTY = frozenset({"numpy", "cv2", "numba"})

#: Top-level oak-d packages that ``sky.*`` must never depend on. Importing any of
#: these from inside ``sky`` would invert the dependency arrow and make the
#: library un-movable / un-portable.
FORBIDDEN_PACKAGES = frozenset({
    "imu_camera", "depth", "vio", "slam", "ui", "launcher",
    "verification", "baseline",
})


def assert_import_clean() -> None:
    """Fail if importing ``sky`` dragged in any forbidden oak-d package.

    Call this from a fresh interpreter that has done nothing but ``import
    sky`` (and the ``sky.*`` sub-modules under test). It scans
    ``sys.modules`` for any top-level package in :data:`FORBIDDEN_PACKAGES`
    and raises ``AssertionError`` listing the offenders. This is the
    executable form of the movability rule documented above; the consolidation
    self-tests run it after a bare ``import sky.*`` to keep the package a leaf.
    """
    leaked = sorted(
        name for name in sys.modules
        if name.split(".", 1)[0] in FORBIDDEN_PACKAGES
    )
    assert not leaked, (
        "sky.* must not import any oak-d process/comms/io module; "
        f"these leaked into sys.modules: {leaked}"
    )
