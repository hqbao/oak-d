"""Pre-compile VIO's Numba JIT kernel (KLT) off the critical path.

The first call to the KLT inner-loop ``@njit`` kernel pays a one-time LLVM
compile (~1-2 s on a cold cache); on the first frame that stalls the pipeline
while capture is already streaming. :func:`warmup_klt` triggers that compile with
a tiny synthetic pair on a background thread at boot, overlapping it with the
calib handshake + the first frames, so the frame path is already machine code.

VIO owns ONLY the KLT frontend -- it CONSUMES ``frame.depth`` from capture and so
does NOT run SGM (that is the ``imu_camera`` project, which warms its own SGM
kernel). This is the KLT half of the pre-split ``ours.lib.misc.warmup.warmup_jit``.
It NEVER changes results; any failure (numba absent, etc.) is swallowed so it can
never break a run.
"""
from __future__ import annotations

import numpy as np

from vio.comms.runtime import NUMBA_PARALLEL_LOCK


def warmup_klt(klt_cfg=None) -> bool:
    """Compile the KLT numba kernel via a tiny dummy call.

    ``klt_cfg`` should be the config the live path will use so the compiled type
    signatures + code paths match (numba specialises per argument *type*, which
    this tiny call reproduces; the config *values* only need to exercise the same
    code paths). Defaults to the library default. Returns ``True`` if it
    compiled, ``False`` if numba is unavailable or anything went wrong (always
    safe -- the real path then compiles on the first frame exactly as before).
    """
    try:
        from sky.front.klt_numba import HAVE_NUMBA
        if not HAVE_NUMBA:
            return False
        from sky.front.frontend import FrontendConfig
        from sky.front.klt import calc_optical_flow_pyr_lk

        klt = klt_cfg or FrontendConfig()

        # A small textured pair so the KLT solver actually runs its inner loops
        # (a flat image would short-circuit before the kernel).
        rng = np.random.default_rng(0)
        left = rng.integers(0, 255, (64, 96)).astype(np.float32)
        right = np.roll(left, -2, axis=1).astype(np.float32)
        pts = np.array([[30.0, 30.0], [50.0, 25.0]], dtype=np.float32)

        with NUMBA_PARALLEL_LOCK:
            calc_optical_flow_pyr_lk(
                left, right, pts,
                win_size=int(klt.win_size), max_level=int(klt.max_level))
        return True
    except Exception:                                    # noqa: BLE001
        return False
