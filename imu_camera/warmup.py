"""Pre-compile imu_camera's Numba JIT kernel (SGM) off the critical path.

The first call to the SGM census/cost/aggregation/WTA ``@njit`` kernels pays a
one-time LLVM compile (~1-2 s on a cold cache); on the first live frame that
stalls the viewer while the OAK-D is already streaming. :func:`warmup_sgm`
triggers that compile with a tiny synthetic pair on a background thread at
device-open time, overlapping it with the OAK-D boot + the startup IMU
still-window, so frame one is already machine code.

imu_camera owns ONLY the stereo/SGM kernel — it does NOT run KLT (that is the
``vio`` project, which warms its own KLT kernel). So this warms SGM only. It
NEVER changes results; any failure (numba absent, etc.) is swallowed so it can
never break a run.
"""
from __future__ import annotations

import numpy as np

from imu_camera.comms.runtime import NUMBA_PARALLEL_LOCK


def warmup_sgm(sgm_cfg=None) -> bool:
    """Compile the SGM numba kernel via a tiny dummy call.

    ``sgm_cfg`` should be the config the live path will use so the compiled type
    signatures + code paths (e.g. SGM downscale) match; defaults to the library
    default. Returns ``True`` if it compiled, ``False`` if numba is unavailable
    or anything went wrong (always safe — the real path then compiles on frame
    one exactly as before).
    """
    try:
        from sky.depth.stereo import SGMConfig, sgm_disparity
        sgm = sgm_cfg or SGMConfig()
        # A small textured pair so the matcher actually runs its inner loops (a
        # flat image would short-circuit before the kernel).
        rng = np.random.default_rng(0)
        left = rng.integers(0, 255, (64, 96)).astype(np.float32)
        right = np.roll(left, -2, axis=1).astype(np.float32)
        with NUMBA_PARALLEL_LOCK:
            sgm_disparity(left, right, sgm)
        return True
    except Exception:                                    # noqa: BLE001
        return False
