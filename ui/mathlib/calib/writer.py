"""calib.json writer (Phase 3 -- the calibration math core), round-trip safe.

Serialises a :class:`ui.mathlib.calib.solve.StereoCalibResult` to the exact JSON
schema that :meth:`imu_camera.io.reader.StereoCalib.from_json` consumes, so a calib
produced by the wizard loads byte-compatibly into the live pipeline.

CRITICAL -- TRANSLATION UNITS (the cm/m trap)
---------------------------------------------
``StereoCalib.from_json`` does ``T[:3,3] *= 0.01`` on load: it expects the JSON
``T_left_right`` translation in **centimetres** (depthai's convention) and converts
to metres. The solve produces the translation in **metres**, so this writer
MULTIPLIES the translation by 100 (m -> cm) before writing. Skip that and the
parsed baseline comes back 100x too small (e.g. 7.5 cm -> 0.075 cm). The self-test
round-trips ``write -> StereoCalib.from_json`` and asserts the reloaded baseline
equals the solved baseline to within 1e-6 m.

The rotation block of ``T_left_right`` is dimensionless and written unchanged.

SCHEMA written
--------------
::

    {
      "intrinsics_left":  {fx, fy, cx, cy, K (3x3), dist (list), width, height},
      "intrinsics_right": { ... same ... },
      "T_left_right": 4x4 list  # rotation as-is; translation in CENTIMETRES
    }

cv2 POLICY
----------
This module is pure-Python (json + numpy) and imports NO cv2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .solve import StereoCalibResult


def _intrinsics_dict(K: np.ndarray, dist: np.ndarray,
                     image_size: tuple[int, int]) -> dict:
    """Build one camera's intrinsics block for the calib JSON.

    ``image_size`` is ``(width, height)`` (cv2 convention). ``K`` is written both as
    the flat ``fx/fy/cx/cy`` fields the loader reads AND as the full 3x3 ``K`` list
    (kept for downstream tools / human inspection; the loader rebuilds K from the
    flat fields, so the two are guaranteed consistent here).
    """
    K = np.asarray(K, dtype=np.float64)
    w, h = image_size
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "K": K.tolist(),
        "dist": np.asarray(dist, dtype=np.float64).ravel().tolist(),
        "width": int(w),
        "height": int(h),
    }


def calib_to_dict(result: StereoCalibResult,
                  image_size: tuple[int, int]) -> dict:
    """Convert a solved result to the calib-JSON dict (translation in CENTIMETRES).

    Factored out of :func:`write_calib_json` so callers/tests can round-trip the
    structure in memory without touching the filesystem.
    """
    # T_left_right: rotation as-is, translation m -> cm so the loader's *=0.01
    # restores metres. This is THE round-trip-critical line.
    T_cm = result.T_left_right.copy()
    T_cm[:3, 3] *= 100.0  # metres -> centimetres (undone by from_json's *0.01)

    return {
        "intrinsics_left": _intrinsics_dict(result.K_l, result.dist_l, image_size),
        "intrinsics_right": _intrinsics_dict(result.K_r, result.dist_r, image_size),
        "T_left_right": T_cm.tolist(),
    }


def write_calib_json(result: StereoCalibResult,
                     image_size: tuple[int, int],
                     path: str | Path) -> Path:
    """Write a solved stereo calibration to ``path`` as reader-compatible calib.json.

    Parameters
    ----------
    result:
        The solved :class:`~ui.mathlib.calib.solve.StereoCalibResult`.
    image_size:
        ``(width, height)`` in pixels (cv2 convention).
    path:
        Destination file. Parent directories are created if missing.

    Returns
    -------
    pathlib.Path
        The resolved path written, so callers can chain (e.g. into calib_check).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calib_to_dict(result, image_size), indent=2))
    return out.resolve()
