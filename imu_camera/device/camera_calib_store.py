"""Persisted per-device STEREO CAMERA calibration (the operator's own calib).

The calibration WIZARD (:mod:`ui.qt.camera_calib_dialog`) lets the operator solve
their OWN stereo intrinsics + left->right extrinsic. This module persists that solve
so the LIVE pipeline can auto-apply it on the next capture start instead of falling
back to the OAK-D FACTORY calibration -- exactly mirroring how
:mod:`sky.sensors.calib_store` persists the per-device IMU calibration.

Why a SEPARATE module from the IMU ``calib_store``
--------------------------------------------------
The IMU store lives in the shared :mod:`sky.sensors` library because it stores
IMU-domain data (gyro bias + accel affine) and depends on
:class:`~sky.sensors.accel_calib.AccelCalibration`. This store keeps CAMERA-domain
data (a :class:`~imu_camera.io.reader.StereoCalib`), so it belongs next to its sole
live consumer, :mod:`live_calib`, in this project's ``device/`` layer.

Device-agnostic by construction (so the UI can import it)
---------------------------------------------------------
This module pulls in ONLY ``json`` + :class:`~imu_camera.io.reader.StereoCalib`
(which itself imports no depthai). So the cv2-free, depthai-free UI can import it to
SAVE a wizard solve, keyed by the abstract ``device_id`` -- the same allowed
dependency the UI calib dialogs already take on the IMU ``calib_store``. No OAK-D /
depthai specifics ever reach this file, keeping the UI multi-chip-generic.

On-disk shape -- one tiny JSON under the (gitignored) repo ``.cache`` dir, keyed by
device id so several cameras never clobber each other::

    {"<device_id>": {
        "calib": { ... the wizard's calib.json dict ... },
        "ts": 1234567890.0
    }}

The stored ``calib`` dict is EXACTLY the schema
:meth:`imu_camera.io.reader.StereoCalib.from_json` consumes (translation in
centimetres, the depthai convention -- see :mod:`sky.calib.writer`), so a
saved wizard solve loads byte-compatibly into the live pipeline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from imu_camera.io.reader import StereoCalib

# Repo-root/.cache/camera_calib.json (.cache is gitignored). This file is
# imu_camera/device/camera_calib_store.py, so parents[2] is the repo root
# (matching the IMU store's parents[2] from sky/sensors/calib_store.py).
_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
_DEFAULT_PATH = _CACHE_DIR / "camera_calib.json"


def default_path() -> Path:
    """Where the camera calibration cache lives (repo ``.cache/camera_calib.json``)."""
    return _DEFAULT_PATH


def _load_all(path: Path) -> dict:
    """Load the whole cache dict, or ``{}`` if absent/corrupt (never raises)."""
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_all(path: Path, data: dict) -> Path:
    """Atomically write the whole cache dict (mirrors the IMU store's ``_save_all``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(path)        # atomic on POSIX -> never a half-written cache
    return path


def _entry(data: dict, device_id: str) -> dict:
    e = data.get(str(device_id))
    return e if isinstance(e, dict) else {}


def save_camera_calib(device_id: str, calib_dict: dict,
                      path: Path | None = None) -> Path:
    """Persist the wizard's calib.json ``dict`` for ``device_id`` (merges into the file).

    ``calib_dict`` is the schema :meth:`StereoCalib.from_json` consumes (built by
    :func:`sky.calib.writer.calib_to_dict`) -- stored verbatim so the live
    pipeline reloads it byte-identically.
    """
    p = path or _DEFAULT_PATH
    data = _load_all(p)
    data[str(device_id)] = {"calib": dict(calib_dict), "ts": time.time()}
    return _save_all(p, data)


def load_camera_calib(device_id: str,
                      path: Path | None = None) -> StereoCalib | None:
    """Return the saved :class:`StereoCalib` for ``device_id`` or ``None``.

    Returns ``None`` when there is no saved calib for this device, or when the file
    is absent/corrupt or the stored dict does not parse -- never raises, so a missing
    or damaged cache is a clean "fall back to factory", not a crash on the live path.
    """
    e = _entry(_load_all(path or _DEFAULT_PATH), device_id)
    calib = e.get("calib")
    if not isinstance(calib, dict):
        return None
    try:
        return StereoCalib.from_json(calib)
    except (KeyError, ValueError, TypeError):
        return None
