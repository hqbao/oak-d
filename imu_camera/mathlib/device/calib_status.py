"""UNIFIED, device-agnostic calibration status -- the one place that answers
"is this device calibrated?" for ALL three calibrations at once.

Today the operator calibrates gyro, accel, and stereo camera through three SEPARATE
wizards, each with its own per-device store and its own ad-hoc "is it done?" check.
This module folds those three checks into ONE structured query so the UI can show a
single status view + nag the operator about whatever is still missing (flying
uncalibrated = inaccurate).

Device-agnostic + UI-importable (no cv2, no depthai)
----------------------------------------------------
This pulls in ONLY the three persisted-calib LOADERS, each of which is itself
cv2-free and depthai-free:

* :func:`imu_camera.mathlib.imu.calib_store.load_gyro_bias`
* :func:`imu_camera.mathlib.imu.calib_store.load_accel_calib`
* :func:`imu_camera.mathlib.device.camera_calib_store.load_camera_calib`

So the multi-chip-generic UI can import :func:`calibration_status` directly, keyed by
the abstract ``device_id`` -- no OAK-D / depthai specifics ever reach this file. The
loaders never raise on a missing/corrupt cache (they return ``None``), so this query
never raises either: an unknown device is simply "nothing calibrated".

The three items' semantics (be accurate -- the operator acts on these strings)
------------------------------------------------------------------------------
* **gyro** -- ``calibrated`` iff a bias is cached. When MISSING this is NOT a hard
  blocker: capture AUTO-measures the gyro bias from the first still samples on every
  capture start, so an uncalibrated gyro self-corrects shortly after launch (the
  cached value just skips that warm-up). Detail says so.
* **accel** -- ``calibrated`` iff the six-position affine is cached. When MISSING the
  pipeline falls back to RAW accel for leveling, which biases the gravity / attitude
  reference -- a real accuracy hit, no auto-fix. Detail says so.
* **camera** -- INFORMATIONAL, never a "missing" item. The live pipeline uses the
  trusted FACTORY calibration by DEFAULT; the operator's own stereo solve is applied
  only when running with ``--use-camera-calib``. So an empty store is a perfectly
  valid, non-error state (factory is the intended default) and the camera item is
  ALWAYS reported ``calibrated=True`` -- it never lands in ``missing`` and never nags.
  Its ``detail`` just tells the operator which calib is in play: "user calib saved
  (enable with --use-camera-calib)" when a store entry exists, else "using factory
  calib (default)".
"""
from __future__ import annotations

from imu_camera.mathlib.device.camera_calib_store import load_camera_calib
from imu_camera.mathlib.imu.calib_store import load_accel_calib, load_gyro_bias

# Stable item keys/names. The dialog renders rows in THIS order (gyro, accel,
# camera), and ``missing`` preserves it, so the operator always sees the same layout.
GYRO = "gyro"
ACCEL = "accel"
CAMERA = "camera"

# Detail strings, split done/missing so callers never have to compose them. Kept
# here (not in the UI) so the semantics live with the loaders they describe.
_DETAIL_GYRO_DONE = "bias cached"
_DETAIL_GYRO_MISSING = "not yet measured — auto-measured on first capture start"
_DETAIL_ACCEL_DONE = "6-position done"
_DETAIL_ACCEL_MISSING = "not calibrated — leveling uses raw accel (attitude drift)"
# Camera is informational (never "missing"): factory is the default, a saved user
# calib is opt-in via --use-camera-calib. Both details describe a VALID state.
_DETAIL_CAMERA_STORED = "user calib saved (enable with --use-camera-calib)"
_DETAIL_CAMERA_FACTORY = "using factory calib (default)"


def _item(name: str, calibrated: bool, detail: str) -> dict:
    """One status row: name + calibrated flag + a human-readable detail string."""
    return {"name": name, "calibrated": bool(calibrated), "detail": detail}


def calibration_status(device_id: str) -> dict:
    """Return the unified calibration status for ``device_id``.

    Re-reads the three on-disk caches on EVERY call (cheap -- three tiny JSON
    reads), so a caller that re-queries after a wizard finishes immediately sees the
    fresh result. Never raises: the loaders treat a missing/corrupt cache as "not
    calibrated".

    Returns a dict::

        {
            "device_id": <str>,
            "items": [ {"name", "calibrated": bool, "detail": str}, ... ],  # 3 rows
            "all_calibrated": bool,        # every item calibrated
            "missing": [<name>, ...],      # names of the uncalibrated items, in order
        }

    The three ``items`` are always present and always in (gyro, accel, camera)
    order.
    """
    dev = str(device_id)

    gyro_ok = load_gyro_bias(dev) is not None
    accel_ok = load_accel_calib(dev) is not None
    # Whether the operator has a saved stereo calib in the store. This only
    # influences the camera item's DETAIL text -- the camera item is ALWAYS reported
    # calibrated (informational, never "missing") because factory is a valid default.
    camera_stored = load_camera_calib(dev) is not None

    items = [
        _item(GYRO, gyro_ok,
              _DETAIL_GYRO_DONE if gyro_ok else _DETAIL_GYRO_MISSING),
        _item(ACCEL, accel_ok,
              _DETAIL_ACCEL_DONE if accel_ok else _DETAIL_ACCEL_MISSING),
        # Camera is informational: calibrated=True ALWAYS so it never enters
        # `missing` and never triggers the startup nag. The detail just reports
        # which calib the live pipeline will use (factory by default, user calib
        # only with --use-camera-calib).
        _item(CAMERA, True,
              _DETAIL_CAMERA_STORED if camera_stored else _DETAIL_CAMERA_FACTORY),
    ]
    missing = [it["name"] for it in items if not it["calibrated"]]
    return {
        "device_id": dev,
        "items": items,
        "all_calibrated": not missing,
        "missing": missing,
    }
