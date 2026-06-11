#!/usr/bin/env python3
"""Self-test for the UNIFIED calibration status API
(:func:`imu_camera.mathlib.device.calib_status.calibration_status`).

Fully OFFLINE and CACHE-SAFE: it monkeypatches the three persisted-calib loaders
that :mod:`calib_status` imported (``load_gyro_bias`` / ``load_accel_calib`` /
``load_camera_calib``) so it simulates "calibrated" / "not calibrated" WITHOUT
touching the real ``.cache``. It asserts:

CAMERA is INFORMATIONAL: factory is the default, the user calib is opt-in via
``--use-camera-calib``, so the camera item is ALWAYS ``calibrated=True`` and NEVER
in ``missing`` -- its store presence only changes its DETAIL text. gyro/accel keep
the old semantics (they genuinely need calibration and still appear in ``missing``).

  (a) gyro+accel present (camera store empty) -> all_calibrated True, missing == [],
      every item ✓; camera detail = "using factory calib (default)",
  (b) gyro+accel absent (camera store empty) -> all_calibrated False,
      missing == [gyro, accel] (camera NOT listed despite empty store), camera ✓,
  (c) accel missing only -> missing == [accel]; camera ✓ (factory detail),
  (d) camera STORE present but gyro/accel still drive missing; camera detail flips
      to the "user calib saved" wording yet camera stays OUT of `missing`,
  (e) the API NEVER raises and the loaders receive the device_id.

Run::

    .venv/bin/python -m imu_camera.tests.calib_status_selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imu_camera.mathlib.device import calib_status as cs       # noqa: E402


# --------------------------------------------------------------------------- #
# Patch helpers: replace the three loaders in the calib_status namespace with
# stubs that return a sentinel "calib present" object or None. Each stub also
# records the device_id it was called with, so we can assert it's threaded through.
# --------------------------------------------------------------------------- #
def _install_loaders(gyro_ok: bool, accel_ok: bool, camera_ok: bool) -> dict:
    seen: dict[str, str] = {}

    def _mk(name: str, ok: bool):
        def _loader(device_id: str):
            seen[name] = device_id
            return object() if ok else None
        return _loader

    cs.load_gyro_bias = _mk("gyro", gyro_ok)        # type: ignore[assignment]
    cs.load_accel_calib = _mk("accel", accel_ok)    # type: ignore[assignment]
    cs.load_camera_calib = _mk("camera", camera_ok)  # type: ignore[assignment]
    return seen


def _by_name(status: dict) -> dict:
    return {it["name"]: it for it in status["items"]}


# --------------------------------------------------------------------------- #
def main() -> int:
    # Keep the originals so we restore them and never leave the module patched.
    orig = (cs.load_gyro_bias, cs.load_accel_calib, cs.load_camera_calib)
    try:
        # (a) gyro+accel present, camera store empty -> fully OK; camera ✓ with
        #     the factory-default detail (camera is never an error). ----------- #
        _install_loaders(True, True, False)
        s = calibration = cs.calibration_status("devA")
        assert s["all_calibrated"] is True, s
        assert s["missing"] == [], s
        assert [it["name"] for it in s["items"]] == ["gyro", "accel", "camera"], s
        items = _by_name(s)
        assert all(items[n]["calibrated"] for n in ("gyro", "accel", "camera"))
        assert items["gyro"]["detail"] == "bias cached"
        assert items["accel"]["detail"] == "6-position done"
        assert "factory" in items["camera"]["detail"], items["camera"]
        assert "default" in items["camera"]["detail"], items["camera"]
        assert s["device_id"] == "devA"
        print("[a] gyro+accel ok, no camera store -> all_calibrated, camera ✓     OK")
        del calibration

        # (b) gyro+accel absent, camera store empty -> missing = [gyro, accel]
        #     ONLY. Camera is ✓ + NOT in missing despite the empty store. ------ #
        seen = _install_loaders(False, False, False)
        s = cs.calibration_status("devB")
        assert s["all_calibrated"] is False, s
        assert s["missing"] == ["gyro", "accel"], s      # camera NOT listed
        items = _by_name(s)
        assert items["gyro"]["calibrated"] is False
        assert items["accel"]["calibrated"] is False
        assert items["camera"]["calibrated"] is True, items["camera"]
        # Missing-detail wording carries the auto-measure / accuracy-risk semantics.
        assert "auto-measured" in items["gyro"]["detail"], items["gyro"]
        assert "raw accel" in items["accel"]["detail"], items["accel"]
        assert "factory" in items["camera"]["detail"], items["camera"]
        # device_id threaded through to every loader.
        assert seen == {"gyro": "devB", "accel": "devB", "camera": "devB"}, seen
        print("[b] gyro+accel missing -> missing=[gyro,accel] (camera ✓, not in)  OK")

        # (c) accel missing only -> missing == [accel]; camera ✓ (factory). ---- #
        _install_loaders(True, False, False)
        s = cs.calibration_status("devC")
        assert s["all_calibrated"] is False, s
        assert s["missing"] == ["accel"], s
        items = _by_name(s)
        assert items["gyro"]["calibrated"] is True
        assert items["accel"]["calibrated"] is False
        assert items["camera"]["calibrated"] is True
        assert items["gyro"]["detail"] == "bias cached"
        assert "raw accel" in items["accel"]["detail"]
        assert "factory" in items["camera"]["detail"]
        print("[c] partial (accel missing) -> missing=[accel], camera ✓ factory   OK")

        # (d) camera STORE present -> camera detail flips to the "user calib
        #     saved" wording, but camera STILL stays out of `missing`; gyro
        #     drives the only missing item. Item order is preserved. ----------- #
        _install_loaders(False, True, True)
        s = cs.calibration_status("devD")
        assert [it["name"] for it in s["items"]] == ["gyro", "accel", "camera"], s
        assert s["missing"] == ["gyro"], s               # camera NOT in missing
        items = _by_name(s)
        assert items["camera"]["calibrated"] is True
        assert "user calib saved" in items["camera"]["detail"], items["camera"]
        assert "--use-camera-calib" in items["camera"]["detail"], items["camera"]
        print("[d] camera store present -> detail 'user calib saved', still ✓      OK")

        # (e) Non-str device_id is coerced; API never raises. ---------------- #
        _install_loaders(True, True, True)
        s = cs.calibration_status(12345)                 # type: ignore[arg-type]
        assert s["device_id"] == "12345", s
        print("[e] non-str device_id coerced to str; no raise               OK")
    finally:
        cs.load_gyro_bias, cs.load_accel_calib, cs.load_camera_calib = orig

    print("\nALL calib_status API CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
