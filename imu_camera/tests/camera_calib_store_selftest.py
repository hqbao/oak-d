#!/usr/bin/env python3
"""Self-test for the per-device CAMERA-calib store + the live override selection.

Two halves, both fully OFFLINE (no OAK-D, no depthai) and writing ONLY to a tmp
cache path so the real ``.cache/camera_calib.json`` is never touched:

1. STORE round-trip
   (:func:`~imu_camera.device.camera_calib_store.save_camera_calib` /
   :func:`load_camera_calib`):
     (a) save a calib dict for a device, reload it -> a ``StereoCalib`` whose
         intrinsics + baseline match the source dict,
     (b) a DIFFERENT device id returns ``None`` (no cross-device clobber),
     (c) a MISSING file returns ``None`` (clean fall-back, no crash),
     (d) a CORRUPT file returns ``None`` (clean fall-back, no crash).

2. OVERRIDE selection
   (:func:`~imu_camera.device.live_calib.select_camera_calib` -- the live
   decision factored out so it runs HEADLESS with a stubbed device-calib read).
   FACTORY is the DEFAULT; the user calib is opt-in via ``use_camera_calib``:
     (e) flag OFF (default) + a stored calib present -> still FACTORY (the store is
         ignored) and NO warning,
     (f) flag ON + a stored calib present -> the result uses the STORED K +
         StereoCalib, NOT the factory one, logged "factory overridden",
     (g) flag ON + no stored calib -> factory K + StereoCalib AND the prominent
         "asked for user camera calib ... but none saved" warning to stderr.

Run::

    .venv/bin/python -m imu_camera.tests.camera_calib_store_selftest
"""
from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imu_camera.io.reader import StereoCalib                  # noqa: E402
from imu_camera.device.camera_calib_store import (    # noqa: E402
    load_camera_calib, save_camera_calib)
from imu_camera.device.live_calib import select_camera_calib  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic calib dicts in the EXACT schema StereoCalib.from_json consumes
# (translation in CENTIMETRES, the depthai/writer convention).
# --------------------------------------------------------------------------- #
def _intr(fx, fy, cx, cy, w=640, h=400) -> dict:
    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "dist": [0.0] * 8, "width": int(w), "height": int(h)}


def _calib_dict(fx, baseline_mm) -> dict:
    """A calib.json dict with the given left fx and L->R baseline (mm).

    ``T_left_right`` translation is in CENTIMETRES (from_json multiplies by 0.01),
    so ``baseline_mm`` mm = ``baseline_mm / 10`` cm along -X.
    """
    T = np.eye(4)
    T[0, 3] = -(baseline_mm / 10.0)            # mm -> cm, right cam to the right
    return {
        "intrinsics_left": _intr(fx, fx, 320.0, 200.0),
        "intrinsics_right": _intr(fx, fx, 320.0, 200.0),
        "T_left_right": T.tolist(),
    }


# --------------------------------------------------------------------------- #
# 1. Store round-trip.
# --------------------------------------------------------------------------- #
def test_store_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "camera_calib.json"
        dev = "device-A"
        src = _calib_dict(fx=300.0, baseline_mm=75.0)

        # (a) save -> load returns a StereoCalib whose intrinsics + baseline match.
        out = save_camera_calib(dev, src, path=path)
        assert out == path and path.exists(), out
        calib = load_camera_calib(dev, path=path)
        assert isinstance(calib, StereoCalib), type(calib)
        assert abs(calib.left.fx - 300.0) < 1e-9, calib.left.fx
        assert abs(calib.left.cx - 320.0) < 1e-9, calib.left.cx
        assert abs(calib.baseline_m - 0.075) < 1e-9, calib.baseline_m

        # (b) a different device id returns None (no cross-device clobber).
        assert load_camera_calib("device-B", path=path) is None

        # A second device coexists without overwriting the first.
        save_camera_calib("device-B", _calib_dict(fx=280.0, baseline_mm=60.0),
                          path=path)
        a = load_camera_calib(dev, path=path)
        b = load_camera_calib("device-B", path=path)
        assert abs(a.left.fx - 300.0) < 1e-9 and abs(b.left.fx - 280.0) < 1e-9
        assert abs(b.baseline_m - 0.060) < 1e-9, b.baseline_m
        print(f"[ok] store round-trip: fx={calib.left.fx:.1f}, baseline="
              f"{calib.baseline_m * 1000:.1f} mm; 2 devices coexist; other id -> None")

        # (c) missing file -> None (no crash).
        missing = Path(tmp) / "does_not_exist.json"
        assert load_camera_calib(dev, path=missing) is None

        # (d) corrupt file -> None (no crash).
        corrupt = Path(tmp) / "corrupt.json"
        corrupt.write_text("{ this is not valid json ]]")
        assert load_camera_calib(dev, path=corrupt) is None
        print("[ok] store robustness: missing file -> None, corrupt file -> None "
              "(no crash, clean fall-back to factory)")


# --------------------------------------------------------------------------- #
# 2. Override selection (factored decision, headless).
# --------------------------------------------------------------------------- #
def test_flag_off_keeps_factory_even_with_stored_calib() -> None:
    factory = StereoCalib.from_json(_calib_dict(fx=285.0, baseline_mm=75.0))
    factory_K = factory.left.K
    user = StereoCalib.from_json(_calib_dict(fx=311.0, baseline_mm=68.0))

    # (e) DEFAULT (flag off): factory even though a stored calib is supplied, and
    # NO warning (factory is the intended default, not an error). The caller would
    # normally pass user_calib=None when the flag is off, but pass the user calib
    # here to prove the gate ignores it regardless.
    buf = io.StringIO()
    with redirect_stderr(buf):
        K, calib = select_camera_calib("dev-D", factory_K, factory, user,
                                       use_camera_calib=False)
    assert calib is factory, "default must return the FACTORY StereoCalib"
    assert K is factory_K, "default must return the FACTORY K"
    log = buf.getvalue()
    assert log.strip() == "", f"default path must be silent, got: {log!r}"
    print(f"[ok] flag OFF (default): factory kept (fx={K[0, 0]:.1f}, not user 311) "
          f"despite a stored calib; no warning")


def test_flag_on_uses_stored_calib() -> None:
    factory = StereoCalib.from_json(_calib_dict(fx=285.0, baseline_mm=75.0))
    factory_K = factory.left.K
    user = StereoCalib.from_json(_calib_dict(fx=311.0, baseline_mm=68.0))

    # (f) flag ON + stored calib present -> result uses the STORED K + calib.
    buf = io.StringIO()
    with redirect_stderr(buf):
        K, calib = select_camera_calib("dev-X", factory_K, factory, user,
                                       use_camera_calib=True)
    assert calib is user, "override must return the USER StereoCalib object"
    assert abs(K[0, 0] - 311.0) < 1e-9, f"K must be the user's fx: {K[0, 0]}"
    assert abs(calib.baseline_m - 0.068) < 1e-9, calib.baseline_m
    log = buf.getvalue()
    assert "using SAVED camera calibration" in log and "overridden" in log, log
    assert "68.0 mm" in log, log
    print(f"[ok] flag ON + stored: K fx={K[0, 0]:.1f} (user, not factory 285), "
          f"baseline={calib.baseline_m * 1000:.1f} mm; log={log.strip()!r}")


def test_flag_on_no_stored_calib_warns_and_uses_factory() -> None:
    factory = StereoCalib.from_json(_calib_dict(fx=285.0, baseline_mm=75.0))
    factory_K = factory.left.K

    # (g) flag ON but no stored calib -> factory K + calib AND a warning naming the
    # flag to stderr so the operator knows their request could not be honoured.
    buf = io.StringIO()
    with redirect_stderr(buf):
        K, calib = select_camera_calib("dev-Y", factory_K, factory, None,
                                       use_camera_calib=True)
    assert calib is factory, "fall-back must return the FACTORY StereoCalib"
    assert K is factory_K, "fall-back must return the FACTORY K"
    log = buf.getvalue()
    assert "asked for user camera calib" in log, log
    assert "--use-camera-calib" in log, log
    assert "none saved" in log, log
    assert "using factory" in log, log
    assert "dev-Y" in log, log
    print(f"[ok] flag ON + no stored: factory kept + flag-named warning emitted; "
          f"log={log.strip()!r}")


def main() -> int:
    test_store_round_trip()
    test_flag_off_keeps_factory_even_with_stored_calib()
    test_flag_on_uses_stored_calib()
    test_flag_on_no_stored_calib_warns_and_uses_factory()
    print("\nALL CAMERA CALIB STORE + OVERRIDE SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
