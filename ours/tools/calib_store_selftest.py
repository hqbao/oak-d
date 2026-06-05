"""Self-test for the unified IMU calibration store (gyro bias + accel calib).

Exercises the on-disk persistence the operator relies on between flights:
save/load round-trips, multi-device isolation, corrupt-file tolerance, the two
legacy-format migrations (gyro-only file + gyro-at-top-level entry), and that the
``bias_store`` compatibility shim still resolves. All offline, no hardware.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.imu import bias_store, calib_store  # noqa: E402
from ours.lib.imu.accel_calib import (  # noqa: E402
    G_STANDARD,
    AccelCalibration,
    solve_accel_calibration,
)


def _sample_accel_cal() -> AccelCalibration:
    faces = np.array([
        [+1, 0, 0], [-1, 0, 0], [0, +1, 0],
        [0, -1, 0], [0, 0, +1], [0, 0, -1]], dtype=float)
    M = np.array([[1.018, 0.012, -0.009],
                  [-0.007, 0.982, 0.015],
                  [0.010, -0.013, 1.005]])
    b = np.array([0.21, -0.15, 0.08])
    caps = [M @ (f * G_STANDARD) + b for f in faces]
    return solve_accel_calibration(caps)


def main() -> int:
    ok = True
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "imu_calib.json"

    # --- gyro round-trip + multi-device isolation -------------------------
    bA = np.array([0.01, -0.02, 0.003])
    assert calib_store.load_gyro_bias("devA", p) is None
    calib_store.save_gyro_bias("devA", bA, 137, p)
    calib_store.save_gyro_bias("devB", np.array([1., 2., 3.]), 10, p)
    gotA = calib_store.load_gyro_bias("devA", p)
    ok &= gotA is not None and np.allclose(gotA, bA)
    ok &= np.allclose(calib_store.load_gyro_bias("devB", p), [1, 2, 3])
    print(f"gyro round-trip + isolation: {'OK' if ok else 'FAIL'}")

    # --- accel round-trip; coexists with gyro in the same entry -----------
    cal = _sample_accel_cal()
    assert calib_store.load_accel_calib("devA", p) is None
    calib_store.save_accel_calib("devA", cal, 6, p)
    got = calib_store.load_accel_calib("devA", p)
    ok_acc = got is not None and np.allclose(got.T, cal.T) and \
        np.allclose(got.bias, cal.bias)
    # saving accel must not wipe the gyro bias already stored for the device
    ok_acc &= np.allclose(calib_store.load_gyro_bias("devA", p), bA)
    # and vice-versa: re-saving gyro keeps the accel calibration
    calib_store.save_gyro_bias("devA", bA * 2, 50, p)
    ok_acc &= calib_store.load_accel_calib("devA", p) is not None
    print(f"accel round-trip + coexistence: {'OK' if ok_acc else 'FAIL'}")
    ok &= ok_acc

    # --- corrupt file -> None, no crash -----------------------------------
    p.write_text("{ not valid json")
    ok_corrupt = (calib_store.load_gyro_bias("devA", p) is None
                  and calib_store.load_accel_calib("devA", p) is None)
    print(f"corrupt-file tolerance: {'OK' if ok_corrupt else 'FAIL'}")
    ok &= ok_corrupt

    # --- legacy migration 1: gyro-at-top-level entry shape ----------------
    p.write_text(json.dumps({"devLegacy": {"bias": [0.1, 0.2, 0.3],
                                           "n": 99, "ts": 1.0}}))
    leg = calib_store.load_gyro_bias("devLegacy", p)
    ok_leg1 = leg is not None and np.allclose(leg, [0.1, 0.2, 0.3])
    print(f"legacy top-level-bias migration: {'OK' if ok_leg1 else 'FAIL'}")
    ok &= ok_leg1

    # --- legacy migration 2: auto-read the old imu_bias.json filename -----
    tmp2 = Path(tempfile.mkdtemp())
    # point the store at <tmp2>/imu_calib.json but only write the legacy file
    new_path = tmp2 / "imu_calib.json"
    legacy_path = tmp2 / "imu_bias.json"
    legacy_path.write_text(json.dumps({"devX": {"bias": [0.4, 0.5, 0.6],
                                                "n": 5, "ts": 2.0}}))
    # monkeypatch the module default so _load_all's legacy fallback triggers
    orig_default, orig_legacy = calib_store._DEFAULT_PATH, calib_store._LEGACY_PATH
    calib_store._DEFAULT_PATH = new_path
    calib_store._LEGACY_PATH = legacy_path
    try:
        leg2 = calib_store.load_gyro_bias("devX", new_path)
    finally:
        calib_store._DEFAULT_PATH, calib_store._LEGACY_PATH = orig_default, orig_legacy
    ok_leg2 = leg2 is not None and np.allclose(leg2, [0.4, 0.5, 0.6])
    print(f"legacy imu_bias.json auto-read: {'OK' if ok_leg2 else 'FAIL'}")
    ok &= ok_leg2

    # --- compatibility shim still resolves --------------------------------
    ok_shim = (bias_store.load_gyro_bias is calib_store.load_gyro_bias
               and bias_store.save_gyro_bias is calib_store.save_gyro_bias)
    print(f"bias_store shim: {'OK' if ok_shim else 'FAIL'}")
    ok &= ok_shim

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
