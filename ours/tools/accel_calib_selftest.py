"""Self-test for the six-position accelerometer calibration solver.

Strategy: invent a *known* sensor distortion (bias + scale + misalignment),
synthesise the raw readings the six tumble poses would produce through it, run
the solver blind, and check it both (a) drives every corrected pose back onto
the gravity sphere and (b) recovers a correction that UNDOES the planted
distortion on independent test orientations. A sign/convention or Jacobian bug
shows up here, offline, instead of as a tilted world frame on the device.

Because the calibrated frame's absolute orientation is a free gauge (any global
rotation of ``T`` still satisfies ``|a_cal| = g``), we do NOT compare ``T``
entry-by-entry against the planted matrix. We compare the *physically meaningful*
quantity instead: the corrected magnitude must be ``g`` and the corrected vector
must have the right direction up to one fixed rotation common to all poses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.imu.accel_calib import (  # noqa: E402
    G_STANDARD,
    AccelCalibration,
    solve_accel_calibration,
)

# Gravity along each of the six faces (true specific force at rest, m/s^2).
_FACES = np.array([
    [+1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, +1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, +1.0], [0.0, 0.0, -1.0],
]) * G_STANDARD


def _planted_distortion():
    """A realistic-magnitude true sensor model: a_raw = M @ a_true + b_raw."""
    # ~2% scale errors, ~1 deg cross-axis leakage, ~0.2 m/s^2 bias.
    M = np.array([
        [1.018, 0.012, -0.009],
        [-0.007, 0.982, 0.015],
        [0.010, -0.013, 1.005],
    ])
    b_true = np.array([0.21, -0.15, 0.08])      # bias in TRUE units
    # Raw bias is what the solver sees (b in a_raw = M a_true + b_raw):
    b_raw = b_true.copy()
    return M, b_raw


def _synth_raw(true_force, M, b_raw, rng, noise=0.0):
    """Raw reading the sensor produces for a given true specific force."""
    a = M @ true_force + b_raw
    if noise > 0.0:
        a = a + rng.normal(0.0, noise, size=3)
    return a


def main() -> int:
    rng = np.random.default_rng(0)
    M, b_raw = _planted_distortion()

    # --- 1. Exact recovery from the 6 canonical faces (no noise) -----------
    caps = [_synth_raw(f, M, b_raw, rng) for f in _FACES]
    cal = solve_accel_calibration(caps)
    corrected = np.array([cal.apply(c) for c in caps])
    mags = np.linalg.norm(corrected, axis=1)
    sphere_err = float(np.max(np.abs(mags - G_STANDARD)))
    print("exact six-face recovery (no noise)")
    print(f"  RMS sphere residual reported = {cal.residual_g:.3e} m/s^2")
    print(f"  max |corrected| - g          = {sphere_err:.3e} m/s^2")

    # Direction check: the corrected faces must match the true faces up to ONE
    # common rotation R (the gauge). Solve R by Procrustes and check the fit.
    U, _, Vt = np.linalg.svd(corrected.T @ _FACES)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    dir_err = float(np.max(np.linalg.norm(corrected @ R.T - _FACES, axis=1)))
    print(f"  max direction error (gauge-aligned) = {dir_err:.3e} m/s^2")
    ok_exact = sphere_err < 1e-6 and dir_err < 1e-6

    # --- 2. Undo planted distortion on INDEPENDENT tilted poses ------------
    # Random rest orientations (gravity along a random unit direction): the
    # calibration learned from the faces must still sphere these unseen poses.
    test_dirs = rng.normal(size=(40, 3))
    test_dirs /= np.linalg.norm(test_dirs, axis=1, keepdims=True)
    test_forces = test_dirs * G_STANDARD
    test_raw = np.array([_synth_raw(f, M, b_raw, rng) for f in test_forces])
    test_corr = cal.apply(test_raw)
    test_mag_err = float(np.max(np.abs(
        np.linalg.norm(test_corr, axis=1) - G_STANDARD)))
    print("generalisation to 40 unseen tilted poses")
    print(f"  max |corrected| - g = {test_mag_err:.3e} m/s^2")
    ok_generalise = test_mag_err < 1e-6

    # --- 3. Noisy data: still well within sensor noise --------------------
    # 8 poses (6 faces + 2 tilts), realistic 0.02 m/s^2 white noise on the mean.
    # Tilted poses pass their KNOWN direction explicitly (a guided wizard always
    # knows the intended orientation -- only the canonical faces are auto-snapped).
    extra = (rng.normal(size=(2, 3)))
    extra = (extra / np.linalg.norm(extra, axis=1, keepdims=True))
    noisy_dirs = np.vstack([_FACES / G_STANDARD, extra])
    noisy_forces = noisy_dirs * G_STANDARD
    noisy_caps = [_synth_raw(f, M, b_raw, rng, noise=0.02) for f in noisy_forces]
    cal_n = solve_accel_calibration(noisy_caps, directions=noisy_dirs)
    nc = cal_n.apply(np.array(noisy_caps))
    noisy_err = float(np.sqrt(np.mean(
        (np.linalg.norm(nc, axis=1) - G_STANDARD) ** 2)))
    print("noisy 8-pose fit (0.02 m/s^2 noise, known directions)")
    print(f"  RMS sphere residual = {noisy_err:.3e} m/s^2")
    ok_noisy = noisy_err < 0.05      # at or below the injected noise level

    # --- 4. Round-trip serialisation --------------------------------------
    d = cal.to_dict()
    cal2 = AccelCalibration.from_dict(d)
    rt = float(np.max(np.abs(cal.apply(test_raw) - cal2.apply(test_raw))))
    print(f"serialisation round-trip max diff = {rt:.3e}")
    ok_serial = rt < 1e-12

    # --- 5. Guards --------------------------------------------------------
    ok_guard = True
    try:
        solve_accel_calibration(caps[:5])     # < 6 captures must raise
        ok_guard = False
        print("guard: FAILED to reject < 6 captures")
    except ValueError:
        pass
    # Identity calibration is a pass-through.
    idn = AccelCalibration.identity()
    ok_guard = ok_guard and np.allclose(idn.apply(test_raw), test_raw)

    ok = (ok_exact and ok_generalise and ok_noisy and ok_serial and ok_guard)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
