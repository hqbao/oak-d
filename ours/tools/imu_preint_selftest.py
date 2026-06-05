"""Self-test for IMU preintegration (oakd.vio.imu.preintegrate_imu).

Phase 1 guard for the tight-coupled VIO build. It checks the preintegrated
increments ``dR, dv, dp`` against a CLOSED-FORM ground-truth trajectory, so a
convention or sign error shows up here instead of as silent drift on the device.

Ground truth (constant body angular velocity ``w``, constant world-frame linear
acceleration ``A``, gravity ``g``):

    R(t) = R0 Exp(w t)
    v(t) = v0 + A t
    p(t) = p0 + v0 t + 0.5 A t^2

The simulated IMU then reads (body frame):

    gyro(t)  = w                       (+ true bias)
    accel(t) = R(t)^T (A - g)          (+ true bias)   [specific force]

and the preintegration must satisfy

    R_j = R_i dR
    v_j = v_i + g dt + R_i dv
    p_j = p_i + v_i dt + 0.5 g dt^2 + R_i dp

A second block perturbs the bias and checks the first-order ``corrected()``
update matches a full re-integration with the perturbed bias.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.vio.imu import (  # noqa: E402
    ImuPreintegration,
    preintegrate_imu,
    so3_exp,
    so3_log,
)


def _make_imu(R0, w, v0, A, g, T, rate_hz, bg=None, ba=None):
    """Sample the closed-form trajectory's IMU over [0, T] at rate_hz."""
    bg = np.zeros(3) if bg is None else np.asarray(bg, float)
    ba = np.zeros(3) if ba is None else np.asarray(ba, float)
    n = int(round(T * rate_hz)) + 1
    t = np.linspace(0.0, T, n)
    ts_ns = np.round(t * 1e9).astype(np.int64)
    gyro = np.zeros((n, 3))
    accel = np.zeros((n, 3))
    for k in range(n):
        Rk = R0 @ so3_exp(w * t[k])
        gyro[k] = w + bg
        accel[k] = Rk.T @ (A - g) + ba
    return ts_ns, gyro, accel


def main() -> int:
    rng = np.random.default_rng(0)
    R0 = so3_exp(np.array([0.2, -0.35, 0.5]))
    w = np.array([0.3, -0.2, 1.6])          # fast-yaw-dominated body rates
    v0 = np.array([0.4, -0.1, 0.2])
    A = np.array([0.6, -0.9, 0.3])          # real world linear acceleration
    g = np.array([0.0, 0.0, -9.81])
    T = 0.5
    rate = 4000.0
    p0 = np.array([1.0, 2.0, -0.5])

    ts, gyro, accel = _make_imu(R0, w, v0, A, g, T, rate)
    pre = preintegrate_imu(ts, gyro, accel, np.zeros(3), np.zeros(3))

    dt = T
    R_i, v_i, p_i = R0, v0, p0
    R_j = R0 @ so3_exp(w * T)
    v_j = v0 + A * T
    p_j = p0 + v0 * T + 0.5 * A * T * T

    # --- check the three increments against closed form --------------------
    eR = float(np.linalg.norm(so3_log(R_j.T @ (R_i @ pre.dR))))
    ev = float(np.linalg.norm(v_j - (v_i + g * dt + R_i @ pre.dv)))
    ep = float(np.linalg.norm(p_j - (p_i + v_i * dt + 0.5 * g * dt * dt
                                     + R_i @ pre.dp)))
    print("preintegration vs closed-form ground truth")
    print(f"  rotation residual |rR| = {np.degrees(eR):.4e} deg")
    print(f"  velocity residual |rv| = {ev:.4e} m/s")
    print(f"  position residual |rp| = {ep:.4e} m")

    # Bounds are physical: rotation exact (machine eps), velocity < 1 mm/s and
    # position < 0.5 mm over 0.5 s of aggressive motion -- the residual here is
    # pure forward-Euler discretisation (O(dt), shrinks with the sample rate),
    # far below real IMU noise. A convention/sign error would blow these up.
    ok_incr = (eR < 1e-3) and (ev < 1e-3) and (ep < 5e-4)

    # --- bias Jacobian: first-order correction vs full re-integration ------
    bg0 = np.array([0.01, -0.02, 0.015])    # linearisation bias
    ba0 = np.array([0.05, 0.03, -0.04])
    ts2, gyro2, accel2 = _make_imu(R0, w, v0, A, g, T, rate,
                                   bg=bg0, ba=ba0)   # true bias baked in
    pre0 = preintegrate_imu(ts2, gyro2, accel2, bg0, ba0)   # at linearisation pt

    dbg = np.array([0.004, -0.003, 0.002])
    dba = np.array([-0.02, 0.01, 0.015])
    bg1, ba1 = bg0 + dbg, ba0 + dba
    dR_c, dv_c, dp_c = pre0.corrected(bg1, ba1)             # first-order
    pre1 = preintegrate_imu(ts2, gyro2, accel2, bg1, ba1)   # full re-integration

    eR_b = float(np.linalg.norm(so3_log(pre1.dR.T @ dR_c)))
    ev_b = float(np.linalg.norm(pre1.dv - dv_c))
    ep_b = float(np.linalg.norm(pre1.dp - dp_c))
    print("bias first-order correction vs full re-integration")
    print(f"  d(rotation) = {np.degrees(eR_b):.4e} deg")
    print(f"  d(velocity) = {ev_b:.4e} m/s")
    print(f"  d(position) = {ep_b:.4e} m")

    # First-order error is O(|db|^2); these bounds are comfortably above that
    # but far below the full effect of the bias (which is ~10x larger).
    ok_bias = (eR_b < 1e-3) and (ev_b < 1e-3) and (ep_b < 1e-4)

    ok = ok_incr and ok_bias
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
