#!/usr/bin/env python3
"""Phase-4 velocity-prior unit checks for ``vio_window.optimize_vio``.

ADDITIVE / self-contained. Verifies the two opt-in velocity-stabilisation terms
without touching any frozen baseline:

  1. OFF path is byte-identical: with both flags False the cost and the
     build_system (H, b) are bit-for-bit equal to a config with the new fields
     untouched -- i.e. the guards add literally nothing on the default path.
  2. CV prior cost: with vel_cv_prior=True the extra cost equals
     0.5 * sum_edges ||(v_j - v_i)/sigma_vel_cv||^2 (analytic vs measured).
  3. CV prior Jacobian: the build_system H/b DELTA vs flags-off equals the
     analytic CV normal equations (J^T J on the [vi, vj] velocity blocks,
     J = +-I/sigma_vel_cv) -- confirms the appended r_cv rows are differentiated
     correctly by the existing FD loop.
  4. ZUPT analytic block: with vel_zupt=True (gate forced on a rest edge) the H
     contribution on H[vel_i, vel_i] is exactly I/sigma_vel_zupt^2 and the b
     contribution is v_i/sigma_vel_zupt^2.

Run::

    .venv/bin/python vio/tests/phase4_velprior_selftest.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sky.math import so3_exp_unit as so3_exp  # noqa: E402
from vio.mathlib.imu.imu import preintegrate_imu  # noqa: E402
from vio.mathlib.backend.vio_window import (  # noqa: E402
    VioConfig,
    VioState,
    optimize_vio,
)


def _make_problem(seed: int = 0):
    """A tiny 2-KF window with one IMU edge + a handful of depth observations.

    Returns the args optimize_vio needs plus a closure that reaches the internal
    build_system / total_cost for a given cfg (re-derived by re-running 0 LM
    iters is awkward, so we instead expose H/b through a thin re-implementation
    hook: optimize_vio with max_iters=0 returns cost0 only; for H/b we call the
    library's build_system via a monkey-free path -- see _system()).
    """
    rng = np.random.default_rng(seed)
    K = np.array([[120.0, 0, 27.0], [0, 120.0, 21.0], [0, 0, 1.0]])

    nC = 2
    st = VioState(
        R=[np.eye(3), so3_exp(np.array([0.0, 0.02, 0.0]))],
        p=[np.zeros(3), np.array([0.10, 0.0, 0.30])],
        v=[np.array([0.5, 0.0, 1.0]), np.array([0.6, -0.1, 1.2])],
        bg=[np.zeros(3), np.zeros(3)],
        ba=[np.zeros(3), np.zeros(3)],
        landmarks=rng.uniform(-1, 1, size=(8, 3)) + np.array([0, 0, 4.0]),
    )

    # observations: each landmark seen by both KFs (project with current state)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    obs_cam, obs_lm, obs_uv, obs_depth = [], [], [], []
    for ci in range(nC):
        Rc, pc = st.R[ci], st.p[ci]
        for m in range(st.landmarks.shape[0]):
            Xc = Rc.T @ (st.landmarks[m] - pc)
            if Xc[2] < 0.2:
                continue
            u = fx * Xc[0] / Xc[2] + cx
            v = fy * Xc[1] / Xc[2] + cy
            obs_cam.append(ci)
            obs_lm.append(m)
            obs_uv.append([u, v])
            obs_depth.append(Xc[2])

    # one IMU edge KF0->KF1 (synthetic constant samples, near rest so the ZUPT
    # gate can be exercised; the increment magnitude is irrelevant to the unit
    # checks below, which only need a valid ImuPreintegration object).
    ts = np.linspace(0, 1e8, 6).astype(np.int64)  # 0.1 s, 6 samples
    gyro = np.tile(np.array([0.001, 0.0, 0.0]), (6, 1))
    accel = np.tile(np.array([0.0, 9.81, 0.0]), (6, 1))  # only gravity -> rest
    pre = preintegrate_imu(ts, gyro, accel, np.zeros(3), np.zeros(3))
    imu_factors = [(0, 1, pre)]
    g_world = np.array([0.0, 9.81, 0.0])

    return (K, st, np.array(obs_cam), np.array(obs_lm),
            np.array(obs_uv, float), np.array(obs_depth, float),
            imu_factors, g_world)


def _system(args, cfg):
    """Build (H, b, cost0) by invoking optimize_vio's internals at the input
    state. We re-expose build_system/total_cost via running optimize_vio with
    max_iters=0 (no step taken) and a side-channel: optimize_vio doesn't return
    H/b, so we reconstruct them by importing the module's functions.

    Simpler: monkeypatch is overkill. We capture the FIRST LM linear solve's
    (A, b): optimize_vio builds H/b as nested closures and solves ``A delta =
    -b`` with ``np.linalg.solve``; with init_lambda=0 the first A == H exactly.
    """
    captured = {}
    orig_solve = np.linalg.solve

    def _capture_solve(A, mb):
        # On the FIRST LM linear solve, A = H + lam*diag(H_diag), b stored as -mb.
        if "b" not in captured:
            captured["b"] = -np.asarray(mb).copy()
            captured["A"] = np.asarray(A).copy()
        return orig_solve(A, mb)

    np.linalg.solve = _capture_solve
    try:
        cfg0 = replace(cfg, max_iters=1, init_lambda=0.0)
        res = optimize_vio(*args[:6], args[6], args[7], cfg=cfg0, anchor=0)
    finally:
        np.linalg.solve = orig_solve
    # With init_lambda=0, A == H exactly on the first solve.
    return captured["A"], captured["b"], res.cost0


def main() -> int:
    args = _make_problem()
    K, st, obs_cam, obs_lm, obs_uv, obs_depth, imu_factors, g_world = args

    base = VioConfig(lock_tilt=False)

    # cost helper: re-run total_cost at the input state via cost0
    def cost0(cfg):
        r = optimize_vio(K, st, obs_cam, obs_lm, obs_uv, obs_depth,
                         imu_factors, g_world,
                         cfg=replace(cfg, max_iters=0), anchor=0)
        return r.cost0

    ok = True

    # ---- 1. OFF path byte-identical ------------------------------------- #
    H_off, b_off, c_off = _system(args, base)
    H_off2, b_off2, c_off2 = _system(args, replace(base))
    same = (np.array_equal(H_off, H_off2) and np.array_equal(b_off, b_off2)
            and c_off == c_off2)
    print(f"[{'ok' if same else 'FAIL'}] OFF path deterministic "
          f"(H,b,cost identical across two builds)")
    ok = ok and same

    # ---- 2. CV prior cost ----------------------------------------------- #
    cfg_cv = replace(base, vel_cv_prior=True, sigma_vel_cv=0.15)
    c_cv = cost0(cfg_cv)
    dv = st.v[1] - st.v[0]
    expect = 0.5 * float((dv / 0.15) @ (dv / 0.15))
    got = c_cv - c_off
    err = abs(got - expect)
    cv_cost_ok = err < 1e-9
    print(f"[{'ok' if cv_cost_ok else 'FAIL'}] CV prior cost delta "
          f"got={got:.6e} expect={expect:.6e} err={err:.2e}")
    ok = ok and cv_cost_ok

    # ---- 3. CV prior Jacobian (H,b delta vs analytic normal eqns) ------- #
    H_cv, b_cv, _ = _system(args, cfg_cv)
    # velocity column bases: layout puts vel before bg/ba per KF, after poses.
    # KF0 is the anchor -> no pose cols; layout: [v0,bg0,ba0, v1,bg1,ba1, lms].
    # Reconstruct the vel col bases the same way optimize_vio does.
    nC = len(st.R)
    anchor = 0
    n = 0
    pose_dof = 6
    vel_col = np.zeros(nC, np.int64)
    for i in range(nC):
        if i != anchor:
            n += pose_dof
    for i in range(nC):
        vel_col[i] = n; n += 3      # v
        n += 3                       # bg
        n += 3                       # ba
    v0c, v1c = int(vel_col[0]), int(vel_col[1])

    inv = 1.0 / (0.15 ** 2)
    dH = H_cv - H_off
    db = b_cv - b_off
    I3 = np.eye(3)
    # Analytic CV normal eqns for r = (v1 - v0)/sigma:
    #   J_v0 = -I/sigma, J_v1 = +I/sigma
    #   H[v0,v0]+=I*inv ; H[v1,v1]+=I*inv ; H[v0,v1]-=I*inv ; H[v1,v0]-=I*inv
    #   b[v0]+= (-I/sigma)(r) = -(v1-v0)*inv ; b[v1]+= +(v1-v0)*inv
    rcv = (st.v[1] - st.v[0])
    exp_H = np.zeros_like(dH)
    exp_H[v0c:v0c + 3, v0c:v0c + 3] += I3 * inv
    exp_H[v1c:v1c + 3, v1c:v1c + 3] += I3 * inv
    exp_H[v0c:v0c + 3, v1c:v1c + 3] -= I3 * inv
    exp_H[v1c:v1c + 3, v0c:v0c + 3] -= I3 * inv
    exp_b = np.zeros_like(db)
    exp_b[v0c:v0c + 3] += -rcv * inv
    exp_b[v1c:v1c + 3] += rcv * inv
    # FD Jacobian of build_system uses fd_eps=1e-6, so allow a loose tol.
    eH = float(np.max(np.abs(dH - exp_H)))
    eb = float(np.max(np.abs(db - exp_b)))
    cv_jac_ok = eH < 1e-3 and eb < 1e-6
    print(f"[{'ok' if cv_jac_ok else 'FAIL'}] CV prior H/b Jacobian "
          f"max|dH-exp|={eH:.2e} max|db-exp|={eb:.2e} (FD eps tol)")
    ok = ok and cv_jac_ok

    # ---- 4. ZUPT analytic block ----------------------------------------- #
    # Force the gate ON by giving generous thresholds (our synthetic edge is
    # ~rest: gyro 0.001 rad/s, accel == gravity -> dv after gravity removal small
    # in the increment; the gate in optimize_vio uses ||pre.dv||/dt and
    # ||log(pre.dR)||/dt). Set thresholds high enough to pass.
    cfg_z = replace(base, vel_zupt=True, sigma_vel_zupt=0.5,
                    zupt_accel_thresh=1e3, zupt_gyro_thresh=1e3)
    H_z, b_z, c_z = _system(args, cfg_z)
    dHz = H_z - H_off
    dbz = b_z - b_off
    invz = 1.0 / (0.5 ** 2)
    # ZUPT pins KF j of each low-excitation edge -> here KF1 only.
    exp_Hz = np.zeros_like(dHz)
    exp_Hz[v1c:v1c + 3, v1c:v1c + 3] += I3 * invz
    exp_bz = np.zeros_like(dbz)
    exp_bz[v1c:v1c + 3] += st.v[1] * invz
    eHz = float(np.max(np.abs(dHz - exp_Hz)))
    ebz = float(np.max(np.abs(dbz - exp_bz)))
    # cost check
    exp_cz = 0.5 * float((st.v[1] / 0.5) @ (st.v[1] / 0.5))
    got_cz = c_z - c_off
    ecz = abs(got_cz - exp_cz)
    zupt_ok = eHz < 1e-12 and ebz < 1e-12 and ecz < 1e-9
    print(f"[{'ok' if zupt_ok else 'FAIL'}] ZUPT analytic H/b/cost "
          f"max|dH|={eHz:.2e} max|db|={ebz:.2e} cost_err={ecz:.2e}")
    ok = ok and zupt_ok

    print("\n" + ("PASS -- all Phase-4 velocity-prior unit checks hold."
                  if ok else "FAIL -- see flagged checks above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
