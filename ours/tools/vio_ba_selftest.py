"""Self-test for the tight-coupled VIO window optimizer (ours.vio.vio_window).

Phase 2 gate for the Basalt-style build. It manufactures a synthetic world with
a KNOWN answer, so a convention/sign error in the joint visual+inertial solve
shows up here -- not as drift on the device.

Construction
------------
A multi-segment trajectory: each segment has a constant body angular velocity and
a constant world-frame linear acceleration, so the closed-form pose/velocity at
every segment boundary (= keyframe) is exact, and the IMU the body would feel is
``gyro = w (+bias)``, ``accel = R(t)^T (A - g) (+bias)``. We:

  1. place 3D landmarks, project them into every keyframe (pinhole + depth) to
     get the visual measurements,
  2. preintegrate the sampled IMU between consecutive keyframes,
  3. PERTURB the keyframe poses / velocities / biases / landmarks away from
     ground truth,
  4. run ``optimize_vio`` and check it recovers the ground truth.

Scenario A: zero true bias -- tests the pose/velocity/landmark coupling.
Scenario B: non-zero true gyro+accel bias baked into the IMU, preintegrated at a
zero linearisation point -- tests that the optimizer estimates the biases (via
the first-order ``corrected`` update) while recovering the states.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.imu.imu import preintegrate_imu, so3_exp, so3_log  # noqa: E402
from ours.lib.backend.vio_window import (  # noqa: E402
    VioConfig,
    VioState,
    optimize_vio,
)

K = np.array([[400.0, 0.0, 320.0],
              [0.0, 400.0, 240.0],
              [0.0, 0.0, 1.0]])
W, H = 640, 480
G = np.array([0.0, 0.0, -9.81])           # gravity acceleration, z-down world


def make_world(true_bg=np.zeros(3), true_ba=np.zeros(3), seed=1):
    rng = np.random.default_rng(seed)
    # --- segments: (duration, body angular velocity, world accel) ----------
    segs = [
        (0.30, np.array([0.05, 0.6, 0.1]), np.array([0.3, -0.2, 0.1])),
        (0.30, np.array([-0.1, 0.5, -0.05]), np.array([-0.4, 0.1, 0.2])),
        (0.30, np.array([0.0, -0.7, 0.15]), np.array([0.2, 0.3, -0.15])),
        (0.30, np.array([0.08, 0.4, 0.0]), np.array([-0.1, -0.25, 0.05])),
    ]
    rate = 500.0

    R = np.eye(3)
    p = np.array([0.0, 0.0, 0.0])
    v = np.array([0.2, -0.1, 0.05])
    kf_R = [R.copy()]; kf_p = [p.copy()]; kf_v = [v.copy()]
    imu_factors_raw = []          # (ts, gyro, accel) per segment

    for (T, w, A) in segs:
        n = int(round(T * rate)) + 1
        t = np.linspace(0.0, T, n)
        ts_ns = np.round(t * 1e9).astype(np.int64)
        gyro = np.zeros((n, 3)); accel = np.zeros((n, 3))
        for k in range(n):
            Rk = R @ so3_exp(w * t[k])
            gyro[k] = w + true_bg
            accel[k] = Rk.T @ (A - G) + true_ba
        imu_factors_raw.append((ts_ns, gyro, accel))
        # advance closed form to segment end
        p = p + v * T + 0.5 * A * T * T
        v = v + A * T
        R = R @ so3_exp(w * T)
        kf_R.append(R.copy()); kf_p.append(p.copy()); kf_v.append(v.copy())

    nKF = len(kf_R)

    # --- landmarks: spread in front of the camera path ---------------------
    lms = []
    while len(lms) < 60:
        X = rng.uniform([-2.0, -1.5, 2.0], [2.0, 1.5, 6.0])
        # accept if visible in at least 2 keyframes
        vis = 0
        for i in range(nKF):
            Xc = kf_R[i].T @ (X - kf_p[i])
            if Xc[2] > 0.3:
                u = K[0, 0] * Xc[0] / Xc[2] + K[0, 2]
                v_ = K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
                if 0 <= u < W and 0 <= v_ < H:
                    vis += 1
        if vis >= 2:
            lms.append(X)
    lms = np.array(lms)

    obs_cam = []; obs_lm = []; obs_uv = []; obs_d = []
    for i in range(nKF):
        for m in range(lms.shape[0]):
            Xc = kf_R[i].T @ (lms[m] - kf_p[i])
            if Xc[2] <= 0.3:
                continue
            u = K[0, 0] * Xc[0] / Xc[2] + K[0, 2]
            v_ = K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
            if not (0 <= u < W and 0 <= v_ < H):
                continue
            obs_cam.append(i); obs_lm.append(m)
            obs_uv.append([u, v_]); obs_d.append(Xc[2])

    return dict(
        kf_R=kf_R, kf_p=kf_p, kf_v=kf_v, lms=lms,
        obs_cam=np.array(obs_cam), obs_lm=np.array(obs_lm),
        obs_uv=np.array(obs_uv), obs_d=np.array(obs_d),
        imu_raw=imu_factors_raw,
    )


def build_factors(imu_raw, bg_lin, ba_lin):
    """Preintegrate each segment at the given linearisation bias."""
    factors = []
    for s, (ts, gyro, accel) in enumerate(imu_raw):
        pre = preintegrate_imu(ts, gyro, accel, bg_lin, ba_lin)
        factors.append((s, s + 1, pre))
    return factors


def run_scenario(name, true_bg, true_ba, gate_bias, lock_tilt=False):
    w = make_world(true_bg=true_bg, true_ba=true_ba)
    nKF = len(w["kf_R"])
    rng = np.random.default_rng(7)

    # ground-truth state
    gt = VioState(R=[r.copy() for r in w["kf_R"]],
                  p=[x.copy() for x in w["kf_p"]],
                  v=[x.copy() for x in w["kf_v"]],
                  bg=[true_bg.copy() for _ in range(nKF)],
                  ba=[true_ba.copy() for _ in range(nKF)],
                  landmarks=w["lms"].copy())

    # world-up axis (gravity is z-down here -> up = +z); the tilt-locked solve
    # only moves yaw about this axis, so the perturbation must keep roll/pitch at
    # ground truth (a tilt error would be unrecoverable by design).
    up = -G / np.linalg.norm(G)

    # perturbed initial guess (anchor KF0 left at truth -> gauge fixed)
    st = gt.copy()
    for i in range(1, nKF):
        if lock_tilt:
            dyaw = rng.normal(0, np.radians(3.0))
            st.R[i] = so3_exp(up * dyaw) @ st.R[i]   # yaw-only, tilt preserved
        else:
            st.R[i] = st.R[i] @ so3_exp(rng.normal(0, np.radians(3.0), 3))
        st.p[i] = st.p[i] + rng.normal(0, 0.05, 3)
        st.v[i] = st.v[i] + rng.normal(0, 0.1, 3)
    st.v[0] = gt.v[0] + rng.normal(0, 0.1, 3)
    for i in range(nKF):
        st.bg[i] = np.zeros(3)          # guess starts at zero
        st.ba[i] = np.zeros(3)
    st.landmarks = st.landmarks + rng.normal(0, 0.03, st.landmarks.shape)

    # factors preintegrated at the zero linearisation bias (matches guess)
    factors = build_factors(w["imu_raw"], np.zeros(3), np.zeros(3))

    cfg = VioConfig(max_iters=40, lock_tilt=lock_tilt)
    res = optimize_vio(K, st, w["obs_cam"], w["obs_lm"], w["obs_uv"], w["obs_d"],
                       factors, G, cfg, anchor=0)
    out = res.state

    # --- errors vs ground truth -------------------------------------------
    rot_err = max(np.degrees(np.linalg.norm(so3_log(gt.R[i].T @ out.R[i])))
                  for i in range(nKF))
    pos_err = max(float(np.linalg.norm(gt.p[i] - out.p[i])) for i in range(nKF))
    vel_err = max(float(np.linalg.norm(gt.v[i] - out.v[i])) for i in range(nKF))
    lm_err = float(np.max(np.linalg.norm(gt.landmarks - out.landmarks, axis=1)))
    bg_err = max(float(np.linalg.norm(true_bg - out.bg[i])) for i in range(nKF))
    ba_err = max(float(np.linalg.norm(true_ba - out.ba[i])) for i in range(nKF))

    print(f"\n=== scenario {name} ===")
    print(f"  cost {res.cost0:.3e} -> {res.cost1:.3e} "
          f"({res.iters} it, mean reproj {res.mean_reproj_px:.3f}px)")
    print(f"  max rotation err = {rot_err:.4e} deg")
    print(f"  max position err = {pos_err:.4e} m")
    print(f"  max velocity err = {vel_err:.4e} m/s")
    print(f"  max landmark err = {lm_err:.4e} m")
    print(f"  max gyrobias err = {bg_err:.4e} rad/s")
    print(f"  max accelbias err= {ba_err:.4e} m/s^2")

    ok = (rot_err < 5e-2 and pos_err < 2e-3 and vel_err < 1e-2
          and lm_err < 3e-3 and bg_err < gate_bias[0] and ba_err < gate_bias[1])
    print("  " + ("OK" if ok else "FAIL"))
    return ok


def main() -> int:
    okA = run_scenario("A: zero bias", np.zeros(3), np.zeros(3),
                       gate_bias=(1e-3, 1e-2))
    okB = run_scenario("B: with bias",
                       np.array([0.004, -0.003, 0.005]),
                       np.array([0.03, -0.02, 0.04]),
                       gate_bias=(2e-3, 2e-2))
    okC = run_scenario("C: tilt-locked (yaw+pos only)",
                       np.zeros(3), np.zeros(3),
                       gate_bias=(1e-3, 1e-2), lock_tilt=True)
    ok = okA and okB and okC
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
