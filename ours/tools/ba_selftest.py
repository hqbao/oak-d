#!/usr/bin/env python3
"""Synthetic self-test for the sliding-window BA core (ours.vio.bundle).

We build a known scene (random 3D landmarks + a smooth camera trajectory),
project the landmarks into every camera to get *exact* pixel observations, then
corrupt the poses and landmarks with noise and ask the optimiser to recover the
truth. With the first camera fixed as the gauge anchor and noise-free
observations, BA should drive the reprojection error to ~0.

This proves the solver maths (Jacobians, Schur complement, SE3 update) are
correct *before* we trust it on real session data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.backend.bundle import BAConfig, optimize, se3_exp  # noqa: E402


def project(K, T_cw, Xw):
    Xc = T_cw[:3, :3] @ Xw + T_cw[:3, 3]
    u = K[0, 0] * Xc[0] / Xc[2] + K[0, 2]
    v = K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
    return np.array([u, v])


def main() -> int:
    rng = np.random.default_rng(0)
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 200.0], [0, 0, 1.0]])

    # Truth landmarks in a slab 3-7 m in front of the start.
    M = 80
    Xw_true = np.column_stack([
        rng.uniform(-3, 3, M),
        rng.uniform(-2, 2, M),
        rng.uniform(3, 7, M),
    ])

    # Truth trajectory: 6 cameras translating sideways + slight yaw.
    nC = 6
    poses_true = []
    for i in range(nC):
        ang = np.deg2rad(4.0 * i)
        xi = np.array([0.15 * i, 0.0, 0.0, 0.0, ang, 0.0])  # move +x, yaw
        T_wc = se3_exp(xi)               # camera->world
        poses_true.append(np.linalg.inv(T_wc))  # store world->camera
    # Re-anchor so camera 0 is exactly identity (world == cam0).
    T0_inv = np.linalg.inv(poses_true[0])
    poses_true = [T @ T0_inv for T in poses_true]

    # Observations: every landmark seen by every camera (if in front).
    obs_cam, obs_lm, obs_uv = [], [], []
    for c in range(nC):
        for l in range(M):
            Xc = poses_true[c][:3, :3] @ Xw_true[l] + poses_true[c][:3, 3]
            if Xc[2] <= 0.1:
                continue
            obs_cam.append(c)
            obs_lm.append(l)
            obs_uv.append(project(K, poses_true[c], Xw_true[l]))
    obs_cam = np.array(obs_cam)
    obs_lm = np.array(obs_lm)
    obs_uv = np.array(obs_uv)

    # Corrupt: keep cam0 fixed (truth), perturb the rest + landmarks.
    poses0 = [poses_true[0].copy()]
    for c in range(1, nC):
        noise = np.concatenate([rng.normal(0, 0.10, 3), rng.normal(0, 0.03, 3)])
        poses0.append(se3_exp(noise) @ poses_true[c])
    Xw0 = Xw_true + rng.normal(0, 0.15, Xw_true.shape)
    fixed = [c == 0 for c in range(nC)]

    def reproj_rmse(poses, lms):
        errs = []
        for n in range(len(obs_cam)):
            p = project(K, poses[obs_cam[n]], lms[obs_lm[n]])
            errs.append(np.linalg.norm(p - obs_uv[n]))
        return float(np.sqrt(np.mean(np.square(errs))))

    before = reproj_rmse(poses0, Xw0)
    res = optimize(K, poses0, fixed, Xw0, obs_cam, obs_lm, obs_uv,
                   cfg=BAConfig(max_iters=30, huber_px=5.0))
    after = reproj_rmse(res.poses, res.landmarks)

    # Pose recovery error (translation) vs truth.
    pos_err = np.mean([
        np.linalg.norm(np.linalg.inv(res.poses[c])[:3, 3]
                       - np.linalg.inv(poses_true[c])[:3, 3])
        for c in range(nC)
    ])

    print(f"observations      : {len(obs_cam)}")
    print(f"reproj RMSE before : {before:8.3f} px")
    print(f"reproj RMSE after  : {after:8.3f} px   (iters={res.iters})")
    print(f"cost  {res.cost0:.2f} -> {res.cost1:.2f}")
    print(f"mean camera pos err: {pos_err*1000:7.2f} mm")

    ok = after < 0.1 and pos_err < 0.01
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
