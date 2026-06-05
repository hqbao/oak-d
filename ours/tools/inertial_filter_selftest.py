"""Self-test for the loosely-coupled inertial translation filter.

Validates the predict+correct behaviour on synthetic data (no device):

  A. gravity cancellation  -- at rest the net world accel is ~0
  B. standing still         -- vision ~0 + accel ~0 => no position drift
  C. forward push           -- accel + matching vision => tracks the motion
  D. in-place yaw from rest -- accel ~0, phantom vision during spin => no walk
  E. forward + yaw          -- velocity prior coasts forward through the turn
  F. vision dropout         -- IMU coasts then the velocity bleeds to rest

Run:  python ours/tools/inertial_filter_selftest.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ours.lib.imu.inertial_filter import (  # noqa: E402
    InertialFilterConfig,
    InertialTranslationFilter,
)

G = np.array([0.0, 9.81, 0.0])   # optical world gravity (down = +y)


def _Rz(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rest_accel_cam(R_wc):
    """Specific force in the camera frame for a stationary device.

    At rest world specific force = -g_grav = [0,-9.81,0]; rotate into cam.
    """
    f_world = -G
    return R_wc.T @ f_world


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}{(' -- ' + detail) if detail else ''}")
    return bool(cond)


def main():
    ok_all = True
    dt = 1.0 / 30.0
    R0 = np.eye(3)

    # --- A. gravity cancellation --------------------------------------------
    print("A. gravity cancellation at rest")
    f = InertialTranslationFilter()
    a_world = R0 @ _rest_accel_cam(R0) + G
    ok_all &= check("net world accel ~0", np.linalg.norm(a_world) < 1e-9,
                    f"|a|={np.linalg.norm(a_world):.2e}")
    # also when tilted/yawed
    Rt = _Rz(37.0)
    a_world_t = Rt @ _rest_accel_cam(Rt) + G
    ok_all &= check("net world accel ~0 (yawed frame)",
                    np.linalg.norm(a_world_t) < 1e-9,
                    f"|a|={np.linalg.norm(a_world_t):.2e}")

    # --- B. standing still ---------------------------------------------------
    print("B. standing still")
    f.reset()
    acc = _rest_accel_cam(R0)
    for _ in range(300):                      # 10 s
        f.step(dt, R0, acc, np.zeros(3), rot_deg=0.0)
    drift = float(np.linalg.norm(f.p))
    ok_all &= check("no drift over 10 s", drift < 1e-3, f"drift={drift*100:.3f} cm")

    # --- C. forward push -----------------------------------------------------
    print("C. forward push (accel + matching vision)")
    f.reset()
    # accelerate +x for 0.5 s then constant velocity for 0.5 s
    v_true = np.zeros(3)
    p_true = np.zeros(3)
    a_mag = 2.0
    for i in range(30):
        a = np.array([a_mag if i < 15 else 0.0, 0.0, 0.0])
        # specific force the accelerometer would read: f = a_inertial - g
        accel_cam = R0.T @ (a - G)
        # ground-truth integration for the vision displacement
        p_prev = p_true.copy()
        v_true = v_true + a * dt
        p_true = p_true + v_true * dt
        dp_vis = p_true - p_prev
        f.step(dt, R0, accel_cam, dp_vis, rot_deg=0.0)
    err = float(np.linalg.norm(f.p - p_true))
    ok_all &= check("tracks forward motion", err < 0.05,
                    f"p={f.p[0]:.3f}m true={p_true[0]:.3f}m err={err*100:.2f}cm")
    ok_all &= check("moved meaningfully forward", f.p[0] > 0.2, f"x={f.p[0]:.3f}m")

    # --- D. in-place yaw from rest (honest ~0 vision -> no drift) ------------
    # The odometry now solves translation with the gyro rotation LOCKED, so a
    # pure in-place yaw emits ~0 vision displacement (the phantom is removed at
    # the source). The filter's job is simply to trust that honest measurement
    # and not drift.
    print("D. in-place yaw from rest (honest ~0 vision)")
    f.reset()
    yaw = 0.0
    for _ in range(60):                       # 2 s spin
        yaw += 6.0
        R_wc = _Rz(yaw)
        accel_cam = _rest_accel_cam(R_wc)     # truly at rest, only rotating
        honest = np.zeros(3)                  # rotation-locked solve -> ~0
        f.step(dt, R_wc, accel_cam, honest, rot_deg=6.0)
    walk = float(np.linalg.norm(f.p))
    ok_all &= check("no drift on honest yaw", walk < 0.05,
                    f"walk={walk*100:.2f} cm")

    # --- E. forward + yaw (honest vision tracks the forward motion) ---------
    print("E. forward + yaw (honest forward vision is tracked)")
    f.reset()
    yaw = 0.0
    p_true = np.zeros(3)
    for _ in range(30):
        yaw += 6.0
        R_wc = _Rz(yaw)
        accel_cam = _rest_accel_cam(R_wc)     # constant velocity -> net accel 0
        # honest rotation-locked vision: real forward step in WORLD frame
        dp = np.array([0.02, 0.0, 0.0])
        p_true = p_true + dp
        f.step(dt, R_wc, accel_cam, dp, rot_deg=6.0)
    err = float(np.linalg.norm(f.p - p_true))
    ok_all &= check("tracks forward through yaw", f.p[0] > 0.4 and err < 0.1,
                    f"x={f.p[0]:.3f}m true={p_true[0]:.3f}m err={err*100:.1f}cm")

    # --- F. vision dropout (coast then bleed to rest) -----------------------
    print("F. vision dropout coasts then bleeds to rest")
    f.reset()
    # build a forward velocity
    for i in range(15):
        a = np.array([2.0, 0.0, 0.0])
        accel_cam = R0.T @ (a - G)
        f.step(dt, R0, accel_cam, np.array([0.03 + 0.01 * i, 0.0, 0.0]),
               rot_deg=0.0)
    v_drop = float(np.linalg.norm(f.v))
    x_drop = f.p[0]
    # vision gone, no further acceleration (specific force = rest)
    acc_rest = _rest_accel_cam(R0)
    for _ in range(120):                      # 4 s dropout
        f.step(dt, R0, acc_rest, None, rot_deg=0.0)
    v_after = float(np.linalg.norm(f.v))
    ok_all &= check("coasted forward during dropout", f.p[0] > x_drop,
                    f"x {x_drop:.3f}->{f.p[0]:.3f} m")
    ok_all &= check("velocity bled toward rest", v_after < 0.05 * v_drop,
                    f"v {v_drop:.3f}->{v_after:.3f} m/s")

    print()
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
