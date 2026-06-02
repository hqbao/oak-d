"""IMU integration for a visual-inertial motion prior (pure numpy).

The recorded sessions carry a 200 Hz IMU (gyro rad/s + accel m/s^2) on the same
device clock as the camera frames, plus the IMU<->camera extrinsics in
``calib.json``. That is exactly what a VIO needs to predict how the camera moved
*between* two image frames, before looking at the images at all.

This module does the cheap, robust half of that: **gyro preintegration**. Given
the gyro samples that fall between two frame timestamps, it integrates them into
a single rotation increment and expresses it in the camera frame using the
IMU->camera extrinsics. That rotation is then used to seed PnP, which is the part
that matters most when the camera rotates fast (the regime where pure-vision KLT
struggles).

Accelerometer double-integration for translation is deliberately *not* done here:
without estimating accel bias + gravity in a proper filter it adds more drift
than it removes, and metric stereo depth already gives us translation. We only
take the well-conditioned, bias-tolerant signal (short-interval gyro rotation).

Measured benefit on the recorded gold sessions (2026-06-02)
----------------------------------------------------------
As a *seed* this is currently a no-op: with the well-synchronised stereo depth,
vision PnP already converges on every frame (0 failures across all sessions), so
the starting rotation doesn't change the converged solution. Forcing the gyro
rotation as a *hard* constraint is strictly worse (gyro bias drift exceeds the
vision rotation error). It is kept ON because it is theoretically correct and a
cheap robustness fallback for when vision degrades (dropped frames, motion blur,
feature-starved views). A real accuracy gain from IMU needs tight coupling with
online bias estimation (preintegration factors in a sliding-window bundle
adjustment) -- a larger build than this seed.
"""
from __future__ import annotations

import numpy as np


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map: rotation vector (rad) -> 3x3 rotation matrix."""
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class GyroPreintegrator:
    """Integrates gyro samples into inter-frame rotation, in the camera frame.

    Parameters
    ----------
    ts_ns, gyro:
        The full IMU stream for a session: ``ts_ns`` shape ``(N,)`` (device-clock
        nanoseconds, sorted), ``gyro`` shape ``(N, 3)`` rad/s in the IMU frame.
    T_imu_cam:
        4x4 IMU->camera extrinsic (the recorded ``T_imu_left``). Its rotation maps
        a vector from the IMU frame into the camera frame.
    gyro_bias:
        Optional constant gyro bias (rad/s) subtracted from every sample. If None
        and ``estimate_bias_window_s`` > 0, the bias is estimated from the first
        seconds of the stream (assumed near-static at startup).
    """

    def __init__(self, ts_ns: np.ndarray, gyro: np.ndarray, T_imu_cam: np.ndarray,
                 gyro_bias: np.ndarray | None = None,
                 estimate_bias_window_s: float = 1.0):
        order = np.argsort(ts_ns)
        self.ts = np.asarray(ts_ns, dtype=np.int64)[order]
        self.gyro = np.asarray(gyro, dtype=np.float64)[order]
        self.R_imu_cam = np.asarray(T_imu_cam, dtype=np.float64)[:3, :3]

        if gyro_bias is not None:
            self.bias = np.asarray(gyro_bias, dtype=np.float64)
        elif estimate_bias_window_s > 0 and len(self.ts) > 1:
            t0 = self.ts[0]
            win = self.ts <= t0 + int(estimate_bias_window_s * 1e9)
            self.bias = self.gyro[win].mean(axis=0) if win.any() else np.zeros(3)
        else:
            self.bias = np.zeros(3)

    def delta_rotation(self, t0_ns: int, t1_ns: int) -> np.ndarray:
        """Camera-frame rotation R_cam(t0->t1) from gyro between two timestamps.

        Integrates angular velocity over the IMU samples in ``[t0, t1]`` (trapezoid
        in time), forming an IMU-frame rotation, then conjugates by the IMU->cam
        extrinsic so the result rotates points in the camera frame:
        ``R_cam = R_imu_cam @ R_imu @ R_imu_cam^T``.
        Returns identity if the interval is empty or degenerate.
        """
        if t1_ns <= t0_ns:
            return np.eye(3)
        lo = np.searchsorted(self.ts, t0_ns, side="left")
        hi = np.searchsorted(self.ts, t1_ns, side="right")
        idx = np.arange(max(lo - 1, 0), min(hi + 1, len(self.ts)))
        if idx.size < 2:
            return np.eye(3)

        R_imu = np.eye(3)
        ts = self.ts[idx]
        w = self.gyro[idx] - self.bias
        for j in range(len(idx) - 1):
            # clamp the segment to the requested [t0, t1] window
            a = max(int(ts[j]), t0_ns)
            b = min(int(ts[j + 1]), t1_ns)
            dt = (b - a) * 1e-9
            if dt <= 0:
                continue
            w_mid = 0.5 * (w[j] + w[j + 1])  # trapezoidal angular velocity
            R_imu = R_imu @ so3_exp(w_mid * dt)

        return self.R_imu_cam @ R_imu @ self.R_imu_cam.T


def gravity_aligned_R0(accel_cam: np.ndarray) -> np.ndarray:
    """Initial camera->world rotation that levels the optical world to gravity.

    ``accel_cam`` is the accelerometer specific-force reading (m/s^2) expressed
    in the camera **optical** frame (x right, y down, z forward), averaged over a
    near-static startup window. At rest the accelerometer measures +g along the
    *upward* axis, so gravity ("down") in the camera frame is ``-accel_cam``.

    The returned rotation ``R0`` (camera->world, i.e. the value to seed
    ``RGBDVisualOdometry.pose[:3, :3]`` with) defines a world frame whose optical
    "down" axis (+y) is aligned with real gravity and whose forward axis (+z) is
    the horizontal projection of the camera's starting forward direction. Yaw is
    left at the camera's starting heading -- there is no magnetometer, so absolute
    yaw is undefined (this matches Basalt, which also leaves yaw free).

    Verified on the gold sessions: the resulting startup roll/pitch agrees with
    Basalt's gravity-leveled attitude to < 1 deg on near-static starts.
    """
    a = np.asarray(accel_cam, dtype=np.float64)
    na = float(np.linalg.norm(a))
    if na < 1e-6:
        return np.eye(3)
    down = -a / na                          # gravity dir in cam = world +y (down)
    fwd = np.array([0.0, 0.0, 1.0])         # camera forward (optical +z)
    fwd = fwd - (fwd @ down) * down         # horizontalise (perp to gravity)
    if np.linalg.norm(fwd) < 1e-6:          # camera staring straight up/down
        fwd = np.array([1.0, 0.0, 0.0])
        fwd = fwd - (fwd @ down) * down
    fwd /= np.linalg.norm(fwd)
    right = np.cross(down, fwd)             # optical x = y (down) cross z (fwd)
    right /= np.linalg.norm(right)
    # Columns = world axes (right, down, fwd) expressed in the camera frame,
    # i.e. R_{cam<-world}. The initial camera->world pose rotation is its inverse.
    R_cam_from_world = np.column_stack([right, down, fwd])
    return R_cam_from_world.T
