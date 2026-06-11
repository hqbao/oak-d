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

# SO(3) primitives live in the shared ``sky.math`` kernel. The IMU module uses the
# "unit" exponential (exact identity at zero); numerics are byte-identical to the
# former local copies.
from sky.math import skew as _skew
from sky.math import so3_exp_unit as so3_exp
from sky.math import so3_right_jacobian


class ImuPreintegration:
    """Result of preintegrating IMU between two times (body/IMU frame).

    Holds the preintegrated rotation/velocity/position increments ``dR, dv, dp``
    over the interval ``dt`` seconds, plus the first-order Jacobians w.r.t. the
    gyro/accel biases used at integration time, so a slightly changed bias
    estimate can correct the deltas WITHOUT re-integrating the raw samples
    (Forster et al., "On-Manifold Preintegration", TRO 2017).

    All quantities are in the IMU/body frame; the extrinsic to the camera is
    applied by the optimizer, not here.
    """

    __slots__ = ("dR", "dv", "dp", "dt", "bg", "ba",
                 "dR_dbg", "dv_dbg", "dv_dba", "dp_dbg", "dp_dba")

    def __init__(self, dR, dv, dp, dt, bg, ba,
                 dR_dbg, dv_dbg, dv_dba, dp_dbg, dp_dba):
        self.dR = dR
        self.dv = dv
        self.dp = dp
        self.dt = dt
        self.bg = bg          # gyro bias used at integration (linearisation pt)
        self.ba = ba          # accel bias used at integration
        self.dR_dbg = dR_dbg
        self.dv_dbg = dv_dbg
        self.dv_dba = dv_dba
        self.dp_dbg = dp_dbg
        self.dp_dba = dp_dba

    def corrected(self, bg_new: np.ndarray, ba_new: np.ndarray):
        """First-order bias-corrected ``(dR, dv, dp)`` for a new bias estimate."""
        dbg = np.asarray(bg_new, np.float64) - self.bg
        dba = np.asarray(ba_new, np.float64) - self.ba
        dR = self.dR @ so3_exp(self.dR_dbg @ dbg)
        dv = self.dv + self.dv_dbg @ dbg + self.dv_dba @ dba
        dp = self.dp + self.dp_dbg @ dbg + self.dp_dba @ dba
        return dR, dv, dp


def preintegrate_imu(ts_ns: np.ndarray, gyro: np.ndarray, accel: np.ndarray,
                     bg: np.ndarray, ba: np.ndarray) -> ImuPreintegration:
    """Preintegrate a contiguous block of IMU samples (body frame).

    Parameters
    ----------
    ts_ns : (K,) int64 device-clock nanoseconds, strictly increasing.
    gyro  : (K,3) rad/s in the IMU frame.
    accel : (K,3) m/s^2 specific force in the IMU frame.
    bg, ba: (3,) gyro / accel bias to subtract (linearisation point).

    Returns an :class:`ImuPreintegration`. The increments satisfy, for body
    poses ``(R_i,p_i,v_i)`` at the first sample and ``(R_j,p_j,v_j)`` at the
    last, with world gravity ``g``::

        R_j  ~= R_i @ dR
        v_j  ~= v_i + g*dt + R_i @ dv
        p_j  ~= p_i + v_i*dt + 0.5*g*dt^2 + R_i @ dp

    (forward-Euler segment integration; error -> 0 as the sample rate rises).
    """
    ts = np.asarray(ts_ns, np.int64)
    g = np.asarray(gyro, np.float64)
    a = np.asarray(accel, np.float64)
    bg = np.asarray(bg, np.float64)
    ba = np.asarray(ba, np.float64)

    dR = np.eye(3)
    dv = np.zeros(3)
    dp = np.zeros(3)
    dR_dbg = np.zeros((3, 3))
    dv_dbg = np.zeros((3, 3))
    dv_dba = np.zeros((3, 3))
    dp_dbg = np.zeros((3, 3))
    dp_dba = np.zeros((3, 3))
    t_acc = 0.0

    for k in range(len(ts) - 1):
        dt = (int(ts[k + 1]) - int(ts[k])) * 1e-9
        if dt <= 0:
            continue
        # Midpoint sample over the segment (trapezoidal in the raw signal).
        w = 0.5 * (g[k] + g[k + 1]) - bg
        acc = 0.5 * (a[k] + a[k + 1]) - ba

        # Position + velocity increments use the CURRENT dR (= dR_{i,k}); update
        # position before velocity (it uses the pre-update dv). Same ordering for
        # the bias Jacobians.
        aR = dR @ acc
        dp = dp + dv * dt + 0.5 * aR * dt * dt
        dv = dv + aR * dt

        Rk_sk = dR @ _skew(acc)
        dp_dba = dp_dba + dv_dba * dt - 0.5 * dR * dt * dt
        dp_dbg = dp_dbg + dv_dbg * dt - 0.5 * (Rk_sk @ dR_dbg) * dt * dt
        dv_dba = dv_dba - dR * dt
        dv_dbg = dv_dbg - (Rk_sk @ dR_dbg) * dt

        # Rotation increment + its gyro-bias Jacobian recursion.
        phi = w * dt
        dR_inc = so3_exp(phi)
        Jr = so3_right_jacobian(phi)
        dR_dbg = dR_inc.T @ dR_dbg - Jr * dt
        dR = dR @ dR_inc
        t_acc += dt

    return ImuPreintegration(dR, dv, dp, t_acc, bg.copy(), ba.copy(),
                             dR_dbg, dv_dbg, dv_dba, dp_dbg, dp_dba)



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


def integrate_gyro_camera(imu_ts: np.ndarray, gyro: np.ndarray,
                          R_imu_cam: np.ndarray) -> np.ndarray | None:
    """Camera-frame rotation from a short, self-contained gyro block.

    Unlike :class:`GyroPreintegrator` (which indexes a whole-session stream by
    absolute timestamps), this integrates the samples carried inside a single
    ``ImuCamPacket`` -- the gyro covering exactly one inter-frame interval. The
    samples are assumed already bias-corrected (ApplyCalibration removes the
    cached bias), so nothing is subtracted here.

    Parameters
    ----------
    imu_ts : (M,) int64 device-clock nanoseconds for the packet, increasing.
    gyro   : (M,3) rad/s in the IMU frame (calibrated).
    R_imu_cam : 3x3 rotation mapping IMU-frame vectors into the camera frame.

    Returns the trapezoidal camera-frame rotation ``R_imu_cam @ R_imu @
    R_imu_cam^T`` for the interval, or ``None`` when fewer than two samples are
    available (no rotation can be formed).
    """
    ts = np.asarray(imu_ts, dtype=np.int64)
    if ts.size < 2:
        return None
    w = np.asarray(gyro, dtype=np.float64)
    R = np.asarray(R_imu_cam, dtype=np.float64)
    R_imu = np.eye(3)
    for j in range(ts.size - 1):
        dt = (int(ts[j + 1]) - int(ts[j])) * 1e-9
        if dt <= 0:
            continue
        w_mid = 0.5 * (w[j] + w[j + 1])  # trapezoidal angular velocity
        R_imu = R_imu @ so3_exp(w_mid * dt)
    return R @ R_imu @ R.T


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
