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
# "unit" exponential (exact identity at zero) -- re-exported here under the names
# the odometry / PnP / windowed-BA importers expect (``so3_exp`` / ``so3_log`` /
# ``so3_right_jacobian``). Numerics are byte-identical to the former local copies.
from sky.math import se3_exp_unit as _se3_exp
from sky.math import se3_log as _se3_log
from sky.math import skew as _skew
from sky.math import so3_exp_unit as so3_exp
from sky.math import so3_log, so3_right_jacobian


class ImuNoise:
    """Continuous-time IMU white-noise densities used to weight the IMU factor.

    These are the per-axis spectral densities of the gyro / accel measurement
    noise (the same numbers Basalt / VINS call ``gyro_noise_std`` and
    ``accel_noise_std``). A continuous density ``sigma_c`` (units ``rad/s/sqrt(Hz)``
    for the gyro, ``m/s^2/sqrt(Hz)`` for the accel) becomes a per-segment discrete
    covariance ``sigma_c^2 / dt`` over a step of length ``dt`` -- the form used in
    the covariance propagation below.

    The bias random-walk densities are NOT used here: the bias states live on the
    keyframes and are tied by the bias-random-walk residual in the optimizer
    (``vio_window._bias_rw_residual``); this class only covers the additive
    measurement noise that drives the 9-state preintegration covariance.

    Defaults are reasonable order-of-magnitude values for the OAK-D's Bosch
    BMI270 (consumer MEMS): they make the covariance physically sane (cm-scale
    position 1-sigma over a ~0.25 s keyframe interval). They are config knobs, not
    constants -- tune per device once a noise study is available.
    """

    __slots__ = ("sigma_g", "sigma_a")

    def __init__(self, sigma_g: float = 1.5e-3, sigma_a: float = 2.0e-2):
        self.sigma_g = float(sigma_g)   # gyro  noise density [rad/s/sqrt(Hz)]
        self.sigma_a = float(sigma_a)   # accel noise density [m/s^2/sqrt(Hz)]


# Default measurement-noise densities used when ``preintegrate_imu`` is called
# without an explicit ``noise=`` (keeps the existing call sites working while
# still producing a sane covariance).
DEFAULT_IMU_NOISE = ImuNoise()


class ImuPreintegration:
    """Result of preintegrating IMU between two times (body/IMU frame).

    Holds the preintegrated rotation/velocity/position increments ``dR, dv, dp``
    over the interval ``dt`` seconds, plus the first-order Jacobians w.r.t. the
    gyro/accel biases used at integration time, so a slightly changed bias
    estimate can correct the deltas WITHOUT re-integrating the raw samples
    (Forster et al., "On-Manifold Preintegration", TRO 2017).

    It ALSO carries the 9x9 preintegration covariance ``cov`` of the noise on the
    9-state delta ``eta = [dphi(3); dvel(3); dpos(3)]`` (same ordering as the IMU
    residual in :func:`vio.mathlib.backend.vio_window._imu_residual`), plus its
    information square root ``sqrt_info`` such that ``sqrt_info.T @ sqrt_info ==
    inv(cov)``. ``sqrt_info`` is the whitening matrix ``Omega_I = cov^-1``'s upper
    Cholesky factor: whitening a raw 9-residual ``r`` with ``sqrt_info @ r`` gives
    a residual with identity covariance, i.e. the correct IMU-factor weight. These
    covariance fields are ADDITIVE -- ``dR/dv/dp/dt`` and the five bias Jacobians
    are byte-unchanged from the pre-covariance version, so every existing consumer
    (the loose path, ``corrected()``, the byte-parity oracle) is unaffected.

    All quantities are in the IMU/body frame; the extrinsic to the camera is
    applied by the optimizer, not here.
    """

    __slots__ = ("dR", "dv", "dp", "dt", "bg", "ba",
                 "dR_dbg", "dv_dbg", "dv_dba", "dp_dbg", "dp_dba",
                 "cov", "sqrt_info")

    def __init__(self, dR, dv, dp, dt, bg, ba,
                 dR_dbg, dv_dbg, dv_dba, dp_dbg, dp_dba,
                 cov=None, sqrt_info=None):
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
        # 9x9 covariance of [dphi; dvel; dpos] and its sqrt-information. Both are
        # ``None`` only for an empty / degenerate interval (no usable segment).
        self.cov = cov
        self.sqrt_info = sqrt_info

    def corrected(self, bg_new: np.ndarray, ba_new: np.ndarray):
        """First-order bias-corrected ``(dR, dv, dp)`` for a new bias estimate."""
        dbg = np.asarray(bg_new, np.float64) - self.bg
        dba = np.asarray(ba_new, np.float64) - self.ba
        dR = self.dR @ so3_exp(self.dR_dbg @ dbg)
        dv = self.dv + self.dv_dbg @ dbg + self.dv_dba @ dba
        dp = self.dp + self.dp_dbg @ dbg + self.dp_dba @ dba
        return dR, dv, dp


def _sqrt_information(cov: np.ndarray) -> np.ndarray:
    """Upper-triangular whitening matrix ``L`` with ``L.T @ L == inv(cov)``.

    Computed via an LDL^T / Cholesky factorisation of the covariance (Basalt uses
    ``ldlt().matrixL()`` of the information matrix; here we factor the SPD
    covariance, which is numerically the same whitening up to the transpose). A
    tiny jitter is added on the diagonal if ``cov`` is borderline singular so the
    factorisation never fails on a degenerate (e.g. extremely short) interval.

    Given ``cov = U.T @ U`` (upper Cholesky), ``inv(cov) = inv(U) @ inv(U).T`` and
    ``L = inv(U).T`` is upper-triangular with ``L.T @ L = inv(U) @ inv(U).T =
    inv(cov)``, so whitening ``r -> L @ r`` yields unit covariance.
    """
    n = cov.shape[0]
    eye = np.eye(n)
    jitter = 0.0
    for _ in range(8):
        try:
            U = np.linalg.cholesky(cov + jitter * eye).T   # upper factor: U.T@U=cov
            Uinv = np.linalg.solve(U, eye)                 # inv(U)
            return Uinv.T                                  # L = inv(U).T, upper-tri
        except np.linalg.LinAlgError:
            jitter = 1e-12 if jitter == 0.0 else jitter * 10.0
    # Last-resort pseudo-inverse whitening (kept SPD by symmetrising).
    info = np.linalg.pinv(0.5 * (cov + cov.T))
    w, V = np.linalg.eigh(0.5 * (info + info.T))
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def preintegrate_imu(ts_ns: np.ndarray, gyro: np.ndarray, accel: np.ndarray,
                     bg: np.ndarray, ba: np.ndarray,
                     noise: "ImuNoise | None" = None) -> ImuPreintegration:
    """Preintegrate a contiguous block of IMU samples (body frame).

    Parameters
    ----------
    ts_ns : (K,) int64 device-clock nanoseconds, strictly increasing.
    gyro  : (K,3) rad/s in the IMU frame.
    accel : (K,3) m/s^2 specific force in the IMU frame.
    bg, ba: (3,) gyro / accel bias to subtract (linearisation point).
    noise : optional :class:`ImuNoise` measurement-noise densities used ONLY for
            the (additive) 9x9 covariance propagation. Defaults to
            :data:`DEFAULT_IMU_NOISE`. It does NOT affect ``dR/dv/dp`` or the
            bias Jacobians, so passing it never changes the loose-path numerics.

    Returns an :class:`ImuPreintegration`. The increments satisfy, for body
    poses ``(R_i,p_i,v_i)`` at the first sample and ``(R_j,p_j,v_j)`` at the
    last, with world gravity ``g``::

        R_j  ~= R_i @ dR
        v_j  ~= v_i + g*dt + R_i @ dv
        p_j  ~= p_i + v_i*dt + 0.5*g*dt^2 + R_i @ dp

    (forward-Euler segment integration; error -> 0 as the sample rate rises).

    Covariance
    ----------
    Alongside the deltas it propagates the 9x9 covariance ``cov`` of the noise on
    ``eta = [dphi; dvel; dpos]`` (the IMU-residual ordering) via the standard
    Forster discrete recursion ``cov <- A cov A^T + B (Q/dt) B^T`` per segment,
    where ``A = d eta_{k+1} / d eta_k``, ``B = d eta_{k+1} / d (gyro,accel
    noise)`` use the SAME midpoint sample and the SAME dp-before-dv ordering as
    the delta update above (so the weight matches the residual exactly). The
    sqrt-information ``sqrt_info`` (``sqrt_info.T @ sqrt_info == inv(cov)``) is
    exposed for whitening the IMU factor; ``Omega_I = inv(cov)``.
    """
    ts = np.asarray(ts_ns, np.int64)
    g = np.asarray(gyro, np.float64)
    a = np.asarray(accel, np.float64)
    bg = np.asarray(bg, np.float64)
    ba = np.asarray(ba, np.float64)
    noise = noise if noise is not None else DEFAULT_IMU_NOISE

    dR = np.eye(3)
    dv = np.zeros(3)
    dp = np.zeros(3)
    dR_dbg = np.zeros((3, 3))
    dv_dbg = np.zeros((3, 3))
    dv_dba = np.zeros((3, 3))
    dp_dbg = np.zeros((3, 3))
    dp_dba = np.zeros((3, 3))
    t_acc = 0.0

    # 9x9 covariance of [dphi; dvel; dpos] (residual order). Continuous
    # densities -> per-segment discrete covariances Q/dt = sigma_c^2 / dt.
    cov = np.zeros((9, 9))
    sg2 = noise.sigma_g * noise.sigma_g     # gyro  density^2 [ (rad/s)^2 / Hz ]
    sa2 = noise.sigma_a * noise.sigma_a     # accel density^2 [ (m/s^2)^2 / Hz ]
    I3 = np.eye(3)
    had_step = False

    for k in range(len(ts) - 1):
        dt = (int(ts[k + 1]) - int(ts[k])) * 1e-9
        if dt <= 0:
            continue
        # Midpoint sample over the segment (trapezoidal in the raw signal).
        w = 0.5 * (g[k] + g[k + 1]) - bg
        acc = 0.5 * (a[k] + a[k + 1]) - ba

        # --- covariance propagation (uses dR == dR_{i,k} BEFORE its update) ---
        # State-transition A_k and noise-input B_k for eta = [dphi; dvel; dpos],
        # consistent with the dp-before-dv delta update below:
        #   dphi_{k+1} = dR_inc^T dphi_k                       + (Jr dt) n_g
        #   dvel_{k+1} = -dR skew(acc) dt dphi_k + dvel_k      + (dR dt) n_a
        #   dpos_{k+1} = -0.5 dR skew(acc) dt^2 dphi_k
        #                + dt dvel_k + dpos_k                  + (0.5 dR dt^2) n_a
        # (position uses the PRE-update dvel, matching ``dp += dv*dt``.)
        phi = w * dt
        dR_inc = so3_exp(phi)
        Jr = so3_right_jacobian(phi)
        Rk_sk = dR @ _skew(acc)                  # dR_k @ skew(acc)

        A_k = np.zeros((9, 9))
        A_k[0:3, 0:3] = dR_inc.T
        A_k[3:6, 0:3] = -Rk_sk * dt
        A_k[3:6, 3:6] = I3
        A_k[6:9, 0:3] = -0.5 * Rk_sk * dt * dt
        A_k[6:9, 3:6] = I3 * dt
        A_k[6:9, 6:9] = I3

        B_k = np.zeros((9, 6))                    # cols [n_g(3) | n_a(3)]
        B_k[0:3, 0:3] = Jr * dt
        B_k[3:6, 3:6] = dR * dt
        B_k[6:9, 3:6] = 0.5 * dR * dt * dt

        # Discrete noise covariance over this step: Q/dt with Q = diag(sg2,sa2).
        Qd = np.zeros((6, 6))
        Qd[0:3, 0:3] = (sg2 / dt) * I3
        Qd[3:6, 3:6] = (sa2 / dt) * I3
        cov = A_k @ cov @ A_k.T + B_k @ Qd @ B_k.T

        # --- delta + bias-Jacobian update (BIT-UNCHANGED from before) ---------
        # Position + velocity increments use the CURRENT dR (= dR_{i,k}); update
        # position before velocity (it uses the pre-update dv). Same ordering for
        # the bias Jacobians.
        aR = dR @ acc
        dp = dp + dv * dt + 0.5 * aR * dt * dt
        dv = dv + aR * dt

        dp_dba = dp_dba + dv_dba * dt - 0.5 * dR * dt * dt
        dp_dbg = dp_dbg + dv_dbg * dt - 0.5 * (Rk_sk @ dR_dbg) * dt * dt
        dv_dba = dv_dba - dR * dt
        dv_dbg = dv_dbg - (Rk_sk @ dR_dbg) * dt

        # Rotation increment + its gyro-bias Jacobian recursion.
        dR_dbg = dR_inc.T @ dR_dbg - Jr * dt
        dR = dR @ dR_inc
        t_acc += dt
        had_step = True

    # Only build the (symmetric) sqrt-information when at least one real segment
    # was integrated; an empty interval leaves cov/sqrt_info as None so callers
    # can detect a degenerate factor.
    cov_out = None
    sqrt_info = None
    if had_step:
        cov_out = 0.5 * (cov + cov.T)            # enforce exact symmetry
        sqrt_info = _sqrt_information(cov_out)

    return ImuPreintegration(dR, dv, dp, t_acc, bg.copy(), ba.copy(),
                             dR_dbg, dv_dbg, dv_dba, dp_dbg, dp_dba,
                             cov_out, sqrt_info)



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


def imu_at_rest(gyro: np.ndarray, accel: np.ndarray,
                gravity: float = 9.81,
                gyro_thresh: float = 0.15,
                accel_dev_thresh: float = 0.3) -> bool:
    """Zero-velocity (ZUPT) detector: True iff the IMU block reads "at rest".

    Used by the live tight-coupled forward-propagation to decide when to apply a
    Zero-Velocity Update: at rest the only honest motion estimate is "no motion",
    so holding velocity at zero (and freezing translation) for that frame removes
    the accel-bias / gravity-residual drift that otherwise walks the dead-reckoned
    pose off while the device is sitting still. This is what keeps the static
    drift better than a pure forward-integrating filter -- the per-frame IMU
    propagation moves the pose ONLY when there is real motion.

    The two-sided test is the standard MEMS at-rest gate (e.g. used by foot-mounted
    INS ZUPT and by VINS-style stationarity checks):

    * angular rate is small (mean ``|gyro|`` below ``gyro_thresh`` rad/s), AND
    * specific force magnitude is close to gravity (mean ``|accel|`` within
      ``accel_dev_thresh`` m/s^2 of ``gravity``) -- i.e. the only force the
      accelerometer feels is gravity, so the linear acceleration is ~0.

    Both conditions must hold: a fast pure rotation has |accel| ~ g but high gyro
    (NOT at rest), and a free-fall / strong push has low gyro but |accel| far from
    g (NOT at rest).

    Known limitation (deliberate bias toward NOT freezing real motion)
    -----------------------------------------------------------------
    The accel test is on the magnitude only, so a translation whose acceleration
    is *perpendicular* to gravity barely changes ``|accel|`` (a lateral push of
    ``a`` reads ``sqrt(g^2 + a^2) - g ~= a^2/(2g)``: a 2 m/s^2 lateral push moves
    the magnitude only 0.2 m/s^2). The band is therefore set TIGHT
    (``accel_dev_thresh = 0.3``, vs PreintegratePrior's looser 0.6 keyframe-gravity
    gate): this rejects any push of >= ~2.4 m/s^2 lateral (and ANY forward/vertical
    push >= 0.3) as motion so the dead-reckoning is NOT frozen, while a true-rest
    accel bias (~0.05-0.1 m/s^2 on consumer MEMS) stays comfortably inside it. The
    residual blind spot -- a very gentle (<2.4 m/s^2) PURELY lateral creep with no
    rotation -- is the safe failure mode here: ZUPT freezes for that frame and the
    next clear motion / keyframe re-anchor recovers it (preferring "hold still" to
    "drift" is exactly the static-drift win). A magnetometer-free single-frame
    detector cannot do better without a velocity-consistency state.

    Parameters
    ----------
    gyro  : (M,3) rad/s samples over the frame interval (any frame: IMU or camera
            frame -- the magnitude is frame-invariant).
    accel : (M,3) m/s^2 specific-force samples over the frame interval.
    """
    g = np.asarray(gyro, np.float64)
    a = np.asarray(accel, np.float64)
    if a.size == 0:
        return False
    gyro_mag = 0.0 if g.size == 0 else float(np.linalg.norm(g, axis=1).mean())
    accel_mag = float(np.linalg.norm(a, axis=1).mean())
    gyro_still = gyro_mag < gyro_thresh
    accel_still = abs(accel_mag - gravity) < accel_dev_thresh
    return bool(gyro_still and accel_still)


def predict_state(R_wb: np.ndarray, p_wb: np.ndarray, v_w: np.ndarray,
                  ts_ns: np.ndarray, gyro: np.ndarray, accel: np.ndarray,
                  bg: np.ndarray, ba: np.ndarray, g_world: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-propagate one nav-state over a raw IMU block (Basalt predictState).

    This is the per-frame inertial dead-reckoning the loosely-coupled path never
    had: given a starting body->world nav-state ``(R_wb, p_wb, v_w)`` and the raw
    IMU samples spanning the next frame interval, it integrates the gyro into the
    rotation and the (gravity-removed) accelerometer into velocity then position,
    so the live pose KEEPS MOVING from the IMU alone when vision is absent
    (covered camera) or too weak to solve (white wall). It is the discrete
    forward integration matched to :func:`preintegrate_imu`'s increment
    convention -- the same midpoint sample, the same dp-before-dv ordering -- so a
    keyframe's vision-corrected nav-state and the per-frame propagation between
    keyframes are consistent.

    The samples are assumed already expressed in the body == camera optical frame
    (the caller rotates raw IMU by ``R_imu_cam`` before calling, exactly as
    :class:`vio.mathlib.backend.vio_window.WindowedVIOMap` expects). Biases are
    subtracted per sample (gyro/accel bias linearisation point).

    Integration per segment ``k -> k+1`` (forward-Euler on the midpoint sample)::

        w   = 0.5*(gyro[k]+gyro[k+1]) - bg            # bias-corrected rate
        acc = 0.5*(accel[k]+accel[k+1]) - ba          # bias-corrected force (body)
        a_w = R_wb @ acc + g_world                    # world-frame acceleration
        p  += v*dt + 0.5*a_w*dt^2                      # position BEFORE velocity
        v  += a_w*dt                                   # ... so p uses the pre-step v
        R_wb = R_wb @ Exp(w*dt)                        # rotation increment

    ``g_world`` is the gravity ACCELERATION vector (optical-world "down" = +y, so
    ``[0, +9.81, 0]``); since the accelerometer measures specific force (it reads
    ``+g`` upward at rest), the world acceleration is ``R_wb @ accel + g_world``
    (gravity adds, removing the reaction the sensor reports). This matches the
    ``v_j = v_i + g*dt + R_i @ dv`` convention in :func:`preintegrate_imu`.

    Returns the propagated ``(R_wb, p_wb, v_w)``. With fewer than two samples (no
    ``dt`` to integrate) the inputs are returned unchanged.
    """
    ts = np.asarray(ts_ns, np.int64)
    g = np.asarray(gyro, np.float64)
    a = np.asarray(accel, np.float64)
    bg = np.asarray(bg, np.float64)
    ba = np.asarray(ba, np.float64)
    gw = np.asarray(g_world, np.float64)
    R = np.asarray(R_wb, np.float64).copy()
    p = np.asarray(p_wb, np.float64).copy()
    v = np.asarray(v_w, np.float64).copy()
    if ts.size < 2:
        return R, p, v
    for k in range(ts.size - 1):
        dt = (int(ts[k + 1]) - int(ts[k])) * 1e-9
        if dt <= 0:
            continue
        w = 0.5 * (g[k] + g[k + 1]) - bg          # midpoint, bias-corrected
        acc = 0.5 * (a[k] + a[k + 1]) - ba
        a_w = R @ acc + gw                        # world-frame linear accel
        # Position uses the PRE-step velocity (matches preintegrate_imu's
        # dp-before-dv ordering), then velocity is advanced, then the rotation.
        p = p + v * dt + 0.5 * a_w * dt * dt
        v = v + a_w * dt
        R = R @ so3_exp(w * dt)
    return R, p, v


def complementary_correct(R_wb: np.ndarray, p_wb: np.ndarray, v_w: np.ndarray,
                          R_vis: np.ndarray, p_vis: np.ndarray,
                          dt_anchor: float,
                          k_pos: float, k_vel: float, k_rot: float
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth complementary-filter pull of an IMU dead-reckoned nav-state toward
    a fresh vision fix -- the soft replacement for a hard ``p = p_vis`` re-anchor.

    The live ``(R_wb, p_wb, v_w)`` dead-reckons from the IMU *continuously*; this
    nudges it a BOUNDED FRACTION of the way toward the vision pose each time a
    fresh absolute fix arrives, instead of snapping the whole pose (and injecting
    a one-shot ``(p_vis - anchor)/dt`` velocity) at keyframes. The result is the
    instant, smooth response the user asked for: a fast push shows up immediately
    in the dead-reckoned pose, and vision pulls the accumulated drift back over a
    few keyframes with no visible snap and no overshoot from a bad injected vel.

    The correction is a first-order complementary blend (the standard
    error-state feedback form ``x <- x + K (z - x)``)::

        e_p   = p_vis - p_wb                     # position error toward vision
        p_wb <- p_wb + k_pos * e_p               # close a fraction of it now
        v_w  <- v_w  + k_vel * e_p / dt_anchor   # bleed the SAME error into vel
        R_wb <- R_wb @ Exp( k_rot * Log(R_wb^T R_vis) )   # SLERP-toward vision

    * ``k_pos`` (0..1) is the fraction of the *position* error closed this update.
      ``< 1`` so the live pose never jumps to the vision pose -- the drift is
      removed gradually but fully over a few corrections (geometric decay).
    * ``k_vel`` (0..1) feeds the position error back into the *velocity* over the
      anchor interval ``dt_anchor`` (units 1/s via the ``/dt_anchor``). A small
      positive ``k_vel`` damps the drift RATE without the destabilising full
      ``v = displacement/dt`` injection of the old hard re-anchor. With
      ``dt_anchor <= 0`` (degenerate / first interval) the velocity term is
      skipped (no division), leaving velocity untouched.
    * ``k_rot`` (0..1) is the rotational analogue: a fraction of the geodesic
      rotation error ``Log(R_wb^T R_vis)`` applied on the right (a bounded SLERP
      toward the vision attitude), so attitude is vision-anchored without a hard
      orientation jump.

    All gains in ``[0, 1]`` keep the filter stable (each error component decays
    geometrically, never overshoots). The returned ``(R, p, v)`` are fresh arrays;
    inputs are not mutated. Vision-frame inputs must already be in the body==camera
    optical world frame (the caller converts ``T_cw`` -> ``(R_wb, p_wb)``).
    """
    R = np.asarray(R_wb, np.float64)
    p = np.asarray(p_wb, np.float64).copy()
    v = np.asarray(v_w, np.float64).copy()
    R_vis = np.asarray(R_vis, np.float64)
    p_vis = np.asarray(p_vis, np.float64)

    # --- position: close a bounded fraction of the error toward vision --------
    e_p = p_vis - p
    p = p + k_pos * e_p
    # --- velocity: bleed the SAME position error in as a damped rate term ------
    # (1/dt_anchor turns the metres of error into a m/s correction; skipped when
    # the interval is degenerate so there is no divide-by-zero / huge velocity.)
    if dt_anchor > 1e-6 and k_vel != 0.0:
        v = v + (k_vel / dt_anchor) * e_p
    # --- rotation: bounded SLERP of the attitude toward the vision attitude ----
    if k_rot != 0.0:
        e_phi = so3_log(R.T @ R_vis)          # body-frame geodesic error (rad)
        R = R @ so3_exp(k_rot * e_phi)
    else:
        R = R.copy()
    return R, p, v


def loop_correction_delta(R_pre: np.ndarray, p_pre: np.ndarray,
                          R_corr: np.ndarray, p_corr: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """World-frame SE(3) delta that maps a keyframe's PRE-correction body->world
    pose onto its SLAM-corrected body->world pose.

    SLAM's pose-graph optimisation rewrites a keyframe's pose from ``(R_pre,
    p_pre)`` to ``(R_corr, p_corr)`` (both body->world == camera->world here). The
    correction is the LEFT (world-frame) transform ``T_delta`` such that
    ``T_corr = T_delta @ T_pre``; applying the SAME ``T_delta`` to the live pose
    (which has dead-reckoned PAST that keyframe carrying the same accumulated
    drift) pulls the live trajectory back onto the loop-corrected one.

    Because both poses describe the same physical keyframe, ``T_delta`` is the
    accumulated drift the loop closure removed::

        T_delta = T_corr @ inv(T_pre)
        R_delta = R_corr @ R_pre^T
        p_delta = p_corr - R_delta @ p_pre

    Returns ``(R_delta, p_delta)`` -- the rotation + translation of the world-frame
    delta. Applying it to any body->world pose ``(R, p)`` is
    ``(R_delta @ R, R_delta @ p + p_delta)`` (see :func:`apply_se3_left`).
    """
    R_pre = np.asarray(R_pre, np.float64)
    p_pre = np.asarray(p_pre, np.float64)
    R_corr = np.asarray(R_corr, np.float64)
    p_corr = np.asarray(p_corr, np.float64)
    R_delta = R_corr @ R_pre.T
    p_delta = p_corr - R_delta @ p_pre
    return R_delta, p_delta


def scale_se3_delta(R_delta: np.ndarray, p_delta: np.ndarray, gain: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """A BOUNDED fraction ``gain`` of a world-frame SE(3) delta, on-manifold.

    Interpolates the SE(3) transform ``(R_delta, p_delta)`` toward identity by
    ``gain`` in ``[0, 1]`` via the matrix exponential of a scaled twist:
    ``T_step = Exp(gain * Log(T_delta))``. ``gain = 1`` returns the full delta,
    ``gain = 0`` returns identity. This is the SMOOTH partial application used to
    bleed a loop-closure correction onto the live pose over several frames instead
    of a hard one-shot snap -- the geodesic interpolation keeps rotation +
    translation coupled correctly (no shear from scaling R and p independently).

    Returns ``(R_step, p_step)`` -- the partial world-frame delta to left-apply.
    """
    g = float(np.clip(gain, 0.0, 1.0))
    R_delta = np.asarray(R_delta, np.float64)
    p_delta = np.asarray(p_delta, np.float64)
    if g <= 0.0:
        return np.eye(3), np.zeros(3)
    # se3 twist of the full delta, scaled by the per-frame gain, re-exponentiated.
    T = np.eye(4)
    T[:3, :3] = R_delta
    T[:3, 3] = p_delta
    xi = _se3_log(T)
    T_step = _se3_exp(g * xi)
    return T_step[:3, :3].copy(), T_step[:3, 3].copy()


def apply_se3_left(R_delta: np.ndarray, p_delta: np.ndarray,
                   R: np.ndarray, p: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Left-multiply a world-frame SE(3) delta onto a body->world pose ``(R, p)``.

    ``(R', p') = T_delta @ (R, p)`` == ``(R_delta @ R, R_delta @ p + p_delta)``.
    Returns fresh arrays; inputs are not mutated.
    """
    R_delta = np.asarray(R_delta, np.float64)
    p_delta = np.asarray(p_delta, np.float64)
    R_new = R_delta @ np.asarray(R, np.float64)
    p_new = R_delta @ np.asarray(p, np.float64) + p_delta
    return R_new, p_new


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
