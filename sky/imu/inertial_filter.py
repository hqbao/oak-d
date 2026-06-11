"""Loosely-coupled inertial translation filter for the live display.

This is the principled replacement for the pile of per-frame heuristics
(``t_trust`` / rotation-gated damping / velocity coast / correction freeze) that
were fighting each other on the device. It mirrors the structure of Basalt's
``SqrtKeypointVioEstimator::measure()`` -- **predict the state forward with the
IMU every frame, then correct it with vision** -- but as a lightweight
complementary filter on translation/velocity instead of a full sliding-window
optimiser, matching the "loosely-coupled EKF MVP" prescribed in
``docs/SKYSLAM_RESEARCH.md``.

Why a velocity state fixes everything the heuristics could not
-------------------------------------------------------------
The single missing ingredient was a persistent **world velocity**. With it:

* **Still** -> the accelerometer net (specific force + gravity) is ~0, so the
  prediction does not move; vision keeps measuring ~0 displacement and pins the
  velocity to ~0. No "drift while standing still".
* **Fast forward push** -> the accelerometer immediately shows real
  acceleration, so the prediction advances the position the SAME frame (no
  waiting for a slow async correction). Vision then refines it. No "stall / lag".
* **In-place yaw from rest** -> the accelerometer shows ~0 net acceleration, so
  the velocity stays ~0; the phantom translation vision reports during the spin
  is down-weighted (it is exactly when vision translation is least trustworthy),
  so the position does not walk off.
* **Forward + yaw** -> the velocity already holds the real forward motion, so
  even while the vision translation is down-weighted during the turn the body
  keeps coasting forward. No freeze.

The accelerometer is the honest discriminator between a real translation and a
rotation-induced phantom -- which is precisely why the rotation-magnitude-only
heuristics could never separate the two.

Frame conventions
-----------------
Everything is in the camera-optical world frame used by ``RGBDVisualOdometry``
(world "down" = +y). ``g_world`` is the gravitational acceleration vector,
``[0, +9.81, 0]``. The accelerometer specific force in the camera frame,
``accel_cam``, relates to the net inertial acceleration by

    a_world = R_wc @ accel_cam + g_world

(at rest ``R_wc @ accel_cam`` points up ~``[0,-9.81,0]`` and cancels ``g_world``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InertialFilterConfig:
    # Gravitational acceleration vector in the optical world (down = +y).
    g_world: tuple = (0.0, 9.81, 0.0)
    # How much the vision displacement measurement is trusted each frame when it
    # is available (0..1). The complement is how much the IMU prediction is kept.
    # The vision displacement is now HONEST (the odometry solves translation with
    # the gyro rotation locked, so a pure yaw yields ~0 -- the phantom is gone at
    # the source), so we trust it heavily and use the accelerometer only as a
    # light predictive feed-forward for responsiveness + to coast through vision
    # dropouts. No rotation-magnitude gate -- that proxy suppressed REAL motion
    # that happened to carry hand-rotation, the "doesn't translate in realtime"
    # symptom.
    vision_trust: float = 0.8
    # Use the accelerometer as a predictive feed-forward for position. DEFAULT
    # OFF, and here is the honest reason: a useful IMU position prediction needs
    # estimated accel bias + scale and a gravity vector tracked in a tightly-
    # coupled optimiser (what Basalt actually does). Our naive
    # ``R_wc @ accel + g`` with an EMA-levelled, drifting ``g_ref`` is NOT that --
    # measured on device + offline it roughly DOUBLES
    # the trajectory jitter (path/net 44.6 vs 27.0 vision-only, Basalt 21.9) and
    # worsens ATE, because the residual gravity/bias error integrates into a
    # spurious velocity that vision then has to fight every frame. So we keep the
    # accelerometer for ATTITUDE leveling only and let vision own translation.
    # Tight-coupled IMU translation lives in the ``vio`` backend's windowed
    # preintegration, where the bias is actually estimated. Flip this on only
    # once that calibration exists.
    use_accel_prediction: bool = False
    # Clip the net world acceleration magnitude (m/s^2) when prediction is on.
    accel_clip: float = 3.0
    # Velocity decay applied when vision is UNAVAILABLE (per frame): a straight
    # move with a brief KLT dropout dead-reckons on the IMU then gently bleeds.
    vel_damp: float = 0.9
    # Hard cap on speed (m/s) to fence off any divergence.
    max_speed: float = 5.0


class InertialTranslationFilter:
    """Predict translation with the IMU, correct it with vision (per frame)."""

    def __init__(self, cfg: InertialFilterConfig | None = None):
        self.cfg = cfg or InertialFilterConfig()
        self.g_world = np.asarray(self.cfg.g_world, dtype=np.float64)
        self.p = np.zeros(3)           # world position (optical)
        self.v = np.zeros(3)           # world velocity (optical)

    def reset(self, p0: np.ndarray | None = None) -> None:
        self.p = np.zeros(3) if p0 is None else np.asarray(p0, float).copy()
        self.v = np.zeros(3)

    def step(self, dt: float, R_wc: np.ndarray, accel_cam: np.ndarray | None,
             dp_vis_world: np.ndarray | None, rot_deg: float = 0.0) -> np.ndarray:
        """Advance the filter one frame and return the world position.

        Parameters
        ----------
        dt : inter-frame time (s).
        R_wc : 3x3 camera-optical -> world rotation (``vo.pose[:3,:3]``).
        accel_cam : specific force in the camera frame (already gravity-rotated,
            i.e. ``R_imu_cam @ accel``), or ``None`` if no IMU this frame.
        dp_vis_world : vision-measured world displacement since the last frame,
            or ``None`` when vision failed (then the IMU prediction coasts). This
            is expected to be the rotation-locked (honest) translation, so it is
            trusted directly without any rotation-magnitude gate.
        rot_deg : unused (kept for call-site compatibility).
        """
        dt = float(max(dt, 1e-4))

        # --- PREDICT: optional accelerometer feed-forward (default OFF) -------
        # See ``use_accel_prediction`` in the config for why this is off: on this
        # hardware the un-calibrated accel adds jitter rather than removing lag.
        if self.cfg.use_accel_prediction and accel_cam is not None:
            a_world = R_wc @ np.asarray(accel_cam, float) + self.g_world
            n = float(np.linalg.norm(a_world))
            if n > self.cfg.accel_clip:           # bound leveling/centripetal spikes
                a_world *= self.cfg.accel_clip / n
            v_pred = self.v + a_world * dt
        else:
            v_pred = self.v

        # --- CORRECT: fuse the (honest) vision displacement ------------------
        if dp_vis_world is not None:
            v_meas = np.asarray(dp_vis_world, float) / dt
            w = self.cfg.vision_trust
            self.v = (1.0 - w) * v_pred + w * v_meas
        else:
            # No vision: dead-reckon on the IMU prediction, gently damped so a
            # straight-move dropout coasts then bleeds instead of diverging.
            self.v = v_pred * self.cfg.vel_damp

        # speed fence
        sp = float(np.linalg.norm(self.v))
        if sp > self.cfg.max_speed:
            self.v *= self.cfg.max_speed / sp

        self.p = self.p + self.v * dt
        return self.p
