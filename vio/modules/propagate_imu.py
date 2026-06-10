"""``propagate_imu`` step (TIGHT path only): IMU forward-propagate the live pose.

The gap this closes
-------------------
On the loose path the live displayed position is ``pose.odom``, published EVERY
frame by :class:`~vio.modules.publish_pose.PublishPose` from the per-frame
VISION-ONLY odometry (PnP). When vision is absent (covered camera) or too weak to
solve (white wall) the PnP either fails or freezes translation, so the live pose
FREEZES even while the device is physically moving -- the "covered camera + move =
stays still" symptom. Basalt does not freeze: it propagates the IMU every frame
(predictState) so the live pose keeps reacting to motion (and drifts) until vision
re-locks and pulls it back.

This step adds exactly that, but ONLY on the ``--tight`` path (gated on
``retain_imu`` -- the same flag that turns on per-frame IMU retention). It owns a
live body->world nav-state ``(R, p, v, bg, ba)`` plus the fixed world gravity, and
on EVERY frame the live pose DEAD-RECKONS CONTINUOUSLY from the IMU; vision is
applied only as a SMOOTH partial correction. The three pieces:

1. **Gap-free forward-propagation (instant, full-magnitude response).** The
   retained raw IMU block for this frame is integrated forward under gravity
   (:func:`vio.mathlib.imu.imu.predict_state`). To avoid dropping the segment
   BETWEEN this block's first sample and the previous block's last sample (the
   per-frame packet cut ``(prev_ts, ts]`` shares no boundary sample, so a naive
   per-block integration silently loses ~1-of-N inter-sample segments -> the live
   pose only captures a FRACTION of the true displacement), the previous block's
   final sample is prepended to this block. The interval integrated is therefore
   exactly ``(prev_block_last_ts, this_block_last_ts]`` with no gap -- the full
   accel double-integral, so a fast push shows up at 100 %, not ~50 %.

2. **Velocity-gated ZUPT (no mid-motion pause).** A Zero-Velocity Update freezes
   translation ONLY when the IMU is GENUINELY at rest: accel ~ g AND gyro ~ 0
   (:func:`vio.mathlib.imu.imu.imu_at_rest`) AND the live velocity estimate is
   small (``|v| < _ZUPT_VEL``), sustained for a few frames (hysteresis via
   ``zupt_run``). Accel+gyro ALONE cannot tell "at rest" from "cruising at
   constant velocity" -- both read ``|accel| ~ g``, ``|gyro| ~ 0`` -- so the old
   accel/gyro-only gate froze the pose mid-push during the constant-velocity
   cruise (the PAUSE). Adding the velocity gate keeps the IMU integrating through
   the cruise (``|v|`` is large -> no ZUPT), so the pose tracks the full motion;
   after the push the decel drives ``|v| -> 0`` and ZUPT re-engages -> no rest
   drift (the static-drift win is preserved).

3. **Smooth complementary vision correction (no snap / overshoot).** When a fresh
   vision fix is available (every keyframe), the live nav-state is nudged a
   BOUNDED FRACTION of the way toward it
   (:func:`vio.mathlib.imu.imu.complementary_correct`) -- position, velocity, and
   attitude each get a small error-state feedback term -- instead of the old hard
   ``p = p_vis`` jump + ``v = displacement/dt`` velocity injection. The live pose
   dead-reckons continuously; vision pulls the accumulated drift back gradually
   over a few keyframes. No visible snap, no overshoot from a bad injected
   velocity, drift still fully corrected.

The propagated pose REPLACES ``step.pose`` so the downstream
:class:`PublishPose` emits the IMU-propagated pose on ``pose.odom`` -- the live
marker dead-reckons through any blind interval instead of freezing.

LOOSE path: ``retain_imu`` is False, so this step is a pass-through no-op (it never
allocates a nav-state and never touches ``step.pose``). The byte-parity oracle is
therefore untouched -- ``pose.odom`` stays the vision-only odometry pose.

Placement: this step runs AFTER ``CorrectTilt`` (so ``step.pose`` is the final
vision pose used for the correction) and BEFORE ``PublishPose`` (so the published
``pose.odom`` is the IMU-propagated pose). It also OWNS the keyframe-cadence
counter and stamps ``ctx.state["is_kf_frame"]`` so the later ``EmitKeyframe`` does
not duplicate the cadence (single source of truth).
"""
from __future__ import annotations

import numpy as np

from vio.comms import Step as StepBase
from vio.mathlib.backend.vio_window import T_cw_to_body_world, body_world_to_T_cw
from vio.mathlib.imu.imu import (
    complementary_correct, imu_at_rest, predict_state)
from .step import Step

# --- velocity-gated ZUPT tuning -------------------------------------------- #
# At-rest velocity gate: ZUPT only fires when the live speed estimate is below
# this (m/s). During a push |v| is well above it, so the IMU keeps integrating
# through the constant-velocity cruise (no mid-motion freeze); a true rest sits
# at ~0 m/s, comfortably under the gate. 0.05 m/s = 5 cm/s -- below any real hand
# push, above the residual velocity noise a damped at-rest state carries.
_ZUPT_VEL = 0.05
# Hysteresis: require the accel/gyro at-rest gate to hold for this many
# CONSECUTIVE frames before ZUPT engages, so a single quiet frame in the middle
# of a motion (e.g. the instant accel crosses zero between accel and decel)
# cannot flicker the pose to a frozen state.
_ZUPT_HOLD = 3

# --- complementary vision-correction gains (all in [0, 1]; bounded => stable) #
# The correction runs EVERY frame whose vision solve is valid (a fresh per-frame
# PnP fix is available from EstimateMotion -- not only at keyframes), so each gain
# is the fraction of the error closed PER FRAME (~40 Hz). Small per-frame gains
# give a smooth, continuous pull that bleeds off the (bias-free) dead-reckoning
# drift without any visible snap, while the gap-free IMU integration carries the
# instant high-frequency response between (and through) corrections.
#
# Fraction of the POSITION error closed toward the vision pose per VALID-vision
# frame. 0.25/frame => the error half-life is ~2.4 frames (~60 ms): firmly
# vision-anchored (drift cannot run away) yet smooth (no snap). On a covered /
# failed-vision frame NO correction is applied -- the pose pure-dead-reckons.
_K_POS = 0.25
# Position error bled into VELOCITY as a damped rate (1/s after the /dt_anchor in
# the helper). Small: just enough to pull the phantom drift VELOCITY down (the
# bias double-integral) without the destabilising full ``v = displacement/dt``
# injection of the old hard re-anchor.
_K_VEL = 0.05
# Fraction of the ATTITUDE (geodesic) error slerped toward the vision attitude
# per valid-vision frame. Vision rotation is already excellent (gyro-fused PnP),
# so anchor it firmly.
_K_ROT = 0.25
# Minimum PnP inliers for the vision fix to be trusted for the correction. Below
# this (covered camera / textureless wall) vision is treated as ABSENT and the
# pose pure-dead-reckons from the IMU (the covered-camera-keeps-moving win).
_MIN_VIS_INLIERS = 8


class PropagateImu(StepBase):
    name = "propagate_imu"

    def run(self, ctx, step: Step):
        # LOOSE / oracle path: retain_imu is False -> pure pass-through. Never
        # allocate state, never touch step.pose (byte-identical pose.odom).
        if not ctx.state.get("retain_imu"):
            return step

        # --- keyframe-cadence (single source of truth, shared with EmitKeyframe)
        # PropagateImu runs FIRST in the tail of the chain, so it owns the kf
        # counter and stamps the boolean EmitKeyframe consumes. This avoids two
        # steps independently tracking kf_every (which would desync the vision
        # correction from the actual keyframe emission).
        n = ctx.state.get("kf_count", 0) + 1
        is_kf = n >= ctx.state["kf_every"]
        ctx.state["kf_count"] = 0 if is_kf else n
        ctx.state["is_kf_frame"] = bool(is_kf)

        g_world = np.asarray(
            ctx.state.get("g_world", (0.0, 9.81, 0.0)), np.float64)

        # Live nav-state: body->world (R, p), world velocity v, biases bg/ba.
        nav = ctx.state.get("live_nav")
        # Vision pose for this frame (camera->world == body->world here, body ==
        # camera optical frame) -> body->world (R, p) for the nav-state.
        R_vis, p_vis = T_cw_to_body_world(np.linalg.inv(step.pose))

        # Is THIS frame's vision solve a trustworthy absolute fix? EstimateMotion
        # stamps step.info with the per-frame PnP result; a covered camera /
        # textureless wall fails the solve (ok == False) or returns too few
        # inliers, in which case the live pose must PURE-DEAD-RECKON (no pull
        # toward a stale / garbage vision pose) -- the covered-camera win.
        info = step.info or {}
        vis_ok = bool(info.get("ok", True)) and \
            int(info.get("n_inliers", _MIN_VIS_INLIERS)) >= _MIN_VIS_INLIERS

        if nav is None:
            # First frame on the tight path: anchor the live state to the vision
            # pose with zero velocity and zero bias. From here on it dead-reckons
            # continuously and is pulled toward vision by a smooth correction.
            ctx.state["live_nav"] = {
                "R": R_vis, "p": p_vis, "v": np.zeros(3),
                "bg": np.zeros(3), "ba": np.zeros(3),
                # anchor_dt accumulates wall time since the last vision
                # correction (used to scale the velocity-feedback term).
                "anchor_dt": 0.0,
                # zupt_run counts consecutive accel/gyro-at-rest frames for the
                # ZUPT hysteresis.
                "zupt_run": 0,
                # prev_tail holds the LAST raw IMU sample (ts, gyro_cam,
                # accel_cam) of the previously integrated block, prepended to the
                # next block so the inter-block segment is never dropped.
                "prev_tail": None,
            }
            return step

        # --- pull this frame's retained raw IMU block (camera optical frame) ----
        # PreintegratePrior stores an EMPTY segment (size-0 arrays) for a frame
        # whose packet carried no IMU samples, so guard on the sample count.
        seg = ctx.state["imu_segs"].get(step.frame.seq)
        has_imu = seg is not None and np.asarray(seg[0]).size >= 1

        if has_imu:
            ts_raw, gyro_raw, accel_raw = (
                np.asarray(seg[0], np.int64), np.asarray(seg[1], np.float64),
                np.asarray(seg[2], np.float64))
            # --- (1) gap-free interval: prepend the previous block's tail -------
            # The per-frame packet cut is (prev_ts, ts], so consecutive blocks
            # share NO boundary sample; prepending the previous block's last
            # sample makes the integrated interval exactly
            # (prev_block_last_ts, this_block_last_ts] with no dropped segment.
            tail = nav.get("prev_tail")
            if tail is not None and int(tail[0]) < int(ts_raw[0]):
                ts = np.concatenate(([np.int64(tail[0])], ts_raw))
                gyro = np.vstack((tail[1][None, :], gyro_raw))
                accel = np.vstack((tail[2][None, :], accel_raw))
            else:
                ts, gyro, accel = ts_raw, gyro_raw, accel_raw
            # Remember this block's last sample for the next frame's boundary.
            nav["prev_tail"] = (int(ts_raw[-1]), gyro_raw[-1].copy(),
                                accel_raw[-1].copy())
        else:
            ts = gyro = accel = None

        if ts is None or ts.size < 2:
            # No usable IMU for this frame: dead-reckoning cannot advance without
            # samples. Still apply the smooth vision correction when vision is
            # valid (so the drift is pulled back), then hold/publish the nav pose.
            if vis_ok:
                self._vision_correct(nav, R_vis, p_vis)
            step.pose = np.linalg.inv(body_world_to_T_cw(nav["R"], nav["p"]))
            return step

        # --- (2) velocity-gated ZUPT: only freeze when GENUINELY at rest --------
        # imu_at_rest uses raw |gyro|/|accel| magnitudes (frame-invariant), so the
        # camera-frame samples give the same verdict as the IMU-frame ones. But
        # accel+gyro alone cannot tell rest from constant-velocity cruise (both
        # read |accel|~g, |gyro|~0), so we ALSO require the live speed to be small
        # and the at-rest gate to have held for a few frames (hysteresis).
        accel_rest = imu_at_rest(
            gyro, accel, gravity=float(np.linalg.norm(g_world)))
        nav["zupt_run"] = nav["zupt_run"] + 1 if accel_rest else 0
        speed = float(np.linalg.norm(nav["v"]))
        zupt = (accel_rest and speed < _ZUPT_VEL
                and nav["zupt_run"] >= _ZUPT_HOLD)

        if zupt:
            # Genuinely at rest: hold velocity at zero, freeze translation, but
            # still integrate rotation so a slow at-rest yaw is tracked without
            # the position walking off (the static-drift win).
            nav["v"] = np.zeros(3)
            R_new, _, _ = predict_state(
                nav["R"], nav["p"], np.zeros(3), ts, gyro, accel,
                nav["bg"], nav["ba"], np.zeros(3))
            nav["R"] = R_new
        else:
            # --- (3) forward-propagate the IMU (real motion or cruise) ----------
            R_new, p_new, v_new = predict_state(
                nav["R"], nav["p"], nav["v"], ts, gyro, accel,
                nav["bg"], nav["ba"], g_world)
            nav["R"], nav["p"], nav["v"] = R_new, p_new, v_new

        # accumulate the interval for the velocity-feedback scaling.
        nav["anchor_dt"] += (int(ts[-1]) - int(ts[0])) * 1e-9

        # --- (4) smooth vision correction EVERY valid-vision frame --------------
        # Replaces the old hard keyframe re-anchor: a small per-frame
        # complementary pull toward the fresh PnP fix bleeds off the dead-reckoning
        # drift continuously (no snap, no overshoot). On a covered / failed-vision
        # frame this is skipped, so the pose pure-dead-reckons through the blind
        # interval (keeps moving) until vision re-locks and pulls it back.
        if vis_ok:
            self._vision_correct(nav, R_vis, p_vis)

        # Replace the published live pose with the IMU-propagated one (camera->world).
        step.pose = np.linalg.inv(body_world_to_T_cw(nav["R"], nav["p"]))
        return step

    @staticmethod
    def _vision_correct(nav: dict, R_vis: np.ndarray, p_vis: np.ndarray) -> None:
        """Pull the live nav-state a bounded fraction toward the vision fix.

        Smooth complementary correction (NOT a hard ``p = p_vis`` snap): closes a
        per-frame fraction of the position/velocity/attitude error toward the
        fresh vision pose, then resets the anchor-interval accumulator so the next
        velocity-feedback term is scaled by the next inter-correction interval.
        Mutates ``nav`` in place.
        """
        R_new, p_new, v_new = complementary_correct(
            nav["R"], nav["p"], nav["v"], R_vis, p_vis,
            float(nav.get("anchor_dt", 0.0)), _K_POS, _K_VEL, _K_ROT)
        nav["R"], nav["p"], nav["v"] = R_new, p_new, v_new
        nav["anchor_dt"] = 0.0
