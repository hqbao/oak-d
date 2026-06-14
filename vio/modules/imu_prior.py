"""IMU prior + gravity chain feeding the visual solve.

The IMU-side steps of the odometry frame-chain, in pipeline order:

* :func:`preintegrate_prior` -- on the ``imucam.sample`` edge: build this frame's
  :class:`~vio.comms.messages.ImuPrior` (gyro inter-frame rotation + gravity accel
  + at-rest / moving flags) and stash it by ``seq`` for the matching depth frame.
* :func:`align_gravity` -- one-shot startup attitude leveling to the front-end's
  ``accel_align`` reference; fires once then never again.
* :func:`pull_prior` -- the IMU<->vision join: pop the prior preintegrated for this
  frame's ``seq`` and thread it forward on the :class:`~vio.modules.carriers.Primed`
  carrier the visual solve consumes.
* :func:`correct_tilt` -- LIVE-only: continuously level roll/pitch from gravity
  while at rest, AFTER the motion solve (no-op on the offline / oracle path).
"""
from __future__ import annotations

import numpy as np

from sky.front.odometry import RGBDVisualOdometry
from sky.vio.imu import integrate_gyro_camera

from vio.comms.messages import ImuCamPacket, ImuPrior
from .carriers import Primed, Step, Tracked

# Stillness gate for the keyframe gravity prior: low angular rate and accel close
# to 1 g. Mirrors the live capture flow's at-rest thresholds.
_STILL_GYRO = 0.15      # rad/s
_GRAVITY = 9.81         # m/s^2
_STILL_ACCEL_DEV = 0.6  # m/s^2 deviation of |accel| from gravity
# Loose "definitely moving" gate for the IMU-vetoed low-inlier freeze (see
# OdometryConfig.min_inliers_for_translation): higher than the still gates so the
# middle band stays "neither". 0.3 rad/s ~ 17 deg/s (any real hand turn); 0.5
# m/s^2 ~ 5 % of g (any real linear push).
_MOVING_GYRO = 0.3      # rad/s
_MOVING_ACCEL_DEV = 0.5  # m/s^2 deviation of |accel| from gravity


def preintegrate_prior(state: dict, msg: ImuCamPacket) -> None:
    """Build + stash this frame's :class:`~vio.comms.messages.ImuPrior`.

    Was ``PreintegratePrior(Step)``; the worker's shared state dict
    (``ctx.state``) is passed explicitly. Reads ``R_imu_cam`` / ``use_gyro`` /
    ``retain_imu`` and mutates ``priors`` (always) + ``imu_segs`` (tight path).
    Returns ``None`` -- terminal on the imucam.sample edge (was the chain's
    short-circuit-on-None).
    """
    R_imu_cam = state.get("R_imu_cam")
    use_gyro = state.get("use_gyro", True)

    gyro = np.asarray(msg.gyro, dtype=np.float64)
    accel = np.asarray(msg.accel, dtype=np.float64)

    R_prior = None
    if use_gyro and R_imu_cam is not None and gyro.shape[0] >= 2:
        R_prior = integrate_gyro_camera(msg.imu_ts, gyro, R_imu_cam)

    accel_cam = None
    at_rest = False
    imu_moving = False
    if R_imu_cam is not None and accel.size:
        a_mean = accel.mean(axis=0)
        accel_cam = np.asarray(R_imu_cam, dtype=np.float64) @ a_mean
        gyro_mag = (0.0 if gyro.size == 0
                    else float(np.linalg.norm(gyro, axis=1).mean()))
        accel_dev = abs(float(np.linalg.norm(a_mean)) - _GRAVITY)
        gyro_still = (gyro.size == 0 or gyro_mag < _STILL_GYRO)
        accel_still = accel_dev < _STILL_ACCEL_DEV
        at_rest = bool(gyro_still and accel_still)
        # Loose "definitely moving" flag: vetoes the low-inlier translation
        # freeze when motion blur (not a textureless wall) is starving PnP.
        imu_moving = bool(gyro_mag > _MOVING_GYRO
                          or accel_dev > _MOVING_ACCEL_DEV)

    state["priors"][msg.seq] = ImuPrior(
        msg.seq, R_prior, accel_cam, at_rest, imu_moving)
    # Safety cap: in the normal full-fidelity path each prior is popped by the
    # matching depth frame, so the dict stays ~1 entry and this never fires.
    # Under a latest-only visualiser graph, frames whose depth was coalesced
    # away leave their prior un-popped; drop the oldest so the dict can't grow
    # without bound over a long live session.
    priors = state["priors"]
    if len(priors) > 256:
        for seq in sorted(priors)[:-256]:
            priors.pop(seq, None)

    # TIGHT and DIRECT paths: retain this interval's RAW IMU samples (rotated
    # into the camera optical frame) keyed by frame seq. The TIGHT path hands the
    # inter-keyframe block to its backend preintegrator (via emit_keyframe); the
    # DIRECT path feeds each per-frame block to its IMU dead-reckon seed
    # (process_frame_direct). This is a no-op for the LOOSE / oracle path (both
    # ``retain_imu`` and ``direct`` default False), so it never allocates on or
    # perturbs the byte-identical loose path. The body frame == camera optical
    # frame, hence the per-sample ``R_imu_cam @ v`` rotation here (the gyro prior
    # above already uses the same extrinsic via integrate_gyro_camera).
    if (state.get("retain_imu") or state.get("direct")) and R_imu_cam is not None:
        R_ic = np.asarray(R_imu_cam, dtype=np.float64)
        imu_segs = state["imu_segs"]
        if gyro.shape[0]:
            gyro_cam = gyro @ R_ic.T          # (M,3) IMU-frame -> camera frame
            accel_arr = accel @ R_ic.T
            imu_segs[msg.seq] = (
                np.asarray(msg.imu_ts, dtype=np.int64).copy(),
                gyro_cam.copy(), accel_arr.copy())
        else:
            imu_segs[msg.seq] = (np.zeros(0, np.int64),
                                 np.zeros((0, 3)), np.zeros((0, 3)))
        # Same bounded-growth cap as ``priors`` -- emit_keyframe pops the
        # consumed segments, but a coalesced latest-only graph can leave gaps.
        if len(imu_segs) > 512:
            for seq in sorted(imu_segs)[:-512]:
                imu_segs.pop(seq, None)
    return None


def align_gravity(vo: RGBDVisualOdometry, state: dict, tracked: Tracked) -> Tracked:
    """One-shot: level the initial attitude to the startup gravity reference.

    Was ``AlignGravity(Step)``. ``vo`` is the odometry instance; ``state`` is the
    worker's shared state dict (``ctx.state``), holding the one-shot ``aligned``
    latch and the ``accel_align`` seed. Passes the carrier through unchanged.
    """
    if not state.get("aligned") and "accel_align" in state:
        vo.align_to_gravity(state["accel_align"])
        state["aligned"] = True
    return tracked


def pull_prior(priors: dict, tracked: Tracked) -> Primed:
    """Pop this frame's preintegrated IMU prior and join it onto the carrier.

    Was ``PullPrior(Step)``; ``priors`` (the worker's ``ctx.state["priors"]``
    seq->prior dict) is passed explicitly. ``None`` when none was preintegrated.
    """
    prior = priors.pop(tracked.frame.seq, None)
    return Primed(tracked.frame, tracked.obs, prior)


def correct_tilt(vo: RGBDVisualOdometry, level_tilt: bool, step: Step) -> Step:
    """At-rest roll/pitch leveling (LIVE-only); return the carrier (pose updated).

    Was ``CorrectTilt(StepBase)``; the odometry instance + the ``level_tilt``
    gate (was ``ctx.state["level_tilt"]``) are passed explicitly. A no-op on the
    offline / oracle path (``level_tilt`` False) -> byte-identical ``pose.odom``.
    """
    if (level_tilt and step.at_rest and step.accel_cam is not None):
        if vo.correct_tilt(step.accel_cam):
            step.pose = vo.pose.copy()       # publish the leveled attitude
    return step
