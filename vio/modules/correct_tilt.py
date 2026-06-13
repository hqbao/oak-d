"""``correct_tilt`` step: continuously level roll/pitch from gravity, at rest.

Runs right after :func:`~vio.modules.estimate_motion.estimate_motion`. When the
camera is at rest (so the accelerometer reads gravity, not motion) it nudges the
current attitude's roll/pitch toward the measured gravity direction
(:meth:`~sky.front.odometry.RGBDVisualOdometry.correct_tilt`, a slow EMA
gated by ``g_tol``). This is what lets the live view self-level **without** a
hold-still window at startup: the one-shot :func:`align_gravity` seed can be rough
(or absent) because any still moment during the run pulls roll/pitch back to level.

LIVE-ONLY: enabled only when ``level_tilt`` is set (the live builder turns it on).
The offline replay/scoring path leaves it off, so ``pose.odom`` stays
byte-identical there. It only rotates the attitude block (not the accumulated
translation), so a small per-frame correction never retroactively bends the path.
"""
from __future__ import annotations

from sky.front.odometry import RGBDVisualOdometry
from .step import Step


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
