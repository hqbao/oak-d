"""``correct_tilt`` task: continuously level roll/pitch from gravity, at rest.

Runs right after :class:`EstimateMotion`. When the camera is at rest (so the
accelerometer reads gravity, not motion) it nudges the current attitude's
roll/pitch toward the measured gravity direction
(:meth:`~sky.front.odometry.RGBDVisualOdometry.correct_tilt`, a slow EMA
gated by ``g_tol``). This is what lets the live view self-level **without** a
hold-still window at startup: the one-shot :class:`AlignGravity` seed can be rough
(or absent) because any still moment during the run pulls roll/pitch back to level.

LIVE-ONLY: enabled only when ``ctx.state["level_tilt"]`` is set (the live builder
turns it on). The offline replay/scoring path leaves it off, so ``pose.odom`` stays
byte-identical there. It only rotates the attitude block (not the accumulated
translation), so a small per-frame correction never retroactively bends the path.
"""
from __future__ import annotations

from vio.comms import Step as StepBase
from sky.front.odometry import RGBDVisualOdometry
from .step import Step


class CorrectTilt(StepBase):
    name = "correct_tilt"

    def run(self, ctx, step: Step):
        if (ctx.state.get("level_tilt") and step.at_rest
                and step.accel_cam is not None):
            vo: RGBDVisualOdometry = ctx.state["vo"]
            if vo.correct_tilt(step.accel_cam):
                step.pose = vo.pose.copy()       # publish the leveled attitude
        return step
