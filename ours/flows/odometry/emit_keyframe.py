"""``emit_keyframe`` task: every ``kf_every`` frames, publish a ``keyframe``.

The keyframe carries the pose, image, depth, the current track snapshot and --
only when the camera was at rest -- the gravity accel for the back-end (a moving
keyframe's lateral acceleration would bias the gravity direction).

Returns the :class:`Step` unchanged so a tail task (``SignalDone``) can run after
it; it is no longer the last task in the chain.
"""
from __future__ import annotations

from ...lib.flow import topics
from ...lib.flow.messages import Keyframe
from ...lib.flow.task import Task
from ...lib.odometry.odometry import RGBDVisualOdometry
from .step import Step


class EmitKeyframe(Task):
    name = "emit_keyframe"

    def run(self, ctx, step: Step):
        n = ctx.state.get("kf_count", 0) + 1
        if n < ctx.state["kf_every"]:
            ctx.state["kf_count"] = n
            return step
        ctx.state["kf_count"] = 0
        vo: RGBDVisualOdometry = ctx.state["vo"]
        tr = vo.frontend.tracks
        ids = tr.ids.copy() if tr is not None and tr.ids is not None else None
        px = tr.points.copy() if tr is not None and tr.points is not None else None
        accel = step.accel_cam if step.at_rest else None
        ctx.bus.publish(topics.KEYFRAME,
                        Keyframe(step.frame.seq, step.pose,
                                 step.frame.gray_left, step.frame.depth_m,
                                 track_ids=ids, track_px=px, accel=accel))
        return step
