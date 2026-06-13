"""``publish_gyrofuse`` step: emit the frame's gyro-fusion diagnostic on
``frame.gyrofuse`` (ALGORITHMS.md #5).

Runs right after :func:`~vio.modules.estimate_motion.estimate_motion` in the
frame-chain, so the gyro fusion has already run and recorded the per-frame
observation into ``last_info``:

* ``vision_rot_deg`` -- RAW PnP inter-frame rotation magnitude (deg), pre-fusion.
* ``gyro_rot_deg`` -- gyro inter-frame rotation magnitude (deg).
* ``disagree_deg`` -- vision-vs-gyro rotation disagreement (deg).
* ``gain`` -- the resulting vision-correction gain (0..1).
* ``t_trust`` -- the translation-trust this frame (0..1).

It bundles those with the config gate thresholds (``gyro_disagree_deg`` /
``gyro_disagree_span_deg``, read off the live odometry instance) so the UI strip
chart can draw the matching reference lines without knowing the config, and
publishes a :class:`~vio.comms.messages.FrameGyroFuse` -- a REAL odometry output,
never a re-derivation.

Skip / NO-publish contract
--------------------------
The diagnostic fields exist in ``last_info`` ONLY on frames where the gyro fusion
actually ran (``gyro_fuse`` on AND a PnP solve with a rotation prior). On every
other frame -- gyro off, PnP failed, too-few-points, bootstrap, the low-inlier
freeze -- the keys are absent and this step publishes NOTHING (it does not even
tick the topic), so the chart never receives a garbage / all-zero frame that
would misrepresent "no disagreement" as a real measurement. The
:class:`~vio.modules.step.Step` carrier is always passed through unchanged so
``publish_pose`` / ``emit_keyframe`` still run.

The gyrofuse selftest drives this step directly with a tiny ``_Ctx`` exposing
``.bus`` + ``.state["vo"]``; the function keeps that single-``ctx`` signature so
the selftest's ``PublishGyroFuse().run(ctx, step)`` call is preserved.
"""
from __future__ import annotations

from typing import Any

from vio.comms import topics
from vio.comms.messages import FrameGyroFuse
from .step import Step

#: The five per-frame fusion diagnostic keys the odometry records together inside
#: the gyro-fusion branch. All five are present iff the fusion ran this frame, so
#: a single presence check on the first is sufficient -- but we require ALL of
#: them so a partial/legacy info dict can never publish a half-filled message.
_REQUIRED = ("vision_rot_deg", "gyro_rot_deg", "disagree_deg", "gain", "t_trust")


def publish_gyrofuse(ctx: Any, step: Step) -> Step:
    """Publish the gyro-fusion diagnostic on ``frame.gyrofuse`` (or stay silent).

    Was ``PublishGyroFuse(Step)``. ``ctx`` is any object exposing ``.bus`` +
    ``.state["vo"]`` (the odometry worker's
    :class:`~vio.comms.module.ModuleContext`, or the selftest's ``_Ctx``); kept as
    the single arg so the gyrofuse selftest's direct ``run(ctx, step)`` call is
    byte-compatible.
    """
    info = step.info
    # Gyro fusion did not run this frame (gyro off / PnP failed / bootstrap /
    # freeze): no honest diagnostic to publish, so stay silent. The topic
    # only ticks on genuinely gyro-fused frames.
    if any(k not in info for k in _REQUIRED):
        return step
    # Gate thresholds come straight off the live odometry config so the UI
    # draws reference lines that always match the running fusion (no second
    # source of truth). The odometry instance lives in ctx.state["vo"].
    cfg = ctx.state["vo"].cfg
    frame = step.frame
    ctx.bus.publish(
        topics.FRAME_GYROFUSE,
        FrameGyroFuse(
            seq=frame.seq, ts_ns=frame.ts_ns,
            vision_rot_deg=float(info["vision_rot_deg"]),
            gyro_rot_deg=float(info["gyro_rot_deg"]),
            disagree_deg=float(info["disagree_deg"]),
            gain=float(info["gain"]),
            t_trust=float(info["t_trust"]),
            gate_deg=float(cfg.gyro_disagree_deg),
            span_deg=float(cfg.gyro_disagree_span_deg)))
    return step
