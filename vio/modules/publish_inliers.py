"""``publish_inliers`` step: emit the frame's PnP reproj diagnostic on ``frame.inliers``.

Runs right after :func:`~vio.modules.estimate_motion.estimate_motion` in
the frame-chain, so the RGB-D PnP has already solved and recorded -- per PnP
correspondence -- its track id, the reprojection of its prev-frame 3D point
through the solved ``(R, t)``, and whether RANSAC kept it as an inlier
(``info["pnp_ids"/"pnp_reproj"/"pnp_inlier"]`` on the :class:`~vio.modules.step.Step`
it produced). It publishes all three for the keypoint-depth visualiser to draw
the measured-pixel -> reprojected-pixel stub per point -- a REAL odometry output,
never a re-derivation -- and passes the ``Step`` carrier through unchanged so
``publish_pose`` / ``emit_keyframe`` still run on it. When PnP failed (the keys
are absent) it publishes empty arrays so the topic still ticks once per frame.

The reproj-stub selftest drives this step directly, so the publishing object is
exposed as a tiny ``_Ctx``-compatible callable: it takes a context whose ``.bus``
is the local bus (mirrors the old ``ctx.bus``); the function keeps that signature
so the selftest's ``PublishInliers().run(ctx, step)`` call is preserved.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vio.comms import topics
from vio.comms.messages import FrameInliers
from .step import Step


def publish_inliers(ctx: Any, step: Step) -> Step:
    """Publish the PnP inlier reproj diagnostic on ``frame.inliers``; pass on.

    Was ``PublishInliers(Step)``. ``ctx`` is any object exposing ``.bus`` (the
    odometry worker's :class:`~vio.comms.module.ModuleContext`, or the selftest's
    tiny ``_Ctx``); kept as the single arg so the reproj-stub selftest's direct
    ``run(ctx, step)`` call is byte-compatible.
    """
    info = step.info
    ids = info.get("pnp_ids")
    reproj = info.get("pnp_reproj")
    inlier = info.get("pnp_inlier")
    # PnP failed / too-few-points: emit empty, correctly-shaped arrays so the
    # consumer's join + draw_overlay short-circuit cleanly (M == 0).
    if ids is None or reproj is None or inlier is None:
        ids = np.empty((0,), dtype=np.int64)
        reproj = np.empty((0, 2), dtype=np.float32)
        inlier = np.empty((0,), dtype=bool)
    frame = step.frame
    ctx.bus.publish(topics.FRAME_INLIERS,
                    FrameInliers(frame.seq, frame.ts_ns,
                                 np.asarray(ids, dtype=np.int64),
                                 np.asarray(reproj, dtype=np.float32),
                                 np.asarray(inlier, dtype=bool)))
    return step
