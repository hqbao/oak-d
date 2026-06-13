"""``publish_frontend_viz`` step: emit the frontend-internals snapshot.

OPT-IN (``--frontend-viz``). The default odometry worker does NOT wire this step
(it builds the plain ``KLTFrontend`` and leaves ``frontend_viz`` off), so the
deterministic / oracle path never captures this and the returned tracks stay
byte-identical -- the byte-parity oracle is UNAFFECTED.

When enabled, the odometry worker builds a
:class:`sky.front.frontend.CaptureKLTFrontend` (instead of the plain
``KLTFrontend``) which, as a pure side effect of
:func:`~vio.modules.track_features.track_features`, stashes one
:class:`sky.front.frontend.FrontendVizSnap` per frame on its side-car. This step
is chained right AFTER :func:`~vio.modules.publish_tracks.publish_tracks` (so the
frontend has already run for this frame) and BEFORE the motion solve; it pops the
snap (``take_viz_snap``), caps the flow field to the viz budget, fills in the
frame's ``seq`` / ``ts_ns``, and publishes one
:class:`~vio.comms.messages.FrameFrontend` on ``frame.frontend`` for the UI's
"Frontend Internals" view. The ``tracked`` carrier passes through UNCHANGED so the
motion estimate still runs on the same features (this step never touches them).

There are NO full-resolution images on the wire (the heatmap is quantised
producer-side inside the capture frontend, mirrors ``ba.window`` / ``slam.loop``).
"""
from __future__ import annotations

import numpy as np

from vio.comms import LocalPubSub, topics
from vio.comms.messages import FrameFrontend
from sky.front.frontend import CaptureKLTFrontend
from sky.front.odometry import RGBDVisualOdometry
from .tracked import Tracked

#: Max flow vectors put on the wire per frame. The full track set is ~200-400;
#: the cap bounds the per-frame message size for the 20 Hz live stream and lives
#: HERE (the capture) only -- the frontend's returned tracks are never capped.
#: When more tracks exist we keep the CULLED ones first (the interesting ones for
#: "how tracking culls bad/occluded points") then fill with kept tracks.
_FRONTEND_VIZ_MAX_TRACKS = 400


def publish_frontend_viz(vo: RGBDVisualOdometry, bus: LocalPubSub,
                         tracked: Tracked) -> Tracked:
    """Publish the frontend-internals snapshot on ``frame.frontend``; pass on.

    Was ``PublishFrontendViz(Step)``; the odometry instance + bus are passed
    explicitly. A no-op (carrier through unchanged) unless the capture frontend
    is wired and stashed a snap this frame.
    """
    fe = vo.frontend
    # Only the capture frontend stashes a snap; anything else is a no-op
    # (defensive -- the worker only wires this step with the capture frontend).
    if not isinstance(fe, CaptureKLTFrontend):
        return tracked
    snap = fe.take_viz_snap()
    if snap is None:
        return tracked

    frame = tracked.frame
    # Cap the flow field for the wire (capture-only). Keep culled tracks first
    # so the cull behaviour stays visible even when we drop kept tracks.
    fid = np.asarray(snap.flow_id, np.int64).reshape(-1)
    n = fid.shape[0]
    if n > _FRONTEND_VIZ_MAX_TRACKS:
        culled = np.asarray(snap.flow_culled, bool).reshape(-1)
        order = np.concatenate([np.nonzero(culled)[0], np.nonzero(~culled)[0]])
        sel = order[:_FRONTEND_VIZ_MAX_TRACKS]
        sel.sort()                                  # keep prev-point order
        fid = fid[sel]
        fprev = np.asarray(snap.flow_prev, np.float32).reshape(-1, 2)[sel]
        fnext = np.asarray(snap.flow_next, np.float32).reshape(-1, 2)[sel]
        ffb = np.asarray(snap.flow_fb_err, np.float32).reshape(-1)[sel]
        fcull = culled[sel]
    else:
        fprev = np.asarray(snap.flow_prev, np.float32).reshape(-1, 2)
        fnext = np.asarray(snap.flow_next, np.float32).reshape(-1, 2)
        ffb = np.asarray(snap.flow_fb_err, np.float32).reshape(-1)
        fcull = np.asarray(snap.flow_culled, bool).reshape(-1)

    bus.publish(topics.FRAME_FRONTEND, FrameFrontend(
        seq=int(frame.seq), ts_ns=int(frame.ts_ns),
        resp_q=snap.resp_q, resp_max=float(snap.resp_max),
        resp_h=int(snap.resp_h), resp_w=int(snap.resp_w),
        corner_xy=np.asarray(snap.corner_xy, np.float32).reshape(-1, 2),
        min_distance=float(snap.min_distance),
        quality_level=float(snap.quality_level),
        bucketed=bool(snap.bucketed),
        grid_rows=int(snap.grid_rows), grid_cols=int(snap.grid_cols),
        flow_id=fid, flow_prev=fprev, flow_next=fnext,
        flow_fb_err=ffb, flow_culled=fcull,
        fb_threshold=float(snap.fb_threshold)))
    return tracked
