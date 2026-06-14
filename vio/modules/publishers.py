"""Thin "emit one result on a bus topic" steps.

Every function here takes the carrier flowing through its frame-chain (or a pose
message), publishes one message on a bus topic, and forwards the carrier UNCHANGED
so the chain continues. They hold no state and never re-derive anything -- each is
a faithful tap of a REAL pipeline output for the UI / downstream subscribers.

Grouped by where they fire in the pipeline:

Frontend / per-frame (odometry worker)
    * :func:`publish_tracks`        -> ``frame.tracks``   (after track_features)
    * :func:`publish_frontend_viz`  -> ``frame.frontend`` (OPT-IN --frontend-viz)
    * :func:`publish_inliers`       -> ``frame.inliers``  (after estimate_motion)
    * :func:`publish_gyrofuse`      -> ``frame.gyrofuse`` (after estimate_motion)
    * :func:`publish_pose`          -> ``pose.odom``      (the VIO pose, per frame)
    * :func:`publish_vo`            -> ``pose.vo``        (pure vision, LIVE-only)

Back end
    * :func:`publish_refined`       -> ``pose.refined``   (BA-refined pose, terminal)
    * :func:`publish_ba_window`     -> ``ba.window``      (OPT-IN --ba-window)

OPT-IN publishers (``publish_frontend_viz`` / ``publish_ba_window``) are never
wired on the default / oracle path, so the byte-parity oracle is UNAFFECTED.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sky.front.frontend import CaptureKLTFrontend
from sky.front.odometry import RGBDVisualOdometry

from vio.comms import LocalPubSub, topics
from vio.comms.messages import (
    BaWindow, FrameFrontend, FrameGyroFuse, FrameInliers, FrameTracks, PoseMsg)
from vio.engine import Engine
from vio.engine.ba_capture import BaWindowSnap
from .carriers import Step, Tracked

#: Max flow vectors put on the wire per frame by ``publish_frontend_viz``. The
#: full track set is ~200-400; the cap bounds the per-frame message size for the
#: 20 Hz live stream and lives HERE (the capture) only -- the frontend's returned
#: tracks are never capped. When more tracks exist we keep the CULLED ones first
#: (the interesting ones for "how tracking culls bad/occluded points") then fill
#: with kept tracks.
_FRONTEND_VIZ_MAX_TRACKS = 400

#: The five per-frame fusion diagnostic keys the odometry records together inside
#: the gyro-fusion branch (consumed by ``publish_gyrofuse``). All five are present
#: iff the fusion ran this frame, so a single presence check on the first is
#: sufficient -- but we require ALL of them so a partial/legacy info dict can never
#: publish a half-filled message.
_REQUIRED = ("vision_rot_deg", "gyro_rot_deg", "disagree_deg", "gain", "t_trust")


def publish_tracks(bus: LocalPubSub, tracked: Tracked) -> Tracked:
    """Publish the frame's KLT tracks on ``frame.tracks``; pass the carrier on.

    Was ``PublishTracks(Step)``; identical publish, the bus passed explicitly.
    """
    obs = tracked.obs
    if obs:
        ids_list = list(obs.keys())
        ids = np.array(ids_list, dtype=np.int64)
        points = np.array([obs[k] for k in ids_list],
                          dtype=np.float32).reshape(-1, 2)
    else:
        ids = np.empty((0,), dtype=np.int64)
        points = np.empty((0, 2), dtype=np.float32)
    frame = tracked.frame
    bus.publish(topics.FRAME_TRACKS,
                FrameTracks(frame.seq, frame.ts_ns, ids, points))
    return tracked


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


def publish_pose(bus: LocalPubSub, step: Step) -> Step:
    """Publish the per-frame VIO pose on ``pose.odom``; pass the carrier on.

    Was ``PublishPose(Step)``; identical publish, the bus passed explicitly.
    """
    bus.publish(topics.POSE_ODOM,
                PoseMsg(step.frame.seq, step.frame.ts_ns,
                        step.pose, step.info))
    return step


def publish_vo(vo: RGBDVisualOdometry, bus: LocalPubSub, step: Step) -> Step:
    """Publish the pure-vision accumulated pose on ``pose.vo``; pass on the carrier.

    Was ``PublishVo(Step)``; identical publish, the odometry instance + bus passed
    explicitly instead of read off ``ctx.state``.
    """
    # Copy the accumulator: the same instance keeps mutating pose_vo on the
    # next frame, so the published message must own an independent snapshot
    # (the wire/bridge layer reads it asynchronously).
    bus.publish(topics.POSE_VO,
                PoseMsg(step.frame.seq, step.frame.ts_ns,
                        vo.pose_vo.copy(), step.info))
    return step


def publish_refined(bus: LocalPubSub, msg: PoseMsg) -> None:
    """Publish the BA-refined pose on ``pose.refined`` (terminal step).

    Was ``PublishRefined(Step)``; identical publish, the bus passed explicitly.
    """
    bus.publish(topics.POSE_REFINED, msg)
    return None


def publish_ba_window(engine: Engine, bus: LocalPubSub, msg: PoseMsg) -> PoseMsg:
    """Publish the BA-window solve snapshot on ``ba.window``; forward the pose.

    Was ``PublishBaWindow(Step)``; the engine + bus are passed explicitly. The
    refined pose carrier is forwarded UNCHANGED so ``publish_refined`` emits it
    identically to the no-capture path.
    """
    snap = engine.poll_overlay()
    # The overlay is a BaWindowSnap ONLY on the capture engine + a keyframe
    # whose solve ran; anything else (None on warmup) is simply skipped.
    if isinstance(snap, BaWindowSnap):
        bus.publish(topics.BA_WINDOW, BaWindow(
            seq=int(snap.seq), ts_ns=int(snap.ts_ns),
            kf_ids=snap.kf_ids, kf_quat=snap.kf_quat, kf_pos=snap.kf_pos,
            lm_ids=snap.lm_ids, lm_xyz=snap.lm_xyz,
            obs_kf=snap.obs_kf, obs_lm=snap.obs_lm, obs_uv=snap.obs_uv,
            obs_reproj_px=snap.obs_reproj_px,
            ba_reproj_px=float(snap.ba_reproj_px),
            kf_quat_pre=snap.kf_quat_pre, kf_pos_pre=snap.kf_pos_pre,
            lm_xyz_pre=snap.lm_xyz_pre,
            n_kf=int(snap.n_kf), n_lm=int(snap.n_lm)))
    # Forward the refined pose UNCHANGED so publish_refined emits it identically.
    return msg
