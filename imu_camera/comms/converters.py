"""Convert between local (in-proc) dataclasses and wire messages.

A converter is two halves:

* ``to_wire(local_msg, rings, endpoint)`` -- copy any large arrays into shared
  memory and build the matching :mod:`comms.wire` dataclass.
* ``to_local(wire_msg, rings)`` -- ``read_copy`` every shared-memory ref into a
  private ``np.ndarray`` and reconstruct the in-proc dataclass.

Both halves know the per-topic ring naming convention (gray_left, gray_right,
depth_m, kf_gray, kf_depth) so the bridge modules just pick the right converter
by topic.

INVARIANT: every numpy array on the local side that came from shared memory is
``read_copy``-ed (an independent allocation) before any downstream step sees it.
The slot is then free to be reused by the producer's next frame, and no module
can ever read a half-overwritten slot.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from . import topics
from .messages import (
    BaWindow, CamSync, DepthFrame, FrameFrontend, FrameGyroFuse, FrameInliers,
    FrameTracks, ImuCamPacket, ImuRaw, Keyframe, LoopCorrection, LoopMatch,
    PoseMsg, SlamOverlay, END,
)
from .wire import (
    WireBaWindow, WireCamSync, WireDepthFrame, WireEnd, WireFrameFrontend,
    WireFrameInliers, WireFrameTracks, WireGyroFuse, WireImuCamPacket, WireImuRaw,
    WireKeyframe, WireLoopCorrection, WireLoopMatch, WirePoseMsg, WireSlamMap,
)
from .ring_registry import RingRegistry


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_to_ring(rings: RingRegistry, ring_name: str, slot: int, arr):
    """Copy ``arr`` into the named ring's slot; return the wire ref."""
    return rings.get(ring_name).write(slot, arr)


def _read_from_ring(rings: RingRegistry, ref):
    """Return a private copy of the array the ref points to."""
    return rings.get(ref.ring_name).read_copy(ref)


# --------------------------------------------------------------------------- #
# Per-topic converters. Each registered as (topic, to_wire, to_local) below.
# --------------------------------------------------------------------------- #
def _cam_sync_to_wire(msg: CamSync, rings: RingRegistry, endpoint: str):
    slot = int(msg.seq) % rings.get(f"{endpoint}.gray_left").slots
    left_ref = _write_to_ring(rings, f"{endpoint}.gray_left", slot, msg.gray_left)
    right_ref = None
    if msg.gray_right is not None:
        right_ref = _write_to_ring(rings, f"{endpoint}.gray_right", slot, msg.gray_right)
    return WireCamSync(seq=int(msg.seq), ts_ns=int(msg.ts_ns),
                       gray_left_ref=left_ref, gray_right_ref=right_ref)


def _cam_sync_to_local(wm: WireCamSync, rings: RingRegistry) -> CamSync:
    return CamSync(seq=int(wm.seq), ts_ns=int(wm.ts_ns),
                   gray_left=_read_from_ring(rings, wm.gray_left_ref),
                   gray_right=(None if wm.gray_right_ref is None
                               else _read_from_ring(rings, wm.gray_right_ref)))


def _imucam_to_wire(msg: ImuCamPacket, rings: RingRegistry, endpoint: str):
    slot = int(msg.seq) % rings.get(f"{endpoint}.gray_left").slots
    left_ref = _write_to_ring(rings, f"{endpoint}.gray_left", slot, msg.gray_left)
    right_ref = None
    if msg.gray_right is not None:
        right_ref = _write_to_ring(rings, f"{endpoint}.gray_right", slot, msg.gray_right)
    return WireImuCamPacket(seq=int(msg.seq), ts_ns=int(msg.ts_ns),
                            gray_left_ref=left_ref, gray_right_ref=right_ref,
                            imu_ts=msg.imu_ts, gyro=msg.gyro, accel=msg.accel)


def _imucam_to_local(wm: WireImuCamPacket, rings: RingRegistry) -> ImuCamPacket:
    return ImuCamPacket(
        seq=int(wm.seq), ts_ns=int(wm.ts_ns),
        gray_left=_read_from_ring(rings, wm.gray_left_ref),
        gray_right=(None if wm.gray_right_ref is None
                    else _read_from_ring(rings, wm.gray_right_ref)),
        imu_ts=wm.imu_ts, gyro=wm.gyro, accel=wm.accel)


def _imu_raw_to_wire(msg: ImuRaw, rings: RingRegistry, endpoint: str):
    return WireImuRaw(seq=int(msg.seq), ts_ns=int(msg.ts_ns),
                      imu_ts=msg.imu_ts, gyro=msg.gyro, accel=msg.accel)


def _imu_raw_to_local(wm: WireImuRaw, rings: RingRegistry) -> ImuRaw:
    return ImuRaw(seq=int(wm.seq), ts_ns=int(wm.ts_ns),
                  imu_ts=wm.imu_ts, gyro=wm.gyro, accel=wm.accel)


def _depth_to_wire(msg: DepthFrame, rings: RingRegistry, endpoint: str):
    slot = int(msg.seq) % rings.get(f"{endpoint}.gray_left").slots
    left_ref = _write_to_ring(rings, f"{endpoint}.gray_left", slot, msg.gray_left)
    depth_ref = _write_to_ring(rings, f"{endpoint}.depth_m", slot, msg.depth_m)
    return WireDepthFrame(seq=int(msg.seq), ts_ns=int(msg.ts_ns),
                          gray_left_ref=left_ref, depth_ref=depth_ref)


def _depth_to_local(wm: WireDepthFrame, rings: RingRegistry) -> DepthFrame:
    return DepthFrame(seq=int(wm.seq), ts_ns=int(wm.ts_ns),
                      gray_left=_read_from_ring(rings, wm.gray_left_ref),
                      depth_m=_read_from_ring(rings, wm.depth_ref))


def _tracks_to_wire(msg: FrameTracks, rings: RingRegistry, endpoint: str):
    # Pure POD: ids + points only. The gray_left + depth_m needed to render the
    # overlay arrive on `frame.depth` (capture's rings, capture is the SINGLE
    # writer there). Touching the rings here would race capture for the same
    # slot. See FrameTracks docstring for the single-writer ring contract.
    del rings, endpoint                                # no ring slot needed
    return WireFrameTracks(seq=int(msg.seq), ts_ns=int(msg.ts_ns),
                           ids=msg.ids, points=msg.points)


def _tracks_to_local(wm: WireFrameTracks, rings: RingRegistry) -> FrameTracks:
    del rings                                          # no ring slot needed
    return FrameTracks(seq=int(wm.seq), ts_ns=int(wm.ts_ns),
                       ids=wm.ids, points=wm.points)


def _inliers_to_wire(msg: FrameInliers, rings: RingRegistry, endpoint: str):
    del rings, endpoint                                # pure POD, no ring slot
    return WireFrameInliers(seq=int(msg.seq), ts_ns=int(msg.ts_ns),
                            ids=msg.ids, reproj=msg.reproj, inlier=msg.inlier)


def _inliers_to_local(wm: WireFrameInliers, rings: RingRegistry) -> FrameInliers:
    del rings                                          # pure POD, no ring slot
    return FrameInliers(seq=int(wm.seq), ts_ns=int(wm.ts_ns),
                        ids=wm.ids, reproj=wm.reproj, inlier=wm.inlier)


def _gyrofuse_to_wire(msg: FrameGyroFuse, rings: RingRegistry, endpoint: str):
    del rings, endpoint                                # pure POD, no ring slot
    return WireGyroFuse(seq=int(msg.seq), ts_ns=int(msg.ts_ns),
                        vision_rot_deg=float(msg.vision_rot_deg),
                        gyro_rot_deg=float(msg.gyro_rot_deg),
                        disagree_deg=float(msg.disagree_deg),
                        gain=float(msg.gain), t_trust=float(msg.t_trust),
                        gate_deg=float(msg.gate_deg), span_deg=float(msg.span_deg))


def _gyrofuse_to_local(wm: WireGyroFuse, rings: RingRegistry) -> FrameGyroFuse:
    del rings                                          # pure POD, no ring slot
    return FrameGyroFuse(seq=int(wm.seq), ts_ns=int(wm.ts_ns),
                         vision_rot_deg=float(wm.vision_rot_deg),
                         gyro_rot_deg=float(wm.gyro_rot_deg),
                         disagree_deg=float(wm.disagree_deg),
                         gain=float(wm.gain), t_trust=float(wm.t_trust),
                         gate_deg=float(wm.gate_deg), span_deg=float(wm.span_deg))


def _pose_to_wire(msg: PoseMsg, rings: RingRegistry, endpoint: str):
    return WirePoseMsg(seq=int(msg.seq), ts_ns=int(msg.ts_ns),
                       T_world_cam=msg.T_world_cam, info=dict(msg.info))


def _pose_to_local(wm: WirePoseMsg, rings: RingRegistry) -> PoseMsg:
    return PoseMsg(seq=int(wm.seq), ts_ns=int(wm.ts_ns),
                   T_world_cam=wm.T_world_cam, info=dict(wm.info))


def _keyframe_to_wire(msg: Keyframe, rings: RingRegistry, endpoint: str):
    # Keyframes ride VIO's dedicated kf_* rings, not capture's frame rings, so
    # SLAM can pick them up at its own slower cadence without racing capture.
    slot = int(msg.seq) % rings.get(f"{endpoint}.kf_gray").slots
    left_ref = _write_to_ring(rings, f"{endpoint}.kf_gray", slot, msg.gray_left)
    depth_ref = _write_to_ring(rings, f"{endpoint}.kf_depth", slot, msg.depth_m)
    return WireKeyframe(seq=int(msg.seq), T_world_cam=msg.T_world_cam,
                        gray_left_ref=left_ref, depth_ref=depth_ref,
                        track_ids=msg.track_ids, track_px=msg.track_px,
                        accel=msg.accel, inlier_ids=msg.inlier_ids)


def _keyframe_to_local(wm: WireKeyframe, rings: RingRegistry) -> Keyframe:
    return Keyframe(seq=int(wm.seq), T_world_cam=wm.T_world_cam,
                    gray_left=_read_from_ring(rings, wm.gray_left_ref),
                    depth_m=_read_from_ring(rings, wm.depth_ref),
                    track_ids=wm.track_ids, track_px=wm.track_px,
                    accel=wm.accel, inlier_ids=wm.inlier_ids)


def _loop_corr_to_wire(msg: LoopCorrection, rings: RingRegistry, endpoint: str):
    return WireLoopCorrection(seq=int(msg.seq),
                              kf_poses=dict(msg.kf_poses),
                              n_loops=int(msg.n_loops))


def _loop_corr_to_local(wm: WireLoopCorrection, rings: RingRegistry) -> LoopCorrection:
    return LoopCorrection(seq=int(wm.seq), kf_poses=dict(wm.kf_poses),
                          n_loops=int(wm.n_loops))


def _slam_overlay_to_wire(msg: SlamOverlay, rings: RingRegistry, endpoint: str):
    # Pure POD: the keyframe positions are small (a handful of (3,) vectors), so
    # they ride the message itself, no shared-memory ring. kf_ids carry the REAL
    # source frame seqs so the UI can match each corrected keyframe to its dense
    # VIO pose (the rubber-sheet "corrected VIO" line). Fall back to arange when
    # the seqs are missing/mismatched so a malformed overlay can never crash the
    # bridge (the dots still render by POSITION).
    pos = np.asarray(msg.kf_positions, dtype=np.float64).reshape(-1, 3)
    seqs = (None if msg.kf_seqs is None
            else np.asarray(msg.kf_seqs, dtype=np.int64).reshape(-1))
    kf_ids = (seqs if seqs is not None and len(seqs) == len(pos)
              else np.arange(len(pos), dtype=np.int64))
    return WireSlamMap(kf_ids=kf_ids,
                       kf_positions=pos, n_loops=int(msg.n_loops),
                       last_match=(None if msg.last_match is None
                                   else np.asarray(msg.last_match, dtype=np.float64)))


def _slam_map_to_local(wm: WireSlamMap, rings: RingRegistry) -> SlamOverlay:
    return SlamOverlay(kf_positions=np.asarray(wm.kf_positions, dtype=np.float64),
                       n_loops=int(wm.n_loops), last_match=wm.last_match,
                       kf_seqs=np.asarray(wm.kf_ids, dtype=np.int64))


def _loop_match_to_wire(msg: LoopMatch, rings: RingRegistry, endpoint: str):
    # Pure POD: the matched ORB pixel pairs are a few hundred (2,) floats, so they
    # ride the message itself (no shared-memory ring) -- and there are NO keyframe
    # images on the wire (SLAM does not retain the gray; the UI joins by seq to its
    # own keyframe-gray buffer). Force the canonical dtypes so the bytes are stable
    # across hosts (the codec keys off dtype.name).
    return WireLoopMatch(
        cur_seq=int(msg.cur_seq), old_seq=int(msg.old_seq),
        cur_px=np.asarray(msg.cur_px, dtype=np.float32).reshape(-1, 2),
        old_px=np.asarray(msg.old_px, dtype=np.float32).reshape(-1, 2),
        stage=np.asarray(msg.stage, dtype=np.uint8).reshape(-1),
        n_appearance=int(msg.n_appearance), n_fmat=int(msg.n_fmat),
        n_pnp=int(msg.n_pnp), rot_deg=float(msg.rot_deg),
        rot_gate_deg=float(msg.rot_gate_deg), accepted=bool(msg.accepted))


def _loop_match_to_local(wm: WireLoopMatch, rings: RingRegistry) -> LoopMatch:
    return LoopMatch(
        cur_seq=int(wm.cur_seq), old_seq=int(wm.old_seq),
        cur_px=np.asarray(wm.cur_px, dtype=np.float32).reshape(-1, 2),
        old_px=np.asarray(wm.old_px, dtype=np.float32).reshape(-1, 2),
        stage=np.asarray(wm.stage, dtype=np.uint8).reshape(-1),
        n_appearance=int(wm.n_appearance), n_fmat=int(wm.n_fmat),
        n_pnp=int(wm.n_pnp), rot_deg=float(wm.rot_deg),
        rot_gate_deg=float(wm.rot_gate_deg), accepted=bool(wm.accepted))


def _ba_window_to_wire(msg: BaWindow, rings: RingRegistry, endpoint: str):
    # Pure POD: the windowed-BA snapshot (<= 8 keyframes, <= 100 landmarks, a few
    # hundred observation rays) rides the message itself -- no shared-memory ring,
    # no images (mirrors _loop_match_to_wire). Force the canonical dtypes (the
    # codec keys off dtype.name) so the bytes are stable across hosts.
    del rings, endpoint                                # pure POD, no ring slot
    return WireBaWindow(
        seq=int(msg.seq), ts_ns=int(msg.ts_ns),
        kf_ids=np.asarray(msg.kf_ids, dtype=np.int64).reshape(-1),
        kf_quat=np.asarray(msg.kf_quat, dtype=np.float64).reshape(-1, 4),
        kf_pos=np.asarray(msg.kf_pos, dtype=np.float64).reshape(-1, 3),
        lm_ids=np.asarray(msg.lm_ids, dtype=np.int64).reshape(-1),
        lm_xyz=np.asarray(msg.lm_xyz, dtype=np.float64).reshape(-1, 3),
        obs_kf=np.asarray(msg.obs_kf, dtype=np.int32).reshape(-1),
        obs_lm=np.asarray(msg.obs_lm, dtype=np.int32).reshape(-1),
        obs_uv=np.asarray(msg.obs_uv, dtype=np.float32).reshape(-1, 2),
        obs_reproj_px=np.asarray(msg.obs_reproj_px, dtype=np.float32).reshape(-1),
        ba_reproj_px=float(msg.ba_reproj_px),
        kf_quat_pre=np.asarray(msg.kf_quat_pre, dtype=np.float64).reshape(-1, 4),
        kf_pos_pre=np.asarray(msg.kf_pos_pre, dtype=np.float64).reshape(-1, 3),
        lm_xyz_pre=np.asarray(msg.lm_xyz_pre, dtype=np.float64).reshape(-1, 3),
        n_kf=int(msg.n_kf), n_lm=int(msg.n_lm))


def _ba_window_to_local(wm: WireBaWindow, rings: RingRegistry) -> BaWindow:
    del rings                                          # pure POD, no ring slot
    return BaWindow(
        seq=int(wm.seq), ts_ns=int(wm.ts_ns),
        kf_ids=np.asarray(wm.kf_ids, dtype=np.int64).reshape(-1),
        kf_quat=np.asarray(wm.kf_quat, dtype=np.float64).reshape(-1, 4),
        kf_pos=np.asarray(wm.kf_pos, dtype=np.float64).reshape(-1, 3),
        lm_ids=np.asarray(wm.lm_ids, dtype=np.int64).reshape(-1),
        lm_xyz=np.asarray(wm.lm_xyz, dtype=np.float64).reshape(-1, 3),
        obs_kf=np.asarray(wm.obs_kf, dtype=np.int32).reshape(-1),
        obs_lm=np.asarray(wm.obs_lm, dtype=np.int32).reshape(-1),
        obs_uv=np.asarray(wm.obs_uv, dtype=np.float32).reshape(-1, 2),
        obs_reproj_px=np.asarray(wm.obs_reproj_px, dtype=np.float32).reshape(-1),
        ba_reproj_px=float(wm.ba_reproj_px),
        kf_quat_pre=np.asarray(wm.kf_quat_pre, dtype=np.float64).reshape(-1, 4),
        kf_pos_pre=np.asarray(wm.kf_pos_pre, dtype=np.float64).reshape(-1, 3),
        lm_xyz_pre=np.asarray(wm.lm_xyz_pre, dtype=np.float64).reshape(-1, 3),
        n_kf=int(wm.n_kf), n_lm=int(wm.n_lm))


def _frontend_to_wire(msg: FrameFrontend, rings: RingRegistry, endpoint: str):
    # Pure POD: the quantised heatmap (<= 240 px longest side, uint8) + the capped
    # per-track flow arrays ride the message itself -- no shared-memory ring, no
    # full-resolution image (mirrors _ba_window_to_wire). Force the canonical
    # dtypes (the codec keys off dtype.name) so the bytes are stable across hosts.
    del rings, endpoint                                # pure POD, no ring slot
    return WireFrameFrontend(
        seq=int(msg.seq), ts_ns=int(msg.ts_ns),
        resp_q=np.asarray(msg.resp_q, dtype=np.uint8).reshape(
            msg.resp_q.shape[0], -1),
        resp_max=float(msg.resp_max),
        resp_h=int(msg.resp_h), resp_w=int(msg.resp_w),
        corner_xy=np.asarray(msg.corner_xy, dtype=np.float32).reshape(-1, 2),
        min_distance=float(msg.min_distance),
        quality_level=float(msg.quality_level),
        bucketed=bool(msg.bucketed),
        grid_rows=int(msg.grid_rows), grid_cols=int(msg.grid_cols),
        flow_id=np.asarray(msg.flow_id, dtype=np.int64).reshape(-1),
        flow_prev=np.asarray(msg.flow_prev, dtype=np.float32).reshape(-1, 2),
        flow_next=np.asarray(msg.flow_next, dtype=np.float32).reshape(-1, 2),
        flow_fb_err=np.asarray(msg.flow_fb_err, dtype=np.float32).reshape(-1),
        flow_culled=np.asarray(msg.flow_culled, dtype=bool).reshape(-1),
        fb_threshold=float(msg.fb_threshold))


def _frontend_to_local(wm: WireFrameFrontend, rings: RingRegistry) -> FrameFrontend:
    del rings                                          # pure POD, no ring slot
    return FrameFrontend(
        seq=int(wm.seq), ts_ns=int(wm.ts_ns),
        resp_q=np.asarray(wm.resp_q, dtype=np.uint8).reshape(
            wm.resp_q.shape[0], -1),
        resp_max=float(wm.resp_max),
        resp_h=int(wm.resp_h), resp_w=int(wm.resp_w),
        corner_xy=np.asarray(wm.corner_xy, dtype=np.float32).reshape(-1, 2),
        min_distance=float(wm.min_distance),
        quality_level=float(wm.quality_level),
        bucketed=bool(wm.bucketed),
        grid_rows=int(wm.grid_rows), grid_cols=int(wm.grid_cols),
        flow_id=np.asarray(wm.flow_id, dtype=np.int64).reshape(-1),
        flow_prev=np.asarray(wm.flow_prev, dtype=np.float32).reshape(-1, 2),
        flow_next=np.asarray(wm.flow_next, dtype=np.float32).reshape(-1, 2),
        flow_fb_err=np.asarray(wm.flow_fb_err, dtype=np.float32).reshape(-1),
        flow_culled=np.asarray(wm.flow_culled, dtype=bool).reshape(-1),
        fb_threshold=float(wm.fb_threshold))


# --------------------------------------------------------------------------- #
# Registry: topic -> (to_wire, to_local). Bridges pick converters by topic.
# Map-overlay + calib-bundle topics travel WITHOUT a local-side reconstruction
# (they're not flow messages); the UI subscribes to them directly off the wire.
# --------------------------------------------------------------------------- #
ToWire = Callable[[Any, RingRegistry, str], Any]
ToLocal = Callable[[Any, RingRegistry], Any]

#: Topics whose local <-> wire conversion is registered here. Adding a new
#: cross-process topic = add an entry below + (often) a new wire message in
#: ``comms.wire``.
CONVERTERS: dict[str, tuple[ToWire, ToLocal]] = {
    topics.CAM_SYNC:        (_cam_sync_to_wire, _cam_sync_to_local),
    topics.IMU_RAW:         (_imu_raw_to_wire,  _imu_raw_to_local),
    topics.IMUCAM_SAMPLE:   (_imucam_to_wire,   _imucam_to_local),
    topics.FRAME_DEPTH:     (_depth_to_wire,    _depth_to_local),
    topics.FRAME_TRACKS:    (_tracks_to_wire,   _tracks_to_local),
    topics.FRAME_INLIERS:   (_inliers_to_wire,  _inliers_to_local),
    topics.FRAME_GYROFUSE:  (_gyrofuse_to_wire, _gyrofuse_to_local),
    topics.POSE_ODOM:       (_pose_to_wire,     _pose_to_local),
    topics.POSE_VO:         (_pose_to_wire,     _pose_to_local),
    topics.POSE_REFINED:    (_pose_to_wire,     _pose_to_local),
    topics.KEYFRAME:        (_keyframe_to_wire, _keyframe_to_local),
    topics.LOOP_CORRECTION: (_loop_corr_to_wire, _loop_corr_to_local),
    topics.SLAM_MAP:        (_slam_overlay_to_wire, _slam_map_to_local),
    topics.SLAM_LOOP:       (_loop_match_to_wire, _loop_match_to_local),
    topics.BA_WINDOW:       (_ba_window_to_wire, _ba_window_to_local),
    topics.FRAME_FRONTEND:  (_frontend_to_wire, _frontend_to_local),
}


def to_wire(topic: str, msg: Any, rings: RingRegistry, endpoint: str) -> Any:
    """Convert ``msg`` for ``topic`` to its wire form, writing large arrays
    into ``rings`` under ``endpoint``."""
    if msg is END:
        return WireEnd(topic)
    fn, _ = CONVERTERS[topic]
    return fn(msg, rings, endpoint)


def to_local(topic: str, wm: Any, rings: RingRegistry) -> Any:
    """Convert ``wm`` back into the in-proc dataclass for ``topic``."""
    if isinstance(wm, WireEnd):
        return END
    _, fn = CONVERTERS[topic]
    return fn(wm, rings)
