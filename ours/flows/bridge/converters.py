"""Convert between local (in-proc) dataclasses and wire messages.

A converter is two halves:

* ``to_wire(local_msg, rings, seq)`` -- copy any large arrays into shared memory
  and build the matching :mod:`ours.lib.ipc.messages` dataclass.
* ``to_local(wire_msg, rings)`` -- ``read_copy`` every shared-memory ref into a
  private ``np.ndarray`` and reconstruct the in-proc dataclass.

Both halves know the per-topic ring naming convention (gray_left, gray_right,
depth_m, kf_gray, kf_depth) so the bridge flows just pick the right converter
by topic.

INVARIANT: every numpy array on the local side that came from shared memory is
``read_copy``-ed (an independent allocation) before any downstream task sees
it. The slot is then free to be reused by the producer's next frame, and no
flow can ever read a half-overwritten slot. See ``docs/PROC4_ARCHITECTURE.md``
§9 invariant 5.
"""
from __future__ import annotations

from typing import Any, Callable

from ...lib.flow import topics
from ...lib.flow.messages import (
    CamSync, DepthFrame, FrameInliers, FrameTracks, ImuCamPacket, ImuRaw,
    Keyframe, LoopCorrection, PoseMsg, END,
)
from ...lib.ipc.messages import (
    WireCamSync, WireDepthFrame, WireEnd, WireFrameInliers, WireFrameTracks,
    WireImuCamPacket, WireImuRaw, WireKeyframe, WireLoopCorrection, WirePoseMsg,
    WireCalibBundle, WireVioMap, WireSlamMap,
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
    return WireFrameInliers(seq=int(msg.seq), ts_ns=int(msg.ts_ns), ids=msg.ids)


def _inliers_to_local(wm: WireFrameInliers, rings: RingRegistry) -> FrameInliers:
    return FrameInliers(seq=int(wm.seq), ts_ns=int(wm.ts_ns), ids=wm.ids)


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


# --------------------------------------------------------------------------- #
# Registry: topic -> (to_wire, to_local). Bridges pick converters by topic.
# Map-overlay + calib-bundle topics travel WITHOUT a local-side reconstruction
# (they're not flow messages); the UI subscribes to them directly.
# --------------------------------------------------------------------------- #
ToWire = Callable[[Any, RingRegistry, str], Any]
ToLocal = Callable[[Any, RingRegistry], Any]

#: Topics whose local <-> wire conversion is registered here. Adding a new
#: cross-process topic = add an entry below + (often) a new wire message in
#: ``ours.lib.ipc.messages``.
CONVERTERS: dict[str, tuple[ToWire, ToLocal]] = {
    topics.CAM_SYNC:        (_cam_sync_to_wire, _cam_sync_to_local),
    topics.IMU_RAW:         (_imu_raw_to_wire,  _imu_raw_to_local),
    topics.IMUCAM_SAMPLE:   (_imucam_to_wire,   _imucam_to_local),
    topics.FRAME_DEPTH:     (_depth_to_wire,    _depth_to_local),
    topics.FRAME_TRACKS:    (_tracks_to_wire,   _tracks_to_local),
    topics.FRAME_INLIERS:   (_inliers_to_wire,  _inliers_to_local),
    topics.POSE_ODOM:       (_pose_to_wire,     _pose_to_local),
    topics.POSE_REFINED:    (_pose_to_wire,     _pose_to_local),
    topics.KEYFRAME:        (_keyframe_to_wire, _keyframe_to_local),
    topics.LOOP_CORRECTION: (_loop_corr_to_wire, _loop_corr_to_local),
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
