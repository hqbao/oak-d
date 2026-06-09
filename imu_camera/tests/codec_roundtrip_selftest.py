"""Codec round-trip + cross-copy byte-parity self-test (Phase 1a).

Proves the canonical :mod:`imu_camera.comms.codec` is correct and STABLE:

1. ROUND-TRIP: build one instance of EVERY ``Wire*`` (including the
   SharedArrayRef-bearing ones, the dict-bearing one, Optional-None cases, an
   empty-ndarray IMU interval, NaN/Inf floats, and the WireEnd sentinel) and
   assert ``decode(encode(topic, msg)) == msg`` with deep / array-aware equality.

2. DIGEST FREEZE (the cross-copy byte-parity ORACLE): sha256 the ``encode()``
   bytes of each FIXED test vector and write them to ``codec_vectors.json`` the
   first time; on every subsequent run assert the digests match the saved file,
   so any silent field-order / dtype / endianness drift is caught. The SAME
   self-test vendored into each of the 5 projects must produce identical digests
   (CI compares the JSON byte-for-byte across copies).

3. SHARED-MEMORY RING: create a ring, write a gradient gray frame + an
   ``arange`` depth frame, ``read_copy`` them back, and assert binary equality
   (the one-block-per-ring layout is intact).

Run::

    .venv/bin/python -m imu_camera.tests.codec_roundtrip_selftest

Exit code 0 on success, 1 on any failure.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from imu_camera.comms import topics
from imu_camera.comms.codec import decode, encode
from imu_camera.comms.shared_array import SharedArrayRef, SharedArrayRing
from imu_camera.comms.wire import (
    WireCalibBundle, WireCamSync, WireDepthFrame, WireEnd, WireFrameInliers,
    WireFrameTracks, WireImuCamPacket, WireImuRaw, WireKeyframe,
    WireLoopCorrection, WirePoseMsg, WireSlamMap, WireVioMap,
)

#: Where the frozen sha256 digests live (the cross-copy byte-parity oracle).
VECTORS_PATH = Path(__file__).with_name("codec_vectors.json")


# --------------------------------------------------------------------------- #
# Deep / array-aware equality (dataclasses hold ndarrays / nested dicts).
# --------------------------------------------------------------------------- #
def _deep_equal(a, b) -> bool:
    """True iff ``a`` and ``b`` are structurally equal, array-aware.

    Handles ndarrays (shape + dtype + bitwise content, NaN treated as equal),
    SharedArrayRef, WireEnd, dataclass instances (by field), dicts and None.
    """
    if a is None or b is None:
        return a is b
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.size == 0:
            return True
        # equal_nan only valid for floats; use it there so NaN==NaN, and exact
        # bit compare otherwise (ints / bools).
        if np.issubdtype(a.dtype, np.floating):
            return bool(np.array_equal(a, b, equal_nan=True))
        return bool(np.array_equal(a, b))
    if isinstance(a, SharedArrayRef) and isinstance(b, SharedArrayRef):
        return (a.ring_name == b.ring_name and int(a.slot) == int(b.slot)
                and tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype)
    if isinstance(a, WireEnd) and isinstance(b, WireEnd):
        return a.topic == b.topic
    # Scalar float NaN: codec preserves it bitwise via struct('>d'), so NaN must
    # compare equal to NaN here (plain ``nan == nan`` is False).
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:           # both NaN
            return True
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_equal(a[k], b[k]) for k in a)
    # Dataclass instances: compare field by field (covers all Wire* types).
    if hasattr(a, "__dataclass_fields__") and hasattr(b, "__dataclass_fields__"):
        if type(a).__name__ != type(b).__name__:
            return False
        fields = a.__dataclass_fields__
        return all(_deep_equal(getattr(a, f), getattr(b, f)) for f in fields)
    return bool(a == b)


# --------------------------------------------------------------------------- #
# Fixed test vectors -- one per topic / edge case. ORDER and CONTENT are part of
# the frozen oracle: changing a vector changes its digest (so keep them fixed).
# --------------------------------------------------------------------------- #
def _ref(name: str, slot: int, shape, dtype: str) -> SharedArrayRef:
    return SharedArrayRef(ring_name=name, slot=slot, shape=tuple(shape), dtype=dtype)


def build_vectors() -> list[tuple[str, str, object]]:
    """Return ``[(vector_id, topic, wire_msg), ...]`` -- the frozen set.

    ``vector_id`` keys the digest in ``codec_vectors.json`` (stable across runs
    and across the 5 vendored copies); ``topic`` is the wire topic the codec
    keys on; ``wire_msg`` is the ``Wire*`` instance to round-trip.
    """
    gl = _ref("oak.capture.gray_left", 3, (400, 640), "uint8")
    gr = _ref("oak.capture.gray_right", 3, (400, 640), "uint8")
    dp = _ref("oak.capture.depth_m", 3, (400, 640), "float32")
    kf_gl = _ref("oak.vio.kf_gray", 1, (400, 640), "uint8")
    kf_dp = _ref("oak.vio.kf_depth", 1, (400, 640), "float32")

    # IMU interval of M=2 samples (gyro/accel (M,3) float64; ts (M,) int64).
    imu_ts = np.array([1_000, 2_000], dtype=np.int64)
    gyro = np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]], dtype=np.float64)
    accel = np.array([[0.0, 0.0, 9.81], [0.01, -0.02, 9.80]], dtype=np.float64)

    # Empty IMU interval (M=0) -- the first-frame / dropped-interval edge.
    empty_ts = np.zeros((0,), dtype=np.int64)
    empty_3 = np.zeros((0, 3), dtype=np.float64)

    K = np.array([[600.0, 0.0, 320.0],
                  [0.0, 600.0, 200.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [1.5, -2.5, 3.0]

    track_ids = np.array([10, 20, 30], dtype=np.int64)
    track_px = np.array([[100.5, 200.25], [110.0, 210.0], [120.0, 220.0]],
                        dtype=np.float32)
    inlier_ids = np.array([10, 30], dtype=np.int64)

    # FrameInliers per-PnP-point diagnostic: all M point ids + reprojected px +
    # inlier mask (id 20 is the outlier with a long stray reprojection).
    pnp_reproj = np.array([[100.4, 200.30], [137.0, 198.0], [120.1, 219.9]],
                          dtype=np.float32)
    pnp_inlier = np.array([True, False, True], dtype=bool)

    # PoseMsg.info: a dict with NaN / Inf float values to prove float fidelity.
    info = {"n_inliers": 42, "tracking_ok": True,
            "reproj_rms": float("nan"), "depth_max": float("inf")}

    # LoopCorrection.kf_poses: dict[int, (4,4) ndarray].
    pose_a = np.eye(4, dtype=np.float64); pose_a[0, 3] = 1.0
    pose_b = np.eye(4, dtype=np.float64); pose_b[1, 3] = -2.0
    kf_poses = {5: pose_a, 12: pose_b}

    slam_pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]], dtype=np.float64)
    slam_ids = np.array([0, 7], dtype=np.int64)
    last_match = np.array([[1.0, 0.5, -0.5]], dtype=np.float64)

    return [
        # --- SharedArrayRef-bearing (stereo + depth + keyframe) -------------- #
        ("cam_sync", topics.CAM_SYNC,
         WireCamSync(seq=3, ts_ns=123_456_789, gray_left_ref=gl, gray_right_ref=gr)),
        ("cam_sync_mono", topics.CAM_SYNC,
         WireCamSync(seq=4, ts_ns=987_654_321, gray_left_ref=gl, gray_right_ref=None)),
        ("imucam_packet", topics.IMUCAM_SAMPLE,
         WireImuCamPacket(seq=3, ts_ns=123_456_789, gray_left_ref=gl,
                          gray_right_ref=gr, imu_ts=imu_ts, gyro=gyro, accel=accel)),
        ("imucam_packet_empty_imu", topics.IMUCAM_SAMPLE,
         WireImuCamPacket(seq=0, ts_ns=0, gray_left_ref=gl, gray_right_ref=None,
                          imu_ts=empty_ts, gyro=empty_3, accel=empty_3)),
        ("depth_frame", topics.FRAME_DEPTH,
         WireDepthFrame(seq=3, ts_ns=123_456_789, gray_left_ref=gl, depth_ref=dp)),
        ("keyframe_full", topics.KEYFRAME,
         WireKeyframe(seq=7, T_world_cam=T, gray_left_ref=kf_gl, depth_ref=kf_dp,
                      track_ids=track_ids, track_px=track_px,
                      accel=np.array([0.0, 0.0, 9.81], dtype=np.float64),
                      inlier_ids=inlier_ids)),
        ("keyframe_optional_none", topics.KEYFRAME,
         WireKeyframe(seq=8, T_world_cam=T, gray_left_ref=kf_gl, depth_ref=kf_dp,
                      track_ids=None, track_px=None, accel=None, inlier_ids=None)),

        # --- pure POD (IMU / tracks / inliers / pose) ----------------------- #
        ("imu_raw", topics.IMU_RAW,
         WireImuRaw(seq=3, ts_ns=123_456_789, imu_ts=imu_ts, gyro=gyro, accel=accel)),
        ("imu_raw_empty", topics.IMU_RAW,
         WireImuRaw(seq=0, ts_ns=0, imu_ts=empty_ts, gyro=empty_3, accel=empty_3)),
        ("frame_tracks", topics.FRAME_TRACKS,
         WireFrameTracks(seq=3, ts_ns=123_456_789, ids=track_ids, points=track_px)),
        ("frame_inliers", topics.FRAME_INLIERS,
         WireFrameInliers(seq=3, ts_ns=123_456_789, ids=track_ids,
                          reproj=pnp_reproj, inlier=pnp_inlier)),
        ("pose_odom", topics.POSE_ODOM,
         WirePoseMsg(seq=3, ts_ns=123_456_789, T_world_cam=T, info=info)),
        ("pose_vo_empty_info", topics.POSE_VO,
         WirePoseMsg(seq=4, ts_ns=1, T_world_cam=T, info={})),
        ("pose_refined", topics.POSE_REFINED,
         WirePoseMsg(seq=5, ts_ns=2, T_world_cam=T, info={"refined": True})),

        # --- dict[int, ndarray] (loop correction) --------------------------- #
        ("loop_correction", topics.LOOP_CORRECTION,
         WireLoopCorrection(seq=12, kf_poses=kf_poses, n_loops=2)),
        ("loop_correction_empty", topics.LOOP_CORRECTION,
         WireLoopCorrection(seq=0, kf_poses={}, n_loops=0)),

        # --- SLAM map (Optional last_match present + None) ------------------- #
        ("slam_map_with_match", topics.SLAM_MAP,
         WireSlamMap(kf_ids=slam_ids, kf_positions=slam_pos, n_loops=1,
                     last_match=last_match)),
        ("slam_map_no_match", topics.SLAM_MAP,
         WireSlamMap(kf_ids=slam_ids, kf_positions=slam_pos, n_loops=0,
                     last_match=None)),

        # --- retained / read-directly (calib + vio map), all-optional + set -- #
        ("calib_bundle_minimal", topics.CALIB_BUNDLE,
         WireCalibBundle(K=K, width=640, height=400, fps=30)),
        ("calib_bundle_full", topics.CALIB_BUNDLE,
         WireCalibBundle(K=K, width=640, height=400, fps=30,
                         T_imu_left=T, R_imu_cam=np.eye(3, dtype=np.float64),
                         accel_align=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                         gyro_bias=np.array([1e-3, -2e-3, 3e-3], dtype=np.float64),
                         device_id="OAK-D-PRO-12345")),
        ("vio_map", topics.VIO_MAP,
         WireVioMap(kf_ids=slam_ids, kf_positions=slam_pos)),

        # --- END sentinel (handled out-of-band via tag 0x0A) ---------------- #
        ("wire_end", topics.IMUCAM_SAMPLE, WireEnd(topics.IMUCAM_SAMPLE)),
    ]


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def _check_roundtrips(vectors) -> list[str]:
    """Assert decode(encode()) == orig for every vector. Returns failure list."""
    failures: list[str] = []
    for vid, topic, msg in vectors:
        try:
            raw = encode(topic, msg)
            dec_topic, dec_msg = decode(raw)
        except Exception as e:                                     # noqa: BLE001
            failures.append(f"{vid}: encode/decode raised {e!r}")
            continue
        if dec_topic != topic:
            failures.append(f"{vid}: topic {dec_topic!r} != {topic!r}")
        if not _deep_equal(dec_msg, msg):
            failures.append(f"{vid}: round-trip mismatch\n  in : {msg}\n  out: {dec_msg}")
    return failures


def _digests(vectors) -> dict[str, str]:
    """sha256 of encode() for each vector, keyed by vector id (sorted output)."""
    out: dict[str, str] = {}
    for vid, topic, msg in vectors:
        out[vid] = hashlib.sha256(encode(topic, msg)).hexdigest()
    return dict(sorted(out.items()))


def _check_digests(vectors) -> list[str]:
    """Freeze digests on first run; thereafter assert they match. Failure list."""
    current = _digests(vectors)
    if not VECTORS_PATH.exists():
        VECTORS_PATH.write_text(json.dumps(current, indent=2) + "\n")
        print(f"[freeze] wrote {len(current)} digests -> {VECTORS_PATH.name}")
        return []
    saved = json.loads(VECTORS_PATH.read_text())
    failures: list[str] = []
    missing = set(saved) - set(current)
    extra = set(current) - set(saved)
    if missing:
        failures.append(f"vectors removed since freeze: {sorted(missing)}")
    if extra:
        failures.append(f"vectors added since freeze (re-freeze if intended): "
                        f"{sorted(extra)}")
    for vid in sorted(set(saved) & set(current)):
        if saved[vid] != current[vid]:
            failures.append(
                f"DIGEST DRIFT {vid}: saved {saved[vid][:12]}.. "
                f"!= now {current[vid][:12]}.. (wire-format break)")
    return failures


def _check_ring() -> list[str]:
    """Write gradient gray + arange depth into a ring, read_copy, assert equal."""
    failures: list[str] = []
    h, w, slots = 8, 12, 4
    name = "codec_selftest.ring"
    SharedArrayRing.cleanup_stale(name, slots)
    gray_ring = SharedArrayRing.create(name, slots, (h, w), np.uint8)
    depth_name = "codec_selftest.depth"
    SharedArrayRing.cleanup_stale(depth_name, slots)
    depth_ring = SharedArrayRing.create(depth_name, slots, (h, w), np.float32)
    try:
        gradient = (np.add.outer(np.arange(h), np.arange(w)) % 256).astype(np.uint8)
        depth = np.arange(h * w, dtype=np.float32).reshape(h, w)
        gslot = gray_ring.slot_for(3)
        dslot = depth_ring.slot_for(3)
        gref = gray_ring.write(gslot, gradient)
        dref = depth_ring.write(dslot, depth)
        gback = gray_ring.read_copy(gref)
        dback = depth_ring.read_copy(dref)
        if not np.array_equal(gback, gradient):
            failures.append("ring gray read_copy != written gradient")
        if not np.array_equal(dback, depth):
            failures.append("ring depth read_copy != written arange")
        # The ref metadata must round-trip through the codec too (SharedArrayRef).
        raw = encode(topics.FRAME_DEPTH,
                     WireDepthFrame(seq=3, ts_ns=1, gray_left_ref=gref, depth_ref=dref))
        _, wm = decode(raw)
        if not (_deep_equal(wm.gray_left_ref, gref) and _deep_equal(wm.depth_ref, dref)):
            failures.append("SharedArrayRef did not round-trip through codec")
    finally:
        gray_ring.unlink(); gray_ring.close()
        depth_ring.unlink(); depth_ring.close()
    return failures


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    vectors = build_vectors()
    all_failures: list[str] = []

    rt = _check_roundtrips(vectors)
    print(f"[round-trip] {len(vectors) - len(rt)}/{len(vectors)} vectors OK")
    all_failures += rt

    ring = _check_ring()
    print(f"[ring] {'OK' if not ring else 'FAIL'} (binary layout + ref codec)")
    all_failures += ring

    dig = _check_digests(vectors)
    if not dig:
        print(f"[digests] {len(vectors)} digests match {VECTORS_PATH.name}")
    all_failures += dig

    if all_failures:
        print("\nFAILURES:")
        for f in all_failures:
            print(f"  - {f}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
