#!/usr/bin/env python3
"""Cross-project ``comms`` byte-parity gate for the 5-project split.

The split vendors ONE comms contract (``comms/``) bit-identically into all five
projects (imu_camera / vio / slam / ui / launcher). This test proves the contract
did NOT silently diverge between copies, on three independent axes:

1. SOURCE PARITY -- ``diff -r --exclude=__pycache__ <proj>/comms imu_camera/comms``
   must be EMPTY for proj in {vio, slam, ui, launcher}. (imu_camera is the anchor.)

2. CODEC PARITY -- import EACH copy's ``codec`` + ``wire``, encode a FIXED set of
   ``Wire*`` test vectors, and compare a sha256 digest of the produced bytes
   ACROSS ALL 5 COPIES. Identical source guarantees identical bytes; this is the
   live functional proof (catches a codec that would diverge even if the dir-diff
   were somehow fooled, e.g. an import-root that changed encoding). Each copy must
   ALSO decode every other copy's bytes back to an equal Wire object (the whole
   point of the class-path-independent codec).

3. TRANSPORT ROUND-TRIP -- a SharedArrayRing write/read_copy/wrap-around check,
   then a FULL bridge round-trip (local bus -> IPCPublisher -> wire/codec + shared
   memory -> IPCSubscriber -> local bus) over a real Unix-domain socket, asserting
   the reconstructed message is byte-equal to the published one. Proves the
   vendored bridge + codec + rings preserve the message contract end to end.

Runs on stdlib + numpy only (no depthai / Qt). FAILS LOUDLY on any divergence.

Run::

    .venv/bin/python verification/ipc_comms_selftest.py
"""
from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ANCHOR = "imu_camera"
COPIES = ("imu_camera", "vio", "slam", "ui", "launcher")


def _check(cond: bool, msg: str) -> bool:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    return bool(cond)


# --------------------------------------------------------------------------- #
# 1) SOURCE PARITY -- dir diff of every comms copy vs the anchor.
# --------------------------------------------------------------------------- #
def test_source_parity() -> bool:
    print("\n[1] comms/ source parity (diff -r --exclude=__pycache__)")
    ok = True
    anchor_dir = REPO / ANCHOR / "comms"
    for proj in ("vio", "slam", "ui", "launcher"):
        proj_dir = REPO / proj / "comms"
        res = subprocess.run(
            ["diff", "-r", "--exclude=__pycache__",
             str(proj_dir), str(anchor_dir)],
            capture_output=True, text=True)
        empty = (res.returncode == 0 and not res.stdout.strip()
                 and not res.stderr.strip())
        ok &= _check(empty, f"{proj}/comms == {ANCHOR}/comms (diff EMPTY)")
        if not empty:
            print("        --- diff output ---")
            for line in (res.stdout + res.stderr).splitlines():
                print(f"        {line}")
    return ok


# --------------------------------------------------------------------------- #
# 2) CODEC PARITY -- 5-copy encode digest + cross-decode.
# --------------------------------------------------------------------------- #
def _build_vectors(mod):
    """Fixed Wire* test vectors for ONE copy's wire module.

    Deterministic content (no RNG) so the digest is reproducible. Exercises every
    codec type tag: int / float / bool / str / None / dict (str+int keys) /
    ndarray (uint8, int64, float64) / SharedArrayRef / WireEnd.
    """
    w = mod
    K = (np.arange(9, dtype=np.float64).reshape(3, 3) + 0.5)
    T = np.eye(4, dtype=np.float64); T[:3, 3] = [1.25, -2.5, 3.75]
    ids = np.arange(5, dtype=np.int64)
    pts = (np.arange(10, dtype=np.float64).reshape(5, 2) * 1.5)
    imu_ts = np.arange(4, dtype=np.int64) * 5_000_000
    gyro = (np.arange(12, dtype=np.float64).reshape(4, 3) * 0.001)
    accel = (np.arange(12, dtype=np.float64).reshape(4, 3) * 0.01)
    ref_l = w.SharedArrayRef(ring_name="oak.capture.gray_left", slot=3,
                             shape=(400, 640), dtype="uint8")
    ref_d = w.SharedArrayRef(ring_name="oak.capture.depth_m", slot=3,
                             shape=(400, 640), dtype="float32")
    info = {"ok": True, "n_inliers": 42, "reproj_px": 0.75,
            "mode": "vio", "absent": None}
    kf_poses = {0: T.copy(), 7: (T @ T).copy()}

    # (topic, Wire instance) per registered data topic + the END sentinel.
    return [
        (w.topics.IMU_RAW,
         w.WireImuRaw(seq=11, ts_ns=123_456_789, imu_ts=imu_ts,
                      gyro=gyro, accel=accel)),
        (w.topics.IMUCAM_SAMPLE,
         w.WireImuCamPacket(seq=12, ts_ns=987_654_321,
                            gray_left_ref=ref_l, gray_right_ref=None,
                            imu_ts=imu_ts, gyro=gyro, accel=accel)),
        (w.topics.CAM_SYNC,
         w.WireCamSync(seq=13, ts_ns=42, gray_left_ref=ref_l,
                       gray_right_ref=ref_l)),
        (w.topics.FRAME_DEPTH,
         w.WireDepthFrame(seq=14, ts_ns=43, gray_left_ref=ref_l,
                          depth_ref=ref_d)),
        (w.topics.FRAME_TRACKS,
         w.WireFrameTracks(seq=15, ts_ns=44, ids=ids, points=pts)),
        (w.topics.FRAME_INLIERS,
         w.WireFrameInliers(seq=16, ts_ns=45, ids=ids,
                            reproj=pts.astype(np.float32),
                            inlier=(ids % 2 == 0))),
        (w.topics.POSE_ODOM,
         w.WirePoseMsg(seq=17, ts_ns=46, T_world_cam=T, info=info)),
        (w.topics.POSE_VO,
         w.WirePoseMsg(seq=18, ts_ns=47, T_world_cam=T, info={})),
        (w.topics.POSE_REFINED,
         w.WirePoseMsg(seq=19, ts_ns=48, T_world_cam=T, info=info)),
        (w.topics.KEYFRAME,
         w.WireKeyframe(seq=20, T_world_cam=T, gray_left_ref=ref_l,
                        depth_ref=ref_d, track_ids=ids, track_px=pts,
                        accel=accel[0], inlier_ids=ids)),
        (w.topics.LOOP_CORRECTION,
         w.WireLoopCorrection(seq=21, kf_poses=kf_poses, n_loops=3)),
        (w.topics.SLAM_MAP,
         w.WireSlamMap(kf_ids=ids, kf_positions=pts[:, :1].repeat(3, axis=1),
                       n_loops=2, last_match=None)),
        (w.topics.CALIB_BUNDLE,
         w.WireCalibBundle(K=K, width=640, height=400, fps=20,
                           T_imu_left=T, R_imu_cam=K, accel_align=accel[0],
                           gyro_bias=gyro[0], device_id="dev-01")),
        (w.topics.VIO_MAP,
         w.WireVioMap(kf_ids=ids,
                      kf_positions=pts[:, :1].repeat(3, axis=1))),
        ("pose.odom", w.WireEnd(topic="pose.odom")),   # END sentinel (tag 0x0A)
    ]


def _encode_all(copy_name: str):
    """Import one copy's codec+wire, encode the fixed vectors -> (digest, blobs)."""
    codec = importlib.import_module(f"{copy_name}.comms.codec")
    wire = importlib.import_module(f"{copy_name}.comms.wire")
    vectors = _build_vectors(wire)
    h = hashlib.sha256()
    blobs = []
    for topic, msg in vectors:
        b = codec.encode(topic, msg)
        h.update(len(topic).to_bytes(4, "big"))
        h.update(topic.encode("utf-8"))
        h.update(len(b).to_bytes(4, "big"))
        h.update(b)
        blobs.append((topic, b))
    return h.hexdigest(), blobs


def _wire_eq(a, b) -> bool:
    """Structural equality of two decoded Wire dataclasses."""
    from dataclasses import fields, is_dataclass
    if type(a).__name__ != type(b).__name__:
        return False
    if not is_dataclass(a):
        return a == b
    for f in fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if va is None or vb is None:
                if va is not vb:
                    return False
            elif not np.array_equal(va, vb):
                return False
        elif isinstance(va, dict):
            if set(va) != set(vb):
                return False
            for k in va:
                x, y = va[k], vb[k]
                if isinstance(x, np.ndarray):
                    if not np.array_equal(x, y):
                        return False
                elif x != y:
                    return False
        elif is_dataclass(va):
            if not _wire_eq(va, vb):
                return False
        elif va != vb:
            return False
    return True


def test_codec_parity() -> bool:
    print("\n[2] codec byte-parity across all 5 copies (sha256 of encoded vectors)")
    ok = True
    digests = {}
    blobs_by_copy = {}
    for copy in COPIES:
        try:
            digest, blobs = _encode_all(copy)
        except Exception as e:                                       # noqa: BLE001
            ok &= _check(False, f"{copy}: encode raised {e!r}")
            continue
        digests[copy] = digest
        blobs_by_copy[copy] = blobs
        print(f"        {copy:11s} sha256 = {digest}")

    if len(digests) == len(COPIES):
        anchor_dig = digests[ANCHOR]
        for copy in COPIES:
            ok &= _check(digests[copy] == anchor_dig,
                         f"{copy} encode-digest == {ANCHOR} "
                         f"({len(blobs_by_copy[copy])} vectors)")

    # Cross-decode: every copy must decode every other copy's bytes to an EQUAL
    # Wire object (the class-path-independent codec's entire raison d'etre).
    print("        -- cross-decode (each copy decodes every other's bytes) --")
    for producer in COPIES:
        if producer not in blobs_by_copy:
            continue
        for consumer in COPIES:
            try:
                dec = importlib.import_module(f"{consumer}.comms.codec")
                wmod = importlib.import_module(f"{consumer}.comms.wire")
                ref_vectors = dict(
                    (i, v) for i, v in enumerate(_build_vectors(wmod)))
                all_eq = True
                for i, (topic, blob) in enumerate(blobs_by_copy[producer]):
                    dtopic, dmsg = dec.decode(blob)
                    _, expected = ref_vectors[i]
                    if dtopic != topic or not _wire_eq(dmsg, expected):
                        all_eq = False
                        break
            except Exception as e:                                  # noqa: BLE001
                all_eq = False
                print(f"        decode {producer}->{consumer} raised {e!r}")
            ok &= _check(all_eq, f"{consumer} decodes {producer}'s bytes -> equal")
    return ok


# --------------------------------------------------------------------------- #
# 3) TRANSPORT ROUND-TRIP -- SharedArrayRing + full bridge over a real socket.
# --------------------------------------------------------------------------- #
def test_shared_array_ring() -> bool:
    print("\n[3a] SharedArrayRing write / read_copy / wrap-around")
    from imu_camera.comms.shared_array import SharedArrayRing
    ok = True
    name = f"vt.{os.getpid() & 0xFFFF:x}.r"
    SharedArrayRing.cleanup_stale(name, 4)
    ring = SharedArrayRing.create(name, slots=4, shape=(32, 48), dtype="uint8")
    try:
        arr0 = np.full((32, 48), 7, dtype=np.uint8)
        ref0 = ring.write(0, arr0)
        out0 = ring.read_copy(ref0)
        ok &= _check(np.array_equal(arr0, out0), "round-trip slot 0 equal")
        ok &= _check(out0.ctypes.data != arr0.ctypes.data, "read_copy is a copy")
        arr1 = np.full((32, 48), 199, dtype=np.uint8)
        ring.write(0, arr1)
        ok &= _check(np.array_equal(arr1, ring.read_copy(ref0)),
                     "wrap-around overwrites slot")
        raised = False
        try:
            ring.write(1, np.zeros((10, 10), dtype=np.uint8))
        except ValueError:
            raised = True
        ok &= _check(raised, "shape mismatch raises ValueError")
    finally:
        ring.unlink()
        ring.close()
    return ok


def _make_packet(seq: int, h: int = 32, w: int = 48):
    from imu_camera.comms.messages import ImuCamPacket
    base = (np.arange(h * w, dtype=np.int32).reshape(h, w) + int(seq)) % 256
    left = base.astype(np.uint8)
    right = ((base + 7) % 256).astype(np.uint8)
    M = 4
    ts0 = int(seq) * 50_000_000
    imu_ts = ts0 + np.arange(M, dtype=np.int64) * 5_000_000
    gyro = (np.arange(M * 3, dtype=np.float64).reshape(M, 3) + seq) * 0.001
    accel = (np.arange(M * 3, dtype=np.float64).reshape(M, 3) + seq) * 0.01
    return ImuCamPacket(seq=int(seq), ts_ns=ts0, gray_left=left, gray_right=right,
                        imu_ts=imu_ts, gyro=gyro, accel=accel)


def test_bridge_roundtrip() -> bool:
    print("\n[3b] full bridge round-trip (local -> IPCPublisher -> wire/codec + "
          "shm -> IPCSubscriber -> local)")
    from imu_camera.comms import topics
    from imu_camera.comms.pubsub import LocalPubSub
    from imu_camera.comms.ipc import IPCPubSub
    from imu_camera.comms.bridge import IPCPublisher, IPCSubscriber
    from imu_camera.comms.ring_registry import (
        RingRegistry, default_capture_specs)

    ok = True
    endpoint = f"vt.{os.getpid() & 0xFFFF:x}.br"
    h, w, n_msgs = 32, 48, 4

    pub_rings = RingRegistry().create_all(
        default_capture_specs(endpoint=endpoint, width=w, height=h, slots=8))
    sub_rings = RingRegistry().attach_all(
        default_capture_specs(endpoint=endpoint, width=w, height=h, slots=8))

    pub_local = LocalPubSub()
    sub_local = LocalPubSub()
    received: list = []
    done = threading.Event()

    def on_local(msg) -> None:
        received.append(msg)
        if len(received) >= n_msgs:
            done.set()

    sub_local.subscribe(topics.IMUCAM_SAMPLE, on_local)

    server = IPCPubSub(endpoint, role="server")
    client = IPCPubSub(endpoint, role="client")
    pub = IPCPublisher(pub_local, server, pub_rings, [topics.IMUCAM_SAMPLE])
    sub = IPCSubscriber(sub_local, client, sub_rings, [topics.IMUCAM_SAMPLE])

    sent = []
    try:
        pub.start()                       # binds + starts the server socket
        time.sleep(0.3)
        sub.start()                       # connects the client
        time.sleep(0.4)                   # let the subscribe handshake complete
        for s in range(n_msgs):
            pkt = _make_packet(s, h=h, w=w)
            pub_local.publish(topics.IMUCAM_SAMPLE, pkt)
            sent.append(pkt)
            time.sleep(0.05)
        got = done.wait(timeout=5.0)
        ok &= _check(got and len(received) == n_msgs,
                     f"received all {n_msgs} frames (got {len(received)})")
        for r, exp in zip(sorted(received, key=lambda p: p.seq), sent):
            ok &= _check(r.seq == exp.seq, f"seq {exp.seq} matches")
            ok &= _check(np.array_equal(r.gray_left, exp.gray_left),
                         f"gray_left[{exp.seq}] bytes identical")
            ok &= _check(np.array_equal(r.gray_right, exp.gray_right),
                         f"gray_right[{exp.seq}] bytes identical")
            ok &= _check(np.array_equal(r.imu_ts, exp.imu_ts),
                         f"imu_ts[{exp.seq}] identical")
            ok &= _check(np.array_equal(r.gyro, exp.gyro),
                         f"gyro[{exp.seq}] identical")
            ok &= _check(np.array_equal(r.accel, exp.accel),
                         f"accel[{exp.seq}] identical")
    finally:
        try:
            sub.stop()
        except Exception:                                          # noqa: BLE001
            pass
        try:
            pub.stop()
        except Exception:                                          # noqa: BLE001
            pass
        sub_rings.close()
        pub_rings.unlink()
        pub_rings.close()
    return ok


# --------------------------------------------------------------------------- #
def main() -> int:
    print("ipc_comms_selftest -- cross-project comms byte-parity (5 copies)")
    results = {
        "source parity":   test_source_parity(),
        "codec parity":    test_codec_parity(),
        "shared-array ring": test_shared_array_ring(),
        "bridge round-trip": test_bridge_roundtrip(),
    }
    print("\n" + "=" * 70)
    all_ok = all(results.values())
    for name, ok in results.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if all_ok:
        print("\nPASS -- the vendored comms contract is byte-identical across all "
              "5 copies and round-trips intact.")
        return 0
    print("\nFAIL -- a comms copy DIVERGED. VETO: see the [FAIL] lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
