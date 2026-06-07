#!/usr/bin/env python3
"""Offline test for the IPC primitives (``ours.lib.ipc``) + bridge flows.

Validates three layers without touching the OAK-D:

1. :class:`SharedArrayRing` -- create / write / read_copy / wrap-around / cleanup.
2. :class:`IpcServerBus` / :class:`IpcClientBus` -- spawn a subscriber subprocess,
   round-trip an :class:`ImuCamPacket` carrying numpy arrays via shared memory,
   verify byte-for-byte equality.
3. Bridge flow (``IpcPublisherFlow`` + ``IpcSubscriberFlow``) -- the same
   round-trip but driven through the **local-bus to local-bus** ring (two procs)
   the live pipeline uses. Proves the bridge preserves the message contract.

Runs entirely on stdlib + numpy. No depthai, no Qt.

Run::

    python -m ours.tools.ipc_bus_selftest
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.flow import Bus, topics                              # noqa: E402
from ours.lib.flow.messages import ImuCamPacket, ImuRaw            # noqa: E402
from ours.lib.ipc import IpcClientBus, IpcServerBus, SharedArrayRing  # noqa: E402
from ours.flows.bridge import (                                    # noqa: E402
    IpcPublisherFlow, IpcSubscriberFlow,
    RingRegistry,
)
from ours.flows.bridge.ring_registry import default_capture_specs  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def _make_frame(seq: int, h: int = 32, w: int = 48) -> ImuCamPacket:
    """Deterministic synthetic frame -- gradient image + linear IMU rows."""
    base = (np.arange(h * w, dtype=np.int32).reshape(h, w) + int(seq)) % 256
    left = base.astype(np.uint8)
    right = ((base + 7) % 256).astype(np.uint8)
    M = 4
    ts0 = int(seq) * 50_000_000
    imu_ts = ts0 + np.arange(M, dtype=np.int64) * 5_000_000
    gyro = (np.arange(M * 3, dtype=np.float64).reshape(M, 3) + seq) * 0.001
    accel = (np.arange(M * 3, dtype=np.float64).reshape(M, 3) + seq) * 0.01
    return ImuCamPacket(seq=int(seq), ts_ns=ts0,
                        gray_left=left, gray_right=right,
                        imu_ts=imu_ts, gyro=gyro, accel=accel)


# --------------------------------------------------------------------------- #
# 1) SharedArrayRing
# --------------------------------------------------------------------------- #
def test_shared_array_ring() -> None:
    print("\nSharedArrayRing")
    # macOS POSIX shm names are capped at 30 chars (+ leading '/'). Keep the
    # test name short -- short enough that '.N' suffixes fit too.
    name = f"t.{os.getpid() & 0xFFFF:x}.r"
    SharedArrayRing.cleanup_stale(name, 4)
    ring = SharedArrayRing.create(name, slots=4, shape=(32, 48), dtype="uint8")
    try:
        arr0 = np.full((32, 48), 7, dtype=np.uint8)
        ref0 = ring.write(0, arr0)
        out0 = ring.read_copy(ref0)
        _check(np.array_equal(arr0, out0), "round-trip slot 0 equal")
        _check(out0.ctypes.data != arr0.ctypes.data, "read_copy is a copy")

        # Wrap-around: slot index reuses an old slot, must overwrite cleanly.
        arr1 = np.full((32, 48), 199, dtype=np.uint8)
        ring.write(0, arr1)
        out1 = ring.read_copy(ref0)
        _check(np.array_equal(arr1, out1), "wrap-around overwrites slot")

        # Shape / dtype mismatch caught at boundary.
        try:
            ring.write(1, np.zeros((10, 10), dtype=np.uint8))
            ok = False
        except ValueError:
            ok = True
        _check(ok, "shape mismatch raises ValueError")
        try:
            ring.write(1, np.zeros((32, 48), dtype=np.float32))
            ok = False
        except ValueError:
            ok = True
        _check(ok, "dtype mismatch raises ValueError")
    finally:
        ring.unlink()
        ring.close()


# --------------------------------------------------------------------------- #
# 2) IpcServerBus / IpcClientBus round-trip (two processes)
# --------------------------------------------------------------------------- #
def _bus_subscriber_proc(endpoint: str, n: int, out_q: mp.Queue) -> None:
    """Subscriber subprocess: read ``n`` ImuRaw messages, send a summary back."""
    bus = IpcClientBus(endpoint, connect_timeout_s=5.0)
    received = []
    done = mp.Event()

    def on_msg(wm) -> None:
        received.append((int(wm.seq), int(wm.ts_ns),
                         wm.gyro.tolist(), wm.accel.tolist()))
        if len(received) >= n:
            done.set()

    bus.subscribe(topics.IMU_RAW, on_msg)
    try:
        bus.start()
    except Exception as e:                                         # noqa: BLE001
        out_q.put(("ERR", f"client.start: {e}"))
        return
    done.wait(timeout=5.0)
    out_q.put(("OK", received))
    bus.stop()


def test_ipc_bus_roundtrip() -> None:
    print("\nIpcBus round-trip (two processes)")
    # IpcBus endpoint is a unix socket path -> not the 30-char shm cap. Keep
    # it short anyway for tidy logs.
    endpoint = f"t.{os.getpid() & 0xFFFF:x}.b"
    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue()
    n_msgs = 5
    sub = ctx.Process(target=_bus_subscriber_proc,
                      args=(endpoint, n_msgs, out_q),
                      name="ipcbus-sub", daemon=True)
    sub.start()

    server = IpcServerBus(endpoint)
    server.start()
    # Wait briefly for the subscriber to connect (handshake happens in accept).
    time.sleep(0.3)

    sent = []
    for s in range(n_msgs):
        ts = int(s) * 50_000_000
        gyro = (np.arange(3, dtype=np.float64) + s) * 0.001
        accel = (np.arange(3, dtype=np.float64) + s) * 0.01
        msg = ImuRaw(seq=s, ts_ns=ts,
                     imu_ts=np.array([ts], dtype=np.int64),
                     gyro=gyro.reshape(1, 3), accel=accel.reshape(1, 3))
        # Use the same converter the bridge uses so we exercise the on-wire shape.
        from ours.lib.ipc.messages import WireImuRaw
        wm = WireImuRaw(seq=msg.seq, ts_ns=msg.ts_ns,
                        imu_ts=msg.imu_ts, gyro=msg.gyro, accel=msg.accel)
        server.publish(topics.IMU_RAW, wm)
        sent.append((s, ts, msg.gyro.tolist(), msg.accel.tolist()))
        time.sleep(0.02)

    sub.join(timeout=5.0)
    tag, payload = out_q.get(timeout=2.0)
    server.close()
    _check(tag == "OK",
           f"subscriber returned OK (got {tag}: {payload!r})")
    received = payload
    _check(len(received) == n_msgs,
           f"received all {n_msgs} messages (got {len(received)})")
    _check(received == sent, "received content matches sent content")


# --------------------------------------------------------------------------- #
# 3) Full bridge round-trip: local Bus -> publisher -> wire -> subscriber -> local Bus
# --------------------------------------------------------------------------- #
def _bridge_subscriber_proc(endpoint: str, n: int, h: int, w: int,
                            out_q: mp.Queue) -> None:
    """Subscriber subprocess: attach rings, bridge -> local bus, collect frames."""
    rings = RingRegistry().attach_all(default_capture_specs(
        endpoint=endpoint, width=w, height=h, slots=4))
    local = Bus()
    received: list[ImuCamPacket] = []
    done = mp.Event()

    def on_local(msg: ImuCamPacket) -> None:
        received.append(msg)
        if len(received) >= n:
            done.set()

    local.subscribe(topics.IMUCAM_SAMPLE, on_local)
    client = IpcClientBus(endpoint, connect_timeout_s=5.0)
    sub_flow = IpcSubscriberFlow(local, client, rings, [topics.IMUCAM_SAMPLE])
    sub_flow.start()
    done.wait(timeout=5.0)
    # Serialise for IPC return (numpy arrays survive pickle, but easier as POD).
    out_q.put(("OK", [(p.seq, p.ts_ns,
                       p.gray_left.tolist(), p.gray_right.tolist(),
                       p.imu_ts.tolist(), p.gyro.tolist(), p.accel.tolist())
                      for p in received]))
    sub_flow.stop()
    rings.close()


def test_bridge_roundtrip() -> None:
    print("\nBridge flow round-trip (two processes)")
    # Ring names are <endpoint>.<stream>.<slot>. Capture's longest stream is
    # "gray_right" (10 chars) + ".N" (2 chars) + "." (1) = 13 chars budget; the
    # rest goes to the endpoint name. Keep the test endpoint tiny.
    endpoint = f"t.{os.getpid() & 0xFFFF:x}.br"
    h, w = 32, 48
    n_msgs = 4

    # Producer side: create rings, start server, start publisher flow.
    rings = RingRegistry().create_all(default_capture_specs(
        endpoint=endpoint, width=w, height=h, slots=4))
    local = Bus()
    server = IpcServerBus(endpoint)
    pub_flow = IpcPublisherFlow(local, server, rings, [topics.IMUCAM_SAMPLE])
    pub_flow.start()

    payload = None
    tag = None
    try:
        ctx = mp.get_context("spawn")
        out_q: mp.Queue = ctx.Queue()
        sub = ctx.Process(target=_bridge_subscriber_proc,
                          args=(endpoint, n_msgs, h, w, out_q),
                          name="bridge-sub", daemon=True)
        sub.start()
        # Wait for the subscriber to attach + connect.
        time.sleep(0.5)

        sent = []
        for s in range(n_msgs):
            frame = _make_frame(s, h=h, w=w)
            local.publish(topics.IMUCAM_SAMPLE, frame)
            sent.append(frame)
            time.sleep(0.03)

        sub.join(timeout=5.0)
        tag, payload = out_q.get(timeout=2.0)
    finally:
        pub_flow.stop()
        server.close()
        rings.unlink()
        rings.close()

    _check(tag == "OK",
           f"subscriber returned OK (got tag={tag})")
    _check(len(payload) == n_msgs,
           f"received all {n_msgs} frames (got {len(payload)})")
    for (s, ts_ns, left, right, imu_ts, gyro, accel), expected in zip(
            payload, sent):
        _check(s == expected.seq, f"seq matches (got {s} vs {expected.seq})")
        _check(ts_ns == expected.ts_ns, "ts_ns matches")
        _check(np.array_equal(np.asarray(left, dtype=np.uint8),
                              expected.gray_left),
               f"gray_left[{s}] bytes match")
        _check(np.array_equal(np.asarray(right, dtype=np.uint8),
                              expected.gray_right),
               f"gray_right[{s}] bytes match")
        _check(np.array_equal(np.asarray(imu_ts, dtype=np.int64),
                              expected.imu_ts),
               f"imu_ts[{s}] match")
        _check(np.allclose(np.asarray(gyro, dtype=np.float64),
                           expected.gyro),
               f"gyro[{s}] match")
        _check(np.allclose(np.asarray(accel, dtype=np.float64),
                           expected.accel),
               f"accel[{s}] match")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=("ring", "bus", "bridge"),
                    help="run only one of the three tests")
    args = ap.parse_args()

    print("ipc_bus_selftest")
    if args.only in (None, "ring"):
        test_shared_array_ring()
    if args.only in (None, "bus"):
        test_ipc_bus_roundtrip()
    if args.only in (None, "bridge"):
        test_bridge_roundtrip()
    print("\nALL IPC SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
