#!/usr/bin/env python3
"""Offline integration test for the split camera/IMU acquisition front-end.

Wires the real flows -- :class:`~ours.flows.capture.cam_reader.CamReaderFlow` and
:class:`~ours.flows.capture.imu_reader.ImuReaderFlow` -- over a real
:class:`~ours.lib.flow.pubsub.Bus`, fed by a recorded gold session (no device),
and verifies the synchronisation contract end to end:

* one :class:`~ours.lib.flow.messages.ImuCamPacket` per camera frame,
* every packet carries its stereo frames,
* each packet's IMU samples fall in that frame's interval ``(prev_ts, ts]``,
* across the whole run every recorded IMU sample up to the last frame is
  delivered exactly once, in time order (none lost, none double-counted).

Run::

    python -m ours.tools.imucam_sync_selftest
    python -m ours.tools.imucam_sync_selftest --session sessions/gold/lab_loop_30s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.flows.capture.cam_reader import CamReaderFlow              # noqa: E402
from ours.flows.capture.cam_sources import ReplayCamSource          # noqa: E402
from ours.flows.capture.imu_reader import ImuReaderFlow             # noqa: E402
from ours.flows.capture.imu_sources import ReplayImuSource          # noqa: E402
from ours.lib.flow import Bus, Flow, topics                         # noqa: E402
from ours.lib.io.reader import SessionReader                        # noqa: E402


class _Collector(Flow):
    """Sink: gather every ImuCamPacket off the bus until END."""

    def __init__(self, bus: Bus) -> None:
        super().__init__("collector", bus)
        self.packets: list = []
        self.on(topics.IMUCAM_SAMPLE, [self._grab()])

    def _grab(collector_self):
        outer = collector_self

        class _Grab:
            name = "grab"

            def run(self, ctx, msg):
                outer.packets.append(msg)
                return None

        return _Grab()


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def run(session: str, max_frames: int) -> list:
    reader = SessionReader(Path(session))
    bus = Bus()

    imu_flow = ImuReaderFlow(bus, ReplayImuSource(reader))
    cam_flow = CamReaderFlow(
        bus, ReplayCamSource(reader, max_frames=max_frames), fps=20)
    sink = _Collector(bus)
    sink.expected_ends = 1

    # Order matters: the IMU reader (and its source) must be live before the
    # camera reader starts firing triggers, so early frames see a filling buffer.
    sink.start()
    imu_flow.start()
    cam_flow.start()
    cam_flow.join()
    finished = sink.done.wait(timeout=60.0)
    for f in (imu_flow, sink):
        f.stop()
    if not finished:
        raise SystemExit("graph did not drain within timeout")
    return sink.packets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_straight_20s")
    ap.add_argument("--max-frames", type=int, default=40)
    args = ap.parse_args()

    print("imucam_sync_selftest")
    reader = SessionReader(Path(args.session))
    n_frames = len(reader) if args.max_frames <= 0 else min(args.max_frames,
                                                            len(reader))
    imu = reader.load_imu()
    all_imu_ts = imu["ts_ns"].astype(np.int64)

    packets = run(args.session, args.max_frames)

    _check(len(packets) == n_frames,
           f"one packet per frame ({len(packets)}/{n_frames})")

    seqs = [p.seq for p in packets]
    _check(seqs == sorted(seqs), "packets arrive in frame order")

    _check(all(p.gray_left is not None for p in packets),
           "every packet carries the left frame")
    _check(all(p.gray_right is not None for p in packets),
           "every packet carries the right frame")

    # Per-packet interval check: samples lie in (prev_ts, ts].
    prev_ts = None
    interval_ok = True
    for p in packets:
        if p.imu_ts.size:
            lo = -np.inf if prev_ts is None else prev_ts
            if not (np.all(p.imu_ts > lo) and np.all(p.imu_ts <= p.ts_ns)):
                interval_ok = False
                break
            if not np.all(np.diff(p.imu_ts) >= 0):
                interval_ok = False
                break
        prev_ts = p.ts_ns
    _check(interval_ok, "each packet's IMU samples fall in (prev_ts, ts], ordered")

    # Global completeness: concatenated samples == recorded samples up to the
    # last frame timestamp, in order, none lost or duplicated.
    last_ts = packets[-1].ts_ns
    expected = all_imu_ts[all_imu_ts <= last_ts]
    got = np.concatenate([p.imu_ts for p in packets]) if packets else np.empty(0)
    _check(got.size == expected.size,
           f"sample count matches recorded up to last frame "
           f"({got.size}/{expected.size})")
    _check(np.array_equal(np.sort(got), np.sort(expected)),
           "exactly the recorded samples delivered (no loss/dup)")
    _check(np.array_equal(got, np.sort(got)), "delivered globally in time order")

    # Payload sanity: gyro/accel shapes align with timestamps.
    shapes_ok = all(p.gyro.shape == (p.imu_ts.size, 3)
                    and p.accel.shape == (p.imu_ts.size, 3) for p in packets)
    _check(shapes_ok, "gyro/accel shapes align with imu_ts")

    total = sum(p.imu_ts.size for p in packets)
    print(f"  frames={len(packets)} imu_samples={total} "
          f"mean_per_frame={total / max(1, len(packets)):.1f}")
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
