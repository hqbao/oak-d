#!/usr/bin/env python3
"""Offline test for the realtime backpressure admission gate (imu-reader).

The live camera streams faster than the VIO sustains; without a bound the host
backlog of ~0.5 MB stereo packets grows until memory pressure starves the depthai
link and the device firmware watchdog crashes the camera (observed on the bench).
The :mod:`~ours.flows.imu_reader.admission` gate caps frames in flight. This test
proves the gate offline (no device):

1. **Strategy logic** -- :class:`AdmitAll` admits everything; :class:`BudgetAdmission`
   admits exactly ``budget`` before blocking, frees one per ``complete``, and never
   goes negative on a stray completion.
2. **IMU folding across a skip** -- when a frame is skipped the buffer is NOT
   drained, so the skipped interval's inertial samples fold into the next admitted
   frame (gyro preintegration stays gap-free, nothing lost or double-counted).
3. **Cap end-to-end** -- the real :class:`CamReaderFlow` + :class:`ImuReaderFlow`
   over a gold session with a budget of ``N`` and NO completions deliver exactly
   ``N`` packets (the first ``N`` frames, in order): the host backlog is bounded.

Run::

    python -m ours.tools.admission_selftest
    python -m ours.tools.admission_selftest --session sessions/gold/lab_loop_30s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.flows.cam_reader import CamReaderFlow                   # noqa: E402
from ours.flows.cam_reader.sources import ReplayCamSource         # noqa: E402
from ours.flows.imu_reader import ImuReaderFlow                   # noqa: E402
from ours.flows.imu_reader.sources import ReplayImuSource         # noqa: E402
from ours.flows.imu_reader.admission import AdmitAll, BudgetAdmission  # noqa: E402
from ours.flows.imu_reader.admit_frame import AdmitFrame          # noqa: E402
from ours.flows.imu_reader.pack_imucam import PackImuCam          # noqa: E402
from ours.lib.flow import Bus, Flow, topics                       # noqa: E402
from ours.lib.flow.messages import CamSync                        # noqa: E402
from ours.lib.imu.timed_buffer import TimedImuBuffer             # noqa: E402
from ours.lib.io.reader import SessionReader                       # noqa: E402


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def test_strategies() -> None:
    print("strategies")
    a = AdmitAll()
    _check(all(a.try_admit(s) for s in range(5)), "AdmitAll admits everything")
    a.complete(0)  # no-op, must not raise

    b = BudgetAdmission(2)
    _check(b.try_admit(0) and b.try_admit(1), "Budget(2) admits the first two")
    _check(not b.try_admit(2), "Budget(2) blocks the third (in flight == budget)")
    _check(b.in_flight == 2, "in_flight reports 2")
    b.complete(0)
    _check(b.in_flight == 1, "complete frees one credit")
    _check(b.try_admit(2), "next frame admits after a completion")
    b.complete(0)
    b.complete(0)
    b.complete(0)  # extra completions
    _check(b.in_flight == 0, "in_flight never goes negative")


def _dummy_sync(seq: int, ts: int) -> CamSync:
    img = np.zeros((4, 4), np.uint8)
    return CamSync(seq=seq, ts_ns=ts, gray_left=img, gray_right=img)


def test_folding() -> None:
    """A skipped frame must not drain the buffer; its IMU folds forward."""
    print("imu folding across a skip")
    buf = TimedImuBuffer()
    ts = [10, 20, 30, 40, 50]
    for t in ts:
        buf.append(t, np.full(3, t, np.float64), np.full(3, -t, np.float64))

    adm = BudgetAdmission(1)
    admit = AdmitFrame(adm)
    pack = PackImuCam(buf, wait_timeout=0.0)

    # Frame 0 (ts=20): admitted -> drains (.., 20]
    m0 = admit.run(None, _dummy_sync(0, 20))
    _check(m0 is not None, "frame 0 admitted")
    p0 = pack.run(None, m0)
    _check(list(p0.imu_ts) == [10, 20], "packet 0 carries samples <= 20")

    # Frame 1 (ts=30): over budget -> skipped, buffer NOT drained
    m1 = admit.run(None, _dummy_sync(1, 30))
    _check(m1 is None, "frame 1 skipped (in flight == budget)")

    adm.complete(0)  # frame 0 finished -> credit freed

    # Frame 2 (ts=50): admitted -> drains (20, 50], INCLUDING frame 1's 30
    m2 = admit.run(None, _dummy_sync(2, 50))
    _check(m2 is not None, "frame 2 admitted after completion")
    p2 = pack.run(None, m2)
    _check(list(p2.imu_ts) == [30, 40, 50],
           "packet 2 folds the skipped interval's IMU (30) forward")

    delivered = list(p0.imu_ts) + list(p2.imu_ts)
    _check(delivered == ts, "no IMU sample lost or double-counted across the skip")


class _Collector(Flow):
    """Sink: record every ImuCamPacket, never complete (passive consumer)."""

    def __init__(self, bus: Bus) -> None:
        super().__init__("collector", bus)
        self.packets: list = []
        self.on(topics.IMUCAM_SAMPLE, [self._Grab(self.packets)])

    class _Grab:
        name = "grab"

        def __init__(self, out: list) -> None:
            self._out = out

        def run(self, ctx, msg):
            self._out.append(msg)
            return None


def test_cap(session: str, budget: int, n: int) -> None:
    """Real flows, budget N, no completions -> exactly N packets, first N seqs."""
    print(f"cap end-to-end (budget={budget})")
    reader = SessionReader(Path(session))
    bus = Bus()

    imu_flow = ImuReaderFlow(bus, ReplayImuSource(reader),
                             admission=BudgetAdmission(budget))
    cam_flow = CamReaderFlow(bus, ReplayCamSource(reader, max_frames=n), fps=20)
    sink = _Collector(bus)
    sink.expected_ends = 1

    sink.start()
    imu_flow.start()
    cam_flow.start()
    cam_flow.join()
    finished = sink.done.wait(timeout=60.0)
    for f in (imu_flow, sink):
        f.stop()
    if not finished:
        raise SystemExit("graph did not drain within timeout")

    _check(len(sink.packets) == budget,
           f"exactly {budget} packets admitted with no completions "
           f"({len(sink.packets)})")
    _check([p.seq for p in sink.packets] == list(range(budget)),
           f"the admitted packets are the first {budget} frames, in order")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_straight_20s")
    ap.add_argument("--budget", type=int, default=3)
    args = ap.parse_args()

    print("admission_selftest")
    test_strategies()
    test_folding()
    test_cap(args.session, args.budget, n=20)

    print("\nALL ADMISSION SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
