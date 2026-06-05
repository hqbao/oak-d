#!/usr/bin/env python3
"""Headless self-test for the flow framework's latest-only (coalescing) inbox.

The realtime visualiser fix: a reactive :class:`~ours.lib.flow.flow.Flow` built
with ``latest_only=True`` keeps only the newest unprocessed message per topic, so
a consumer slower than the producer always works on the freshest frame and the
backlog is dropped (bounded latency) instead of growing without bound on a FIFO
inbox. This proves, fully offline:

1. **FIFO (default)** processes EVERY message in order (the VIO + replay contract).
2. **latest-only** under a fast producer + slow consumer processes far fewer
   messages, ALWAYS lands the last one, and never falls behind (bounded backlog).
3. ``END`` is never coalesced away -- ``done`` is set even when a data frame was
   pending on the same topic.

Run::

    python -m ours.tools.flow_latest_selftest
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.flow.flow import Flow                                # noqa: E402
from ours.lib.flow.messages import END                            # noqa: E402
from ours.lib.flow.pubsub import Bus                              # noqa: E402
from ours.lib.flow.task import Task                               # noqa: E402

_TOPIC = "t.frame"


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


class _Record(Task):
    """Record each message; optionally sleep to simulate a slow consumer."""

    name = "record"

    def __init__(self, sink: list, delay: float = 0.0) -> None:
        self._sink = sink
        self._delay = delay

    def run(self, ctx, msg):
        if self._delay:
            time.sleep(self._delay)
        self._sink.append(msg)
        return None


def _publish_burst(bus: Bus, n: int, gap: float = 0.0) -> None:
    for i in range(n):
        bus.publish(_TOPIC, i)
        if gap:
            time.sleep(gap)


def test_fifo_processes_every_message() -> None:
    print(" FIFO inbox (default) processes every message in order")
    bus = Bus()
    got: list = []
    flow = Flow("fifo", bus)                       # latest_only defaults False
    flow.on(_TOPIC, [_Record(got)])
    flow.start()
    _publish_burst(bus, 200)
    # Drain: publish END and wait for done.
    bus.publish(_TOPIC, END)
    _check(flow.done.wait(timeout=5.0), "flow drained (done set)")
    flow.stop()
    flow.join(timeout=2.0)
    _check(got == list(range(200)),
           f"all 200 messages processed in order (got {len(got)})")


def test_latest_only_drops_backlog_keeps_freshest() -> None:
    print(" latest-only inbox: slow consumer lands the freshest, drops backlog")
    bus = Bus()
    got: list = []
    # Realtime-like: producer fires 200 frames at ~1 ms spacing (~0.2 s), the
    # consumer is 5x slower at 5 ms/frame. A FIFO inbox would back up to ~160
    # un-processed frames; latest-only must instead stay near the head.
    flow = Flow("latest", bus, latest_only=True)
    flow.on(_TOPIC, [_Record(got, delay=0.005)])
    flow.start()
    time.sleep(0.02)                               # let the consumer thread arm
    _publish_burst(bus, 200, gap=0.001)
    # Let the consumer drain the (coalesced) tail to the freshest value, THEN end.
    time.sleep(0.1)
    bus.publish(_TOPIC, END)
    _check(flow.done.wait(timeout=5.0), "flow drained (done set)")
    flow.stop()
    flow.join(timeout=2.0)
    _check(len(got) > 0, "consumer processed at least one frame")
    _check(len(got) < 200,
           f"backlog was dropped, not processed in full (got {len(got)}/200)")
    _check(got == sorted(got), "processed messages stay monotonic (no reorder)")
    _check(got[-1] == 199,
           f"the freshest message (199) was the last processed (got {got[-1]})")
    # A 5 ms consumer over ~0.3 s can physically process ~60 frames; coalescing
    # must keep it well under the 200 produced (proving the backlog was shed).
    _check(len(got) <= 80,
           f"coalescing kept the consumer near-realtime (got {len(got)})")


def test_latest_only_end_not_dropped() -> None:
    print(" latest-only inbox: END is delivered even over a pending frame")
    bus = Bus()
    got: list = []
    flow = Flow("latest-end", bus, latest_only=True)
    flow.on(_TOPIC, [_Record(got, delay=0.05)])    # slow: first msg holds thread
    flow.start()
    time.sleep(0.02)
    # Fire a data frame then immediately END while the consumer is busy on the
    # first frame: END must overwrite the pending data token but still arrive.
    bus.publish(_TOPIC, 0)
    bus.publish(_TOPIC, 1)
    bus.publish(_TOPIC, END)
    _check(flow.done.wait(timeout=5.0), "END reached the flow (done set)")
    flow.stop()
    flow.join(timeout=2.0)
    _check(END not in got, "END was handled as control, not a data message")


def main() -> int:
    test_fifo_processes_every_message()
    test_latest_only_drops_backlog_keeps_freshest()
    test_latest_only_end_not_dropped()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    print("flow_latest_selftest")
    raise SystemExit(main())
