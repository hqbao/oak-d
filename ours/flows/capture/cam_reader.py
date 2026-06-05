"""Camera-reader flow: pull stereo on a schedule, trigger the IMU pack.

The other half of the split-acquisition front-end. It owns the *schedule*: one
stereo pair per scheduler tick (``fps`` Hz). For each pair it publishes a single
:class:`~ours.lib.flow.messages.CamSync` (the frames + their device timestamp) on
``topics.CAM_SYNC``. That message is the trigger the
:class:`~ours.flows.capture.imu_reader.ImuReaderFlow` reacts to -- it carries the
frames so the IMU flow can both select the inertial interval up to ``ts_ns`` and
bundle the very same frames, with no shared state between the two flows.

``source`` supplies the frames: ``ReplayCamSource`` offline (deterministic, the
recorded device timestamps drive the IMU drain), ``LiveCamSource`` on the bench.
Offline the scheduler runs free (no sleep) so a replay completes immediately;
``realtime=True`` paces ticks to ``fps`` for a live-like visualisation.

It is a :class:`~ours.lib.flow.SourceFlow`: :meth:`produce` yields the frames and
a single publish task emits the trigger; END is forwarded on ``CAM_SYNC`` when
the source is exhausted, draining the graph.
"""
from __future__ import annotations

import time

from ...lib.flow import Bus, SourceFlow, topics
from ...lib.flow.messages import CamSync
from .cam_sources import CamSource


class PublishCamSync:
    """Task: publish one stereo pair as the IMU sync trigger."""

    name = "publish_cam_sync"

    def run(self, ctx, msg: CamSync):
        ctx.bus.publish(topics.CAM_SYNC, msg)
        return None


class CamReaderFlow(SourceFlow):
    """Source flow: emit one :class:`CamSync` per scheduled stereo pair.

    ``fps`` sets the schedule; ``realtime`` paces ticks to it (live-like) versus
    running free (deterministic offline replay).
    """

    def __init__(self, bus: Bus, source: CamSource, *, fps: int = 20,
                 realtime: bool = False) -> None:
        super().__init__("cam-reader", bus, [PublishCamSync()])
        self.source = source
        self.fps = max(1, int(fps))
        self.realtime = bool(realtime)
        self.forwards_to(topics.CAM_SYNC)

    def produce(self):
        self.source.open()
        period = 1.0 / self.fps
        try:
            next_tick = time.monotonic()
            while not self._stop.is_set():
                if self.realtime:
                    now = time.monotonic()
                    if now < next_tick:
                        time.sleep(next_tick - now)
                    next_tick += period
                item = self.source.read()
                if item is None:
                    break
                seq, ts_ns, gray_left, gray_right = item
                yield CamSync(seq=seq, ts_ns=ts_ns,
                              gray_left=gray_left, gray_right=gray_right)
        finally:
            self.source.close()
