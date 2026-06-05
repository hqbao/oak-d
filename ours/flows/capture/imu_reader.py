"""IMU-reader flow: buffer raw IMU, pack it against each camera trigger.

This is one half of the split-acquisition front-end (the camera-reader flow is
the other). It runs on its own thread and does two decoupled things:

1. **Read the IMU continuously** into a timestamped buffer. A background
   ``ImuSource`` pushes every ``(t_ns, gyro, accel)`` sample into a
   :class:`~ours.lib.imu.timed_buffer.TimedImuBuffer` as it arrives -- never
   blocked by the cameras or the consumer.

2. **Answer each camera trigger.** The camera-reader flow publishes a
   :class:`~ours.lib.flow.messages.CamSync` (a stereo pair + its device
   timestamp) on ``topics.CAM_SYNC``. For each one this flow drains the buffer up
   to that timestamp and publishes an
   :class:`~ours.lib.flow.messages.ImuCamPacket` (the frames bundled with exactly
   the inertial samples in that frame's interval) on ``topics.IMUCAM_SAMPLE``.

The two flows therefore share NO state -- only the two bus messages -- honouring
the architecture's hard rule. Selecting IMU by device timestamp is the honest
binding: the IMU carries no frame serial, only a clock shared with the camera.

The flow is reactive (it waits on ``CAM_SYNC``); the continuous IMU read is the
injected source's own thread, started in :meth:`run` and stopped on exit.
"""
from __future__ import annotations

from ...lib.flow import Bus, Flow, topics
from ...lib.flow.messages import CamSync, ImuCamPacket
from ...lib.imu.timed_buffer import TimedImuBuffer
from .imu_sources import ImuSource


class PackImuCam:
    """Task: drain the buffer up to the frame timestamp and build the packet."""

    name = "pack_imucam"

    def __init__(self, buffer: TimedImuBuffer, wait_timeout: float) -> None:
        self._buf = buffer
        self._wait = float(wait_timeout)

    def run(self, ctx, msg: CamSync):
        # Block (bounded) until the IMU stream has covered this frame's time, so
        # the interval is never short-changed by thread scheduling; the last
        # frame (ts past the final IMU sample) just drains what is present.
        self._buf.wait_until(msg.ts_ns, timeout=self._wait)
        imu_ts, gyro, accel = self._buf.drain_until(msg.ts_ns)
        return ImuCamPacket(
            seq=msg.seq, ts_ns=msg.ts_ns,
            gray_left=msg.gray_left, gray_right=msg.gray_right,
            imu_ts=imu_ts, gyro=gyro, accel=accel,
        )


class PublishImuCam:
    """Task: route the packed bundle onto the bus."""

    name = "publish_imucam"

    def run(self, ctx, msg: ImuCamPacket):
        ctx.bus.publish(topics.IMUCAM_SAMPLE, msg)
        return None


class ImuReaderFlow(Flow):
    """Reactive flow: buffers IMU on a side thread, packs it per camera trigger.

    ``source`` supplies the raw IMU (``ReplayImuSource`` offline,
    ``LiveImuSource`` on the bench). ``wait_timeout`` bounds how long packing a
    frame waits for the IMU stream to cover its timestamp before draining what is
    available (so the run never hangs on the final frame).
    """

    def __init__(self, bus: Bus, source: ImuSource, *,
                 buffer_capacity: int = 8192, wait_timeout: float = 0.5) -> None:
        super().__init__("imu-reader", bus)
        self.source = source
        self.buffer = TimedImuBuffer(capacity=buffer_capacity)
        self.forwards_to(topics.IMUCAM_SAMPLE)
        self.on(topics.CAM_SYNC,
                [PackImuCam(self.buffer, wait_timeout), PublishImuCam()])

    def run(self) -> None:
        # Continuous IMU read on the source's own thread; close the buffer when a
        # replay source exhausts so any pending wait_until returns at once.
        self.source.start(self.buffer.append, on_exhausted=self.buffer.close)
        try:
            super().run()
        finally:
            self.source.stop()
            self.buffer.close()
