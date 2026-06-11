"""``pack_synced`` step: drain the buffer to the frame time, build the packet."""
from __future__ import annotations

from imu_camera.comms import Step
from imu_camera.comms.messages import CamSync, ImuCamPacket
from sky.imu.timed_buffer import TimedImuBuffer


class PackSyncedStep(Step):
    """Drain the IMU buffer up to the frame timestamp and bundle the packet."""

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
