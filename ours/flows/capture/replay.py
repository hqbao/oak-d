"""Replay capture: stream a recorded session onto the bus.

This is the offline drop-in for the live OAK-D camera. It reads a session with
:class:`~ours.lib.io.reader.SessionReader` and publishes exactly what the live
capture flow publishes, so every downstream flow is identical live or offline:

* one :class:`~ours.lib.flow.messages.ImuInit` up front (the startup gravity-align
  accelerometer), then
* per frame, one :class:`~ours.lib.flow.messages.ImuPrior` (the gyro rotation prior
  for ``[prev_ts, ts]``) followed by one :class:`~ours.lib.flow.messages.RawFrame`.

The IMU->prior fusion lives HERE (not in the odometry flow) so live and replay
share one odometry flow. The prior is computed with the same
:class:`~ours.lib.imu.imu.GyroPreintegrator` the offline ``vio_run`` driver uses,
so the replayed trajectory stays bit-for-bit identical to that oracle.

It is a :class:`~ours.lib.flow.SourceFlow`: ``produce`` yields those messages and
the single publish task routes each to its topic. When the session is exhausted
the base class emits ``END`` on ``frame.raw`` so the graph drains.
"""
from __future__ import annotations

from ...lib.flow import SourceFlow, Bus, topics
from ...lib.imu.imu import GyroPreintegrator
from ...lib.io.reader import SessionReader
from ...lib.flow.messages import ImuInit, ImuPrior, RawFrame
from .publish_capture import PublishCapture


class ReplayCaptureFlow(SourceFlow):
    def __init__(self, bus: Bus, reader: SessionReader,
                 load_right: bool = True, max_frames: int = 0,
                 use_gyro: bool = True) -> None:
        super().__init__("capture", bus, [PublishCapture()])
        self.reader = reader
        self.load_right = load_right
        self.max_frames = max_frames
        self.use_gyro = use_gyro
        # END travels only down the primary frame path (imu.sample is auxiliary).
        self.forwards_to(topics.FRAME_RAW)

    def produce(self):
        r = self.reader
        # IMU->prior fusion source: the same gyro preintegrator the offline
        # driver uses, plus the startup gravity-align accel. Sessions without
        # IMU extrinsics fall back to pure vision (prior None, no align).
        pre = None
        accel_align = None
        if self.use_gyro and r.calib.has_imu_extrinsics:
            imu = r.load_imu()
            if imu["ts_ns"].size > 1:
                pre = GyroPreintegrator(imu["ts_ns"], imu["gyro"],
                                        r.calib.T_imu_left)
                R_imu_cam = r.calib.T_imu_left[:3, :3]
                t0 = int(imu["ts_ns"][0])
                win = imu["ts_ns"] <= t0 + int(0.3 * 1e9)   # first ~0.3 s
                if win.any():
                    accel_align = R_imu_cam @ imu["accel"][win].mean(axis=0)
        yield ImuInit(accel_align)

        n = len(r) if self.max_frames <= 0 else min(self.max_frames, len(r))
        prev_ts = None
        for i in range(n):
            f = r.load_frame(i, load_right=self.load_right)
            R_prior = (pre.delta_rotation(prev_ts, f.ts_ns)
                       if (pre is not None and prev_ts is not None) else None)
            prev_ts = f.ts_ns
            yield ImuPrior(f.seq, R_prior)
            yield RawFrame(f.seq, f.ts_ns, f.gray_left,
                           f.gray_right if self.load_right else None)
