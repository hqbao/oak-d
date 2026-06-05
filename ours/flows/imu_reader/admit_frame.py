"""``admit_frame`` task: the realtime backpressure gate, first in the chain.

Runs ahead of :class:`~ours.flows.imu_reader.pack_imucam.PackImuCam` so the
admission decision happens BEFORE the IMU buffer is drained. On admit it passes
the :class:`~ours.lib.flow.messages.CamSync` through (reserving an in-flight
credit); over budget it returns ``None``, which stops the chain -- the heavy
packet is never built and, crucially, the buffer is NOT drained, so the skipped
frame's inertial samples remain queued and fold into the next admitted frame's
interval (gyro preintegration stays continuous).

The :class:`~ours.flows.imu_reader.admission.AdmitAll` strategy makes this a
pass-through (replay / offline determinism).
"""
from __future__ import annotations

from ...lib.flow.messages import CamSync
from ...lib.flow.task import Task
from .admission import Admission


class AdmitFrame(Task):
    name = "admit_frame"

    def __init__(self, admission: Admission) -> None:
        self._adm = admission

    def run(self, ctx, msg: CamSync):
        return msg if self._adm.try_admit(msg.seq) else None
