"""``complete_admission`` task: free an in-flight credit on ``frame.done``.

Subscribed to the backpressure control topic ``topics.FRAME_DONE``; for each
:class:`~ours.lib.flow.messages.FrameDone` the odometry tail publishes (one per
processed frame), it tells the shared
:class:`~ours.flows.imu_reader.admission.Admission` that a frame finished, so the
:class:`~ours.flows.imu_reader.admit_frame.AdmitFrame` gate can admit the next.

It runs on the imu-reader's own thread (same inbox as the camera triggers), so
admit/complete are naturally serialised. With the
:class:`~ours.flows.imu_reader.admission.AdmitAll` strategy this is a no-op.
"""
from __future__ import annotations

from ...lib.flow.messages import FrameDone
from ...lib.flow.task import Task
from .admission import Admission


class CompleteAdmission(Task):
    name = "complete_admission"

    def __init__(self, admission: Admission) -> None:
        self._adm = admission

    def run(self, ctx, msg: FrameDone):
        self._adm.complete(msg.seq)
        return None
