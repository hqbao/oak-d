"""The :class:`ImuReaderFlow` -- buffer IMU on a side thread, pack per trigger."""
from __future__ import annotations

from collections.abc import Callable

from ...lib.flow import Bus, Flow, topics
from ...lib.imu.imu_calib import ImuCalibration
from ...lib.imu.timed_buffer import TimedImuBuffer
from .admission import Admission, AdmitAll
from .admit_frame import AdmitFrame
from .apply_calibration import ApplyCalibration
from .complete_admission import CompleteAdmission
from .pack_imucam import PackImuCam
from .publish_imu_raw import PublishImuRaw
from .publish_imucam import PublishImuCam
from .sources import ImuSource


class ImuReaderFlow(Flow):
    """Reactive flow: buffers IMU on a side thread, packs it per camera trigger.

    ``source`` supplies the raw IMU (``ReplayImuSource`` offline,
    ``LiveImuSource`` on the bench). ``wait_timeout`` bounds how long packing a
    frame waits for the IMU stream to cover its timestamp before draining what is
    available (so the run never hangs on the final frame).

    For every camera trigger the flow publishes TWO messages from the same
    drained interval: the uncalibrated samples on ``topics.IMU_RAW`` (honest:
    exactly what the sensor reported) and, on ``topics.IMUCAM_SAMPLE``, the
    frames bundled with the CALIBRATED IMU. ``calibration`` (or the lazy
    ``calibration_provider``, used on the live path where the device id is known
    only after the device opens) supplies the per-device correction; with none,
    the calibrated packet equals the raw one.

    ``admission`` is the realtime backpressure gate (see
    :mod:`~ours.flows.imu_reader.admission`). The default
    :class:`~ours.flows.imu_reader.admission.AdmitAll` admits every frame (replay
    determinism); the live path injects a
    :class:`~ours.flows.imu_reader.admission.BudgetAdmission` so at most ``N``
    frames are in flight. The gate is the FIRST task in the camera chain (it runs
    before the IMU is drained, so a skip folds that interval into the next frame),
    and ``topics.FRAME_DONE`` frees a credit when the odometry tail reports a
    frame finished.

    Note on threads: the *flow* owns one thread (it drains the inbox and runs the
    pack/publish chain). The injected ``source`` runs the continuous high-rate
    IMU read on its OWN I/O thread -- a hardware producer, not a flow, the same
    pattern the calibration ``ImuStream`` uses. No flow logic runs on that
    thread; it only fills the thread-safe buffer.
    """

    def __init__(self, bus: Bus, source: ImuSource, *,
                 buffer_capacity: int = 8192, wait_timeout: float = 0.5,
                 calibration: ImuCalibration | None = None,
                 calibration_provider:
                     Callable[[], ImuCalibration | None] | None = None,
                 admission: Admission | None = None) -> None:
        super().__init__("imu-reader", bus)
        self.source = source
        self.buffer = TimedImuBuffer(capacity=buffer_capacity)
        self.admission = admission or AdmitAll()
        self.forwards_to(topics.IMU_RAW, topics.IMUCAM_SAMPLE)
        self.on(topics.CAM_SYNC, [
            AdmitFrame(self.admission),
            PackImuCam(self.buffer, wait_timeout),
            PublishImuRaw(),
            ApplyCalibration(calibration, provider=calibration_provider),
            PublishImuCam(),
        ])
        # Backpressure control: free a credit per finished frame. Not END-bearing
        # (odometry never forwards END here), so it does not affect drain.
        self.on(topics.FRAME_DONE, [CompleteAdmission(self.admission)])

    def run(self) -> None:
        # Continuous IMU read on the source's own I/O thread; close the buffer
        # when a replay source exhausts so any pending wait_until returns at once.
        self.source.start(self.buffer.append, on_exhausted=self.buffer.close)
        try:
            super().run()
        finally:
            self.source.stop()
            self.buffer.close()
