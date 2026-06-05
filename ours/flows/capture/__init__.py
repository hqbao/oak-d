"""capture flow: ingest sensor data and publish it onto the bus.

Two interchangeable sources publish the *same* topics so every downstream flow
is identical live or offline:

* :class:`ReplayCaptureFlow` -- replays a recorded session (offline validation).
* :class:`LiveCaptureFlow`   -- the OAK-D device (live; validated on hardware).

The device tier also hosts one non-flow helper: :mod:`.imu_stream` exposes
:class:`~ours.flows.capture.imu_stream.ImuStream`, a lightweight IMU-only device
reader (no bus, callback per sample) used by the calibration wizards. It lives
here because it is raw depthai device access, the same concern as live capture.
"""
from .replay import ReplayCaptureFlow

__all__ = ["ReplayCaptureFlow", "LiveCaptureFlow", "LiveCalib"]


def __getattr__(name):
    # Lazy: importing depthai (a heavy optional dep) only when the live capture
    # flow is actually requested, so replay/offline never pays for it.
    if name in ("LiveCaptureFlow", "LiveCalib"):
        from .live import LiveCalib, LiveCaptureFlow
        return {"LiveCaptureFlow": LiveCaptureFlow, "LiveCalib": LiveCalib}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
