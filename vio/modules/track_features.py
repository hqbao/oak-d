"""``track_features`` step: run the KLT frontend on the frame's left image.

First step of the odometry frame-chain. It is the ONLY ``numba parallel=True``
section on the odometry thread (the KLT tracker), so it is the only place that
takes :data:`~vio.comms.runtime.NUMBA_PARALLEL_LOCK` -- serialising just
against the depth matcher (SGM) which runs the same numba layer on the imu_cam
thread. The downstream :func:`~vio.modules.estimate_motion.estimate_motion`
(PnP + fusion) is pure NumPy and runs lock-free, so it overlaps the next frame's
SGM instead of blocking it.
"""
from __future__ import annotations

from vio.comms.messages import DepthFrame
from vio.comms.runtime import NUMBA_PARALLEL_LOCK
from sky.front.odometry import RGBDVisualOdometry
from .tracked import Tracked


def track_features(vo: RGBDVisualOdometry, frame: DepthFrame) -> Tracked:
    """KLT-track the frame's left image; return the :class:`Tracked` carrier.

    Was ``TrackFeatures(Step)``; the odometry instance is passed explicitly
    instead of read off ``ctx.state["vo"]``.
    """
    with NUMBA_PARALLEL_LOCK:            # KLT tracker uses numba parallel=True
        obs = vo.track(frame.gray_left)
    return Tracked(frame, obs)
