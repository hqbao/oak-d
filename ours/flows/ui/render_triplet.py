"""``render_triplet`` task: pair the depth frame with its IMU rows, emit.

Other half of the image|depth|IMU triplet join (see
:mod:`ours.flows.ui.triplet`). When ``frame.depth`` arrives it pops the matching
``seq``'s calibrated IMU rows that
:class:`~ours.flows.ui.stash_imucam.StashImuCam` buffered (always present, since
``imucam.sample`` for a seq is published just before its ``frame.depth``) and
hands the complete (image, depth, gyro, accel) unit to the ``on_triplet``
callback that drives the Qt window.
"""
from __future__ import annotations

import numpy as np

from ...lib.flow.messages import DepthFrame
from ...lib.flow.task import Task


class RenderTriplet(Task):
    name = "render_triplet"

    def run(self, ctx, msg: DepthFrame):
        gyro, accel = ctx.state["imu_rows"].pop(msg.seq, (None, None))
        if gyro is None:
            gyro = np.empty((0, 3), dtype=np.float64)
            accel = np.empty((0, 3), dtype=np.float64)
        ctx.state["on_triplet"](msg.seq, msg.ts_ns, msg.gray_left, msg.depth_m,
                                gyro, accel)
        return None
