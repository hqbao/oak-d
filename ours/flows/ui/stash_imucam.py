"""``stash_imucam`` task: buffer each frame's calibrated IMU rows by seq.

Half of the image|depth|IMU triplet join (see :mod:`ours.flows.ui.triplet`). The
``imu_cam`` flow publishes ``imucam.sample`` (the calibrated IMU for the frame
interval) just BEFORE ``frame.depth`` (the rectified-left image + depth) for the
same ``seq``. This task stashes the IMU rows keyed by ``seq`` so the matching
:class:`~ours.flows.ui.render_triplet.RenderTriplet` can pair them when the depth
frame arrives. Stops the chain (returns ``None``) -- it only records.
"""
from __future__ import annotations

from ...lib.flow.messages import ImuCamPacket
from ...lib.flow.task import Task


class StashImuCam(Task):
    name = "stash_imucam"

    def run(self, ctx, msg: ImuCamPacket):
        buf = ctx.state["imu_rows"]
        buf[msg.seq] = (msg.gyro, msg.accel)
        # Safety cap: each seq is popped by its depth frame, so this stays ~1
        # entry; bound it anyway so a dropped pair can't leak over a long session.
        if len(buf) > 256:
            for seq in sorted(buf)[:-256]:
                buf.pop(seq, None)
        return None
