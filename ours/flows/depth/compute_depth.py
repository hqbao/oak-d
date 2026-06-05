"""``compute_depth`` task: run the SGM matcher on a raw stereo pair."""
from __future__ import annotations

from ...lib.flow.messages import DepthFrame, RawFrame
from ...lib.flow.runtime import NUMBA_PARALLEL_LOCK
from ...lib.flow.task import Task
from ...lib.stereo.stereo import SGMStereoMatcher


class ComputeDepth(Task):
    name = "compute_depth"

    def run(self, ctx, msg: RawFrame):
        matcher: SGMStereoMatcher = ctx.state["matcher"]
        with NUMBA_PARALLEL_LOCK:        # SGM uses numba parallel=True
            # Returns the tracking-grid left + metric depth on the SAME grid.
            # For a replay matcher (rectify_left off) the left passes through
            # unchanged; for the live matcher (rectify_left on, raw cameras) the
            # left is rectified here so depth + tracking share one grid.
            gray_track, depth = matcher.dense_depth_rectified_left(
                msg.gray_left, msg.gray_right)
        return DepthFrame(msg.seq, msg.ts_ns, gray_track, depth)
