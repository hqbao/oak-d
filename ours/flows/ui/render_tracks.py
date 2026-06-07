"""Tasks for the keypoint-depth sink: stash depth frames + join with tracks.

Two tasks, one shared by-seq buffer (``ctx.state["frame_buf"]``):

* :class:`StashFrameDepth` buffers each ``frame.depth`` (gray + depth) by ``seq``
  and stops the chain. It runs on the ``frame.depth`` subscription.
* :class:`RenderTracks` runs on the ``frame.tracks`` subscription: it pops the
  matching ``seq``'s gray + depth from the stash and hands a
  :class:`~ours.flows.ui.tracks.TracksWithFrame` bundle to the ``on_tracks``
  callback. If the matching depth is missing (a latest-only sink coalesced it
  away, or the depth has not yet arrived in 4-proc out-of-order delivery) the
  tracks for that seq are dropped -- the keypoints view is a realtime sink, not
  a logger.
"""
from __future__ import annotations

from ...lib.flow.messages import DepthFrame, FrameTracks
from ...lib.flow.task import Task


# Cap the by-seq depth buffer so a sustained mismatch (latest-only coalescing,
# tracks topic stalled) cannot grow it without bound over a long live session.
_FRAME_BUF_CAP = 256


class StashFrameDepth(Task):
    """Buffer each frame.depth (gray + depth) by seq for the tracks join."""

    name = "stash_frame_depth"

    def run(self, ctx, msg: DepthFrame):
        buf = ctx.state["frame_buf"]
        buf[int(msg.seq)] = (msg.gray_left, msg.depth_m)
        # Safety cap: the matching tracks pop each seq, so this stays ~1 in the
        # nominal full-fidelity path; bound it for the realtime / lossy case.
        if len(buf) > _FRAME_BUF_CAP:
            for seq in sorted(buf)[:-_FRAME_BUF_CAP]:
                buf.pop(seq, None)
        return None


class RenderTracks(Task):
    """Join frame.tracks with its stashed frame.depth, fire on_tracks."""

    name = "render_tracks"

    def run(self, ctx, msg: FrameTracks):
        # Local import to keep the task module free of the sink-flow's dataclass
        # dependency at import-time (avoid an import cycle: tracks.py imports
        # this module).
        from .tracks import TracksWithFrame
        pair = ctx.state["frame_buf"].pop(int(msg.seq), None)
        if pair is None:
            # No matching depth yet (or coalesced away). Drop the tracks for
            # this seq -- a realtime sink may legitimately lose frames. The
            # offline replay path always has the depth ready (the odometry
            # flow consumes frame.depth before publishing frame.tracks for the
            # same seq, so on a single-proc FIFO bus the stash is always set).
            return None
        gray_left, depth_m = pair
        bundle = TracksWithFrame(
            seq=int(msg.seq), ts_ns=int(msg.ts_ns),
            ids=msg.ids, points=msg.points,
            gray_left=gray_left, depth_m=depth_m)
        ctx.state["on_tracks"](bundle)
        return None
