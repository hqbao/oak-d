"""``publish_tracks`` task: emit the frame's KLT tracks on ``frame.tracks``.

Sits between :class:`~ours.flows.odometry.track_features.TrackFeatures` and
:class:`~ours.flows.odometry.estimate_motion.EstimateMotion` in the frame-chain.
It has the freshly tracked ``{id: pixel}`` (from the :class:`Tracked` carrier),
so it publishes the REAL frontend tracks for the keypoint-depth visualiser to
subscribe -- no parallel detector. The matching frame image + depth are NOT
sent on this topic: the UI sink joins ``frame.tracks`` with ``frame.depth`` by
``seq`` (capture is the single writer of the gray/depth shared-memory rings).
``tracked`` passes through unchanged so the motion estimate still runs on the
same carrier.
"""
from __future__ import annotations

import numpy as np

from ...lib.flow import topics
from ...lib.flow.messages import FrameTracks
from ...lib.flow.task import Task
from .tracked import Tracked


class PublishTracks(Task):
    name = "publish_tracks"

    def run(self, ctx, tracked: Tracked):
        obs = tracked.obs
        if obs:
            ids_list = list(obs.keys())
            ids = np.array(ids_list, dtype=np.int64)
            points = np.array([obs[k] for k in ids_list],
                              dtype=np.float32).reshape(-1, 2)
        else:
            ids = np.empty((0,), dtype=np.int64)
            points = np.empty((0, 2), dtype=np.float32)
        frame = tracked.frame
        ctx.bus.publish(topics.FRAME_TRACKS,
                        FrameTracks(frame.seq, frame.ts_ns, ids, points))
        return tracked
