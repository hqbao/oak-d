"""``publish_tracks`` step: emit the frame's KLT tracks on ``frame.tracks``.

Sits between :func:`~vio.modules.track_features.track_features` and
:func:`~vio.modules.estimate_motion.estimate_motion` in the frame-chain.
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

from vio.comms import LocalPubSub, topics
from vio.comms.messages import FrameTracks
from .tracked import Tracked


def publish_tracks(bus: LocalPubSub, tracked: Tracked) -> Tracked:
    """Publish the frame's KLT tracks on ``frame.tracks``; pass the carrier on.

    Was ``PublishTracks(Step)``; identical publish, the bus passed explicitly.
    """
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
    bus.publish(topics.FRAME_TRACKS,
                FrameTracks(frame.seq, frame.ts_ns, ids, points))
    return tracked
