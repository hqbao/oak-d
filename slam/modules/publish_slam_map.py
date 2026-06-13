"""``publish_slam_map`` -- emit the live SLAM keyframe-map on ``slam.map``.

A plain function (not a ``Step`` subclass): the engine + bus are passed in
explicitly. LIVE-ONLY -- the procedural pipeline calls this only when
``publish_map`` is on, so the deterministic ``loop.correction`` scoring path
stays byte-identical. The poll is independent of
:func:`slam.modules.slam_step.slam_submit` (which only emits a
``loop.correction`` ON a confirmed loop): this polls the engine's cheap map
overlay EVERY keyframe. That decoupling is the bug fix: the UI gets continuous
keyframe dots instead of dots only after a loop closes.

It must run AFTER ``slam_submit`` has called ``engine.submit`` for this keyframe
so the polled overlay reflects the keyframe just added (see
:mod:`slam.modules.pipeline` for the call ordering that guarantees this).
"""
from __future__ import annotations

import numpy as np

from slam.comms import LocalPubSub, topics
from slam.comms.messages import SlamOverlay
from slam.engine import Engine


def publish_slam_map(engine: Engine, bus: LocalPubSub) -> None:
    """Poll the engine's map overlay and publish it on ``slam.map`` (if any)."""
    # (kf_seq (N,), kf_pos (N,3) optical, n_loops, match_pos (M,3)) or None
    # when the engine has no overlay yet (subprocess worker not spawned / no
    # keyframe served).
    ov = engine.poll_overlay()
    if ov is None:
        return
    kf_seq, kf_pos, n_loops, match_pos = ov
    bus.publish(topics.SLAM_MAP, SlamOverlay(
        kf_positions=np.asarray(kf_pos, dtype=np.float64),
        n_loops=int(n_loops),
        last_match=(np.asarray(match_pos, dtype=np.float64)
                    if match_pos is not None and len(match_pos) else None),
        kf_seqs=np.asarray(kf_seq, dtype=np.int64)))
