"""``publish_loops`` -- emit per-candidate loop-match funnels on ``slam.loop``.

A plain function (not a ``Step`` subclass): the engine + bus are passed in
explicitly. LIVE-ONLY -- the procedural pipeline calls this only when
``publish_map`` is on, so the deterministic ``loop.correction`` scoring path
stays byte-identical and the offline engine never even captures the funnel. The
poll is independent of :func:`slam.modules.slam_step.slam_submit` (which only
emits a ``loop.correction`` ON a confirmed loop): this polls the engine's
loop-match capture channel EVERY keyframe. For EACH verified candidate (CONFIRMED
or REJECTED) it publishes one :class:`~slam.comms.messages.LoopMatch`, so the
UI's loop-closure window can show WHY a loop fired or got rejected (the matched
ORB pixel pairs + per-match verification stage + funnel counts + rotation-gate
verdict).

It must run AFTER ``slam_submit`` has called ``engine.submit`` for this keyframe
so the polled captures reflect the candidate just verified (see
:mod:`slam.modules.pipeline` for the call ordering that guarantees this).

There are NO keyframe images on the wire (SLAM does not retain the gray); the UI
joins each LoopMatch to the GRAY images it buffers by seq off the ``keyframe``
topic.
"""
from __future__ import annotations

import numpy as np

from slam.comms import LocalPubSub, topics
from slam.comms.messages import LoopMatch
from slam.engine import Engine


def publish_loops(engine: Engine, bus: LocalPubSub) -> None:
    """Drain the engine's loop-match captures and publish one ``slam.loop`` each."""
    # [(cur_seq, old_seq, LoopMatchCapture), ...] for every candidate verified
    # since the last poll (empty unless the engine captures -- live only).
    for cur_seq, old_seq, cap in engine.poll_loops():
        bus.publish(topics.SLAM_LOOP, LoopMatch(
            cur_seq=int(cur_seq), old_seq=int(old_seq),
            cur_px=np.asarray(cap.cur_px, dtype=np.float32).reshape(-1, 2),
            old_px=np.asarray(cap.old_px, dtype=np.float32).reshape(-1, 2),
            stage=np.asarray(cap.stage, dtype=np.uint8).reshape(-1),
            n_appearance=int(cap.n_appearance), n_fmat=int(cap.n_fmat_inliers),
            n_pnp=int(cap.n_pnp_inliers), rot_deg=float(cap.rot_deg),
            rot_gate_deg=float(cap.rot_gate_deg), accepted=bool(cap.accepted)))
