"""``run_ba`` step: submit a keyframe to the BA engine, forward any refined pose.

Offline (in-process engine) ``submit`` runs the solve synchronously and ``poll``
returns this keyframe's refined ``T_cw`` -- identical to the old in-thread path.
Live (subprocess engine) ``submit`` is async and ``poll`` returns the freshest
refined pose the worker has produced (or ``None``); the responsive marker rides
``pose.odom`` and never waits on this.
"""
from __future__ import annotations

import numpy as np

from vio.comms.messages import Keyframe, PoseMsg
from vio.engine import Engine


def run_ba(engine: Engine, tight: bool, kf: Keyframe):
    """Submit the keyframe's snapshot to the BA engine; return the refined pose.

    Was ``RunBA(Step)``; the engine + the ``tight`` snapshot selector (was
    ``ctx.state["engine"]`` / ``ctx.state["tight"]``) are passed explicitly.
    Returns ``None`` (chain short-circuit) when the keyframe has no tracks or the
    engine has no refined pose yet.
    """
    if kf.track_ids is None or kf.track_px is None:
        return None
    T_cw = np.linalg.inv(kf.T_world_cam)
    # Submit the snapshot shaped for whichever backend the worker built.
    # LOOSE (default): the historical 5-tuple ``ba_step`` consumes -- the
    # keyframe's at-rest gravity accel. TIGHT (``--tight``): the SUPERSET
    # 6-tuple ``vio_step`` consumes -- the keyframe timestamp + the raw
    # inter-keyframe IMU block (camera optical frame) for preintegration.
    if tight:
        engine.submit((T_cw, kf.track_ids, kf.track_px, kf.depth_m,
                       kf.ts_ns, kf.imu_seg))
    else:
        engine.submit((T_cw, kf.track_ids, kf.track_px, kf.depth_m,
                       kf.accel))
    post = engine.poll()                     # refined latest T_cw, or None
    if post is None:
        return None
    return PoseMsg(kf.seq, 0, np.linalg.inv(post), {"refined": True})
