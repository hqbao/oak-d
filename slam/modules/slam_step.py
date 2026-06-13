"""``slam_step`` -- submit a keyframe to the SLAM engine, return any closure.

A plain function (not a ``Step`` subclass): the engine is passed in explicitly
rather than fished out of ``ctx.state["engine"]``, so a reader sees the only
input the submit/poll needs. The logic is byte-identical to the old in-thread
``SlamStep.run`` -- only the wrapper changed.

Offline (in-process engine) ``submit`` adds the keyframe and, on a confirmed loop,
optimises synchronously; ``poll`` returns this keyframe's :class:`SlamResult` --
identical to the old in-thread path. Live (subprocess engine) it is async; the
responsive marker rides ``pose.odom`` and never waits on this.
"""
from __future__ import annotations

from slam.comms.messages import Keyframe, LoopCorrection
from slam.engine import Engine, SlamResult


def slam_submit(engine: Engine, kf: Keyframe) -> LoopCorrection | None:
    """Submit ``kf`` to the engine; return a :class:`LoopCorrection` on a closed
    loop, else ``None`` (matching the old ``SlamStep``).
    """
    engine.submit((kf.T_world_cam, kf.gray_left, kf.depth_m, kf.seq))
    res: SlamResult | None = engine.poll()
    if res is None:
        return None
    return LoopCorrection(kf.seq, res.kf_poses, res.n_loops)
