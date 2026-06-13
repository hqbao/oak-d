"""The per-keyframe solve, factored out so in-process and subprocess engines run
*exactly the same code* (the whole offline byte-parity argument depends on this).

``slam_step`` takes a live map object + one keyframe snapshot and returns the
solve result (or ``None`` when there is nothing to publish for that keyframe).
It is a pure function of (map, snapshot): no threads, no queues, no flow/bus
knowledge -- it receives the map instance so the same function drives both the
synchronous :class:`~slam.engine.inprocess.InProcessEngine` and the child of
:class:`~slam.engine.subprocess.SubprocessEngine`.

The logic is lifted verbatim from the old in-thread ``SlamStep`` task so the
offline path stays identical.
"""
from __future__ import annotations

from typing import Any

from .base import SlamResult


def slam_step(slam_map, snap: Any):
    """One SLAM keyframe: add it; on a confirmed loop, optimise the pose graph.

    ``snap`` = ``(T_world_cam, gray_left, depth_m, seq)``. Returns a
    :class:`SlamResult` (rewritten keyframe poses + loop count) only when this
    keyframe closed a loop -- otherwise ``None`` (matching the old ``SlamStep``).
    """
    T_wc, gray, depth_m, seq = snap
    events = slam_map.add_keyframe(T_wc, gray, depth_m, seq=seq)
    if not events:
        return None
    slam_map.optimize()
    kf_poses = {int(slam_map.kf_seq[i]): slam_map.kf_pose[i].copy()
                for i in range(len(slam_map.kf_pose))}
    return SlamResult(kf_poses, len(slam_map.loop_events))


# --------------------------------------------------------------------------- #
# Overlay extractor: a cheap, picklable snapshot of the live MAP for the 3D
# viewer (the visible "refined map behind the responsive marker"). All positions
# are camera world-frame (optical); the UI applies the single optical->NED display
# transform. It reads REAL map outputs (corrected SLAM poses + loop events) --
# never a parallel/derived pipeline.
# --------------------------------------------------------------------------- #

def slam_overlay(slam_map):
    """SLAM map snapshot for the keyframe-dots + loop-flash overlay.

    Returns ``(kf_seq, kf_pos, n_loops, match_pos)``:
    * ``kf_seq``   -- ``(N,)`` int64 source frame seq of each keyframe, in the
      SAME order as ``kf_pos`` (``kf_seq``/``kf_pose`` are parallel arrays on the
      map). Lets the UI match each corrected keyframe back to its dense VIO pose
      (the rubber-sheet "corrected VIO" line).
    * ``kf_pos``   -- ``(N,3)`` current (corrected) keyframe positions.
    * ``n_loops``  -- confirmed loop count (the UI bumps its flash when it grows).
    * ``match_pos``-- ``(M,3)`` the two keyframes of the MOST RECENT loop (cur+old),
      i.e. where the pose just snapped back to (empty if no loop yet).
    """
    import numpy as np
    kf_seq = np.asarray(slam_map.kf_seq, dtype=np.int64)
    kf_pos = (np.array([p[:3, 3] for p in slam_map.kf_pose], dtype=np.float64)
              if slam_map.kf_pose else np.zeros((0, 3)))
    n_loops = len(slam_map.loop_events)
    match_pos = np.zeros((0, 3))
    if slam_map.loop_events:
        ev = slam_map.loop_events[-1]
        idxs = [i for i in (ev.get("cur"), ev.get("old")) if i is not None
                and 0 <= i < len(slam_map.kf_pose)]
        if idxs:
            match_pos = np.array([slam_map.kf_pose[i][:3, 3] for i in idxs],
                                 dtype=np.float64)
    return (kf_seq, kf_pos, n_loops, match_pos)
