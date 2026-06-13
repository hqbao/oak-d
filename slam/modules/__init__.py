"""``slam.modules`` -- the SLAM pipeline (ORB loop closure + pose graph).

PROCEDURAL Python (no reactive ``Module`` / ``Step`` graph). The per-keyframe
work is the plain function :func:`~slam.modules.pipeline.process_keyframe`, which
calls the single-purpose step functions in order:

* :func:`~slam.modules.slam_step.slam_submit` -- submit the keyframe to the SLAM
  engine; on a confirmed loop return the rewritten poses.
* :func:`~slam.modules.publish_correction.publish_correction` -- emit it on
  ``loop.correction``.
* (LIVE only) :func:`~slam.modules.publish_loops.publish_loops` +
  :func:`~slam.modules.publish_slam_map.publish_slam_map` -- poll the engine's
  independent loop-funnel + map-overlay channels for the UI.

:class:`~slam.modules.pipeline.SlamWorker` (exported also under the legacy name
:data:`~slam.modules.pipeline.SlamModule`) is the plain thread that drains the
keyframe inbox -- with the LOAD-BEARING ``latest_only`` coalescing kept explicit
-- and runs ``process_keyframe``. The heavy ORB + pose-graph solve runs behind a
swappable :class:`~slam.engine.base.Engine` (in-process offline,
subprocess live).
"""
from .pipeline import SlamModule, SlamWorker, process_keyframe

__all__ = ["SlamModule", "SlamWorker", "process_keyframe"]
