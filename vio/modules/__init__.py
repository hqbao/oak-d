"""``vio.modules`` -- the VIO pipeline (odometry + windowed BA), procedural.

Data flow, one frame at a time::

    frame --> frontend --> imu_prior --> backend --> publishers
                  (the worker in pipeline.py orchestrates the chain)

The pipeline is plain procedural Python (the old class-heavy Step/Module reactive
framework was dissolved): each stage is a module of single-purpose FUNCTIONS that
take their dependencies explicitly and hand one frame's state to the next via the
small carrier records in :mod:`carriers`.

Files (read in pipeline order)
------------------------------
* :mod:`carriers`      -- the per-frame dataclass records (Tracked / Primed / Step)
                          threaded between stages; never go on the bus.
* :mod:`frontend`      -- sparse visual VO: KLT track -> RGB-D PnP (+ gyro fusion).
* :mod:`imu_prior`     -- IMU prior + gravity chain: preintegrate the per-frame
                          prior, one-shot gravity align, IMU<->vision join, at-rest
                          tilt correction.
* :mod:`backend`       -- keyframe emission + windowed bundle adjustment.
* :mod:`publishers`    -- the thin "emit one result on a bus topic" steps.
* :mod:`pipeline`      -- the orchestration: the two worker threads
                          (:class:`~vio.modules.pipeline.OdometryWorker` joins the
                          IMU + depth edges and runs the per-frame chain;
                          :class:`~vio.modules.pipeline.BackendWorker` runs the
                          sliding-window BA). THE ENTRY POINT.
* :mod:`direct_odometry` -- the ``--direct`` ALT odometry (dense direct RGB-D VO).
* :mod:`propagate_imu` -- the live per-frame IMU dead-reckoning (nav state).
* :mod:`loop_inbox`    -- the SLAM loop-closure correction feedback inbox.

``OdometryModule`` / ``BackendModule`` are kept as public aliases for the workers
(``vio.main`` + the vio/verification selftests import them).
"""
from .pipeline import (
    BackendModule, BackendWorker, OdometryModule, OdometryWorker)

__all__ = ["OdometryModule", "OdometryWorker", "BackendModule", "BackendWorker"]
