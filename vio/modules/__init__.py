"""``vio.modules`` -- the VIO pipeline (odometry + windowed BA), procedural.

The two worker threads and the single-purpose step FUNCTIONS they compose (the
class-heavy Step/Module reactive framework was replaced by plain procedural
Python; see :mod:`vio.modules.pipeline`):

* :class:`~vio.modules.pipeline.OdometryWorker` -- joins ``imucam.sample`` (IMU
  prior preintegration) + ``frame.depth`` (KLT track -> RGB-D PnP -> gyro fusion
  -> pose) and publishes ``pose.odom`` every frame, a ``keyframe`` every few
  frames, plus ``frame.tracks`` / ``frame.inliers`` for the visualiser (and
  ``pose.vo`` when the live builder enables the pure-vision line). It owns the
  2-input multi-END join (``expected_ends == 2``) explicitly.
* :class:`~vio.modules.pipeline.BackendWorker` -- consumes ``keyframe``, runs the
  sliding-window bundle adjustment behind a swappable
  :class:`~vio.mathlib.engine.base.Engine`, and publishes the refined pose on
  ``pose.refined``.

``OdometryModule`` / ``BackendModule`` are kept as public aliases (vio.main + the
vio/verification selftests import them).

The internal carriers (:mod:`~vio.modules.step` / :mod:`~vio.modules.primed` /
:mod:`~vio.modules.tracked`) thread one frame's state through the odometry chain;
they never go on the bus.
"""
from .pipeline import (
    BackendModule, BackendWorker, OdometryModule, OdometryWorker)

__all__ = ["OdometryModule", "OdometryWorker", "BackendModule", "BackendWorker"]
