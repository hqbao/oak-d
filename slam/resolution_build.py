"""Per-project config builder that turns a generic ``ResolutionProfile`` into the
concrete cfg object SLAM's math needs.

The profile itself (:mod:`slam.comms.lib.config.resolution`) is data-only and
headless -- it imports no math. The builder lives HERE so the math import
(loop closure) is owned by the project that uses it: SLAM owns the ORB
loop-closure frontend + the pose-graph backend, so it carries the
``loop_config`` builder, exactly as ``vio`` carries ``frontend_config`` /
``odometry_config`` for the KLT/odometry library it owns and ``imu_camera``
carries ``sgm_config`` for the stereo/SGM library it owns. This is what keeps
the vendored ``comms`` package generic + bit-identical across projects.

The builder is ported VERBATIM from the matching pre-split ``ResolutionProfile``
method (``ours.lib.config.resolution``):

* :func:`loop_config`  <- ``ResolutionProfile.loop``

so the resolution-scaled tuning stays byte-identical to the reference oracle.
"""
from __future__ import annotations

from slam.comms.lib.config.resolution import ResolutionProfile
from sky.slam.loopclosure import LoopConfig


def loop_config(res: ResolutionProfile) -> LoopConfig:
    """ORB loop-closure config at this resolution.

    Scales the ORB budget and the two pixel-unit RANSAC gates (epipolar
    fundamental + PnP reprojection); the depth gates are metric and untouched.
    """
    thr = max(1.0, 2.0 * res.scale)
    return LoopConfig(orb_features=int(res.orb_features),
                      fmat_thresh_px=thr, ransac_reproj_px=thr)
