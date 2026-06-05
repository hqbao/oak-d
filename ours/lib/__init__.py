"""``ours.lib`` -- the from-scratch VIO library: pure algorithms + shared helpers.

This package holds everything that is *logic* (no threads, no pub/sub): the
visual front-end, stereo depth, IMU math, odometry, windowed back-end, loop
closure / pose-graph SLAM, session IO and resolution profiles, plus the core
``Pose`` / frame helpers.

Modules are grouped into subpackages for clarity:

    frontend/  corners, klt, klt_numba, frontend   (feature tracking)
    stereo/    stereo                               (rectify + SGM depth)
    imu/       imu, inertial_filter                 (gyro preint, filters)
    odometry/  odometry, pnp                        (RGB-D VO)
    backend/   bundle, windowed, vio_window         (windowed BA / VIO)
    loop/      orb, loopclosure, posegraph, slam    (loop closure SLAM)
    io/        reader, synced                       (session readers)
    config/    resolution                           (resolution profiles)
    misc/      frames, geometry, pose, pngio        (shared helpers)
    flow/      flow, task, pubsub, messages, ...     (flow architecture)

The flat re-exports below are the stable public API: ``from ours.lib import
RGBDVisualOdometry, ORB, SessionReader, ...``. Live-pipeline orchestration
(threads + pub/sub) lives in ``ours.lib.flow`` and the ``ours.flows`` package;
offline tools call this library directly.
"""
from .frontend.frontend import FrontendConfig, KLTFrontend, TrackState
from .misc.geometry import backproject, valid_mask
from .imu.imu import GyroPreintegrator, gravity_aligned_R0, so3_exp
from .imu.inertial_filter import InertialFilterConfig, InertialTranslationFilter
from .loop.loopclosure import KeyframeAppearance, LoopConfig, LoopDetector
from .odometry.odometry import OdometryConfig, RGBDVisualOdometry, level_attitude
from .loop.orb import (
    ORB,
    OrbConfig,
    find_fundamental_ransac,
    hamming_knn,
    match_ratio_mutual,
)
from .loop.posegraph import PoseGraph, se3_adjoint, se3_inv, se3_log
from .io.reader import CameraCalib, Frame, SessionReader, StereoCalib
from .config.resolution import ResolutionProfile
from .loop.slam import SlamConfig, SlamMap
from .io.synced import ImuSegment, SyncedSample, iter_synced, slice_imu
from .stereo.stereo import (
    LeftRectifier,
    RightRectifier,
    SGMConfig,
    SGMStereoMatcher,
    StereoConfig,
    StereoMatcher,
    rectify_rotations,
)
from .backend.vio_window import (
    WindowedVIOConfig,
    WindowedVIOMap,
    WindowedVIORGBDOdometry,
)
from .backend.windowed import WindowedBAMap, WindowedConfig, WindowedRGBDOdometry

__all__ = [
    "CameraCalib",
    "Frame",
    "SessionReader",
    "StereoCalib",
    "backproject",
    "valid_mask",
    "FrontendConfig",
    "KLTFrontend",
    "TrackState",
    "OdometryConfig",
    "RGBDVisualOdometry",
    "level_attitude",
    "GyroPreintegrator",
    "gravity_aligned_R0",
    "so3_exp",
    "InertialFilterConfig",
    "InertialTranslationFilter",
    "WindowedConfig",
    "WindowedRGBDOdometry",
    "WindowedBAMap",
    "WindowedVIOConfig",
    "WindowedVIOMap",
    "WindowedVIORGBDOdometry",
    "KeyframeAppearance",
    "LoopConfig",
    "LoopDetector",
    "ORB",
    "OrbConfig",
    "find_fundamental_ransac",
    "hamming_knn",
    "match_ratio_mutual",
    "PoseGraph",
    "se3_adjoint",
    "se3_inv",
    "se3_log",
    "SlamConfig",
    "SlamMap",
    "ImuSegment",
    "SyncedSample",
    "iter_synced",
    "slice_imu",
    "ResolutionProfile",
    "StereoConfig",
    "StereoMatcher",
    "SGMConfig",
    "SGMStereoMatcher",
    "LeftRectifier",
    "RightRectifier",
    "rectify_rotations",
]
