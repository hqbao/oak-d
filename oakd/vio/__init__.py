"""From-scratch visual-inertial odometry (work in progress).

This package is being built from the ground up to eventually match the quality
of the Basalt VIO used for the gold sessions. The goal is a fully understood,
offline-testable pipeline driven by *real* recorded data
(``sessions/.../input/``).

Stage 1 (current): read a recorded session and feed a single frame + depth map
through the geometry to prove the data path is correct.
"""
from .frontend import FrontendConfig, KLTFrontend, TrackState
from .geometry import backproject, valid_mask
from .imu import GyroPreintegrator, gravity_aligned_R0, so3_exp
from .inertial_filter import InertialFilterConfig, InertialTranslationFilter
from .loopclosure import KeyframeAppearance, LoopConfig, LoopDetector
from .odometry import OdometryConfig, RGBDVisualOdometry, level_attitude
from .posegraph import PoseGraph, se3_adjoint, se3_inv, se3_log
from .reader import CameraCalib, Frame, SessionReader, StereoCalib
from .resolution import ResolutionProfile
from .slam import SlamConfig, SlamMap
from .synced import ImuSegment, SyncedSample, iter_synced, slice_imu
from .stereo import (
    LeftRectifier,
    RightRectifier,
    SGMConfig,
    SGMStereoMatcher,
    StereoConfig,
    StereoMatcher,
    rectify_rotations,
)
from .vio_window import (
    WindowedVIOConfig,
    WindowedVIOMap,
    WindowedVIORGBDOdometry,
)
from .windowed import WindowedBAMap, WindowedConfig, WindowedRGBDOdometry

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
