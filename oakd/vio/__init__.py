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
from .imu import GyroPreintegrator, so3_exp
from .odometry import OdometryConfig, RGBDVisualOdometry
from .reader import CameraCalib, Frame, SessionReader, StereoCalib
from .windowed import WindowedConfig, WindowedRGBDOdometry

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
    "GyroPreintegrator",
    "so3_exp",
    "WindowedConfig",
    "WindowedRGBDOdometry",
]
