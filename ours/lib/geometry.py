"""Depth geometry helpers for the from-scratch VIO (pure numpy).

Generic RGB-D primitives: back-project an organised depth map to a 3D point
cloud and mask out invalid pixels. These are the building blocks for giving
metric depth to 2D feature tracks.

Camera convention: OpenCV optical frame (+x right, +y down, +z forward).
"""
from __future__ import annotations

import numpy as np


def backproject(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project an organised depth map (metres) to a ``(H, W, 3)`` cloud.

    Invalid pixels (depth <= 0) become ``(0, 0, 0)`` and should be masked via
    :func:`valid_mask`.
    """
    h, w = depth_m.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us = np.arange(w, dtype=np.float32)
    vs = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    z = depth_m.astype(np.float32)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.stack((x, y, z), axis=-1)


def valid_mask(depth_m: np.ndarray) -> np.ndarray:
    """Boolean ``(H, W)`` mask of finite, positive-depth pixels."""
    return np.isfinite(depth_m) & (depth_m > 0.0)
