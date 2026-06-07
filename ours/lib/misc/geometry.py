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


def keyframe_pointcloud(poses, depths, grays, K, *, stride: int = 4,
                        min_depth: float = 0.3, max_depth: float = 6.0):
    """Fuse per-keyframe depth maps into ONE world point cloud.

    The building block of the SLAM 3D-map viewer: each keyframe's depth is
    back-projected to its camera frame, transformed to the world by that
    keyframe's pose, and all keyframes are stacked into a single cloud -- so the
    room is reconstructed from every viewpoint at once.

    * ``poses``  -- list of ``(4,4)`` ``T_world_cam`` (camera->world), one per kf.
    * ``depths`` -- list of ``(H,W)`` metric depth maps (same grid as ``grays``).
    * ``grays``  -- list of ``(H,W)`` intensity images for colour, or ``None``.
    * ``K``      -- ``(3,3)`` intrinsics for the full-resolution depth grid.

    ``stride`` subsamples each map (4 -> 1/16 the points; the room shape survives);
    depth is clipped to ``[min_depth, max_depth]`` because far stereo depth is
    noisy. Returns ``(points (N,3) float32 world, colors (N,3) float32 in [0,1])``
    -- both empty when nothing is valid. Frame: the camera-optical world (the same
    frame ``T_world_cam`` lives in); the viewer applies its own display rotation.
    """
    pts_all: list[np.ndarray] = []
    col_all: list[np.ndarray] = []
    for idx in range(len(poses)):
        depth = np.asarray(depths[idx], dtype=np.float32)
        T = np.asarray(poses[idx], dtype=np.float64)
        cloud = backproject(depth, K)                       # (H,W,3) camera frame
        keep = (valid_mask(depth) & (depth >= min_depth)
                & (depth <= max_depth))[::stride, ::stride]
        cam = cloud[::stride, ::stride][keep]               # (M,3)
        if cam.shape[0] == 0:
            continue
        world = cam @ T[:3, :3].T + T[:3, 3]                # Xw = R Xc + t
        pts_all.append(world.astype(np.float32))
        g = None if grays is None else grays[idx]
        if g is not None:
            gs = np.asarray(g, dtype=np.float32)[::stride, ::stride][keep] / 255.0
            col_all.append(np.repeat(gs[:, None], 3, axis=1).astype(np.float32))
        else:
            col_all.append(np.full((cam.shape[0], 3), 0.7, dtype=np.float32))
    if not pts_all:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
    return np.concatenate(pts_all), np.concatenate(col_all)
