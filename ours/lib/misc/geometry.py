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


def keyframe_landmark_cloud(poses, track_ids, track_px, depths, inlier_ids, K,
                            grays=None, *, min_depth: float = 0.3,
                            max_depth: float = 6.0):
    """Sparse cloud of ONLY the PnP-inlier feature points (the clean landmarks).

    The dense-depth fuse (:func:`keyframe_pointcloud`) back-projects every pixel, so
    a noisy stereo depth map spreads "flying" points before/behind real surfaces.
    Here we keep ONLY the features the RGB-D PnP accepted as inliers this frame --
    motion-consistent points whose depth the solve already trusted -- so the map is
    the sparse landmark set, not the noise the solve rejected.

    Per keyframe: ``track_ids[i]`` / ``track_px[i]`` are the frame's tracks (ids +
    Nx2 pixels), ``inlier_ids[i]`` the subset PnP kept, ``depths[i]`` the depth map,
    ``poses[i]`` the ``T_world_cam``. Returns ``(points (N,3) world, colors (N,3))``.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    pts_all: list[np.ndarray] = []
    col_all: list[np.ndarray] = []
    for i in range(len(poses)):
        ids = track_ids[i] if track_ids is not None else None
        px = track_px[i] if track_px is not None else None
        inl = inlier_ids[i] if inlier_ids is not None else None
        if ids is None or px is None or inl is None or len(ids) == 0 or len(inl) == 0:
            continue
        ids = np.asarray(ids); px = np.asarray(px, dtype=np.float64)
        sel = px[np.isin(ids, np.asarray(inl))]          # (m,2) inlier pixels
        if sel.shape[0] == 0:
            continue
        depth = np.asarray(depths[i], dtype=np.float32)
        h, w = depth.shape
        u = np.round(sel[:, 0]).astype(np.int64)
        v = np.round(sel[:, 1]).astype(np.int64)
        on = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v = u[on], v[on]
        if u.size == 0:
            continue
        z = depth[v, u]
        ok = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
        u, v, z = u[ok], v[ok], z[ok]
        if z.size == 0:
            continue
        cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
        T = np.asarray(poses[i], dtype=np.float64)
        pts_all.append((cam @ T[:3, :3].T + T[:3, 3]).astype(np.float32))
        g = None if grays is None else grays[i]
        if g is not None:
            gs = np.asarray(g, dtype=np.float32)[v, u] / 255.0
            col_all.append(np.repeat(gs[:, None], 3, axis=1).astype(np.float32))
        else:
            col_all.append(np.full((z.size, 3), 0.85, dtype=np.float32))
    if not pts_all:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
    return np.concatenate(pts_all), np.concatenate(col_all)
