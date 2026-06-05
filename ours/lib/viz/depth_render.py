"""Honest depth colourisation shared by the dev tools and the Qt UI.

The metric depth from our SGM is mapped to a TURBO colormap over a **fixed**
range (:data:`D_MIN`..:data:`D_MAX` metres) so colours mean the same distance in
every frame -- a per-frame autoscale would make the scene "breathe" and hide
real depth changes. Invalid pixels (no stereo return, encoded as ``0``) stay
pure black. cv2 is only the colormap backend here; importing this module is what
pulls it, not the base UI.
"""
from __future__ import annotations

import cv2
import numpy as np

# Fixed depth range (metres) for the colormap so colours are stable across
# frames. Matches the odometry/loop-closure usable stereo band.
D_MIN = 0.3
D_MAX = 8.0


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """Metric depth (m, ``0`` == invalid) -> BGR turbo image (near = red/hot)."""
    valid = depth_m > 1e-6
    norm = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid.any():
        z = np.clip(depth_m, D_MIN, D_MAX)
        t = 1.0 - (z - D_MIN) / (D_MAX - D_MIN)          # near = hot
        norm[valid] = (t[valid] * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def depth_scale_bar(height: int, width: int = 16) -> np.ndarray:
    """Vertical TURBO gradient legend (BGR): top = :data:`D_MIN` (near/hot),
    bottom = :data:`D_MAX` (far/cold). The range is fixed, so this is rendered
    once and never changes -- an honest key to the colormap, no per-frame cost.
    """
    rows = np.linspace(0.0, 1.0, max(height, 1), dtype=np.float32)   # 0=near
    norm = ((1.0 - rows) * 255.0).astype(np.uint8)[:, None]
    norm = np.repeat(norm, max(width, 1), axis=1)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
