"""Honest depth colourisation shared by the dev tools and the Qt UI.

The metric depth from our SGM is mapped to a **single-hue khaki ramp** over a
**fixed** range (:data:`D_MIN`..:data:`D_MAX` metres) so brightness means the
same distance in every frame -- a per-frame autoscale would make the scene
"breathe" and hide real depth changes. One hue keeps the view calm: **near =
bright khaki, far = dark olive**. Invalid pixels (no stereo return, encoded as
``0``) stay pure black (distinct from the darkest valid depth, which keeps a
floor brightness). cv2 is only the array backend here; importing this module is
what pulls it, not the base UI.
"""
from __future__ import annotations

import numpy as np

# Fixed depth range (metres) so colours are stable across frames. Matches the
# odometry/loop-closure usable stereo band.
D_MIN = 0.3
D_MAX = 8.0

# Single depth hue: theme ACCENT khaki #c9b97f, stored BGR for the cv2 arrays.
_KHAKI_BGR = np.array([127, 185, 201], dtype=np.float64)
# Darkest valid depth keeps this fraction of the hue's brightness so far points
# stay a visible dark olive, never pure black (which marks invalid/no-stereo).
_FAR_FLOOR = 0.22


def _depth_norm_u8(depth_values: np.ndarray) -> np.ndarray:
    """Map metric depth -> a 0..255 ramp index (``near = 255``, ``far = 0``).

    ``D_MIN..D_MAX`` clamp. Shared by the dense image, the scale-bar and the
    per-keypoint dot so a given distance is always the identical colour.
    """
    z = np.clip(np.asarray(depth_values, dtype=np.float64), D_MIN, D_MAX)
    t = 1.0 - (z - D_MIN) / (D_MAX - D_MIN)
    return (t * 255.0).astype(np.uint8)


_RAMP_LUT: np.ndarray | None = None


def _ramp_lut() -> np.ndarray:
    """``(256, 3)`` BGR single-hue khaki ramp (built once, then cached).

    Index ``255`` (near) = full khaki; index ``0`` (far) = the khaki scaled to
    :data:`_FAR_FLOOR` brightness (a dark olive, not black).
    """
    global _RAMP_LUT
    if _RAMP_LUT is None:
        t = np.arange(256, dtype=np.float64).reshape(256, 1) / 255.0
        bright = _FAR_FLOOR + (1.0 - _FAR_FLOOR) * t
        _RAMP_LUT = np.clip(bright * _KHAKI_BGR, 0, 255).astype(np.uint8)
    return _RAMP_LUT


def depth_colors(depth_values: np.ndarray) -> np.ndarray:
    """Per-value depth -> ``(M, 3)`` uint8 BGR, identical mapping to the image.

    Invalid depths (``<= 0``) clamp to ``D_MIN`` here (the caller masks them out
    and draws a neutral marker instead -- this never invents a colour for them).
    """
    return _ramp_lut()[_depth_norm_u8(depth_values)]


def depth_color(z: float) -> tuple[int, int, int]:
    """Single metric depth -> BGR tuple (same mapping as :func:`colorize_depth`)."""
    b, g, r = _ramp_lut()[int(_depth_norm_u8(np.array([float(z)]))[0])]
    return int(b), int(g), int(r)


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """Metric depth (m, ``0`` == invalid) -> BGR khaki-ramp image (near = bright)."""
    valid = depth_m > 1e-6
    out = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if valid.any():
        out[valid] = _ramp_lut()[_depth_norm_u8(depth_m)[valid]]
    return out


def depth_scale_bar(height: int, width: int = 16) -> np.ndarray:
    """Vertical khaki-ramp legend (BGR): top = :data:`D_MIN` (near/bright),
    bottom = :data:`D_MAX` (far/dark). The range is fixed, so this is rendered
    once and never changes -- an honest key to the ramp, no per-frame cost.
    """
    rows = np.linspace(0.0, 1.0, max(height, 1), dtype=np.float32)   # 0=near
    idx = ((1.0 - rows) * 255.0).astype(np.uint8)
    bar = _ramp_lut()[idx]                                  # (H, 3)
    return np.repeat(bar[:, None, :], max(width, 1), axis=1)
