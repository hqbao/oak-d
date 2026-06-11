"""Per-project config builders that turn a generic ``ResolutionProfile`` into the
concrete cfg objects VIO's math needs.

The profile itself (:mod:`vio.comms.lib.config.resolution`) is data-only and
headless -- it imports no math. The builders live HERE so the math import
(frontend / odometry) is owned by the project that uses it: VIO owns the KLT
frontend + the frame-to-frame RGB-D odometry, so it carries the
``frontend_config`` / ``odometry_config`` / ``ba_huber_px`` builders, exactly as
``imu_camera`` carries ``sgm_config`` for the stereo/SGM library it owns. This is
what keeps the vendored ``comms`` package generic + bit-identical across projects.

Each builder is ported VERBATIM from the matching pre-split
``ResolutionProfile`` method (``ours.lib.config.resolution``):

* :func:`frontend_config`  <- ``ResolutionProfile.frontend``
* :func:`odometry_config`  <- ``ResolutionProfile.odometry``
* :func:`ba_huber_px`      <- ``ResolutionProfile.ba_huber_px``

so the resolution-scaled tuning stays byte-identical to the reference oracle.
"""
from __future__ import annotations

from vio.comms.lib.config.resolution import ResolutionProfile, _round_odd
from sky.front.frontend import FrontendConfig
from sky.front.odometry import OdometryConfig


def frontend_config(res: ResolutionProfile, *, numba: bool) -> FrontendConfig:
    """KLT + Shi-Tomasi config at this resolution.

    Without Numba the pure-NumPy KLT cannot afford the full window/pyramid at
    20 fps, so the window, pyramid depth and corner budget are capped to the
    ``live_own`` budget -- but the *resolution-scaled* corner geometry
    (spacing) is kept either way. At the 640 baseline with Numba this returns
    exactly the full-quality default; without Numba it matches the historical
    ``FrontendConfig.live_own()`` preset.
    """
    win = res.klt_win if numba else _round_odd(min(res.klt_win, 13), lo=7)
    lvl = res.klt_levels if numba else min(res.klt_levels, 2)
    corners = res.max_corners if numba else min(res.max_corners, 200)
    # Detection geometry is a VIO-frontend concern, so it is derived HERE from
    # the width -- NOT stored on the shared ``ResolutionProfile`` (that lives in
    # the vendored ``comms`` package, which must stay byte-identical across every
    # project's copy). Only the low-res regime (<= 160 px wide: the 160x100 floor
    # + the 54x42 ToF sim) switches to a small Shi-Tomasi window + bucketed
    # detection (more, evenly-spread corners); 320/640 keep the historical
    # block_size=7 / non-bucketed path -> 640 stays byte-identical to the oracle.
    low_res = res.width <= 160
    block_size = 3 if low_res else 7
    bucketed = low_res
    return FrontendConfig(max_corners=corners,
                          min_distance=res.min_distance,
                          block_size=block_size,
                          bucketed=bucketed,
                          win_size=win, max_level=lvl)


def odometry_config(res: ResolutionProfile, **guards) -> OdometryConfig:
    """Frame-to-frame RGB-D PnP config at this resolution.

    Scales the reprojection gate (px) and the translation-freeze inlier
    count (tied to the corner budget) from the baseline; ``guards`` carries
    the resolution-independent live safety gates (``gyro_fuse``,
    ``max_translation_speed`` in m/s, ...).
    """
    min_inl = max(6, int(round(12 * res.scale)))
    return OdometryConfig(ransac_reproj_px=res.reproj_px,
                          min_inliers_for_translation=min_inl,
                          **guards)


def ba_huber_px(res: ResolutionProfile) -> float:
    """Sliding-window BA robust (Huber) reprojection scale, in pixels."""
    return max(1.0, 2.0 * res.scale)
