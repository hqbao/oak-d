"""Per-project config builders that turn a generic ``ResolutionProfile`` into the
concrete cfg objects this project's math needs.

The profile itself (``imu_camera.comms.lib.config.resolution``) is data-only and
headless — it imports no math. The builders live HERE so the math import (SGM)
is owned by the project that uses it (imu_camera owns the stereo/SGM library);
vio/slam carry their own builders for frontend/odometry/loop. This is what keeps
the vendored ``comms`` package generic + bit-identical across projects.
"""
from __future__ import annotations

from dataclasses import replace

from imu_camera.comms.lib.config.resolution import ResolutionProfile
from sky.depth.stereo import SGMConfig


def sgm_config(res: ResolutionProfile, *, fast: bool) -> SGMConfig:
    """Dense-SGM depth config at this resolution.

    ``fast`` selects the half-res live preset (cheaper census + 4 paths +
    internal downscale); either way the disparity search range is set from the
    resolution so the metric near-depth bound (``fx*B/num_disparities``) stays
    roughly constant across resolutions (``fx`` scales with width too). Logic is
    verbatim from the pre-split ``ResolutionProfile.sgm`` so depth stays
    byte-identical.
    """
    base = SGMConfig.live() if fast else SGMConfig()
    # The ``live`` preset downsamples SGM by 2x for speed. At a LOW capture
    # resolution that is counter-productive: 160x100 -> a 80x50 internal match
    # whose disparity border + coarse census leave too few valid-depth pixels at
    # the tracked keypoints, so RGB-D PnP starves (measured: only ~20 of ~60 KLT
    # tracks get depth -> PnP fails on the majority -> phantom poses). Below the
    # 160px floor we compute SGM at native resolution (cheap there anyway) for a
    # dense, accurate depth. 320/640 keep the 2x downscale -> 640 byte-identical.
    downscale = 1 if res.width <= 160 else base.downscale
    return replace(base, num_disparities=int(res.num_disparities),
                   downscale=downscale)
