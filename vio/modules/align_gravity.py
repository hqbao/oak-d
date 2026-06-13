"""``align_gravity`` step: one-shot startup gravity alignment.

Third step of the odometry frame-chain (after :func:`track_features` /
:func:`publish_tracks`). It is a one-shot bootstrap, not a per-frame solve: the
first time the front-end's startup gravity reference (``accel_align``, camera
frame) is available it levels the initial attitude via
:meth:`~sky.front.odometry.RGBDVisualOdometry.align_to_gravity`, then
never fires again. Pulled out of :func:`estimate_motion` so the per-frame motion
solve carries no init branch. Passes the :class:`Tracked` carrier through
unchanged. A no-op when there is no usable IMU (no ``accel_align``).
"""
from __future__ import annotations

from sky.front.odometry import RGBDVisualOdometry
from .tracked import Tracked


def align_gravity(vo: RGBDVisualOdometry, state: dict, tracked: Tracked) -> Tracked:
    """One-shot: level the initial attitude to the startup gravity reference.

    Was ``AlignGravity(Step)``. ``vo`` is the odometry instance; ``state`` is the
    worker's shared state dict (``ctx.state``), holding the one-shot ``aligned``
    latch and the ``accel_align`` seed. Passes the carrier through unchanged.
    """
    if not state.get("aligned") and "accel_align" in state:
        vo.align_to_gravity(state["accel_align"])
        state["aligned"] = True
    return tracked
