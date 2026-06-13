"""``pull_prior`` step: the IMU<->vision fusion join.

Fourth step of the odometry frame-chain. The
:func:`~vio.modules.preintegrate_prior.preintegrate_prior` step (on the
``imucam.sample`` edge) buffers one :class:`~vio.comms.messages.ImuPrior`
per frame ``seq`` in the worker's ``priors`` dict; this step pops the matching
prior for the frame now being solved and threads it forward on the :class:`Primed`
carrier. Splitting the pop out of :func:`estimate_motion` names the place the two
front-end edges meet -- the solve downstream just consumes the joined prior. The
prior is ``None`` when none preintegrated for this frame (pure vision / no IMU).
"""
from __future__ import annotations

from .primed import Primed
from .tracked import Tracked


def pull_prior(priors: dict, tracked: Tracked) -> Primed:
    """Pop this frame's preintegrated IMU prior and join it onto the carrier.

    Was ``PullPrior(Step)``; ``priors`` (the worker's ``ctx.state["priors"]``
    seq->prior dict) is passed explicitly. ``None`` when none was preintegrated.
    """
    prior = priors.pop(tracked.frame.seq, None)
    return Primed(tracked.frame, tracked.obs, prior)
