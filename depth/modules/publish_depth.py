"""``publish_depth`` -- emit a computed depth frame on ``frame.depth``.

A plain function (not a ``Step`` subclass): the destination bus is passed in
explicitly rather than reached through ``ctx.bus``. ``bus`` is the in-process
:class:`~depth.comms.LocalPubSub` the :class:`~depth.comms.IPCPublisher` mirrors
onto the wire, so publishing here is exactly what the old step did.
"""
from __future__ import annotations

from depth.comms import LocalPubSub, topics
from depth.comms.messages import DepthFrame


def publish_depth(bus: LocalPubSub, frame: DepthFrame) -> None:
    """Publish ``frame`` on ``topics.FRAME_DEPTH``."""
    bus.publish(topics.FRAME_DEPTH, frame)
