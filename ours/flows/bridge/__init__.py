"""bridge flows -- glue the in-process :class:`~ours.lib.flow.pubsub.Bus`
to the cross-process :mod:`ours.lib.ipc` bus.

The whole point of this package is so the existing flows (``OdometryFlow``,
``BackendFlow``, ``SlamFlow``, every UI sink) **do not change at all** when the
graph is split across processes. Each process keeps a local Bus and runs its
own flows on it; the bridge flows translate at the boundary:

* :class:`IpcPublisherFlow` -- a sink that subscribes to N local topics and
  publishes the matching wire messages on an :class:`~ours.lib.ipc.IpcServerBus`.
  Lives in the producing process.
* :class:`IpcSubscriberFlow` -- a source that pulls wire messages off an
  :class:`~ours.lib.ipc.IpcClientBus` and republishes them on the local Bus.
  Lives in the consuming process.

The mapping between local dataclasses and wire-message types
(:mod:`ours.lib.ipc.messages`) lives in :mod:`.converters`.
"""
from __future__ import annotations

from .ring_registry import RingSpec, RingRegistry, DEFAULT_RING_SPECS
from .publisher_flow import IpcPublisherFlow
from .subscriber_flow import IpcSubscriberFlow

__all__ = ["RingSpec", "RingRegistry", "DEFAULT_RING_SPECS",
           "IpcPublisherFlow", "IpcSubscriberFlow"]
