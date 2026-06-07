"""IpcSubscriberFlow -- bridge a remote IpcClientBus to local-bus topics.

Subscribes to N topics on an :class:`IpcClientBus`; for every wire message it
``read_copy``s the shared-memory arrays into private numpy arrays, reconstructs
the in-proc dataclass, and publishes it on the local :class:`Bus`. Other flows
in this process consume from the local Bus exactly as before -- they never see
the wire layer.

Threading is supplied by the underlying :class:`IpcClientBus.recv_loop` thread:
it invokes the per-topic handler on its own thread, which does the conversion
and the local publish inline. The IpcClientBus is started in this flow's
``run`` (after the standard :meth:`threading.Thread.start`) so a process can
build its graph + bridges synchronously, then call :meth:`start` once to bring
everything up.
"""
from __future__ import annotations

import logging
import threading
from typing import Iterable

from ...lib.flow import Bus
from ...lib.ipc.bus import IpcClientBus
from .converters import to_local
from .ring_registry import RingRegistry

LOG = logging.getLogger("ours.bridge.sub")


class IpcSubscriberFlow(threading.Thread):
    """A "source" flow that mirrors remote topics onto a local :class:`Bus`.

    ``client`` must be an *unstarted* :class:`IpcClientBus` (this flow calls
    ``.subscribe`` for every requested topic, then ``.start``). ``rings`` is
    the consumer-side :class:`RingRegistry` attached to the producer's shared
    memory. ``topics`` is the list of remote topic names to forward to the
    local bus.
    """

    def __init__(self, local_bus: Bus, client: IpcClientBus,
                 rings: RingRegistry, topics: Iterable[str]) -> None:
        super().__init__(name=f"ipc-sub-{client.endpoint}", daemon=True)
        self.local_bus = local_bus
        self.client = client
        self.rings = rings
        self._topics = list(topics)
        self._stop = threading.Event()
        for t in self._topics:
            client.subscribe(t, self._make_forwarder(t))

    # ------------------------------------------------------------------ #
    def _make_forwarder(self, topic: str):
        """Closure that re-hydrates + republishes one wire message."""
        local_bus = self.local_bus
        rings = self.rings

        def _forward(wm) -> None:
            try:
                msg = to_local(topic, wm, rings)
            except Exception as e:                                 # noqa: BLE001
                LOG.warning("ipc-sub %s/%s: convert failed: %s",
                            self.client.endpoint, topic, e)
                return
            local_bus.publish(topic, msg)

        return _forward

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        try:
            self.client.start()
        except Exception as e:                                     # noqa: BLE001
            LOG.error("ipc-sub %s: client.start failed: %s",
                      self.client.endpoint, e)
            return
        self._stop.wait()

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self.client.stop()
        except Exception:                                          # noqa: BLE001
            pass
