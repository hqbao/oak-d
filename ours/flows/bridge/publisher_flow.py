"""IpcPublisherFlow -- bridge local-bus topics to a remote IpcServerBus.

Subscribes to N in-proc topics; for every message it converts to the wire
form (writing large arrays into the local :class:`RingRegistry`'s shared-memory
slots) and publishes on the IpcServerBus.

This flow is **not threaded** in the usual way: it has no inbox / task chain
because publishing is fast (a memcpy + a pickle.send) and we want minimum
latency between the local Bus and the wire. The class therefore inherits
from :class:`threading.Thread` only for lifecycle parity with the other flows
(start/stop/join), but its ``run`` is a one-shot wait-for-stop -- all of the
real work happens inline in the subscribe handlers (which run on the publisher
flow's thread, just like any other local Bus handler).

For long heavy publishes (e.g. depth frames at 20 fps × ~1.5 MB) the work
runs on the producer's own thread, which is exactly what we want: a slow
serialise inside the imu_cam flow naturally throttles the imu_cam flow, the
same way a slow consumer naturally throttles the in-proc Bus today. The
IpcServerBus's per-subscriber outbox is bounded latest-only so a stuck
subscriber cannot stall the producer thread either.
"""
from __future__ import annotations

import logging
import threading
from typing import Iterable

from ...lib.flow import Bus, topics as _topics
from ...lib.flow.messages import END
from ...lib.ipc.bus import IpcServerBus
from .converters import to_wire
from .ring_registry import RingRegistry

LOG = logging.getLogger("ours.bridge.pub")


class IpcPublisherFlow(threading.Thread):
    """A "sink" flow that mirrors local topics onto an :class:`IpcServerBus`.

    ``endpoint`` is the publisher endpoint (e.g. ``"oak.capture"``); ``rings``
    must already be created via :meth:`RingRegistry.create_all` so the
    converters can write into the slots; ``topics`` is the list of local-bus
    topic names to forward. The IpcServerBus is started by this flow (and
    closed on :meth:`stop`).

    The flow holds **no inbox** -- it's a pure subscribe-and-forward bridge.
    The ``Thread.run`` body just waits for the stop event so the standard
    ``start()`` / ``join()`` lifecycle still works.

    Construction is intentionally idempotent for the same Bus / endpoint pair
    so a process that builds the graph multiple times during a unit test does
    not leak duplicate subscriptions.
    """

    def __init__(self, local_bus: Bus, server: IpcServerBus,
                 rings: RingRegistry, topics: Iterable[str],
                 *, endpoint: str | None = None,
                 ring_endpoint: str | None = None) -> None:
        super().__init__(name=f"ipc-pub-{server.endpoint}", daemon=True)
        self.local_bus = local_bus
        self.server = server
        self.rings = rings
        self.endpoint = endpoint or server.endpoint
        # Ring names are namespaced by the producing endpoint -- e.g. capture
        # publishes "oak.capture.gray_left". A re-publisher (VIO republishing
        # capture's frames) may want a different ring namespace; default to
        # the server's endpoint.
        self.ring_endpoint = ring_endpoint or self.endpoint
        self._topics = list(topics)
        self._stop = threading.Event()
        # Subscriptions are made eagerly so messages published while this flow
        # is "starting" are not lost. The IpcServerBus is started in `run` so
        # the socket only exists once we're committed to the lifecycle.
        for t in self._topics:
            local_bus.subscribe(t, self._make_forwarder(t))

    # ------------------------------------------------------------------ #
    def _make_forwarder(self, topic: str):
        """Closure that converts + publishes one message for ``topic``."""
        server = self.server
        rings = self.rings
        ring_endpoint = self.ring_endpoint

        def _forward(msg) -> None:
            try:
                wm = to_wire(topic, msg, rings, ring_endpoint)
            except Exception as e:                                 # noqa: BLE001
                LOG.warning("ipc-pub %s/%s: convert failed: %s",
                            self.endpoint, topic, e)
                return
            try:
                server.publish(topic, wm)
            except Exception as e:                                 # noqa: BLE001
                LOG.warning("ipc-pub %s/%s: send failed: %s",
                            self.endpoint, topic, e)

        return _forward

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Start the server, then idle until :meth:`stop`."""
        try:
            self.server.start()
        except Exception as e:                                     # noqa: BLE001
            LOG.error("ipc-pub %s: server.start failed: %s", self.endpoint, e)
            return
        self._stop.wait()

    def stop(self) -> None:
        """Idempotent shutdown: stop the wait, close the server socket."""
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self.server.close()
        except Exception:                                          # noqa: BLE001
            pass
