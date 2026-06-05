"""Thread-safe publish/subscribe bus for inter-flow communication.

Flows are independent threads. They never call each other directly; they
exchange data only through this bus. A publisher calls ``bus.publish(topic, msg)``
and every handler registered for that topic is invoked synchronously on the
publisher's thread. Flows register a handler that drops the message into their
own inbox queue, so the real work always runs on the *subscribing* flow's thread
(actor model) -- the publish call itself stays cheap and non-blocking.

Topics are plain strings. The canonical set used by the live pipeline lives in
``ours.lib.flow.topics``.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable

Handler = Callable[[Any], None]


class Bus:
    """A minimal thread-safe pub/sub bus."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register ``handler`` to be called for every message on ``topic``."""
        with self._lock:
            self._subs[topic].append(handler)

    def publish(self, topic: str, msg: Any) -> None:
        """Deliver ``msg`` to every subscriber of ``topic``.

        The subscriber list is copied under the lock and the handlers are then
        invoked outside the lock, so a handler may itself publish without
        deadlocking.
        """
        with self._lock:
            handlers = list(self._subs.get(topic, ()))
        for handler in handlers:
            handler(msg)
