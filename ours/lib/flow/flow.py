"""Flow: one thread that runs a fixed list of tasks sequentially.

A Flow owns a single thread. Inside it, tasks run one after another -- the output
of one task is the input of the next. Flows never call each other directly; they
communicate only through the :class:`~ours.lib.flow.pubsub.Bus`.

Two shapes of flow:

``SourceFlow``
    Produces messages on its own (e.g. grabbing frames from the camera).
    Subclass and override :meth:`SourceFlow.produce` to yield raw items; each
    item is pushed through the task chain on the flow's thread.

``Flow`` (reactive)
    Waits for messages from the bus. Register a task chain per input topic with
    :meth:`Flow.on`. Incoming messages are queued and fed through the matching
    chain on this flow's thread, so heavy work never runs on the publisher.

Tasks publish their results to the bus via ``ctx.bus.publish(topic, msg)`` --
usually in a small dedicated "publish" task at the end of the chain.
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Iterable, Sequence

from .messages import END
from .pubsub import Bus
from .task import Task

_SENTINEL = object()


class FlowContext:
    """Shared state handed to every task in a flow.

    Exposes the bus (for publishing) and a flow-local ``state`` dict that tasks
    use to keep stateful helpers (the VO object, the stereo matcher, counters).
    """

    def __init__(self, bus: Bus, name: str) -> None:
        self.bus = bus
        self.name = name
        self.state: dict[str, Any] = {}


class _BaseFlow(threading.Thread):
    def __init__(self, name: str, bus: Bus) -> None:
        super().__init__(name=name, daemon=True)
        self.bus = bus
        self.ctx = FlowContext(bus, name)
        self._stop = threading.Event()
        self._downstream: list[str] = []

    def forwards_to(self, *topics: str) -> "_BaseFlow":
        """Declare the topics this flow publishes, so END is forwarded to them."""
        self._downstream.extend(topics)
        return self

    def _emit_end(self) -> None:
        for topic in self._downstream:
            self.bus.publish(topic, END)

    def stop(self) -> None:
        self._stop.set()

    @staticmethod
    def _run_chain(ctx: FlowContext, tasks: Sequence[Task], msg: Any) -> None:
        for task in tasks:
            msg = task.run(ctx, msg)
            if msg is None:
                return


class Flow(_BaseFlow):
    """A reactive flow: drains an inbox and routes messages by topic."""

    def __init__(self, name: str, bus: Bus) -> None:
        super().__init__(name, bus)
        self._inbox: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._routes: dict[str, list[Task]] = {}
        self.done = threading.Event()  #: set after all expected ENDs are handled
        self.expected_ends = 1  #: a sink subscribing N END-bearing topics sets this to N
        self._ends_seen = 0
        self._emitted_end = False

    def on(self, topic: str, tasks: Sequence[Task]) -> "Flow":
        """Run ``tasks`` (in order) whenever a message arrives on ``topic``."""
        self._routes[topic] = list(tasks)
        self.bus.subscribe(topic, lambda m, t=topic: self._inbox.put((t, m)))
        return self

    def on_end(self) -> None:
        """Hook called once END has been received. Override for custom drain."""

    def stop(self) -> None:
        super().stop()
        self._inbox.put((_SENTINEL, _SENTINEL))  # unblock the queue.get

    def run(self) -> None:
        while not self._stop.is_set():
            topic, msg = self._inbox.get()
            if msg is _SENTINEL:
                break
            if msg is END:
                self._ends_seen += 1
                if not self._emitted_end:
                    self._emitted_end = True
                    self._emit_end()
                self.on_end()
                if self._ends_seen >= self.expected_ends:
                    self.done.set()
                continue
            self._run_chain(self.ctx, self._routes.get(topic, ()), msg)


class SourceFlow(_BaseFlow):
    """A producing flow: pushes self-generated items through one task chain."""

    def __init__(self, name: str, bus: Bus, tasks: Sequence[Task]) -> None:
        super().__init__(name, bus)
        self.tasks = list(tasks)
        self.done = threading.Event()

    def produce(self) -> Iterable[Any]:
        """Yield raw items to feed into the task chain. Override in subclass."""
        return ()

    def run(self) -> None:
        for item in self.produce():
            if self._stop.is_set():
                break
            self._run_chain(self.ctx, self.tasks, item)
        self._emit_end()
        self.done.set()
