"""``ours.lib.flow`` -- the flow-architecture library.

This is the threaded, message-passing substrate the concrete flows in
``ours.flows`` are built on. It is a *library* (reusable machinery), which is why
it lives in ``ours.lib`` next to the other libraries -- the concrete flows in
``ours.flows`` import and use it, just like they use ``ours.lib.stereo`` or
``ours.lib.odometry``.

    flow      Flow / SourceFlow -- one thread running a fixed task chain
    task      Task              -- the smallest input->output step in a chain
    pubsub    Bus               -- thread-safe publish/subscribe between flows
    messages  message carriers  -- one immutable type per topic (the flow contract)
    topics    topic-name constants
    runtime   process-wide guards (e.g. the numba parallel lock)

HARD RULE of the architecture: flows NEVER call each other directly -- they
communicate ONLY by publishing/subscribing messages on ``Bus`` topics. This
library provides the mechanism (``Bus`` + ``topics`` + ``messages``); the
concrete flows provide the behaviour.

The common entry points are re-exported flat so flows can write
``from ...lib.flow import Flow, SourceFlow, Task, Bus``; ``messages``, ``topics``
and ``runtime`` are used as submodules.

Layering: this library depends only on the standard library + numpy. It does NOT
import the concrete flows or the pure-algorithm libraries -- the concrete flow
implementations are what wire the two together.
"""
from .flow import Flow, FlowContext, SourceFlow
from .pubsub import Bus
from .task import Task

__all__ = ["Flow", "SourceFlow", "FlowContext", "Bus", "Task"]
