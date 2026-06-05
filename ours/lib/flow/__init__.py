"""``ours.lib.flow`` -- the flow-architecture primitives.

Everything here is the *threaded, message-passing* substrate that the
``ours.flows`` package is built on (as opposed to the pure algorithms and helpers
elsewhere in ``ours.lib``):

    flow      Flow / SourceFlow -- one thread running a fixed task chain
    task      Task              -- the smallest input->output step in a chain
    pubsub    Bus               -- thread-safe publish/subscribe between flows
    messages  message carriers  -- one immutable type per topic
    topics    topic-name constants
    runtime   process-wide guards (e.g. the numba parallel lock)

The common entry points are re-exported flat so callers can write
``from ours.lib.flow import Flow, SourceFlow, Task, Bus``; ``messages``,
``topics`` and ``runtime`` are used as submodules.
"""
from .flow import Flow, FlowContext, SourceFlow
from .pubsub import Bus
from .task import Task

__all__ = ["Flow", "SourceFlow", "FlowContext", "Bus", "Task"]
