"""``ours.flows.core`` -- the flow-architecture framework (NOT a flow itself).

This is the threaded, message-passing substrate every flow in ``ours.flows`` is
built on. It deliberately lives next to the flows (the orchestration layer), not
in ``ours.lib`` -- ``ours.lib`` is reserved for pure computation libraries.

    flow      Flow / SourceFlow -- one thread running a fixed task chain
    task      Task              -- the smallest input->output step in a chain
    pubsub    Bus               -- thread-safe publish/subscribe between flows
    messages  message carriers  -- one immutable type per topic
    topics    topic-name constants
    runtime   process-wide guards (e.g. the numba parallel lock)

The common entry points are re-exported flat so flows can write
``from ..core import Flow, SourceFlow, Task, Bus``; ``messages``, ``topics`` and
``runtime`` are used as submodules.

Layering: this package depends only on the standard library + numpy. It does NOT
import ``ours.lib`` (pure algorithms) -- the concrete flow implementations are
what wire the two together.
"""
from .flow import Flow, FlowContext, SourceFlow
from .pubsub import Bus
from .task import Task

__all__ = ["Flow", "SourceFlow", "FlowContext", "Bus", "Task"]
