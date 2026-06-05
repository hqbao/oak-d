"""Base class for a Task: one clear input -> one clear output step.

A Task is the smallest unit of work inside a Flow. It receives the flow's shared
context and one message, and returns the message to hand to the next task in the
chain. Returning ``None`` stops the chain for that input (e.g. "no keyframe this
frame, nothing to forward").

Keep tasks small and single-purpose. If a task file grows too long, split the
work into two tasks.
"""
from __future__ import annotations

from typing import Any


class Task:
    """A single step in a flow's task chain."""

    #: Human-readable name, used in logs/diagnostics.
    name: str = "task"

    def run(self, ctx: Any, msg: Any) -> Any:
        """Process ``msg`` and return the value for the next task.

        ``ctx`` is the owning flow's :class:`~ours.lib.flow.flow.FlowContext`
        (gives access to the bus and the flow-local ``state`` dict). Return
        ``None`` to halt the chain for this message.
        """
        raise NotImplementedError
