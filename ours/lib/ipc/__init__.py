"""``ours.lib.ipc`` -- cross-process pub/sub for the 4-process live architecture.

Stdlib-only IPC substrate (no new pip deps). Mirrors the in-process
:class:`ours.lib.flow.pubsub.Bus` API so the existing flows can be reused
unchanged across the process boundary -- a tiny :mod:`ours.flows.bridge` flow
publishes / subscribes at the boundary, every other flow keeps using the
in-process Bus.

The design + decisions are documented in ``docs/PROC4_ARCHITECTURE.md``. This
package implements §3 of that doc:

* :class:`SharedArrayRing` -- a fixed-shape ring of :mod:`multiprocessing.shared_memory`
  slots for one image / depth stream. Producer writes the slot, consumer copies
  it out; metadata travels via :class:`IpcServerBus`.
* :class:`IpcServerBus` / :class:`IpcClientBus` -- pub/sub over
  :class:`multiprocessing.connection.Listener` (Unix-domain socket on macOS /
  Linux). Wire messages are pickled, large numpy arrays travel via
  ``SharedArrayRing`` references inside the wire message.
* :mod:`messages` -- wire-message dataclasses, one per cross-process topic.

OFFLINE replay (``ours.app.run_replay`` + ``flow_replay_selftest``) is unchanged:
it stays single-process and never imports this package.
"""
from __future__ import annotations

from .shared_array import SharedArrayRing, SharedArrayRef
from .bus import IpcServerBus, IpcClientBus

__all__ = ["SharedArrayRing", "SharedArrayRef",
           "IpcServerBus", "IpcClientBus"]
