"""slam pipeline: loop closure + pose graph, as PROCEDURAL Python.

This replaces the old reactive ``SlamModule(Module)`` + ``_RunCorrectionChain``
``Step`` graph. The per-keyframe work is now a plain function
(:func:`process_keyframe`) that calls the slam step functions in the EXACT same
order the reactive chain ran -- so ``loop.correction`` stays byte-identical and
the live ``slam.map`` / ``slam.loop`` overlays poll the SAME engine channels
AFTER the submit. The framework indirection (``Module.on`` / ``_routes`` /
``_run_chain`` short-circuit-on-None) is gone; the data flow reads as
straight-line code.

What did NOT vanish: the keyframe inbox + the ``latest_only`` coalescing. SLAM
consumes ``keyframe`` with ``latest_only=True`` on the LIVE path -- under
real-time load the ORB + pose-graph solve cannot keep up, so the inbox must DROP
intermediate keyframes and always solve the FRESHEST one (a strict-FIFO inbox
would back up without bound and the ``slam.map`` overlay would lag seconds behind
real time). That coalescing is LOAD-BEARING and is replicated EXPLICITLY here in
:class:`SlamWorker` -- a plain thread that owns a single-slot "latest keyframe"
holder + a worker loop, instead of inheriting the comms reactive substrate.
``END`` is never coalesced away, so clean shutdown still propagates. The OFFLINE /
oracle callers build it with ``latest_only=False`` for a strict-FIFO inbox (the
deterministic path must process every keyframe).

``SlamModule`` is kept as the public name (``slam.main`` + the two external
in-process consumers -- ``vio.tests.closed_loop_drift_selftest`` and
``verification.loop_teleport_diag`` -- import it); it is now an alias for the
procedural :class:`SlamWorker`.
"""
from __future__ import annotations

import queue
import threading
from typing import Any

import numpy as np

from slam.comms import LocalPubSub, topics
from slam.comms.messages import END, Keyframe
from slam.engine import Engine, make_slam_engine
from sky.slam.slam import SlamConfig

from .slam_step import slam_submit
from .publish_correction import publish_correction
from .publish_loops import publish_loops
from .publish_slam_map import publish_slam_map

#: Inbox payload marker for the coalescing path: "the real message is the current
#: self._latest". Mirrors the old ``Module._LATEST`` token -- the inbox carries
#: only a wake-up token, the keyframe itself lives in the single-slot holder.
_LATEST = object()
#: Inbox sentinel to unblock ``queue.get`` on ``stop()``. Mirrors ``Module._SENTINEL``.
_SENTINEL = object()


# --------------------------------------------------------------------------- #
def process_keyframe(engine: Engine, bus: LocalPubSub, kf: Keyframe, *,
                     publish_map: bool) -> None:
    """Run the full per-keyframe chain for one keyframe.

    Ordering is byte-identical to the old reactive chain:

    1. ``slam_submit`` -- ``engine.submit`` then ``engine.poll``; on a confirmed
       loop publish the ``loop.correction`` (was ``SlamStep`` ->
       ``PublishCorrection``).
    2. LIVE only (``publish_map``): ``publish_loops`` then ``publish_slam_map``,
       both polling INDEPENDENT engine channels AFTER the submit above (was
       ``PublishLoops`` -> ``PublishSlamMap``, which had to run after the submit;
       the old ``_RunCorrectionChain`` existed only to force that ordering inside
       the single-route reactive chain -- here the order is just the call order).

    Offline (``publish_map=False``) this is exactly ``slam_submit`` +
    conditional ``publish_correction`` -- the deterministic ``loop.correction``
    scoring path, byte-for-byte.
    """
    correction = slam_submit(engine, kf)
    if correction is not None:           # confirmed loop this keyframe
        publish_correction(bus, correction)
    if publish_map:
        # LIVE overlays: poll the loop-match funnel + the map overlay channels,
        # which reflect the submit just done. PublishLoops ran before
        # PublishSlamMap in the old chain; preserve that order (no functional
        # dependency between them, but keep it identical).
        publish_loops(engine, bus)
        publish_slam_map(engine, bus)


# --------------------------------------------------------------------------- #
class SlamWorker(threading.Thread):
    """One thread that drains a keyframe inbox and runs :func:`process_keyframe`.

    A plain procedural replacement for the old reactive ``Module`` -- it owns the
    inbox, the optional ``latest_only`` coalescing, END handling, and the
    downstream-END forwarding, all as explicit code rather than framework hooks.

    The caller wires it up by:

    * subscribing the worker to ``keyframe`` on the bus -- :meth:`submit_keyframe`
      (the bus handler) feeds the inbox;
    * :meth:`start` to launch the worker thread (subscribe BEFORE the keyframe
      source starts so nothing is missed);
    * waiting on :attr:`done` (set after END drains) then :meth:`stop`.
    """

    def __init__(self, bus: LocalPubSub, K: np.ndarray,
                 cfg: SlamConfig | None = None, *,
                 latest_only: bool = False, worker: bool = False,
                 publish_map: bool = False) -> None:
        super().__init__(name="slam", daemon=True)
        self.bus = bus
        self.publish_map = bool(publish_map)
        # capture_loops rides the SAME LIVE-only flag as publish_map: the
        # loop-closure funnel (slam.loop) is a live overlay, so the offline engine
        # never captures and the deterministic correction path stays byte-identical.
        self.engine = make_slam_engine(K, cfg or SlamConfig(), worker=worker,
                                       capture_loops=publish_map)

        self._latest_only = bool(latest_only)
        self._inbox: "queue.Queue" = queue.Queue()
        self._latest: Any = _SENTINEL          # single-slot newest unprocessed kf
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self.done = threading.Event()          #: set after END is handled
        self._emitted_end = False

        # END is forwarded to whatever this worker publishes (mirrors the old
        # Module.forwards_to): LIVE forwards to all three live topics, OFFLINE
        # only to loop.correction (it never publishes the overlays).
        self._downstream = ([topics.LOOP_CORRECTION, topics.SLAM_MAP,
                             topics.SLAM_LOOP] if self.publish_map
                            else [topics.LOOP_CORRECTION])

        # Subscribe the inbox feeder to the keyframe topic. Done in __init__ (not
        # start) so callers that publish between construction and start() -- the
        # in-process selftests do not, but it matches the old Module.on timing --
        # never lose a keyframe.
        self.bus.subscribe(topics.KEYFRAME, self._on_keyframe)

    # -- inbox feeders (run on the PUBLISHER's thread, kept cheap) ----------- #
    def _on_keyframe(self, msg: Any) -> None:
        """Bus handler for ``keyframe``: enqueue (coalescing or strict FIFO)."""
        if not self._latest_only:
            # Strict FIFO: every keyframe (and END) processed in order. Required
            # by the OFFLINE deterministic path -- dropping one corrupts the result.
            self._inbox.put(msg)
            return
        # Coalescing (LIVE): keep only the newest unprocessed keyframe in the
        # single slot; enqueue a wake-up token only when nothing was pending (one
        # token drives one drain) -- EXCEPT END, which always enqueues a token so
        # it is delivered even if it overwrites a pending data frame (losing the
        # last frame is fine; dropping END is not). Byte-for-byte the old
        # Module._coalesce, specialised to this worker's single keyframe topic.
        with self._latest_lock:
            pending = self._latest is not _SENTINEL
            self._latest = msg
            enqueue = (not pending) or (msg is END)
        if enqueue:
            self._inbox.put(_LATEST)

    # -- thread body -------------------------------------------------------- #
    def stop(self) -> None:
        self._stop.set()
        self._inbox.put(_SENTINEL)             # unblock the queue.get

    def run(self) -> None:
        # Close the engine on THIS thread when the loop exits (stop sentinel or
        # _stop), so a subprocess worker is reaped without a cross-thread race.
        try:
            self._loop()
        finally:
            self.engine.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            item = self._inbox.get()
            if item is _SENTINEL:
                break
            if item is _LATEST:
                # Coalescing inbox: the token says "drain the slot". Pull the
                # current newest keyframe (already drained by an earlier token ->
                # _SENTINEL -> skip).
                with self._latest_lock:
                    msg, self._latest = self._latest, _SENTINEL
                if msg is _SENTINEL:
                    continue
            else:
                msg = item                      # strict-FIFO payload
            if msg is END:
                self._handle_end()
                continue
            process_keyframe(self.engine, self.bus, msg,
                             publish_map=self.publish_map)

    def _handle_end(self) -> None:
        # Emit our own END downstream exactly once, then signal done. SLAM is a
        # single-input sink (one keyframe topic), so the first END is terminal --
        # no multi-input join to wait on (the old Module.expected_ends defaulted
        # to 1 for this case).
        if not self._emitted_end:
            self._emitted_end = True
            for topic in self._downstream:
                self.bus.publish(topic, END)
        self.done.set()


#: Public name kept for the three call sites (slam.main + the two external
#: in-process consumers). It is now the procedural worker, not a reactive Module.
SlamModule = SlamWorker
