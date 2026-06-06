"""Out-of-process engine: run the heavy solve in a separate process.

Why a process and not a thread
------------------------------
The windowed-BA refine and the SLAM ORB + pose-graph solve are mostly
**pure-Python** (only the inner ``np.linalg.solve`` releases the GIL). Run on a
thread they steal ~17-30 % of the GIL from the device frame-read loop (measured on
``fast_push_15s``); a starved read loop drains its camera queues to the latest
frame and drops the backlog, so each surviving frame spans a larger motion, the
frame-to-frame PnP under-measures the translation, and the displayed path
stalls / undershoots. Bare ``ours`` has no backend thread and never
shows this. A separate process removes the GIL contention entirely: the read loop
owns its interpreter and keeps full frame rate under solve load.

Discipline
----------
* **Input** queue is ``maxsize=1`` latest-wins: a busy worker drops the older
  pending keyframe (the overlay only ever needs the freshest map; the responsive
  marker rides ``pose.odom`` and never waits on this).
* **Output** queue is drained to the newest result on ``poll``.
* **Reset** is a dedicated :class:`multiprocessing.Event` (not an input-queue
  sentinel) so the UI "clear keyframes" can never be dropped by latest-wins.
* The child imports the heavy map **lazily inside the worker function** (so the
  child bootstrap stays light) and only via direct module paths, never depthai/Qt
  (``import ours.lib`` is depthai-free by invariant, so spawn is safe).

Teardown is the important part under a long-lived Qt parent: ``close`` sets the
stop event, pushes the ``None`` sentinel, ``join``s, and -- if the child is wedged
in a C-level solve -- ``terminate``s and joins again, then ``cancel_join_thread``
on both queues so a buffered item can't hang the parent's atexit queue-feeder.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any, Callable

import numpy as np


def _put_latest(q: "mp.Queue", item: Any) -> None:
    """Put ``item`` keeping only the newest entry (drop the older one if full)."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def _drain_latest(q: "mp.Queue") -> Any:
    """Return the newest item on ``q`` (discarding any older ones), or ``None``."""
    res = None
    while True:
        try:
            res = q.get_nowait()
        except queue.Empty:
            break
    return res


def _serve(make_map: Callable[[], Any], step, overlay,
           in_q, out_q, ov_q, stop_evt, reset_evt) -> None:
    """Child loop: own one map, run ``step`` per snapshot, emit results + overlay.

    Module-level (called from the module-level worker mains below) so the whole
    chain is picklable under the ``spawn`` start method used on macOS. A non-None
    ``step`` result goes on ``out_q`` (the correction the flow consumes); the
    cheap ``overlay`` snapshot goes on ``ov_q`` EVERY keyframe (latest-wins) so the
    live viewer always has the freshest map.
    """
    m = make_map()
    while not stop_evt.is_set():
        if reset_evt.is_set():
            m = make_map()
            reset_evt.clear()
        try:
            snap = in_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if snap is None:                      # stop sentinel
            break
        res = step(m, snap)
        if res is not None:
            _put_latest(out_q, res)
        if overlay is not None:
            _put_latest(ov_q, overlay(m))


def _ba_worker_main(K, cfg, in_q, out_q, ov_q, stop_evt, reset_evt) -> None:
    """Child entry point for windowed BA (module-level => picklable under spawn)."""
    from ..backend.windowed import WindowedBAMap
    from .steps import ba_step, ba_overlay
    _serve(lambda: WindowedBAMap(K, cfg), ba_step, ba_overlay,
           in_q, out_q, ov_q, stop_evt, reset_evt)


def _slam_worker_main(K, cfg, in_q, out_q, ov_q, stop_evt, reset_evt) -> None:
    """Child entry point for loop-closure SLAM (module-level => picklable)."""
    from ..loop.slam import SlamMap
    from .steps import slam_step, slam_overlay
    _serve(lambda: SlamMap(K, cfg), slam_step, slam_overlay,
           in_q, out_q, ov_q, stop_evt, reset_evt)


class SubprocessEngine:
    """Engine that runs ``worker_main`` in a spawned process. See module docstring."""

    def __init__(self, worker_main, K: np.ndarray, cfg) -> None:
        ctx = mp.get_context("spawn")
        self._in_q: "mp.Queue" = ctx.Queue(maxsize=1)    # one pending KF; newest wins
        self._out_q: "mp.Queue" = ctx.Queue(maxsize=2)   # corrections (drain to newest)
        self._ov_q: "mp.Queue" = ctx.Queue(maxsize=2)    # map overlay (drain to newest)
        self._stop_evt = ctx.Event()
        self._reset_evt = ctx.Event()
        self._proc = ctx.Process(
            target=worker_main,
            args=(K, cfg, self._in_q, self._out_q, self._ov_q,
                  self._stop_evt, self._reset_evt),
            name="OursEngineWorker", daemon=True)
        self._proc.start()
        self._closed = False

    def submit(self, snapshot: Any) -> None:
        _put_latest(self._in_q, snapshot)

    def poll(self) -> Any:
        return _drain_latest(self._out_q)

    def poll_overlay(self) -> Any:
        return _drain_latest(self._ov_q)

    def reset(self) -> None:
        self._reset_evt.set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_evt.set()
        _put_latest(self._in_q, None)          # wake the blocking get with the sentinel
        self._proc.join(timeout=1.0)
        if self._proc.is_alive():              # wedged in a C-level solve: force it
            self._proc.terminate()
            self._proc.join(timeout=1.0)
        for q in (self._in_q, self._out_q, self._ov_q):  # don't hang atexit feeder
            try:
                q.cancel_join_thread()
            except Exception:                  # noqa: BLE001
                pass
