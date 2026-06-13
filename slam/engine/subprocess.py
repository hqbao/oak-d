"""Out-of-process engine: run the heavy solve in a separate process.

Why a process and not a thread
------------------------------
The SLAM ORB + pose-graph solve is mostly
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
  (``import slam.engine`` is depthai-free by invariant, so spawn is safe).

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
           in_q, out_q, ov_q, stop_evt, reset_evt, loops_q=None) -> None:
    """Child loop: own one map, run ``step`` per snapshot, emit results + overlay.

    Module-level (called from the module-level worker mains below) so the whole
    chain is picklable under the ``spawn`` start method used on macOS. A non-None
    ``step`` result goes on ``out_q`` (the correction the flow consumes); the
    cheap ``overlay`` snapshot goes on ``ov_q`` EVERY keyframe (latest-wins) so the
    live viewer always has the freshest map.

    ``loops_q`` carries the per-candidate loop-match CAPTURES (the funnel for the
    UI's loop-closure view). Loops are sporadic + must NOT be coalesced away (each
    is a distinct event explaining one candidate), so they are PUSHED individually
    (best-effort, non-blocking) rather than latest-wins; the parent drains the whole
    queue. The drain is guarded by ``getattr`` so a map without
    ``drain_loop_captures`` simply yields nothing.
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
        if loops_q is not None:
            drain = getattr(m, "drain_loop_captures", None)
            if drain is not None:
                for cap in drain():
                    try:
                        loops_q.put_nowait(cap)
                    except queue.Full:
                        pass               # parent fell behind; drop the oldest event


def _slam_worker_main(K, cfg, in_q, out_q, ov_q, stop_evt, reset_evt,
                      loops_q=None) -> None:
    """Child entry point for loop-closure SLAM (module-level => picklable).

    The subprocess engine is LIVE-only, so the map is built with
    ``capture_loops=True``: every verified candidate's match funnel is captured
    and pushed on ``loops_q`` for the UI's loop-closure view. (The OFFLINE path
    uses the in-process engine and never sets this, so determinism is untouched.)
    """
    from sky.slam.slam import SlamMap
    from .steps import slam_step, slam_overlay
    _serve(lambda: SlamMap(K, cfg, capture_loops=True), slam_step, slam_overlay,
           in_q, out_q, ov_q, stop_evt, reset_evt, loops_q)


class SubprocessEngine:
    """Engine that runs ``worker_main`` in a spawned process. See module docstring.

    The worker process is spawned **lazily** on the first :meth:`submit` (or an
    explicit :meth:`start`), NOT in ``__init__``. The live graph builds the engine
    while the OAK-D is open but the camera read loop has not started yet; spawning a
    fresh interpreter in that dead window left the device unread long enough to
    starve the XLink and trip the firmware watchdog (``mutex lock failed`` on the
    crashed device). Deferring the spawn to the first keyframe means the read loop is
    already streaming the device when the (heavy, decoupled) backend flow spawns the
    worker -- the spawn blocks only that flow's thread, never the device readers.
    The queues are created up front so a no-drop feeder can attach after ``start``.
    """

    def __init__(self, worker_main, K: np.ndarray, cfg) -> None:
        self._ctx = mp.get_context("spawn")
        self._worker_main = worker_main
        self._K = np.asarray(K)
        self._cfg = cfg
        self._in_q: "mp.Queue" = self._ctx.Queue(maxsize=1)   # one pending KF; newest wins
        self._out_q: "mp.Queue" = self._ctx.Queue(maxsize=2)  # corrections (drain to newest)
        self._ov_q: "mp.Queue" = self._ctx.Queue(maxsize=2)   # map overlay (drain to newest)
        # Loop-match captures (SLAM only): each is a distinct event (one verified
        # candidate's funnel), so this queue ACCUMULATES (not latest-wins) and the
        # parent drains it whole. Loops are sporadic, so a modest cap absorbs a
        # burst (a keyframe can close up to max_loops_per_kf at once) without
        # unbounded growth; if the parent ever falls behind the oldest is dropped.
        self._loops_q: "mp.Queue" = self._ctx.Queue(maxsize=64)
        self._stop_evt = self._ctx.Event()
        self._reset_evt = self._ctx.Event()
        self._proc = None
        self._closed = False
        self._failed = False

    def start(self) -> None:
        """Spawn the worker process (idempotent; no-op after close/failure).

        A spawn failure is non-fatal: the engine goes inert (``submit``/``poll`` are
        no-ops), so ours-ba/ours-slam keep running with the responsive marker and
        simply show no refined-map overlay -- never a crash on the flow thread.
        """
        if self._proc is not None or self._closed or self._failed:
            return
        try:
            self._proc = self._ctx.Process(
                target=self._worker_main,
                args=(self._K, self._cfg, self._in_q, self._out_q, self._ov_q,
                      self._stop_evt, self._reset_evt, self._loops_q),
                name="OursEngineWorker", daemon=True)
            self._proc.start()
        except Exception as e:                 # noqa: BLE001
            import sys
            self._proc = None
            self._failed = True
            print(f"[engine] worker spawn failed ({e}); running without the "
                  f"map overlay (marker unaffected)", file=sys.stderr)

    def submit(self, snapshot: Any) -> None:
        self.start()                           # lazy: spawn only once data flows
        if self._proc is None:                 # spawn failed -> inert engine
            return
        _put_latest(self._in_q, snapshot)

    def poll(self) -> Any:
        return None if self._proc is None else _drain_latest(self._out_q)

    def poll_overlay(self) -> Any:
        return None if self._proc is None else _drain_latest(self._ov_q)

    def poll_loops(self) -> list:
        """Drain ALL pending loop-match captures (events, not latest-wins)."""
        if self._proc is None:
            return []
        out: list = []
        while True:
            try:
                out.append(self._loops_q.get_nowait())
            except queue.Empty:
                break
        return out

    def reset(self) -> None:
        self._reset_evt.set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc is None:                 # never started -> nothing to reap
            return
        self._stop_evt.set()
        _put_latest(self._in_q, None)          # wake the blocking get with the sentinel
        self._proc.join(timeout=1.0)
        if self._proc.is_alive():              # wedged in a C-level solve: force it
            self._proc.terminate()
            self._proc.join(timeout=1.0)
        for q in (self._in_q, self._out_q, self._ov_q, self._loops_q):  # don't hang atexit feeder
            try:
                q.cancel_join_thread()
            except Exception:                  # noqa: BLE001
                pass
