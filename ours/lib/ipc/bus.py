"""Cross-process pub/sub bus -- the wire side of :mod:`ours.lib.flow.pubsub`.

Each *publisher process* exposes ONE :class:`IpcServerBus` on a Unix-domain
socket (macOS / Linux); each subscriber process opens ONE :class:`IpcClientBus`
per publisher it cares about. The API mirrors the in-process
:class:`ours.lib.flow.pubsub.Bus` so the existing flows + the new bridge flows
look almost identical at the call site::

    # publisher
    bus = IpcServerBus("oak.capture")
    bus.start()
    bus.publish("imucam.sample", wire_msg)
    ...
    bus.close()

    # subscriber
    bus = IpcClientBus("oak.capture")
    bus.subscribe("imucam.sample", on_imucam)
    bus.subscribe("calib.bundle", on_calib)
    bus.start()                             # blocks until close() / END
    ...

Wire protocol
-------------
- Unix-domain socket via :class:`multiprocessing.connection.Listener` /
  :func:`multiprocessing.connection.Client`. Auth disabled (``authkey=None``)
  on Linux/macOS: the socket file lives under ``$TMPDIR/ours_ipc/`` with
  mode 0600 so only the current uid can connect. (Acceptable for a desktop
  dev tool; production would add HMAC.)
- Each connection starts with a one-line :class:`Connection.send` handshake
  carrying ``{"role": "subscriber", "topics": [...]}``. The server records
  the topic list and only forwards matching messages to that connection.
- Every published message is a 3-tuple ``("M", topic, wire_msg)`` pickled by
  :meth:`Connection.send`. Control sentinels are ``("BYE",)``.
- Retained topics (e.g. ``calib.bundle``): when a subscriber connects, the
  server first replays the latest cached message for every retained topic the
  subscriber is interested in, so booting late never misses a one-shot
  configuration.

Threading
---------
``IpcServerBus`` uses one *accept thread* and one *fan-out thread per
connection*. Publish is non-blocking from the caller's perspective: it drops
the wire message into each subscriber's outbound :class:`queue.Queue` (bounded,
latest-wins on overflow so a stuck consumer cannot stall the publisher).

``IpcClientBus`` uses one *receive thread* per upstream connection that reads
messages and invokes the user-registered handlers on that thread (the same
actor model the in-proc Bus uses -- handlers typically drop into a flow's
inbox, so the real work runs on the consuming flow's own thread).

OFFLINE replay never imports this module; the single-process
``ours.app.run_replay`` path is untouched.
"""
from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
from collections import defaultdict
from multiprocessing import connection
from typing import Any, Callable

LOG = logging.getLogger("ours.ipc.bus")

Handler = Callable[[Any], None]

# A subscriber's outbound queue is bounded; if it fills (slow consumer) the
# producer drops the oldest message and inserts the newest. This matches the
# "latest-only" inbox semantics already used by the live in-proc flows, and
# guarantees ``publish`` never blocks the producer thread.
_DEFAULT_OUTBOUND_CAP = 32


# --------------------------------------------------------------------------- #
# Endpoint helpers
# --------------------------------------------------------------------------- #
def _endpoint_path(name: str) -> str:
    """Resolve a logical endpoint name to a Unix-domain socket path.

    All sockets live under ``$TMPDIR/ours_ipc/`` (one directory per user,
    chmod 0700). The socket itself is created by :class:`Listener` with
    chmod 0600 enforced below.
    """
    root = os.path.join(tempfile.gettempdir(), "ours_ipc")
    try:
        os.makedirs(root, mode=0o700, exist_ok=True)
    except OSError:
        pass
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    return os.path.join(root, f"{name}.sock")


# --------------------------------------------------------------------------- #
# Server side -- the publishing process
# --------------------------------------------------------------------------- #
class _ConnState:
    """Per-subscriber-connection state held by the server."""

    __slots__ = ("conn", "topics", "outbox", "thread", "alive")

    def __init__(self, conn: "connection.Connection",
                 topics: set[str], cap: int) -> None:
        self.conn = conn
        self.topics: set[str] = topics
        self.outbox: "queue.Queue" = queue.Queue(maxsize=cap)
        self.thread: threading.Thread | None = None
        self.alive = True


class IpcServerBus:
    """A pub/sub server: one Unix-domain socket, many subscribers."""

    def __init__(self, endpoint: str, *,
                 retain_topics: set[str] | None = None,
                 outbound_cap: int = _DEFAULT_OUTBOUND_CAP,
                 blocking: bool = True) -> None:
        """Construct an unstarted server.

        ``blocking`` controls back-pressure semantics when a subscriber's
        outbox fills up:

        * ``True`` (default) -- :meth:`publish` blocks the caller until space
          is available. The producer is throttled to the slowest consumer's
          rate; nothing is dropped. This is what offline replay + the smoke
          tests need, where every frame must reach VIO for correctness.
        * ``False`` -- the OLDEST queued message is dropped and the new one
          is appended (latest-wins semantics). Use for live operation where
          a stale marker beats a stalled producer.

        Either way, the underlying outbox is bounded at ``outbound_cap`` to
        keep the in-flight memory footprint predictable; in blocking mode the
        bound just turns into a throttle instead of a drop.
        """
        self.endpoint = endpoint
        self._path = _endpoint_path(endpoint)
        self._retain_topics = set(retain_topics or ())
        self._retained: dict[str, Any] = {}      # latest msg per retained topic
        self._outbound_cap = int(outbound_cap)
        self._blocking = bool(blocking)
        self._listener: connection.Listener | None = None
        self._accept_thread: threading.Thread | None = None
        self._conns: list[_ConnState] = []
        self._lock = threading.Lock()
        self._stopped = threading.Event()

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Bind the socket and start accepting subscribers (idempotent).

        Thread-safe: two flows that share one ``IpcServerBus`` may both call
        ``start`` from their own ``run`` thread (see ``ours.proc.vio`` which
        runs two :class:`IpcPublisherFlow` against one server). Only the first
        caller binds; the rest return immediately.
        """
        with self._lock:
            if self._listener is not None:
                return
            # Clear any stale socket from a previous crash.
            try:
                if os.path.exists(self._path):
                    os.unlink(self._path)
            except OSError:
                pass
            # AF_UNIX listener, no authkey -- access controlled by FS perms.
            self._listener = connection.Listener(
                address=self._path, family="AF_UNIX")
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                pass
            self._accept_thread = threading.Thread(
                target=self._accept_loop, name=f"ipc-{self.endpoint}-accept",
                daemon=True)
            self._accept_thread.start()
            LOG.info("IpcServerBus[%s] listening on %s",
                     self.endpoint, self._path)

    # ------------------------------------------------------------------ #
    def publish(self, topic: str, msg: Any) -> None:
        """Deliver ``msg`` to every subscriber registered for ``topic``.

        Non-blocking from the caller's perspective: messages drop into each
        connection's bounded outbox; a fan-out thread drains it onto the
        socket. If a subscriber's outbox is full (slow consumer), the OLDEST
        queued message is dropped so the latest always wins.

        After :meth:`close` (``self._stopped`` is set) publishes become
        no-ops -- the in-flight queue is still drained by the fan-out
        threads, but no further messages enter the system.
        """
        if self._stopped.is_set():
            return
        if topic in self._retain_topics:
            with self._lock:
                self._retained[topic] = msg
        with self._lock:
            conns = [c for c in self._conns if c.alive and topic in c.topics]
        for c in conns:
            self._enqueue(c, ("M", topic, msg))

    def publish_end(self, topic: str) -> None:
        """Wire-side END sentinel for one topic (replay path)."""
        from .messages import WireEnd
        self.publish(topic, WireEnd(topic))

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Stop accepting + close every subscriber connection. Idempotent.

        Drains each subscriber's outbox before tearing down the socket: every
        message that the publisher has already enqueued is delivered, then a
        BYE marker, then the connection is closed. This is what the offline
        replay path needs -- if a producer published N frames and immediately
        closes, we must NOT discard those N frames just because the close ran
        before the fanout thread caught up.

        We do gate further :meth:`publish` calls (via ``_stopped``) so new
        publishes after close are no-ops, but EVERYTHING already in flight
        gets through.
        """
        if self._stopped.is_set():
            return
        self._stopped.set()
        # Stop accepting.
        if self._listener is not None:
            try:
                self._listener.close()
            except Exception:                                      # noqa: BLE001
                pass
            self._listener = None
        # Snapshot connections; signal each to drain + BYE.
        with self._lock:
            conns = list(self._conns)
            self._conns.clear()
        for c in conns:
            try:
                c.outbox.put(("BYE",), timeout=1.0)
            except queue.Full:
                # The outbox was already at capacity; force-make-room then BYE.
                try:
                    c.outbox.get_nowait()
                except queue.Empty:
                    pass
                try:
                    c.outbox.put_nowait(("BYE",))
                except queue.Full:
                    pass
        # Join each fanout thread -- it processes the rest of its outbox in
        # order, hits BYE, sends it, and exits. Generous timeout per conn so a
        # slow subscriber gets its data.
        for c in conns:
            if c.thread is not None:
                c.thread.join(timeout=5.0)
            c.alive = False
            try:
                c.conn.close()
            except Exception:                                      # noqa: BLE001
                pass
        # Best-effort socket unlink.
        try:
            if os.path.exists(self._path):
                os.unlink(self._path)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _accept_loop(self) -> None:
        listener = self._listener
        while not self._stopped.is_set() and listener is not None:
            try:
                conn = listener.accept()
            except OSError:                       # listener closed -> exit
                return
            except Exception as e:                                 # noqa: BLE001
                LOG.warning("IpcServerBus[%s] accept failed: %s",
                            self.endpoint, e)
                return
            # Read the subscribe handshake (one short pickle, blocking).
            try:
                hello = conn.recv()
            except Exception as e:                                 # noqa: BLE001
                LOG.warning("IpcServerBus[%s] handshake failed: %s",
                            self.endpoint, e)
                try:
                    conn.close()
                except Exception:                                  # noqa: BLE001
                    pass
                continue
            topics = set(hello.get("topics", ())) if isinstance(hello, dict) else set()
            state = _ConnState(conn, topics, self._outbound_cap)
            state.thread = threading.Thread(
                target=self._fanout_loop, args=(state,),
                name=f"ipc-{self.endpoint}-out", daemon=True)
            with self._lock:
                self._conns.append(state)
                # Replay retained messages this subscriber asked for, in
                # declaration order (calib first, then everything else).
                retained = [(t, m) for t, m in self._retained.items()
                            if t in topics]
            for t, m in retained:
                self._enqueue(state, ("M", t, m))
            state.thread.start()
            LOG.info("IpcServerBus[%s] subscriber connected for %s",
                     self.endpoint, sorted(topics))

    def _enqueue(self, state: _ConnState, item: tuple) -> None:
        """Drop ``item`` into ``state.outbox``.

        Behaviour on a full outbox depends on the server's ``blocking`` flag:

        * blocking -- :meth:`Queue.put` with a short timeout in a loop, giving
          up only if the subscriber dies or the server is being shut down.
          Throttles the producer to the slowest consumer (offline / replay).
        * non-blocking -- drop oldest, append newest (live latest-wins).
        """
        if not state.alive:
            return
        if self._blocking:
            while True:
                try:
                    state.outbox.put(item, timeout=0.1)
                    return
                except queue.Full:
                    if not state.alive or self._stopped.is_set():
                        return
            return
        # Latest-wins (non-blocking).
        try:
            state.outbox.put_nowait(item)
        except queue.Full:
            try:
                state.outbox.get_nowait()
            except queue.Empty:
                pass
            try:
                state.outbox.put_nowait(item)
            except queue.Full:
                pass

    def _fanout_loop(self, state: _ConnState) -> None:
        """Drain ``state.outbox`` onto the socket until BYE / EOF.

        Order matters: BYE is checked BEFORE ``state.alive``. ``alive`` is
        only set False by send-errors (BrokenPipe etc.); :meth:`close` does
        NOT set it, because we must let the drain finish first. So a normal
        server shutdown:

            publisher -> publish N items -> close()
            close puts BYE on the queue, joins this thread
            this thread: pops item 1..N, sends them, pops BYE, sends BYE, exits

        ...delivers every message that was in flight at close time.
        """
        try:
            while True:
                item = state.outbox.get()
                if item[0] == "BYE":
                    try:
                        state.conn.send(("BYE",))
                    except Exception:                              # noqa: BLE001
                        pass
                    return
                if not state.alive:
                    # Set by a previous send-error: the conn is dead, no point
                    # spending time on remaining items.
                    return
                try:
                    state.conn.send(item)
                except (BrokenPipeError, EOFError, OSError):
                    state.alive = False
                    return
                except Exception as e:                             # noqa: BLE001
                    LOG.warning("IpcServerBus[%s] send failed: %s",
                                self.endpoint, e)
                    state.alive = False
                    return
        finally:
            try:
                state.conn.close()
            except Exception:                                      # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Client side -- the subscribing process
# --------------------------------------------------------------------------- #
class IpcClientBus:
    """A pub/sub client: connect to one publisher, dispatch to local handlers."""

    def __init__(self, endpoint: str, *,
                 connect_timeout_s: float = 10.0,
                 connect_retry_s: float = 0.2) -> None:
        self.endpoint = endpoint
        self._path = _endpoint_path(endpoint)
        self._connect_timeout_s = float(connect_timeout_s)
        self._connect_retry_s = float(connect_retry_s)
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._conn: "connection.Connection | None" = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        return self._error

    # ------------------------------------------------------------------ #
    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register ``handler`` for every message on ``topic``.

        Must be called BEFORE :meth:`start` (the topic list is sent in the
        connect handshake). Calling after start raises ``RuntimeError``.
        """
        if self._started.is_set():
            raise RuntimeError(
                f"IpcClientBus[{self.endpoint}] already started -- "
                f"subscribe before start()")
        self._subs[topic].append(handler)

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Connect to the server and start the receive thread.

        Blocks until either the connection succeeds, the connect timeout
        elapses (raises ``TimeoutError``), or the connect fails (raises
        ``ConnectionError``). The receive thread then runs in the background;
        :meth:`stop` to join it.
        """
        if self._started.is_set():
            return
        conn = self._connect_with_retry()
        # Send the handshake (subscribed topics).
        try:
            conn.send({"role": "subscriber",
                       "topics": list(self._subs.keys())})
        except Exception as e:                                     # noqa: BLE001
            try:
                conn.close()
            except Exception:                                      # noqa: BLE001
                pass
            raise ConnectionError(
                f"IpcClientBus[{self.endpoint}] handshake failed: {e}") from e
        self._conn = conn
        self._thread = threading.Thread(
            target=self._recv_loop, name=f"ipc-{self.endpoint}-in",
            daemon=True)
        self._thread.start()
        self._started.set()

    def stop(self, timeout: float = 2.0) -> None:
        """Close the connection and join the receive thread. Idempotent."""
        self._stop.set()
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:                                      # noqa: BLE001
                pass
            self._conn = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------ #
    def _connect_with_retry(self) -> "connection.Connection":
        """Retry :func:`connection.Client` until the socket file exists.

        The publisher may not have called :meth:`IpcServerBus.start` yet when
        the subscriber boots; rather than crash, we wait up to
        ``connect_timeout_s`` for the socket to appear.
        """
        import time
        deadline = time.monotonic() + self._connect_timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return connection.Client(address=self._path, family="AF_UNIX")
            except (FileNotFoundError, ConnectionRefusedError) as e:
                last_err = e
                time.sleep(self._connect_retry_s)
            except Exception as e:                                 # noqa: BLE001
                last_err = e
                time.sleep(self._connect_retry_s)
        raise TimeoutError(
            f"IpcClientBus[{self.endpoint}] could not connect to "
            f"{self._path} within {self._connect_timeout_s}s "
            f"(last error: {last_err})")

    def _recv_loop(self) -> None:
        conn = self._conn
        try:
            while not self._stop.is_set() and conn is not None:
                try:
                    item = conn.recv()
                except EOFError:
                    return                            # publisher closed
                except (OSError, BrokenPipeError):
                    return
                except Exception as e:                             # noqa: BLE001
                    self._error = f"recv failed: {e}"
                    LOG.warning("IpcClientBus[%s] recv failed: %s",
                                self.endpoint, e)
                    return
                if not isinstance(item, tuple) or not item:
                    continue
                if item[0] == "BYE":
                    return
                if item[0] != "M":
                    continue
                _, topic, msg = item
                for h in list(self._subs.get(topic, ())):
                    try:
                        h(msg)
                    except Exception as e:                         # noqa: BLE001
                        # Handler errors must not kill the receive loop; log
                        # and keep delivering subsequent messages.
                        LOG.warning("IpcClientBus[%s] handler for %s raised: %s",
                                    self.endpoint, topic, e)
        finally:
            self._started.clear()
