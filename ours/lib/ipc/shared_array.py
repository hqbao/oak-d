"""Fixed-shape ring of shared-memory slots for one image / depth stream.

The IPC bus (:mod:`ours.lib.ipc.bus`) carries only metadata across the wire;
large numpy arrays travel through a :class:`SharedArrayRing` -- N pre-allocated
slots of identical shape / dtype that the producer fills in rotation and the
consumer reads by index.

Why a ring and not one-shot blocks
----------------------------------
A live OAK-D stream publishes ~20 frames/s into 3-4 subscribers (VIO, SLAM, UI,
maybe a tool). Allocating + unlinking a fresh ``SharedMemory`` per frame per
subscriber is far too slow (each ``shared_memory.SharedMemory("name", create=True)``
takes ~1-5 ms on macOS). A pre-allocated ring of ``N=8`` slots gives 0.4 s of
slack at 20 fps -- well above the 50-60 ms latest-only inbox cadence downstream,
so the producer rotation never catches a still-reading consumer (the consumer
copies out within its inbox handler, then the slot is free to reuse).

Concurrency model
-----------------
SINGLE-PRODUCER, MULTI-CONSUMER (publishers are processes, not threads -- the
capture process is the only writer to each ring). No locks: the producer
advances ``slot = seq % N`` monotonically. Consumers receive ``(slot, shape,
dtype)`` in the wire metadata; the bus publishes the metadata only AFTER the
slot has been fully written, so by the time a consumer reads the slot it is
already coherent (writes happen-before the pickled send on the socket; the
recv-side pickle load happens-after, all under the connection's TCP-like
ordering guarantee on the local socket).

Worst case: an extremely slow consumer is N frames behind, reads stale data.
That matches the downstream "latest-only" inbox semantics already used by the
live pipeline (the keypoints/triplet/odometry latest_only inboxes coalesce
backlog on purpose), so a stale read just drops a frame the consumer would have
discarded anyway.

Lifecycle
---------
The producer creates the ring with ``SharedArrayRing.create(name, slots, shape,
dtype)`` -- this allocates the underlying :class:`SharedMemory` blocks and
returns a handle that knows how to close + unlink them. Consumers attach to an
existing ring with ``SharedArrayRing.attach(name, slots, shape, dtype)`` -- they
do NOT unlink, only close. On Linux/macOS shared memory persists until every
attached process closes it AND the creator unlinks it.

The owning ``capture`` process is responsible for ``unlink`` at shutdown; if it
crashes the OS cleans up shared memory at the next reboot (acceptable for a
desktop dev tool; production would add an atexit fallback).
"""
from __future__ import annotations

import atexit
import struct
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SharedArrayRef:
    """Wire reference to one slot of a :class:`SharedArrayRing`.

    Travels inside an IPC wire message; the receiver uses it to copy the slot's
    contents out of shared memory into a private ``np.ndarray`` before any
    downstream task runs (see invariant 5 in ``docs/PROC4_ARCHITECTURE.md``).
    """

    ring_name: str
    slot: int
    shape: tuple[int, ...]
    dtype: str               # numpy dtype name, e.g. "uint8" / "float32"


class SharedArrayRing:
    """A ring of ``slots`` identically-shaped shared-memory blocks.

    Use :meth:`create` on the producer side, :meth:`attach` on each consumer.
    """

    def __init__(self, name: str, slots: int, shape: tuple[int, ...],
                 dtype: np.dtype, _shm_blocks, _owner: bool) -> None:
        self.name = name
        self.slots = int(slots)
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self._shm = _shm_blocks           # list[SharedMemory], len == slots
        self._owner = bool(_owner)
        #: Set after the first successful :meth:`unlink` so a stray atexit
        #: callback (defence in depth -- see :meth:`create`) is a no-op.
        self._unlinked = False
        # Pre-build np.ndarray views (one per slot) so the hot publish/poll path
        # only does a memcpy, no fresh ndarray construction.
        self._views: list[np.ndarray] = [
            np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
            for shm in self._shm
        ]

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #
    @classmethod
    def create(cls, name: str, slots: int, shape: Iterable[int],
               dtype) -> "SharedArrayRing":
        """Allocate ``slots`` shared blocks of shape × dtype. Producer-side.

        The slot's actual :class:`SharedMemory` is named ``"{name}.{i}"`` so
        consumers can attach by index.
        """
        shape = tuple(int(s) for s in shape)
        dt = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dt.itemsize
        # macOS limits POSIX shared-memory names to ~31 chars (PSHMNAMLEN, incl.
        # the leading '/'). Fail loudly here rather than getting a cryptic
        # ENAMETOOLONG from shm_open mid-build. Linux is much higher (NAME_MAX
        # 255), so this is the lower of the two and the right gate.
        longest = max(len(f"{name}.{i}") for i in range(int(slots)))
        if longest > 30:
            raise ValueError(
                f"shared-memory name {name!r} too long: '.{slots - 1}' suffix "
                f"would make it {longest} chars; macOS limit is 30. "
                f"Use a shorter endpoint / stream name.")
        blocks = []
        try:
            for i in range(int(slots)):
                blocks.append(shared_memory.SharedMemory(
                    name=f"{name}.{i}", create=True, size=nbytes))
        except FileExistsError as e:
            # Stale ring from a previous run that crashed without unlink.
            # Clean up partial allocations and re-raise so the caller can
            # cleanup_stale + retry.
            for shm in blocks:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:                                  # noqa: BLE001
                    pass
            raise RuntimeError(
                f"shared memory ring {name!r} already exists -- "
                f"call SharedArrayRing.cleanup_stale({name!r}, {slots}) first") from e
        ring = cls(name, slots, shape, dt, blocks, _owner=True)
        # Defence in depth: register an atexit fallback so an unhandled exception
        # path (or any creator-side teardown that forgets `unlink`) still frees
        # the shared blocks instead of leaking them as
        # `resource_tracker: There appear to be N leaked shared_memory objects`.
        # The fallback can't save us from SIGKILL (atexit doesn't run there) --
        # only the clean SIGTERM / exception paths -- but combined with the
        # SIGTERM handlers in `ours.proc.{capture,vio,slam}` this closes the
        # window. `_safe_unlink` is idempotent (guarded by `_unlinked`) so it is
        # a no-op when the caller has already unlinked explicitly.
        atexit.register(ring._safe_unlink)
        return ring

    @classmethod
    def attach(cls, name: str, slots: int, shape: Iterable[int],
               dtype) -> "SharedArrayRing":
        """Open an existing ring created by another process. Consumer-side.

        Passes ``track=False`` to :class:`SharedMemory` so the attaching
        process's :mod:`multiprocessing.resource_tracker` does NOT claim
        ownership of the block (the creator already does). Without this, the
        attacher would print "leaked shared_memory" warnings on exit even
        though only the creator should unlink -- a long-standing footgun in
        Python's stdlib (`issue38119 <https://bugs.python.org/issue38119>`_,
        fixed in 3.13 via the ``track`` parameter).
        """
        shape = tuple(int(s) for s in shape)
        dt = np.dtype(dtype)
        blocks = [shared_memory.SharedMemory(name=f"{name}.{i}", create=False,
                                             track=False)
                  for i in range(int(slots))]
        return cls(name, slots, shape, dt, blocks, _owner=False)

    @staticmethod
    def cleanup_stale(name: str, slots: int) -> None:
        """Unlink leftover shared blocks from a previous run that crashed.

        Best-effort; missing blocks are silently skipped (the normal case).
        """
        for i in range(int(slots)):
            try:
                shm = shared_memory.SharedMemory(
                    name=f"{name}.{i}", create=False)
            except FileNotFoundError:
                continue
            try:
                shm.close()
                shm.unlink()
            except Exception:                                      # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    # Producer / consumer ops
    # ------------------------------------------------------------------ #
    def slot_for(self, seq: int) -> int:
        """The slot index the producer uses for a given monotonic ``seq``."""
        return int(seq) % self.slots

    def write(self, slot: int, arr: np.ndarray) -> SharedArrayRef:
        """Copy ``arr`` into slot ``slot``; return the wire reference.

        ``arr`` must already be the ring's shape + dtype (the producer is
        expected to allocate the camera/depth at that shape, no per-frame
        reshaping). Raises ``ValueError`` otherwise so a wiring bug is caught at
        the boundary, not silently corrupted in shared memory.
        """
        if arr.shape != self.shape:
            raise ValueError(
                f"ring {self.name!r} shape {self.shape} != arr {arr.shape}")
        if arr.dtype != self.dtype:
            raise ValueError(
                f"ring {self.name!r} dtype {self.dtype} != arr {arr.dtype}")
        np.copyto(self._views[int(slot)], arr, casting="no")
        return SharedArrayRef(self.name, int(slot), self.shape, str(self.dtype))

    def read_copy(self, ref: SharedArrayRef) -> np.ndarray:
        """Return a private copy of the slot referenced by ``ref``.

        Always copies -- the caller owns the result and may keep it past the
        next producer rotation. Cheap (~0.1 ms for 640x400 uint8).
        """
        if ref.ring_name != self.name:
            raise ValueError(
                f"ref ring {ref.ring_name!r} != this ring {self.name!r}")
        return self._views[int(ref.slot)].copy()

    # ------------------------------------------------------------------ #
    # Teardown
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Detach this process from the shared blocks (idempotent).

        Does NOT unlink -- only :meth:`unlink` (creator) destroys the memory.
        Consumers always call :meth:`close` only.
        """
        for shm in self._shm:
            try:
                shm.close()
            except Exception:                                      # noqa: BLE001
                pass

    def unlink(self) -> None:
        """Destroy the underlying shared blocks. Creator-only, idempotent.

        After :meth:`unlink` no further reads / writes succeed. Always pair
        :meth:`close` after :meth:`unlink` in the creator to free the local
        handle. Sets ``self._unlinked`` so the atexit fallback registered in
        :meth:`create` becomes a no-op once the caller has cleaned up.
        """
        if not self._owner or self._unlinked:
            return
        self._unlinked = True
        for shm in self._shm:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:                                      # noqa: BLE001
                pass

    def _safe_unlink(self) -> None:
        """atexit fallback: unlink wrapped in try/except so interpreter teardown
        never raises out of the registered callback.

        Idempotent via :attr:`_unlinked`; called automatically when the
        interpreter exits normally (clean exit, SIGTERM with finally-block,
        unhandled exception). Does NOT run on SIGKILL or os._exit -- those paths
        rely on the OS / next reboot to reclaim the shared blocks.
        """
        try:
            self.unlink()
        except Exception:                                          # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Pickle helpers: SharedArrayRef is a plain dataclass so it pickles trivially.
# The ring itself is process-local and never pickled; the producer creates one,
# the consumer attaches its own (same name + dims) and they exchange only refs.
# --------------------------------------------------------------------------- #
def pack_ref(ref: SharedArrayRef) -> bytes:
    """Pack a ref into a compact binary (for cases that want to avoid pickle).

    Layout: 1B name-len | name | 4B slot | 2B ndim | ndim x 4B shape | 1B dtype-len | dtype.
    Used by the wire protocol when bandwidth is a concern; the default IpcBus
    just pickles ``SharedArrayRef`` and that is fast enough.
    """
    name = ref.ring_name.encode("utf-8")
    dt = ref.dtype.encode("utf-8")
    shape = ref.shape
    parts = [
        struct.pack("!B", len(name)),
        name,
        struct.pack("!I", int(ref.slot)),
        struct.pack("!H", len(shape)),
        *(struct.pack("!I", int(s)) for s in shape),
        struct.pack("!B", len(dt)),
        dt,
    ]
    return b"".join(parts)
