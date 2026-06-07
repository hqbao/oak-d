"""Ring registry: every cross-process numpy stream pre-declares its shape.

A :class:`~ours.lib.ipc.shared_array.SharedArrayRing` must be created with a
FIXED shape + dtype so the producer and every consumer can attach to the same
memory layout. The capture process publishes ~3 numpy streams (gray_left,
gray_right, depth_m); VIO additionally publishes (gray_left, depth_m,
track_ids/track_px implicit in keyframe). All consumers of one process need to
know the same registry to attach correctly.

This module centralises that registry so a wiring mistake (subscriber attaches
to wrong shape) is caught at process-boot, not at first frame.

Convention
----------
Ring names are namespaced ``"<endpoint>.<stream>"``, e.g. ``"oak.capture.gray_left"``,
so two publishers on different endpoints cannot accidentally collide.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...lib.ipc.shared_array import SharedArrayRing


@dataclass(frozen=True)
class RingSpec:
    """One ring's declared layout.

    * ``name`` -- ring name (one shared-memory block per slot, suffix ``.{i}``).
    * ``slots`` -- ring depth. 8 slots @ 20 fps = 0.4 s of slack.
    * ``shape`` -- array shape (e.g. ``(400, 640)``).
    * ``dtype`` -- numpy dtype name (e.g. ``"uint8"``, ``"float32"``).
    """

    name: str
    slots: int
    shape: tuple[int, ...]
    dtype: str


class RingRegistry:
    """A keyed bundle of :class:`SharedArrayRing` instances for one process.

    The producer side calls :meth:`create_all` once at boot to allocate every
    ring; the consumer side calls :meth:`attach_all` to attach existing rings.
    Both then use :meth:`get` to look up a ring by stream name (e.g.
    ``"gray_left"``) inside the bridge converters.
    """

    def __init__(self) -> None:
        self._rings: dict[str, SharedArrayRing] = {}
        self._owner = False

    # ------------------------------------------------------------------ #
    def create_all(self, specs: list[RingSpec]) -> "RingRegistry":
        """Allocate every ring in ``specs``. Producer-side."""
        self._owner = True
        for spec in specs:
            try:
                SharedArrayRing.cleanup_stale(spec.name, spec.slots)
                self._rings[spec.name] = SharedArrayRing.create(
                    spec.name, spec.slots, spec.shape, np.dtype(spec.dtype))
            except Exception:
                # Best-effort cleanup of partials before re-raising.
                self.close()
                raise
        return self

    def attach_all(self, specs: list[RingSpec]) -> "RingRegistry":
        """Open every ring in ``specs``. Consumer-side."""
        self._owner = False
        for spec in specs:
            self._rings[spec.name] = SharedArrayRing.attach(
                spec.name, spec.slots, spec.shape, np.dtype(spec.dtype))
        return self

    # ------------------------------------------------------------------ #
    def get(self, name: str) -> SharedArrayRing:
        """Look up a ring by its name (e.g. ``"oak.capture.gray_left"``)."""
        try:
            return self._rings[name]
        except KeyError:
            raise KeyError(
                f"ring {name!r} not in registry "
                f"(have: {sorted(self._rings)})") from None

    def has(self, name: str) -> bool:
        return name in self._rings

    def names(self) -> list[str]:
        return list(self._rings)

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Close every attached ring (consumer + producer). Idempotent."""
        for ring in list(self._rings.values()):
            try:
                ring.close()
            except Exception:                                      # noqa: BLE001
                pass

    def unlink(self) -> None:
        """Unlink every ring (producer-only). Pair with :meth:`close`."""
        if not self._owner:
            return
        for ring in list(self._rings.values()):
            try:
                ring.unlink()
            except Exception:                                      # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Default rings published by the ``capture`` endpoint at the canonical 640x400.
# Other endpoints register their own as needed (see ours.proc.vio etc.).
# --------------------------------------------------------------------------- #
def default_capture_specs(endpoint: str = "oak.capture",
                          width: int = 640, height: int = 400,
                          slots: int = 64) -> list[RingSpec]:
    """Canonical specs for the capture process at the given resolution.

    Three streams cross the boundary:

    * ``<endpoint>.gray_left`` -- rectified-left image, uint8 (H, W).
    * ``<endpoint>.gray_right`` -- raw right image, uint8 (H, W).
    * ``<endpoint>.depth_m`` -- metric depth (rectified-left grid), float32 (H, W).

    ``slots=64`` must stay strictly greater than the IpcServerBus outbound cap
    (default 32) so that a wire message queued in a subscriber's outbox still
    points at a slot the producer has not yet overwritten. See
    ``docs/PROC4_ARCHITECTURE.md`` §9 invariant 6.
    """
    h, w = int(height), int(width)
    return [
        RingSpec(f"{endpoint}.gray_left",  slots, (h, w), "uint8"),
        RingSpec(f"{endpoint}.gray_right", slots, (h, w), "uint8"),
        RingSpec(f"{endpoint}.depth_m",    slots, (h, w), "float32"),
    ]


def default_vio_specs(endpoint: str = "oak.vio",
                      width: int = 640, height: int = 400,
                      slots: int = 64) -> list[RingSpec]:
    """Canonical specs for the VIO process's keyframe rings.

    VIO republishes the keyframe image + depth so SLAM and the UI can read them
    without going back to capture (capture's slots cycle at frame rate; the
    keyframe rate is much lower so a dedicated ring keeps a fresh keyframe slot
    for SLAM to ingest at its own pace). ``slots=64`` for the same reason as
    :func:`default_capture_specs`.

    ``frame.tracks`` carries ONLY the per-frame ids + pixels (pure POD); the
    gray + depth needed to render the overlay are read from capture's
    ``FRAME_DEPTH`` (capture is the single writer of those rings). So no
    image / depth ring lives here for the tracks topic.
    """
    h, w = int(height), int(width)
    return [
        RingSpec(f"{endpoint}.kf_gray",  slots, (h, w), "uint8"),
        RingSpec(f"{endpoint}.kf_depth", slots, (h, w), "float32"),
    ]


#: Convenience: the default-capture specs at the canonical resolution. Procs
#: that need a different resolution call ``default_capture_specs(width=..., height=...)``.
DEFAULT_RING_SPECS = default_capture_specs()
