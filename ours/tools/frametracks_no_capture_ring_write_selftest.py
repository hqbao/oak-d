#!/usr/bin/env python3
"""Regression guard for BUG A: ``frame.tracks`` must NEVER touch capture rings.

Background
----------
In the 4-process topology the capture process is the SINGLE writer of the
``gray_left`` / ``gray_right`` / ``depth_m`` shared-memory rings (see
``docs/PROC4_ARCHITECTURE.md`` §9 invariant 6). BUG A was a regression where
the VIO process republished ``frame.tracks`` carrying the gray + depth payload
in the *same* ring slots capture was actively writing. Two writers per slot =
torn reads on the UI side, and a single-writer-contract violation that breaks
the whole bridge invariant.

The fix restricted ``FrameTracks`` (and its wire counterpart
``WireFrameTracks``) to pure POD: per-frame track ids + pixel coords ONLY. The
UI sink joins them with ``FRAME_DEPTH`` by ``seq`` to render the overlay.

This test guards two structural invariants of the fix:

1. ``WireFrameTracks`` carries NO :class:`~ours.lib.ipc.shared_array.SharedArrayRef`
   field (and no field whose name ends in ``_ref``). If a future change adds an
   image / depth ref, this assertion fires before any process boots.
2. The VIO process publisher topology does NOT advertise ``frame.tracks`` as
   needing capture's image / depth rings. We inspect ``ours.proc.vio``'s
   ``_OUTPUT_TOPICS`` + the converter table to assert that the bridge converter
   registered for ``frame.tracks`` does NOT write into any capture-side ring.

Run::

    python -m ours.tools.frametracks_no_capture_ring_write_selftest
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.flows.bridge.converters import CONVERTERS                  # noqa: E402
from ours.flows.bridge.ring_registry import (                        # noqa: E402
    default_capture_specs, default_vio_specs,
)
from ours.lib.flow import topics                                     # noqa: E402
from ours.lib.ipc.messages import (                                  # noqa: E402
    WireFrameTracks, WireDepthFrame, WireKeyframe, WireImuCamPacket,
    WireCamSync,
)
from ours.lib.ipc.shared_array import SharedArrayRef                 # noqa: E402
from ours.proc import vio as vio_mod                                 # noqa: E402


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def _has_ring_ref_field(cls) -> tuple[bool, list[str]]:
    """Return (any_ref_field, list_of_ref_field_names) for a dataclass.

    A "ref" field is one whose type involves ``SharedArrayRef`` (image / depth
    payload), or whose name ends with ``_ref`` (the convention all
    image/depth-carrying wire messages follow). We check BOTH the type
    annotation and the name to catch a stringised annotation that bypasses the
    type check.
    """
    refs: list[str] = []
    for f in dataclasses.fields(cls):
        ann = f.type
        # `from __future__ import annotations` means f.type is a string.
        is_ref_type = (
            "SharedArrayRef" in str(ann) if not isinstance(ann, type)
            else ann is SharedArrayRef or issubclass(ann, SharedArrayRef)
        )
        if is_ref_type or f.name.endswith("_ref"):
            refs.append(f.name)
    return bool(refs), refs


def main() -> int:
    print("frametracks_no_capture_ring_write_selftest")

    # ------------------------------------------------------------------ #
    # 1. WireFrameTracks dataclass: NO shared-memory ref field
    # ------------------------------------------------------------------ #
    has_ref, ref_names = _has_ring_ref_field(WireFrameTracks)
    _check(not has_ref,
           f"WireFrameTracks carries NO SharedArrayRef field "
           f"(found: {ref_names})")

    # Positive control: wire messages that SHOULD carry refs really do (proves
    # the detector itself works). If this regresses, the negative test above is
    # silently meaningless.
    for cls in (WireDepthFrame, WireKeyframe, WireImuCamPacket, WireCamSync):
        has, names = _has_ring_ref_field(cls)
        _check(has, f"{cls.__name__} still carries at least one ring ref "
                    f"(found: {names})")

    # ------------------------------------------------------------------ #
    # 2. FrameTracks fields are pure POD: seq, ts_ns, ids, points
    # ------------------------------------------------------------------ #
    field_names = sorted(f.name for f in dataclasses.fields(WireFrameTracks))
    _check(field_names == ["ids", "points", "seq", "ts_ns"],
           f"WireFrameTracks has exactly (seq, ts_ns, ids, points) "
           f"(got: {field_names})")

    # ------------------------------------------------------------------ #
    # 3. VIO process topology: FRAME_TRACKS is published, but the converter
    #    must not need any capture-side image / depth ring slot.
    # ------------------------------------------------------------------ #
    _check(topics.FRAME_TRACKS in vio_mod._OUTPUT_TOPICS,
           "vio process declares FRAME_TRACKS as an output (it republishes it)")

    cap_specs = default_capture_specs(endpoint="oak.capture",
                                      width=640, height=400)
    vio_specs = default_vio_specs(endpoint="oak.vio",
                                  width=640, height=400)
    cap_ring_names = {s.name for s in cap_specs}
    vio_ring_names = {s.name for s in vio_specs}

    # ``frame.tracks`` is NOT in either ring spec set: the publisher converter
    # writes nothing into a ring, and the consumer side has nothing to attach.
    _check(not any("frame.tracks" in n or "tracks" in n for n in cap_ring_names),
           "no capture-side ring is named for frame.tracks "
           f"(have: {sorted(cap_ring_names)})")
    _check(not any("frame.tracks" in n or "tracks" in n for n in vio_ring_names),
           "no vio-side ring is named for frame.tracks "
           f"(have: {sorted(vio_ring_names)})")

    # The converter for FRAME_TRACKS exists (otherwise the bridge can't publish
    # the topic at all), but it must not write into capture's rings. We prove
    # this by calling the to_wire converter with an EMPTY ring registry: if
    # it tried to access any ring slot it would raise KeyError.
    from ours.flows.bridge.ring_registry import RingRegistry
    import numpy as np
    from ours.lib.flow.messages import FrameTracks

    to_wire_fn, to_local_fn = CONVERTERS[topics.FRAME_TRACKS]
    msg = FrameTracks(
        seq=0, ts_ns=0,
        ids=np.array([1, 2, 3], dtype=np.int64),
        points=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )
    empty_rings = RingRegistry()      # no rings registered at all
    try:
        wm = to_wire_fn(msg, empty_rings, "oak.capture")
    except KeyError as e:
        _check(False, f"to_wire(FRAME_TRACKS) touched a ring slot: {e!r}")
        return 1                       # unreachable, _check raises
    _check(isinstance(wm, WireFrameTracks),
           "to_wire(FRAME_TRACKS) returns a WireFrameTracks with no rings present")

    # And the same in reverse: to_local does not need to read any ring either.
    try:
        local = to_local_fn(wm, empty_rings)
    except KeyError as e:
        _check(False, f"to_local(FRAME_TRACKS) touched a ring slot: {e!r}")
        return 1
    _check(local.seq == 0 and local.ids.tolist() == [1, 2, 3],
           "to_local(FRAME_TRACKS) round-trip preserved ids without ring read")

    # ------------------------------------------------------------------ #
    # 4. Positive control: a wire converter that legitimately needs a capture
    #    ring (FRAME_DEPTH) MUST raise on an empty registry. Proves the empty-
    #    ring trick above is a real test, not a no-op.
    # ------------------------------------------------------------------ #
    from ours.lib.flow.messages import DepthFrame
    depth_msg = DepthFrame(
        seq=0, ts_ns=0,
        gray_left=np.zeros((400, 640), dtype=np.uint8),
        depth_m=np.zeros((400, 640), dtype=np.float32),
    )
    depth_to_wire, _ = CONVERTERS[topics.FRAME_DEPTH]
    raised = False
    try:
        depth_to_wire(depth_msg, empty_rings, "oak.capture")
    except KeyError:
        raised = True
    _check(raised,
           "control: to_wire(FRAME_DEPTH) DOES need a ring (proves the test "
           "above is real)")

    print("\nALL FRAMETRACKS NO-CAPTURE-RING SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
