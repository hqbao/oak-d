"""``ours.lib.misc`` -- shared, dependency-light helpers.

These are the small cross-cutting utilities that are neither a VIO algorithm nor
part of the flow architecture, grouped here so the library root holds only
packages:

    frames    NED/FRD/optical frame conventions + rigid-body transforms
    geometry  RGB-D back-projection primitives (pure numpy)
    pose      Pose dataclass + fixed-size trajectory ring buffer
    pngio     stdlib-only 8-bit grayscale PNG codec (record/replay frames)

Import the submodules directly, e.g. ``from ours.lib.misc.pose import Pose`` or
``from ours.lib.misc import frames``.
"""
