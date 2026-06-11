#!/usr/bin/env python3
"""Self-test: the launcher FORWARDS ``--stabilize-velocity`` to the VIO subprocess
argv ONLY when ``--tight`` AND ``--stabilize-velocity`` are both set, and NEVER on
the loose path (so the default end-to-end run -- and the offline oracle -- stay
byte-identical).

Fully OFFLINE (no spawning, no device): exercises the pure
:func:`launcher.main.build_vio_args` builder with synthetic namespaces, and confirms
the real launcher argparser registers the flag via ``-m launcher.main --help`` (so a
typo'd action= / dest= is caught). Mirrors ``use_camera_calib_forward_selftest``.

Asserts:
  (a) --tight + --stabilize-velocity SET    -> ``--stabilize-velocity`` IS in vio argv,
  (b) --tight only (no stabilize)           -> ``--stabilize-velocity`` NOT in vio argv,
  (c) --stabilize-velocity WITHOUT --tight  -> NOT forwarded (loose has no vel state),
  (d) neither flag                          -> NOT in vio argv (default OFF),
  (e) the launcher CLI ``--help`` lists ``--stabilize-velocity`` (parser registered).

Run::

    .venv/bin/python -m launcher.tests.stabilize_velocity_forward_selftest
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from launcher.main import build_vio_args                       # noqa: E402


def _ns(**over) -> types.SimpleNamespace:
    """A launcher-args namespace with sane defaults, overridable per test."""
    base = dict(kf_every=5, no_gyro=False, worker=False,
                tight=False, stabilize_velocity=False)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_forwarding() -> None:
    cap, vio, slam = "oak.capture", "oak.vio", "oak.slam"

    # (a) --tight + --stabilize-velocity SET -> forwarded.
    argv = build_vio_args(_ns(tight=True, stabilize_velocity=True),
                          cap, vio, slam, use_worker=False)
    assert "--tight" in argv, argv
    assert "--stabilize-velocity" in argv, argv
    print("[a] --tight + --stabilize-velocity SET -> forwarded to vio argv        OK")

    # (b) --tight only -> stabilize NOT forwarded (tight default = oracle-tuned).
    argv = build_vio_args(_ns(tight=True, stabilize_velocity=False),
                          cap, vio, slam, use_worker=False)
    assert "--tight" in argv, argv
    assert "--stabilize-velocity" not in argv, argv
    print("[b] --tight only (no stabilize)        -> NOT in vio argv               OK")

    # (c) --stabilize-velocity WITHOUT --tight -> dropped (loose has no vel state).
    argv = build_vio_args(_ns(tight=False, stabilize_velocity=True),
                          cap, vio, slam, use_worker=False)
    assert "--tight" not in argv, argv
    assert "--stabilize-velocity" not in argv, argv
    print("[c] --stabilize-velocity WITHOUT --tight -> NOT forwarded (warned)      OK")

    # (d) neither flag -> default OFF end-to-end.
    argv = build_vio_args(_ns(), cap, vio, slam, use_worker=False)
    assert "--tight" not in argv, argv
    assert "--stabilize-velocity" not in argv, argv
    print("[d] neither flag                       -> NOT in vio argv (default OFF) OK")


def test_help_registers_flag() -> None:
    """The real launcher parser must register the flag (catches action=/dest= typos)."""
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-m", "launcher.main", "--help"],
        cwd=str(root), capture_output=True, text=True, check=True)
    assert "--stabilize-velocity" in out.stdout, out.stdout
    print("[e] launcher --help lists --stabilize-velocity                          OK")


if __name__ == "__main__":
    test_forwarding()
    test_help_registers_flag()
    print("\nstabilize_velocity_forward_selftest: ALL PASS")
