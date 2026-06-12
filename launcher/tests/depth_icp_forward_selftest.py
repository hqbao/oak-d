#!/usr/bin/env python3
"""Self-test: the launcher FORWARDS ``--depth-icp`` to the VIO subprocess argv
ONLY when ``--tight`` AND ``--depth-icp`` are both set, and NEVER on the loose
path (so the default end-to-end run -- and the offline oracle -- stay
byte-identical).

Fully OFFLINE (no spawning, no device): exercises the pure
:func:`launcher.main.build_vio_args` builder with synthetic namespaces, and
confirms the launcher argparser registers the flag via ``-m launcher.main
--help``. Mirrors ``stabilize_velocity_forward_selftest``.

Asserts:
  (a) --tight + --depth-icp SET    -> ``--depth-icp`` IS in vio argv,
  (b) --tight only (no depth-icp)  -> ``--depth-icp`` NOT in vio argv,
  (c) --depth-icp WITHOUT --tight  -> NOT forwarded (loose has no factor graph),
  (d) neither flag                 -> NOT in vio argv (default OFF),
  (e) the launcher CLI ``--help`` lists ``--depth-icp`` (parser registered).

Run::

    .venv/bin/python -m launcher.tests.depth_icp_forward_selftest
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from launcher.main import build_vio_args                       # noqa: E402


def _ns(**over) -> types.SimpleNamespace:
    base = dict(kf_every=5, no_gyro=False, worker=False,
                tight=False, stabilize_velocity=False, depth_icp=False,
                ba_window=False, frontend_viz=False)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_forwarding() -> None:
    cap, vio, slam = "oak.capture", "oak.vio", "oak.slam"

    # (a) --tight + --depth-icp SET -> forwarded.
    argv = build_vio_args(_ns(tight=True, depth_icp=True),
                          cap, vio, slam, use_worker=False)
    assert "--tight" in argv, argv
    assert "--depth-icp" in argv, argv
    print("[a] --tight + --depth-icp SET -> forwarded to vio argv               OK")

    # (b) --tight only -> depth-icp NOT forwarded (tight default = oracle-tuned).
    argv = build_vio_args(_ns(tight=True, depth_icp=False),
                          cap, vio, slam, use_worker=False)
    assert "--depth-icp" not in argv, argv
    print("[b] --tight only (no depth-icp)        -> NOT in vio argv             OK")

    # (c) --depth-icp WITHOUT --tight -> dropped (loose has no factor graph).
    argv = build_vio_args(_ns(tight=False, depth_icp=True),
                          cap, vio, slam, use_worker=False)
    assert "--tight" not in argv, argv
    assert "--depth-icp" not in argv, argv
    print("[c] --depth-icp WITHOUT --tight        -> NOT forwarded (warned)      OK")

    # (d) neither flag -> default OFF end-to-end.
    argv = build_vio_args(_ns(), cap, vio, slam, use_worker=False)
    assert "--depth-icp" not in argv, argv
    print("[d] neither flag                       -> NOT in vio argv (default)   OK")


def test_help_registers_flag() -> None:
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-m", "launcher.main", "--help"],
        cwd=str(root), capture_output=True, text=True, check=True)
    assert "--depth-icp" in out.stdout, out.stdout
    print("[e] launcher --help lists --depth-icp                                 OK")


if __name__ == "__main__":
    test_forwarding()
    test_help_registers_flag()
    print("\ndepth_icp_forward_selftest: ALL PASS")
