#!/usr/bin/env python3
"""Self-test: the launcher PARSES ``--use-camera-calib`` and FORWARDS it to the
capture subprocess argv only in the LIVE branch, only when set.

Fully OFFLINE (no spawning, no device): exercises the pure
:func:`launcher.main.build_capture_args` builder with synthetic namespaces, and
confirms the real launcher argparser registers the flag via ``-m launcher.main
--help`` (so a typo'd action= or dest= would be caught). Mirrors how the other
capture flags (``--no-gyro`` / ``--recalibrate-bias``) are forwarded.

Asserts:
  (a) live + flag SET    -> ``--use-camera-calib`` IS in the capture argv,
  (b) live + flag UNSET  -> ``--use-camera-calib`` is NOT in the capture argv,
  (c) REPLAY + flag SET  -> NOT forwarded (the flag is live-only; replay reads the
      session's calib.json, not the per-device store),
  (d) the launcher's CLI ``--help`` lists ``--use-camera-calib`` (parser registered).

Run::

    .venv/bin/python -m launcher.tests.use_camera_calib_forward_selftest
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from launcher.main import build_capture_args                  # noqa: E402


def _ns(**over) -> types.SimpleNamespace:
    """A launcher-args namespace with sane defaults, overridable per test."""
    base = dict(width=640, height=400, fps=20, session=None, max_frames=0,
                no_gyro=False, recalibrate_bias=False, use_camera_calib=False,
                vl53l9cx=False)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_forwarding() -> None:
    cap = "oak.capture"

    # (a) live + flag SET -> forwarded.
    argv = build_capture_args(_ns(use_camera_calib=True), cap)
    assert "--live" in argv, argv
    assert "--use-camera-calib" in argv, argv
    print("[a] live + --use-camera-calib SET   -> forwarded to capture argv      OK")

    # (b) live + flag UNSET -> not forwarded (default off = factory).
    argv = build_capture_args(_ns(use_camera_calib=False), cap)
    assert "--live" in argv, argv
    assert "--use-camera-calib" not in argv, argv
    print("[b] live + flag UNSET               -> NOT in capture argv (factory)   OK")

    # (c) replay + flag SET -> NOT forwarded (live-only; replay uses session calib).
    argv = build_capture_args(
        _ns(session="sessions/gold/lab_loop_30s", use_camera_calib=True), cap)
    assert "--session" in argv and "--live" not in argv, argv
    assert "--use-camera-calib" not in argv, argv
    print("[c] replay + flag SET               -> NOT forwarded (live-only)       OK")


def test_parser_registers_flag() -> None:
    # (d) the real launcher CLI lists the flag in --help, so argparse registered it.
    repo = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-m", "launcher.main", "--help"],
        cwd=repo, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "--use-camera-calib" in out.stdout, out.stdout
    print("[d] launcher --help lists --use-camera-calib (parser registered)       OK")


def main() -> int:
    test_forwarding()
    test_parser_registers_flag()
    print("\nALL launcher --use-camera-calib FORWARDING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
