#!/usr/bin/env python3
"""Self-test: the Frontend-Internals capture defaults ON when the UI runs (it is a
UI tool, so it "just works" without a flag), and OFF when headless (lean flight
path). Unlike the BA Window it is NOT tight-only: the KLT frontend is identical on
the loose and tight paths, so it stays ON under ``--tight`` with the UI.
``--frontend-viz`` forces it on (e.g. headless, for the smoke); ``--no-frontend-viz``
forces it off.

Fully OFFLINE (no spawning, no device): exercises the pure
:func:`launcher.main.resolve_frontend_viz` resolver with synthetic namespaces, the
:func:`launcher.main.build_vio_args` forwarding, and confirms the launcher
argparser registers both flags via ``--help``. Mirrors ``ba_window_forward_selftest``.

The CaptureKLTFrontend returns BYTE-IDENTICAL tracks, so this default is
oracle-safe (the offline ``oracle_replay_selftest`` never goes through this
launcher path).

Run::

    .venv/bin/python -m launcher.tests.frontend_viz_forward_selftest
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from launcher.main import build_vio_args, resolve_frontend_viz        # noqa: E402


def _ns(**over) -> types.SimpleNamespace:
    base = dict(frontend_viz=False, no_frontend_viz=False,
                no_ui=False, tight=False)
    base.update(over)
    return types.SimpleNamespace(**base)


def _vio_ns(**over) -> types.SimpleNamespace:
    """A launcher-args namespace for build_vio_args (the VIO forwarding test)."""
    base = dict(kf_every=5, no_gyro=False, worker=False,
                tight=False, stabilize_velocity=False, depth_icp=False,
                ba_window=False, frontend_viz=False)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_resolution() -> None:
    # (a) default with UI -> ON (the "just works" case).
    assert resolve_frontend_viz(_ns()) is True
    print("[a] default (UI)                       -> ON  (no flag needed)         OK")

    # (b) headless (flight path) -> OFF (no UI to consume; keep it lean).
    assert resolve_frontend_viz(_ns(no_ui=True)) is False
    print("[b] --no-ui (headless / flight)        -> OFF (lean path)              OK")

    # (c) tight path WITH UI -> still ON (frontend is identical loose/tight).
    assert resolve_frontend_viz(_ns(tight=True)) is True
    print("[c] --tight (frontend identical, UI)   -> ON                           OK")

    # (d) --frontend-viz forces it on even headless (e.g. the smoke).
    assert resolve_frontend_viz(_ns(frontend_viz=True, no_ui=True)) is True
    print("[d] --frontend-viz --no-ui (force on)  -> ON                           OK")

    # (e) --no-frontend-viz forces it off even with the UI shown.
    assert resolve_frontend_viz(_ns(no_frontend_viz=True)) is False
    print("[e] --no-frontend-viz (with UI)        -> OFF                          OK")

    # (f) --no-frontend-viz beats an explicit --frontend-viz.
    assert resolve_frontend_viz(
        _ns(frontend_viz=True, no_frontend_viz=True)) is False
    print("[f] --no-frontend-viz beats --frontend-viz                             OK")


def test_vio_forwarding() -> None:
    cap, vio, slam = "oak.capture", "oak.vio", "oak.slam"
    # ON -> --frontend-viz forwarded to vio.main (loose AND tight).
    argv = build_vio_args(_vio_ns(frontend_viz=True), cap, vio, slam,
                          use_worker=False)
    assert "--frontend-viz" in argv, argv
    argv_t = build_vio_args(_vio_ns(frontend_viz=True, tight=True), cap, vio,
                            slam, use_worker=False)
    assert "--frontend-viz" in argv_t, argv_t
    print("[g] build_vio_args forwards --frontend-viz (loose AND tight)           OK")
    # OFF -> NOT forwarded.
    argv_off = build_vio_args(_vio_ns(frontend_viz=False), cap, vio, slam,
                              use_worker=False)
    assert "--frontend-viz" not in argv_off, argv_off
    print("[h] build_vio_args omits --frontend-viz when off                       OK")


def test_help_registers_flags() -> None:
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-m", "launcher.main", "--help"],
        cwd=str(root), capture_output=True, text=True, check=True)
    assert "--frontend-viz" in out.stdout, out.stdout
    assert "--no-frontend-viz" in out.stdout, out.stdout
    print("[i] launcher --help lists --frontend-viz + --no-frontend-viz           OK")


if __name__ == "__main__":
    test_resolution()
    test_vio_forwarding()
    test_help_registers_flags()
    print("\nfrontend_viz_forward_selftest: ALL PASS")
