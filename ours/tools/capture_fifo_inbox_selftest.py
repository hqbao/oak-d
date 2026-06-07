#!/usr/bin/env python3
"""Regression guard for BUG B: capture front-end MUST be FIFO, not latest-only.

Background
----------
``ours.proc.capture`` builds the live front-end via
:func:`ours.app.build_live_frontend`. VIO + deterministic replay require FIFO
on every topic that feeds the odometry compute path (``cam.sync``,
``imucam.sample``, ``frame.depth``). See ARCHITECTURE.md §3 ("VIO + deterministic
replay require FIFO") and ``topics.VIO_PATH_TOPICS``.

BUG B was a regression where the live capture path called
``build_live_frontend(..., latest_only=True)``: a slow downstream subscriber
caused the capture-side imu_cam inbox to coalesce away CAM_SYNC packets,
breaking gyro continuity for ``PreintegratePrior`` and KLT continuity for
``TrackFeatures`` -- silently corrupted poses on a busy machine.

The fix is to force ``latest_only=False`` at the capture flow. Backpressure
belongs at the IPC boundary (the publisher bridge / IpcServerBus outbox), not
at the VIO compute inputs.

This test is purely structural: parse ``ours/proc/capture.py`` with ``ast`` and
assert every call to ``build_live_frontend`` (and ``build_replay_frontend``)
passes ``latest_only=False`` (or no kwarg, since the default is False). It also
asserts ``VIO_PATH_TOPICS`` exists and lists the three FIFO-mandatory topics,
so removing the constant or shrinking the set fires a separate alarm.

Run::

    python -m ours.tools.capture_fifo_inbox_selftest
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.flow import topics                                     # noqa: E402

CAPTURE_PATH = (Path(__file__).resolve().parents[1]
                / "proc" / "capture.py")
APP_PATH = (Path(__file__).resolve().parents[1] / "app.py")


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def _kwarg_value(call: ast.Call, name: str) -> ast.expr | None:
    """Return the AST node for kwarg ``name`` on ``call`` (or None)."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_false_literal(node: ast.expr | None) -> bool:
    """True iff the AST node is the literal ``False``."""
    return (isinstance(node, ast.Constant) and node.value is False)


def _is_true_literal(node: ast.expr | None) -> bool:
    return (isinstance(node, ast.Constant) and node.value is True)


def _find_calls(tree: ast.AST, fn_names: tuple[str, ...]) -> list[ast.Call]:
    """Return every Call whose direct callee name is in ``fn_names``."""
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in fn_names:
            found.append(node)
        elif isinstance(fn, ast.Attribute) and fn.attr in fn_names:
            found.append(node)
    return found


def main() -> int:
    print("capture_fifo_inbox_selftest")

    # ------------------------------------------------------------------ #
    # 1. VIO_PATH_TOPICS exists + lists the three FIFO-mandatory topics.
    # ------------------------------------------------------------------ #
    _check(hasattr(topics, "VIO_PATH_TOPICS"),
           "ours.lib.flow.topics.VIO_PATH_TOPICS exists")
    expected = {topics.CAM_SYNC, topics.IMUCAM_SAMPLE, topics.FRAME_DEPTH}
    _check(set(topics.VIO_PATH_TOPICS) == expected,
           f"VIO_PATH_TOPICS == {{CAM_SYNC, IMUCAM_SAMPLE, FRAME_DEPTH}} "
           f"(got: {sorted(topics.VIO_PATH_TOPICS)})")

    # ------------------------------------------------------------------ #
    # 2. Parse ours/proc/capture.py and assert every call to
    #    build_live_frontend / build_replay_frontend passes latest_only=False
    #    (or omits the kwarg if the default is False -- we check that below).
    # ------------------------------------------------------------------ #
    _check(CAPTURE_PATH.exists(),
           f"capture source file exists at {CAPTURE_PATH}")
    src = CAPTURE_PATH.read_text()
    tree = ast.parse(src, filename=str(CAPTURE_PATH))

    calls = _find_calls(tree, ("build_live_frontend", "build_replay_frontend"))
    _check(len(calls) >= 1,
           f"capture.py calls at least one build_*_frontend "
           f"(found {len(calls)})")

    bad: list[str] = []
    seen_kwarg = False
    for call in calls:
        callee = (call.func.id if isinstance(call.func, ast.Name)
                  else call.func.attr)
        node = _kwarg_value(call, "latest_only")
        if node is None:
            # kwarg omitted -> default value applies; we verify the default
            # below (separately). For capture's structural contract, omitting
            # is fine IFF the default is False.
            continue
        seen_kwarg = True
        if _is_true_literal(node):
            bad.append(f"{callee}(line {call.lineno}): latest_only=True")
        elif not _is_false_literal(node):
            bad.append(f"{callee}(line {call.lineno}): "
                       f"latest_only=<non-literal>, can't verify statically")

    _check(not bad,
           "every build_*_frontend in capture.py uses latest_only=False "
           f"(violations: {bad})")
    _check(seen_kwarg,
           "at least one call passes the latest_only kwarg explicitly "
           "(belt-and-suspenders: live path must spell it out, since the "
           "default could change)")

    # ------------------------------------------------------------------ #
    # 3. Verify the DEFAULT of build_live_frontend / build_replay_frontend is
    #    False, so any call that omits the kwarg still gets FIFO behaviour.
    # ------------------------------------------------------------------ #
    if APP_PATH.exists():
        app_tree = ast.parse(APP_PATH.read_text(), filename=str(APP_PATH))
        for node in ast.walk(app_tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name in ("build_live_frontend",
                                      "build_replay_frontend")):
                # Find the latest_only kwarg in the signature defaults.
                args = node.args
                # kw-only first (most modern style), then regular.
                kwonly_names = [a.arg for a in args.kwonlyargs]
                kwonly_defs = args.kw_defaults
                regular_names = [a.arg for a in args.args]
                regular_defs = args.defaults
                default: ast.expr | None = None
                if "latest_only" in kwonly_names:
                    idx = kwonly_names.index("latest_only")
                    default = kwonly_defs[idx]
                elif "latest_only" in regular_names:
                    # defaults align to the TAIL of args.
                    idx = regular_names.index("latest_only")
                    tail = len(regular_names) - len(regular_defs)
                    if idx >= tail:
                        default = regular_defs[idx - tail]
                if default is not None:
                    _check(_is_false_literal(default),
                           f"ours.app.{node.name}: default latest_only is False "
                           f"(got: {ast.dump(default)})")

    # ------------------------------------------------------------------ #
    # 4. Runtime smoke: build_replay_frontend on a gold session must produce a
    #    flow whose imu_cam inbox is NOT latest_only. We import the module
    #    rather than spin up real OAK-D hardware.
    # ------------------------------------------------------------------ #
    from ours.app import build_replay_frontend
    from ours.lib.flow import Bus
    from ours.lib.io.reader import SessionReader

    session = Path("sessions/gold/lab_loop_30s")
    if session.exists():
        bus = Bus()
        reader = SessionReader(session)
        cam_flow, imu_flow = build_replay_frontend(
            bus=bus, reader=reader, depth_fast=True, max_frames=1)
        try:
            # The Flow.latest_only flag should be False (FIFO) on both. We try
            # several plausible attribute names since this is a guard, not a
            # hard interface contract.
            for f, label in ((cam_flow, "cam_flow"), (imu_flow, "imu_flow")):
                flag = (getattr(f, "latest_only", None)
                        or getattr(getattr(f, "ctx", None), "latest_only", None)
                        or getattr(getattr(f, "inbox", None), "latest_only", None))
                if flag is None:
                    # Some Flow impls store it on _latest_only / queue; skip
                    # silently if we can't introspect.
                    print(f"  [info] {label}: no introspectable latest_only "
                          f"attribute (skipping runtime flag check)")
                    continue
                _check(flag is False,
                       f"{label} built by build_replay_frontend has "
                       f"latest_only=False (got: {flag!r})")
        finally:
            try:
                cam_flow.stop(); imu_flow.stop()
            except Exception:                                          # noqa: BLE001
                pass
    else:
        print(f"  [info] gold session {session} missing -- runtime check skipped")

    print("\nALL CAPTURE FIFO INBOX SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
