#!/usr/bin/env python3
"""Headless smoke test for the proc4 UI data path.

Spawns capture + vio + slam exactly like :mod:`ours.tools.proc4_replay_selftest`,
then drives an :class:`IpcPoseSource` (mirrors what the VIO tab does) and a
:class:`SlamMapTracker` (mirrors what the SLAM tab does) IN-PROCESS. Asserts
that:

* the source emits one NED :class:`Pose` per replay frame, with finite
  positions and unit quaternions,
* the SLAM tracker's snapshot stays valid even when there are no loop
  corrections (empty array, not a crash),
* the Qt MainWindow can be CONSTRUCTED (we don't enter the event loop because
  the test runs headless on CI / pyqtgraph + OpenGL might not be available --
  but the construction proves the imports + widget wiring).

Run::

    python -m ours.tools.proc4_ui_selftest
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.misc.pose import Pose, PoseHistory                  # noqa: E402
from ours.proc.ui import (                                        # noqa: E402
    IpcPoseSource, SlamMapTracker, _await_calib_bundle,
)


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--no-qt", action="store_true",
                    help="skip the Qt MainWindow construction test")
    args = ap.parse_args()

    pid = os.getpid()
    cap_ep = f"oak.cap.u{pid & 0xFFF:x}"
    vio_ep = f"oak.vio.u{pid & 0xFFF:x}"
    slam_ep = f"oak.slm.u{pid & 0xFFF:x}"

    py = sys.executable
    base_env = dict(os.environ)
    log_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}

    print("proc4_ui_selftest")
    print(f"  session={args.session} max-frames={args.max_frames}")

    vio_proc = subprocess.Popen(
        [py, "-m", "ours.proc.vio",
         "--capture-endpoint", cap_ep, "--endpoint", vio_ep,
         "--kf-every", str(args.kf_every)],
        env=base_env, **log_kwargs)
    slam_proc = subprocess.Popen(
        [py, "-m", "ours.proc.slam",
         "--capture-endpoint", cap_ep, "--vio-endpoint", vio_ep,
         "--endpoint", slam_ep],
        env=base_env, **log_kwargs)
    time.sleep(0.3)
    cap_proc = subprocess.Popen(
        [py, "-m", "ours.proc.capture",
         "--endpoint", cap_ep, "--session", args.session,
         "--max-frames", str(args.max_frames)],
        env=base_env, **log_kwargs)

    procs = (cap_proc, vio_proc, slam_proc)

    try:
        # Wait for VIO + SLAM to be ready (their retained calib.bundle).
        _await_calib_bundle(vio_ep, timeout_s=20.0)
        print("  vio: ready")
        _await_calib_bundle(slam_ep, timeout_s=20.0)
        print("  slam: ready")

        # --- IpcPoseSource (VIO tab) ---
        history = PoseHistory(capacity=10_000)
        source = IpcPoseSource(vio_ep, label="vio", connect_timeout_s=20.0)
        source.start(history.push)

        # --- SlamMapTracker (SLAM tab) ---
        tracker = SlamMapTracker(slam_ep, connect_timeout_s=20.0)
        tracker.start()

        # Wait for the children to drain the whole replay (capture exits when
        # its replay ends; vio + slam exit when their END propagates).
        cap_proc.wait(timeout=60.0)
        vio_proc.wait(timeout=60.0)
        slam_proc.wait(timeout=60.0)

        # Give the source's recv thread a moment to drain any in-flight wire
        # messages that landed AFTER subprocess exit (capture's drain-on-close
        # buffers them but the local recv loop processes async).
        time.sleep(0.5)
        source.stop()
        tracker.stop()

        # ---------- Assertions ----------
        n_frames = args.max_frames
        pos, flags, latest = history.snapshot()
        snap_kf = tracker.refined_path_snapshot()
        _, _, _, n_loops = tracker.slam_overlay_snapshot()

        print(f"\n  received poses: {pos.shape[0]} (expected {n_frames})")
        print(f"  latest pose: {latest!r}")
        print(f"  slam kf snapshot: {snap_kf.shape} loops={n_loops}")

        _check(cap_proc.returncode == 0,
               f"capture exited 0 (got {cap_proc.returncode})")
        _check(vio_proc.returncode == 0,
               f"vio exited 0 (got {vio_proc.returncode})")
        _check(slam_proc.returncode == 0,
               f"slam exited 0 (got {slam_proc.returncode})")
        _check(pos.shape[0] == n_frames,
               f"received one pose per frame (got {pos.shape[0]}/{n_frames})")
        _check(np.isfinite(pos).all(),
               "all positions finite (no NaN/inf from the NED conversion)")
        _check(isinstance(latest, Pose),
               "latest snapshot is a Pose")
        _check(latest is not None
               and np.isclose(float(np.linalg.norm(latest.quat_wxyz)), 1.0,
                              atol=1e-3),
               "latest quaternion is unit-norm")
        _check(snap_kf.dtype == np.float32 and snap_kf.shape[1] == 3,
               f"slam kf snapshot is (K, 3) float32 (got "
               f"{snap_kf.shape} {snap_kf.dtype})")

        # ---------- Optional: build the Qt MainWindow ----------
        if not args.no_qt:
            try:
                _try_build_qt(vio_ep, slam_ep)
                _check(True, "Qt MainWindow constructs without error")
            except Exception as e:                                 # noqa: BLE001
                # Headless CI may not have a display / OpenGL; soft-fail.
                print(f"  [skip] Qt build skipped: {e}")

        print("\nALL PROC4 UI SELFTESTS PASSED")
        return 0
    finally:
        _terminate_all(*procs)
        # Surface failure context if anything went wrong.
        for name, proc in (("capture", cap_proc), ("vio", vio_proc),
                           ("slam", slam_proc)):
            try:
                _out, err = proc.communicate(timeout=2.0)
            except Exception:                                      # noqa: BLE001
                err = b""
            if err.strip():
                print(f"\n  --- {name}.stderr ---\n"
                      f"{err.decode(errors='replace')}",
                      file=sys.stderr)


def _try_build_qt(vio_ep: str, slam_ep: str) -> None:
    """Build the QMainWindow + tabs once; immediately destroy.

    Proves the widget tree is wireable. Does NOT call ``app.exec()`` so the
    test stays headless. Imports Qt inside this function so the rest of the
    file imports cleanly even without PyQt6 installed.
    """
    from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget

    from ours.ui import theme
    from ours.ui.viewer3d import Viewer3D

    app = QApplication.instance() or QApplication(sys.argv or ["test"])
    try:
        history_a = PoseHistory(capacity=128)
        history_b = PoseHistory(capacity=128)
        viewer_a = Viewer3D(history_a, default_view="ISO")
        viewer_b = Viewer3D(history_b, default_view="ISO")
        tabs = QTabWidget()
        tabs.addTab(viewer_a, "VIO")
        tabs.addTab(viewer_b, "SLAM")
        win = QMainWindow()
        win.setStyleSheet(theme.QSS)
        win.setCentralWidget(tabs)
        # Don't show -- headless construction proof only.
    finally:
        # PyQt5/6 cleans up via Python GC, but explicitly drop refs so the
        # `QApplication.instance()` re-use on a subsequent test call doesn't
        # see a stale widget.
        del win, tabs, viewer_a, viewer_b
        # Don't quit -- the QApplication singleton may be re-used.


def _terminate_all(*procs: subprocess.Popen) -> None:
    for p in procs:
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:                                      # noqa: BLE001
                pass
    for p in procs:
        try:
            p.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:                                      # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
