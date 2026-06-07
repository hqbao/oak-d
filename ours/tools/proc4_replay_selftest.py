#!/usr/bin/env python3
"""4-process smoke selftest: capture + vio + slam + headless UI sink (replay).

Spawns the three background processes (``capture``, ``vio``, ``slam``) and a
**headless** UI subscriber in-process that mirrors ``pose.odom``,
``pose.refined`` and ``loop.correction`` off their respective IPC endpoints.
Drives capture in replay mode over a recorded session for a bounded number of
frames; asserts that:

* every spawned process exits with 0 once capture's replay ends
* the UI sink receives at least one ``pose.odom`` per processed frame
* the UI sink receives some ``pose.refined`` (one per keyframe, ceil(frames/kf))
* no IPC connect / shared-memory error surfaced

The test is the live counterpart of ``flow_replay_selftest`` -- that single-
process replay must stay byte-identical and is unchanged. This one only proves
the 4-process plumbing routes the same data without dropping or corrupting it.

Run::

    python -m ours.tools.proc4_replay_selftest
    python -m ours.tools.proc4_replay_selftest --session sessions/gold/lab_loop_30s --max-frames 60
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.flow import Bus, topics                              # noqa: E402
from ours.lib.flow.messages import END                             # noqa: E402
from ours.lib.io.reader import SessionReader                       # noqa: E402
from ours.lib.ipc import IpcClientBus                              # noqa: E402
from ours.lib.ipc.messages import WireCalibBundle                  # noqa: E402
from ours.flows.bridge import IpcSubscriberFlow, RingRegistry      # noqa: E402
from ours.flows.bridge.ring_registry import (                      # noqa: E402
    default_capture_specs, default_vio_specs,
)


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
def _await_calib(endpoint: str, timeout_s: float) -> WireCalibBundle:
    bundle: list[WireCalibBundle | None] = [None]
    got = threading.Event()

    def on_calib(wm: WireCalibBundle) -> None:
        bundle[0] = wm
        got.set()

    cli = IpcClientBus(endpoint, connect_timeout_s=timeout_s)
    cli.subscribe("calib.bundle", on_calib)
    cli.start()
    try:
        if not got.wait(timeout=timeout_s):
            raise TimeoutError(f"no calib.bundle from {endpoint!r} in {timeout_s}s")
    finally:
        cli.stop()
    return bundle[0]                                               # type: ignore[return-value]


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--keep-logs", action="store_true",
                    help="print subprocess stdout/stderr instead of capturing")
    args = ap.parse_args()

    pid = os.getpid()
    cap_ep = f"oak.cap.t{pid & 0xFFF:x}"
    vio_ep = f"oak.vio.t{pid & 0xFFF:x}"
    slam_ep = f"oak.slm.t{pid & 0xFFF:x}"

    py = sys.executable
    # Inherit our Python so subprocesses see our package layout.
    base_env = dict(os.environ)

    print("proc4_replay_selftest")
    print(f"  session={args.session} max-frames={args.max_frames}")
    print(f"  endpoints: cap={cap_ep!r} vio={vio_ep!r} slam={slam_ep!r}")

    log_kwargs = ({} if args.keep_logs
                  else {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE})

    # ---------- Boot ORDER MATTERS: VIO + SLAM must come up before capture --
    # (they connect with retried `IpcClientBus.start`, so coming up late is ok
    # too, but bringing them up first means they catch the capture's first
    # frames without dropping any.)
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

    # Give VIO + SLAM a brief window to register their calib-bundle subscribers
    # on capture before capture's accept thread starts (otherwise capture has to
    # boot first and we lose nothing because calib.bundle is retained, but the
    # smaller race window keeps logs tidy).
    time.sleep(0.3)

    cap_proc = subprocess.Popen(
        [py, "-m", "ours.proc.capture",
         "--endpoint", cap_ep, "--session", args.session,
         "--max-frames", str(args.max_frames)],
        env=base_env, **log_kwargs)

    procs = (cap_proc, vio_proc, slam_proc)
    cap_rings = vio_rings = None
    bridges: list = []

    try:
        return _run_assertions(args, cap_ep, vio_ep, slam_ep, procs, bridges)
    finally:
        # Always close bridges + rings + terminate any straggler subprocess.
        for b in bridges:
            try:
                b.stop()
            except Exception:                                      # noqa: BLE001
                pass
        for r in (cap_rings, vio_rings):
            if r is not None:
                try:
                    r.close()
                except Exception:                                  # noqa: BLE001
                    pass
        _terminate_all(*procs)
        if not args.keep_logs:
            for name, proc in (("capture", cap_proc), ("vio", vio_proc),
                               ("slam", slam_proc)):
                try:
                    out, err = proc.communicate(timeout=2.0)
                except Exception:                                  # noqa: BLE001
                    out, err = b"", b""
                if err.strip():
                    print(f"\n  --- {name}.stderr ---\n"
                          f"{err.decode(errors='replace')}",
                          file=sys.stderr)


def _run_assertions(args, cap_ep, vio_ep, slam_ep, procs, bridges):
    cap_proc, vio_proc, slam_proc = procs

    # ---------- Wait for capture's calib so we know its resolution ----------
    bundle = _await_calib(cap_ep, timeout_s=20.0)
    width, height = int(bundle.width), int(bundle.height)
    print(f"  calib: {width}x{height} T_imu={bundle.T_imu_left is not None}")

    # Wait for VIO + SLAM to be ready -- each retains+re-broadcasts the calib
    # bundle on its own endpoint AFTER allocating its rings, so the calib's
    # arrival on `vio_ep` proves VIO's kf_* rings exist and we can attach.
    _await_calib(vio_ep, timeout_s=20.0)
    print("  vio: ready")
    _await_calib(slam_ep, timeout_s=20.0)
    print("  slam: ready")

    # ---------- Headless UI sink: subscribe to VIO + SLAM topics -----------
    # (slot count defaults to the registry-wide default that matches the
    # IpcServerBus outbox cap; do NOT hard-code 8 here -- producers create
    # rings with the default and attach must use the same count.)
    cap_rings = RingRegistry().attach_all(default_capture_specs(
        endpoint=cap_ep, width=width, height=height))
    vio_rings = RingRegistry().attach_all(default_vio_specs(
        endpoint=vio_ep, width=width, height=height))

    local = Bus()

    odom_seqs: list[int] = []
    refined_seqs: list[int] = []
    keyframe_seqs: list[int] = []
    loop_counts: list[int] = []
    ends_seen = {"odom": 0, "refined": 0, "kf": 0, "loop": 0}
    all_done = threading.Event()

    def _maybe_done() -> None:
        # VIO publishes END on pose.odom/pose.refined/keyframe; SLAM on loop.
        # All four ENDs landed -> drain is complete.
        if (ends_seen["odom"] >= 1 and ends_seen["refined"] >= 1
                and ends_seen["kf"] >= 1 and ends_seen["loop"] >= 1):
            all_done.set()

    def on_pose(msg) -> None:
        if msg is END:
            ends_seen["odom"] += 1
            _maybe_done()
            return
        odom_seqs.append(int(msg.seq))

    def on_refined(msg) -> None:
        if msg is END:
            ends_seen["refined"] += 1
            _maybe_done()
            return
        refined_seqs.append(int(msg.seq))

    def on_keyframe(msg) -> None:
        if msg is END:
            ends_seen["kf"] += 1
            _maybe_done()
            return
        keyframe_seqs.append(int(msg.seq))

    def on_loop(msg) -> None:
        if msg is END:
            ends_seen["loop"] += 1
            _maybe_done()
            return
        loop_counts.append(int(msg.n_loops))

    local.subscribe(topics.POSE_ODOM, on_pose)
    local.subscribe(topics.POSE_REFINED, on_refined)
    local.subscribe(topics.KEYFRAME, on_keyframe)
    local.subscribe(topics.LOOP_CORRECTION, on_loop)

    vio_client = IpcClientBus(vio_ep, connect_timeout_s=20.0)
    slam_client = IpcClientBus(slam_ep, connect_timeout_s=20.0)
    kf_client = IpcClientBus(vio_ep, connect_timeout_s=20.0)

    vio_bridge = IpcSubscriberFlow(
        local, vio_client, cap_rings,
        [topics.POSE_ODOM, topics.POSE_REFINED, topics.FRAME_TRACKS,
         topics.FRAME_INLIERS])
    # KEYFRAME rides VIO's own rings.
    kf_bridge = IpcSubscriberFlow(local, kf_client, vio_rings,
                                  [topics.KEYFRAME])
    slam_bridge = IpcSubscriberFlow(local, slam_client, vio_rings,
                                    [topics.LOOP_CORRECTION])

    bridges.extend([vio_bridge, kf_bridge, slam_bridge])
    vio_bridge.start()
    kf_bridge.start()
    slam_bridge.start()

    # ---------- Wait until either everyone sends END or timeout -----------
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if all_done.is_set():
            break
        # Surface a subprocess that died early.
        for name, proc in (("capture", cap_proc), ("vio", vio_proc),
                           ("slam", slam_proc)):
            if proc.poll() is not None and proc.returncode != 0:
                print(f"\n  [FAIL] {name} exited early with rc={proc.returncode}",
                      file=sys.stderr)
                raise SystemExit(1)
        time.sleep(0.1)

    # ---------- Wait for the children to exit cleanly --------------------
    # capture exits when its replay finishes (cam_flow.join returns); VIO when
    # it sees both ENDs from capture; SLAM when it sees END on keyframe.
    cap_proc.wait(timeout=10.0)
    vio_proc.wait(timeout=10.0)
    slam_proc.wait(timeout=10.0)

    # Close ring handles + bridges BEFORE the outer finally has to mop up,
    # so the resource_tracker sees them go away cleanly.
    for b in bridges:
        b.stop()
    bridges.clear()
    cap_rings.close()
    vio_rings.close()

    # ---------- Assertions ----------
    n_frames = (args.max_frames if args.max_frames > 0
                else len(SessionReader(Path(args.session))))
    print(f"\n  received: odom={len(odom_seqs)} refined={len(refined_seqs)} "
          f"kf={len(keyframe_seqs)} loops={loop_counts[-1] if loop_counts else 0}")
    print(f"  ends: {ends_seen}")

    import math
    max_refined = math.ceil(n_frames / args.kf_every)

    _check(cap_proc.returncode == 0,
           f"capture exited 0 (got {cap_proc.returncode})")
    _check(vio_proc.returncode == 0,
           f"vio exited 0 (got {vio_proc.returncode})")
    _check(slam_proc.returncode == 0,
           f"slam exited 0 (got {slam_proc.returncode})")

    _check(len(odom_seqs) == n_frames,
           f"received {n_frames} pose.odom (got {len(odom_seqs)})")
    _check(0 < len(refined_seqs) <= max_refined,
           f"received pose.refined: 1..{max_refined} (got {len(refined_seqs)})")
    _check(0 < len(keyframe_seqs) <= max_refined,
           f"received keyframes: 1..{max_refined} (got {len(keyframe_seqs)})")
    _check(sorted(odom_seqs) == list(range(n_frames)),
           f"pose.odom seqs are dense 0..{n_frames - 1}")

    print("\nALL PROC4 SMOKE SELFTESTS PASSED")
    return 0


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
