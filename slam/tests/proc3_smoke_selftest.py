#!/usr/bin/env python3
"""3-process smoke selftest: imu_camera + vio + slam over a gold loop session.

The integration proof for the Phase-4 split. Spawns the three split projects'
processes back-to-back over IPC:

    imu_camera.main (replay)  --oak.cap-->  vio.main  --oak.vio-->  slam.main

then opens a tiny headless :class:`~slam.comms.IPCPubSub` CLIENT on the SLAM
endpoint that counts ``slam.map`` (the continuous keyframe overlay -- proves the
map advances, kf dots growing) and ``loop.correction`` (the loop-event stream --
``n_loops`` is the oracle-comparable count). It asserts:

* all three spawned processes exit with rc=0 once capture's replay ends;
* ``slam.map`` advances (the keyframe count strictly grows -> the map is being
  built, not stalled);
* the loop session DOES close loops: ``1 <= confirmed loops <= oracle ceiling``.
  The exact count is INHERENTLY NON-DETERMINISTIC on the LIVE path -- SLAM runs
  ``latest_only=True``, whose coalescing keyframe inbox drops a variable number of
  keyframes under real-time load (measured: the pre-split multi-process replay
  reference itself reported 2 / 3 / 4 on ``lab_loop_30s`` on consecutive runs).
  The deterministic byte-parity proof of the loop-closure
  MATH lives in :mod:`slam.tests.loop_closure_selftest`; this smoke proves the
  PLUMBING;
* clean shutdown (no IPC connect / shared-memory error surfaced).

This is the live-plumbing counterpart of :mod:`slam.tests.loop_closure_selftest`
(which proves the math byte-parity in-process). This one proves the 3-process
plumbing routes the keyframes -> corrections without dropping or corrupting them.

Run::

    python -m slam.tests.proc3_smoke_selftest
    python -m slam.tests.proc3_smoke_selftest --session sessions/gold/lab_loop_30s --expect-loops 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slam.comms import IPCPubSub, topics                          # noqa: E402
from slam.comms.messages import END                              # noqa: E402
from slam.comms.wire import WireCalibBundle                      # noqa: E402


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
def _await_calib(endpoint: str, timeout_s: float) -> WireCalibBundle:
    """Open a dedicated client, block until the retained calib bundle arrives."""
    bundle: list[WireCalibBundle | None] = [None]
    got = threading.Event()

    def on_calib(wm: WireCalibBundle) -> None:
        bundle[0] = wm
        got.set()

    cli = IPCPubSub(endpoint, role="client", connect_timeout_s=timeout_s)
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
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = full session (needed for a loop to actually close)")
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--expect-loops", type=int, default=7,
                    help="CEILING for confirmed loops on the session (live "
                         "latest_only SLAM is non-deterministic: the pre-split "
                         "Module-framework reference reported 2..4 on lab_loop_30s; "
                         "after the procedural shell roll-out the workers drop fewer "
                         "keyframes -> slightly higher live throughput, observed up "
                         "to 5 across ~17 runs, so the ceiling is 7 with margin. The "
                         "extra closures are REAL: the loop-closure math is unchanged "
                         "(slam loop_closure_selftest) and the oracle is gap=0. The "
                         "real gates here are >=1 loop + slam.map advances + rc=0; "
                         "this upper bound is only a 'not absurd' sanity. Assert "
                         "1 <= observed <= this ceiling)")
    ap.add_argument("--keep-logs", action="store_true",
                    help="print subprocess stdout/stderr instead of capturing")
    args = ap.parse_args()

    pid = os.getpid()
    cap_ep = f"oak.cap.s{pid & 0xFFF:x}"
    vio_ep = f"oak.vio.s{pid & 0xFFF:x}"
    slam_ep = f"oak.slm.s{pid & 0xFFF:x}"

    py = sys.executable
    base_env = dict(os.environ)

    print("proc3_smoke_selftest (imu_camera + vio + slam)")
    print(f"  session={args.session} max-frames={args.max_frames}")
    print(f"  endpoints: cap={cap_ep!r} vio={vio_ep!r} slam={slam_ep!r}")
    print(f"  expect loops={args.expect_loops}")

    log_kwargs = ({} if args.keep_logs
                  else {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE})

    # ---------- Boot vio + slam first (they connect with retried clients) -----
    vio_proc = subprocess.Popen(
        [py, "-m", "vio.main",
         "--capture-endpoint", cap_ep, "--endpoint", vio_ep,
         "--kf-every", str(args.kf_every)],
        env=base_env, **log_kwargs)
    slam_proc = subprocess.Popen(
        [py, "-m", "slam.main",
         "--vio-endpoint", vio_ep, "--endpoint", slam_ep],
        env=base_env, **log_kwargs)

    # Brief window so vio + slam register their calib subscribers before capture
    # starts accepting (calib is retained, so a later boot is fine too).
    time.sleep(0.3)

    cap_proc = subprocess.Popen(
        [py, "-m", "imu_camera.main",
         "--endpoint", cap_ep, "--session", args.session,
         "--max-frames", str(args.max_frames)],
        env=base_env, **log_kwargs)

    procs = (cap_proc, vio_proc, slam_proc)
    try:
        return _run_assertions(args, slam_ep, procs)
    finally:
        _terminate_all(*procs)
        if not args.keep_logs:
            for name, proc in (("imu_camera", cap_proc), ("vio", vio_proc),
                               ("slam", slam_proc)):
                try:
                    _, err = proc.communicate(timeout=2.0)
                except Exception:                                  # noqa: BLE001
                    _, err = b"", b""
                if err and err.strip():
                    print(f"\n  --- {name}.stderr (tail) ---\n"
                          f"{_tail(err.decode(errors='replace'), 20)}",
                          file=sys.stderr)


def _tail(s: str, n: int) -> str:
    lines = s.rstrip().splitlines()
    return "\n".join(lines[-n:])


def _run_assertions(args, slam_ep, procs):
    cap_proc, vio_proc, slam_proc = procs

    # ---------- ONE persistent headless client on the SLAM endpoint -----------
    # Subscribe to calib.bundle (readiness barrier -- retained, so it arrives the
    # instant SLAM's server is up) AND slam.map + loop.correction in the SAME
    # connection, started ONCE and kept alive for the whole run. A single
    # continuously-connected subscriber catches EVERY published message including
    # the final loop.correction + the END sentinel, so the count is observed
    # deterministically (a late-connecting / early-stopping client could miss the
    # last correction in the SLAM-shutdown window -> a flaky n_loops). slam.map +
    # loop.correction are pure POD (no shared-memory ring), so a plain client with
    # no ring registry receives them directly off the wire.
    loop_counts: list[int] = []        # n_loops carried on each loop.correction
    map_kf_counts: list[int] = []      # keyframe count on each slam.map overlay
    ends = {"map": 0, "loop": 0}
    ready = threading.Event()
    all_done = threading.Event()

    def _maybe_done() -> None:
        if ends["map"] >= 1 and ends["loop"] >= 1:
            all_done.set()

    def on_calib(_wm) -> None:
        ready.set()

    def on_map(msg) -> None:
        if msg is END:
            ends["map"] += 1
            _maybe_done()
            return
        # SlamOverlay.kf_positions is (N,3); N = current keyframe count.
        map_kf_counts.append(int(len(msg.kf_positions)))

    def on_loop(msg) -> None:
        if msg is END:
            ends["loop"] += 1
            _maybe_done()
            return
        loop_counts.append(int(msg.n_loops))

    client = IPCPubSub(slam_ep, role="client", connect_timeout_s=30.0)
    client.subscribe("calib.bundle", on_calib)
    client.subscribe(topics.SLAM_MAP, on_map)
    client.subscribe(topics.LOOP_CORRECTION, on_loop)
    client.start()

    try:
        # ---------- Wait for SLAM readiness (retained calib lands) -----------
        if not ready.wait(timeout=30.0):
            raise TimeoutError(f"no calib.bundle from {slam_ep!r} in 30s")
        print("  slam: ready")

        # ---------- Wait until both ENDs land or timeout ---------------------
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            if all_done.is_set():
                break
            for name, proc in (("imu_camera", cap_proc), ("vio", vio_proc),
                               ("slam", slam_proc)):
                if proc.poll() is not None and proc.returncode != 0:
                    print(f"\n  [FAIL] {name} exited early rc={proc.returncode}",
                          file=sys.stderr)
                    raise SystemExit(1)
            time.sleep(0.1)
    finally:
        client.stop()

    # ---------- Wait for the children to exit cleanly --------------------
    cap_proc.wait(timeout=15.0)
    vio_proc.wait(timeout=15.0)
    slam_proc.wait(timeout=15.0)

    # ---------- Assertions ----------
    # HONEST NOTE on the loop count. SLAM runs with ``latest_only=True`` (the LIVE
    # viewer path), whose coalescing keyframe inbox DROPS a variable number of
    # keyframes under real-time load -- so the confirmed-loop count is INHERENTLY
    # NON-DETERMINISTIC across runs (measured: the pre-split multi-process replay
    # reference itself reported 2 / 3 / 4 loops on
    # lab_loop_30s on consecutive runs). The deterministic byte-parity proof of the
    # loop-closure MATH lives in ``slam.tests.loop_closure_selftest``; this smoke
    # proves the 3-process PLUMBING: the loop session DOES close loops, the map
    # advances, and everyone shuts down cleanly. So we assert ``>= 1`` confirmed
    # loop (the loop session must close at least one), bounded by the oracle's
    # observed ceiling, NOT a fixed equality.
    max_loops = max(loop_counts) if loop_counts else 0
    max_map_kf = max(map_kf_counts) if map_kf_counts else 0
    print(f"\n  received: slam.map={len(map_kf_counts)} (max kf={max_map_kf}) "
          f"loop.correction={len(loop_counts)} (max n_loops={max_loops})")
    print(f"  ends: {ends}")

    _check(cap_proc.returncode == 0,
           f"imu_camera exited 0 (got {cap_proc.returncode})")
    _check(vio_proc.returncode == 0,
           f"vio exited 0 (got {vio_proc.returncode})")
    _check(slam_proc.returncode == 0,
           f"slam exited 0 (got {slam_proc.returncode})")

    _check(len(map_kf_counts) > 0, "slam.map overlay received (kf dots stream)")
    _check(max_map_kf > 1,
           f"slam.map advances: keyframe count grew to {max_map_kf}")
    _check(1 <= max_loops <= args.expect_loops,
           f"loop.correction closed 1..{args.expect_loops} loops "
           f"(oracle ceiling; got {max_loops})")

    print("\nALL PROC3 SMOKE SELFTESTS PASSED")
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
