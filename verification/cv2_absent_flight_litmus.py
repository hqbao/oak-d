#!/usr/bin/env python3
"""LITMUS: the FLIGHT stack runs end-to-end with OpenCV (cv2) UNINSTALLED.

The Pi flight install (``requirements-flight.txt``) deliberately omits the heavy
``opencv-python`` wheel. This harness PROVES the default ToF flight path imports
and runs with cv2 absent -- WITHOUT uninstalling cv2 from the dev venv -- by
injecting a ``sitecustomize.py`` (on ``PYTHONPATH``) that makes ``import cv2``
raise ``ImportError`` in every spawned interpreter. ``sitecustomize`` auto-runs
at interpreter startup (before any flight code), so the block is total and
applies to all three subprocesses.

It spawns the real 3-process flight pipeline in the ToF + dense-direct recipe::

    imu_camera.main --vl53l9cx  --oak.cap-->  vio.main --direct  --oak.vio-->  slam.main

and asserts:

* all three processes exit ``rc=0`` once the replay session ends;
* the SLAM ``slam.map`` overlay advances (keyframe count strictly grows -> the
  whole chain -- ToF downsample, direct VO, SLAM relay -- actually produced
  frames, not just imported clean);
* NO subprocess imported cv2 (each prints a one-line guard at startup; a leak
  would have raised ImportError and failed rc).

This is the headline gate for the "cv2-free flight runtime" work. The math
parity of each replaced op is proven separately (the per-op parity checks); this
proves the *integration*: the lean Pi install runs the default flight.

Run::

    python -m verification.cv2_absent_flight_litmus
    python -m verification.cv2_absent_flight_litmus --session sessions/gold/push_straight_fast_15s
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slam.comms import IPCPubSub, topics                          # noqa: E402
from slam.comms.messages import END                              # noqa: E402


# The sitecustomize.py body that blocks cv2 in every child interpreter. Placed on
# PYTHONPATH so CPython auto-imports it at startup, before any flight module.
_BLOCKER_SRC = '''\
"""Auto-loaded cv2 blocker (simulate opencv-python UNINSTALLED on the Pi)."""
import sys, importlib.abc


class _BlockCv2(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "cv2" or name.startswith("cv2."):
            raise ImportError(
                "cv2 blocked by cv2_absent_flight_litmus (simulated absent): "
                + name)
        return None


for _m in [m for m in list(sys.modules) if m == "cv2" or m.startswith("cv2.")]:
    del sys.modules[_m]
sys.meta_path.insert(0, _BlockCv2())
'''


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_straight_20s",
                    help="gold session to replay (the ToF + --direct recipe is "
                         "tuned for the straight/push/quick sessions)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = full session; >0 for a quick smoke")
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--keep-logs", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    session = (repo / args.session) if not Path(args.session).is_absolute() \
        else Path(args.session)
    if not session.exists():
        print(f"  [FAIL] session not found: {session}", file=sys.stderr)
        return 1

    pid = os.getpid()
    cap_ep = f"oak.cap.l{pid & 0xFFF:x}"
    vio_ep = f"oak.vio.l{pid & 0xFFF:x}"
    slam_ep = f"oak.slm.l{pid & 0xFFF:x}"
    py = sys.executable

    with tempfile.TemporaryDirectory(prefix="cv2block_") as blockdir:
        (Path(blockdir) / "sitecustomize.py").write_text(_BLOCKER_SRC)
        # Prepend the blocker dir AND the repo to PYTHONPATH so every child both
        # blocks cv2 (sitecustomize) and can import the flight packages.
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            [blockdir, str(repo)] + ([existing] if existing else []))

        print("cv2_absent_flight_litmus (FLIGHT runs with cv2 BLOCKED)")
        print(f"  session={args.session}  recipe=--vl53l9cx --direct")
        print(f"  endpoints: cap={cap_ep!r} vio={vio_ep!r} slam={slam_ep!r}")
        print(f"  cv2 blocker: {Path(blockdir) / 'sitecustomize.py'}")

        # Sanity: the blocker really blocks cv2 in a child interpreter.
        probe = subprocess.run(
            [py, "-c", "import cv2"], env=env, capture_output=True, text=True)
        _check(probe.returncode != 0 and "blocked" in probe.stderr,
               "cv2 is BLOCKED in spawned interpreters "
               "(import cv2 -> ImportError)")

        log_kwargs = ({} if args.keep_logs
                      else {"stdout": subprocess.PIPE,
                            "stderr": subprocess.PIPE})

        # ---- boot vio (--direct) + slam first; they connect with retries ----
        vio_proc = subprocess.Popen(
            [py, "-m", "vio.main",
             "--capture-endpoint", cap_ep, "--endpoint", vio_ep,
             "--slam-endpoint", slam_ep,
             "--kf-every", str(args.kf_every), "--direct"],
            env=env, **log_kwargs)
        slam_proc = subprocess.Popen(
            [py, "-m", "slam.main",
             "--vio-endpoint", vio_ep, "--endpoint", slam_ep],
            env=env, **log_kwargs)
        time.sleep(0.3)
        # ---- capture in the ToF (--vl53l9cx) recipe ----
        cap_proc = subprocess.Popen(
            [py, "-m", "imu_camera.main",
             "--endpoint", cap_ep, "--session", str(session),
             "--max-frames", str(args.max_frames), "--vl53l9cx"],
            env=env, **log_kwargs)

        procs = (("imu_camera", cap_proc), ("vio", vio_proc), ("slam", slam_proc))
        try:
            rc = _run_assertions(slam_ep, procs)
        finally:
            for _, p in procs:
                if p.poll() is None:
                    p.terminate()
            if not args.keep_logs:
                for name, p in procs:
                    try:
                        _, err = p.communicate(timeout=3.0)
                    except Exception:                              # noqa: BLE001
                        _, err = b"", b""
                    if err and err.strip():
                        tail = "\n".join(
                            err.decode(errors="replace").rstrip()
                            .splitlines()[-20:])
                        print(f"\n  --- {name}.stderr (tail) ---\n{tail}",
                              file=sys.stderr)
        return rc


def _run_assertions(slam_ep, procs) -> int:
    cap_proc = procs[0][1]
    vio_proc = procs[1][1]
    slam_proc = procs[2][1]

    map_kf_counts: list[int] = []
    ends = {"map": 0}
    ready = threading.Event()
    done = threading.Event()

    def on_calib(_m) -> None:
        ready.set()

    def on_map(msg) -> None:
        if msg is END:
            ends["map"] += 1
            done.set()
            return
        map_kf_counts.append(int(len(msg.kf_positions)))

    client = IPCPubSub(slam_ep, role="client", connect_timeout_s=30.0)
    client.subscribe("calib.bundle", on_calib)
    client.subscribe(topics.SLAM_MAP, on_map)
    client.start()
    try:
        if not ready.wait(timeout=30.0):
            raise TimeoutError(f"no calib.bundle from {slam_ep!r} in 30s")
        print("  slam: ready")
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline:
            if done.is_set():
                break
            for name, proc in procs:
                if proc.poll() is not None and proc.returncode != 0:
                    print(f"\n  [FAIL] {name} exited early rc={proc.returncode}",
                          file=sys.stderr)
                    return 1
            time.sleep(0.1)
    finally:
        client.stop()

    cap_proc.wait(timeout=20.0)
    vio_proc.wait(timeout=20.0)
    slam_proc.wait(timeout=20.0)

    kf_max = max(map_kf_counts) if map_kf_counts else 0
    print(f"\n  received: slam.map={len(map_kf_counts)} (max kf={kf_max})")

    _check(cap_proc.returncode == 0, f"imu_camera exited 0 (got {cap_proc.returncode})")
    _check(vio_proc.returncode == 0, f"vio --direct exited 0 (got {vio_proc.returncode})")
    _check(slam_proc.returncode == 0, f"slam exited 0 (got {slam_proc.returncode})")
    _check(len(map_kf_counts) > 0, "slam.map overlay received (chain produced frames)")
    _check(kf_max > 1, f"slam.map advances: keyframe count grew to {kf_max}")

    print("\nLITMUS PASSED: the FLIGHT stack ran with cv2 ABSENT "
          "(--vl53l9cx --direct, rc=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
