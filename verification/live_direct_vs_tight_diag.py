#!/usr/bin/env python3
"""LIVE diagnostic: does the REAL --direct IPC code path achieve the OFFLINE
--direct bench numbers, and how does live --direct compare to live --tight?

Spawns the REAL 2-process live pipeline over IPC (exactly like proc3_smoke /
the launcher), captures the published ``pose.odom`` stream headlessly, and scores
it against the Basalt reference with the SAME umeyama/ate/load_basalt_positions
helpers loose_vs_tight_bench uses.

    imu_camera.main --vl53l9cx --session SESS --endpoint <cap>
        --(pose.odom)-->  vio.main {--direct|--tight} --capture-endpoint <cap>

Per session it reports Sim3 scale + ATE RMSE (cm) + okfrac (= distinct pose seqs
that overlap the Basalt ref / Basalt frame count -- the real-time frame-drop
indicator; offline this is ~1.0). Read-only: spawns the live processes as-is,
never edits any frozen path.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verification.oracle_replay import ate, load_basalt_positions  # noqa: E402
from vio.comms import IPCPubSub, topics                            # noqa: E402
from vio.comms.messages import END                                 # noqa: E402


def _spawn(session: str, mode: str, *, vl53: bool, keep_logs: bool):
    """Spawn imu_camera + vio for one (session, mode); return (procs, slam_ep|None, captured).

    mode in {"direct", "tight"}.  Returns a live subscriber's captured dict.
    """
    tag = f"{mode}{abs(hash((session, mode))) & 0xFFF:x}"
    cap_ep = f"oak.cap.{tag}"
    vio_ep = f"oak.vio.{tag}"

    py = sys.executable
    env = dict(os.environ)
    log_kwargs = ({} if keep_logs
                  else {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE})

    # --- vio first (retried client connects to capture's retained calib) ----
    vio_cmd = [py, "-m", "vio.main",
               "--capture-endpoint", cap_ep, "--endpoint", vio_ep,
               "--kf-every", "5"]
    if mode == "direct":
        vio_cmd += ["--direct"]
    elif mode == "tight":
        vio_cmd += ["--tight"]
    else:
        raise ValueError(mode)
    vio_proc = subprocess.Popen(vio_cmd, env=env, **log_kwargs)

    time.sleep(0.3)  # let vio register its calib subscriber before capture boots

    cap_cmd = [py, "-m", "imu_camera.main",
               "--endpoint", cap_ep, "--session", session, "--max-frames", "0"]
    if vl53:
        cap_cmd += ["--vl53l9cx"]
    cap_proc = subprocess.Popen(cap_cmd, env=env, **log_kwargs)

    return (cap_proc, vio_proc), vio_ep, vio_cmd, cap_cmd


def _capture_poses(vio_ep: str, procs, *, timeout_s: float = 240.0) -> dict[int, np.ndarray]:
    """Headless client on vio_ep: collect every pose.odom {seq: position}.

    Keeps ONE persistent subscriber for the whole run (catches the final pose +
    the END sentinel). Returns {seq: (3,) world position}.
    """
    cap_proc, vio_proc = procs
    poses: dict[int, np.ndarray] = {}
    info_acc: dict[str, int] = {"n": 0, "rejected": 0, "diverged": 0, "valid_sum": 0.0}
    got_end = threading.Event()

    def on_pose(wm) -> None:
        if wm is END:
            got_end.set()
            return
        try:
            poses[int(wm.seq)] = np.asarray(wm.T_world_cam, dtype=np.float64)[:3, 3].copy()
            info = getattr(wm, "info", None) or {}
            info_acc["n"] += 1
            if info.get("rejected"):
                info_acc["rejected"] += 1
            if info.get("diverged"):
                info_acc["diverged"] += 1
            if "valid_frac" in info:
                info_acc["valid_sum"] += float(info["valid_frac"])
        except Exception:  # noqa: BLE001
            pass

    client = IPCPubSub(vio_ep, role="client", connect_timeout_s=30.0)
    client.subscribe(topics.POSE_ODOM, on_pose)
    client.start()
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if got_end.is_set():
                break
            # bail early if a child died with an error
            for p in (cap_proc, vio_proc):
                if p.poll() is not None and p.returncode not in (0, None):
                    # capture may exit 0 first (replay done) -> not an error; keep
                    # draining until END or both gone.
                    pass
            if cap_proc.poll() is not None and vio_proc.poll() is not None:
                # both exited; give the wire a beat then stop
                time.sleep(0.3)
                break
            time.sleep(0.05)
    finally:
        client.stop()
    return poses, info_acc


def _score(poses: dict[int, np.ndarray], session_dir: Path) -> dict | None:
    basalt = load_basalt_positions(session_dir)
    if not basalt:
        return None
    n_basalt = len(basalt)
    common = sorted(set(poses) & set(basalt))
    okfrac = len(common) / float(n_basalt) if n_basalt else 0.0
    if len(common) < 10:
        return {"ate_cm": None, "scale": None, "okfrac": okfrac,
                "n_poses": len(poses), "n_common": len(common), "n_basalt": n_basalt}
    src = np.array([poses[s] for s in common])
    dst = np.array([basalt[s] for s in common])
    rigid = ate(src, dst, with_scale=False)
    sim = ate(src, dst, with_scale=True)
    return {"ate_cm": rigid["rmse"] * 100.0, "scale": sim["scale"], "okfrac": okfrac,
            "n_poses": len(poses), "n_common": len(common), "n_basalt": n_basalt}


def _terminate(*procs):
    for p in procs:
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass
    for p in procs:
        try:
            p.wait(timeout=5.0)
        except Exception:  # noqa: BLE001
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass


def run_one(session: str, mode: str, *, vl53: bool, keep_logs: bool) -> dict:
    sess_dir = Path(session)
    procs, vio_ep, vio_cmd, cap_cmd = _spawn(session, mode, vl53=vl53, keep_logs=keep_logs)
    print(f"  [{mode}] cap_cmd: {' '.join(cap_cmd)}")
    print(f"  [{mode}] vio_cmd: {' '.join(vio_cmd)}")
    t0 = time.perf_counter()
    try:
        poses, info_acc = _capture_poses(vio_ep, procs)
    finally:
        _terminate(*procs)
        if not keep_logs:
            for name, p in (("imu_camera", procs[0]), ("vio", procs[1])):
                try:
                    _, err = p.communicate(timeout=2.0)
                except Exception:  # noqa: BLE001
                    err = b""
                if err and err.strip():
                    tail = "\n".join(err.decode(errors="replace").rstrip().splitlines()[-8:])
                    print(f"    --- {name}.stderr (tail) ---\n{tail}", file=sys.stderr)
    wall = time.perf_counter() - t0
    m = _score(poses, sess_dir) or {}
    m["wall_s"] = wall
    m["rc"] = (procs[0].returncode, procs[1].returncode)
    n_info = info_acc["n"]
    m["reject_frac"] = (info_acc["rejected"] / n_info) if n_info else 0.0
    m["diverge_frac"] = (info_acc["diverged"] / n_info) if n_info else 0.0
    m["valid_frac_mean"] = (info_acc["valid_sum"] / n_info) if n_info else 0.0
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", nargs="*", default=[
        "sessions/gold/lab_straight_20s",
        "sessions/gold/push_straight_fast_15s",
        "sessions/gold/quick_motion_15s",
    ])
    ap.add_argument("--modes", nargs="*", default=["direct", "tight"])
    ap.add_argument("--no-vl53", action="store_true",
                    help="omit --vl53l9cx (full-res); default is the 54x42 ToF recipe")
    ap.add_argument("--keep-logs", action="store_true")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat each (session,mode) N times (live okfrac is noisy)")
    args = ap.parse_args()

    vl53 = not args.no_vl53
    results: dict[tuple[str, str, int], dict] = {}
    for session in args.sessions:
        for mode in args.modes:
            for r in range(args.repeats):
                label = f"{Path(session).name} / {mode}" + (f" #{r+1}" if args.repeats > 1 else "")
                print(f"\n=== LIVE {label} (vl53={vl53}) ===")
                m = run_one(session, mode, vl53=vl53, keep_logs=args.keep_logs)
                results[(session, mode, r)] = m
                sc = m.get("scale"); at = m.get("ate_cm")
                print(f"    -> scale={sc if sc is None else round(sc,3)} "
                      f"ATE={at if at is None else round(at,1)}cm "
                      f"okfrac={round(m.get('okfrac',0),3)} "
                      f"(poses={m.get('n_poses')} common={m.get('n_common')}/"
                      f"{m.get('n_basalt')}) rc={m.get('rc')} wall={m['wall_s']:.1f}s")
                if mode == "direct":
                    print(f"       direct-info: reject_frac={m.get('reject_frac',0):.3f} "
                          f"diverge_frac={m.get('diverge_frac',0):.3f} "
                          f"valid_frac_mean={m.get('valid_frac_mean',0):.3f}")

    # ---------------- summary table ----------------
    print("\n" + "=" * 92)
    print("LIVE A/B SUMMARY  (scale = Sim3 vs Basalt | ATE = rigid-SE3 RMSE cm | "
          "okfrac = poses/basalt)")
    print("=" * 92)
    hdr = f"{'session':26s} {'mode':8s} {'scale':>7s} {'ATE cm':>8s} {'okfrac':>8s} {'poses':>7s} {'rc':>10s}"
    print(hdr); print("-" * len(hdr))
    for session in args.sessions:
        for mode in args.modes:
            ms = [results[(session, mode, r)] for r in range(args.repeats)]
            for r, m in enumerate(ms):
                sc = m.get("scale"); at = m.get("ate_cm")
                name = Path(session).name + (f"#{r+1}" if args.repeats > 1 else "")
                print(f"{name:26s} {mode:8s} "
                      f"{('--' if sc is None else f'{sc:.3f}'):>7s} "
                      f"{('--' if at is None else f'{at:.1f}'):>8s} "
                      f"{m.get('okfrac',0):>8.3f} {str(m.get('n_poses')):>7s} "
                      f"{str(m.get('rc')):>10s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
