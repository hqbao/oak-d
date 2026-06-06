#!/usr/bin/env python3
"""Self-test: the OUT-OF-PROCESS engine computes byte-identically to the in-process one.

The live ``ours-ba`` / ``ours-slam`` sources run the heavy BA / SLAM solve in a
separate process (:class:`ours.lib.engine.subprocess.SubprocessEngine`) so it never
holds the camera read loop's GIL. This proves the child produces the SAME numbers
as the in-process engine (:class:`ours.lib.engine.inprocess.InProcessEngine`) used
by the deterministic offline path -- i.e. moving the solve across the process
boundary (pickling the keyframe snapshot, incl. the gray/depth images for SLAM)
changes nothing.

Method
------
1. Replay a gold session through the odometry front-end only and capture every
   ``keyframe`` message off the bus (the exact snapshots the back-end/SLAM flows
   would receive).
2. Feed those snapshots, in order, to the in-process engine (synchronous) and the
   subprocess engine, and compare the per-keyframe results.

The subprocess is fed with a BLOCKING put on its input queue (not the lossy
latest-wins ``submit``) so EVERY keyframe is delivered in order -- a deterministic,
no-drop run for the comparison. The live path is deliberately lossy (it only wants
the freshest map behind a responsive marker); that is a separate concern not under
test here.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.app import build_replay                                  # noqa: E402
from ours.lib.io.reader import SessionReader                       # noqa: E402
from ours.lib.flow.messages import END                            # noqa: E402
from ours.lib.flow.pubsub import Bus                              # noqa: E402
from ours.lib.flow import topics                                  # noqa: E402
from ours.lib.backend.bundle import BAConfig                      # noqa: E402
from ours.lib.backend.windowed import WindowedConfig             # noqa: E402
from ours.lib.loop.slam import SlamConfig                         # noqa: E402
from ours.lib.engine import make_ba_engine, make_slam_engine     # noqa: E402
from ours.lib.engine.base import SlamResult                       # noqa: E402


def collect_keyframes(session: str, *, kf_every: int, max_frames: int):
    """Replay ``session`` through odometry only; return ``(K, [Keyframe...])``."""
    reader = SessionReader(Path(session))
    bus = Bus()
    (cam_flow, imu_flow), flows, _ui = build_replay(
        bus, reader, kf_every=kf_every, use_gyro=True, depth_fast=True,
        max_frames=max_frames, with_backend_slam=False)
    kfs: list = []
    bus.subscribe(topics.KEYFRAME, lambda m: None if m is END else kfs.append(m))
    odom = flows[0]
    for f in flows:
        f.start()
    imu_flow.start()
    cam_flow.start()
    cam_flow.join()
    odom.done.wait(timeout=120.0)        # both ENDs drained => all keyframes emitted
    imu_flow.stop()
    for f in flows:
        f.stop()
    return reader.K, kfs


def _ba_snaps(kfs):
    """The (snapshot, valid) list the back-end flow would submit per keyframe."""
    snaps = []
    for kf in kfs:
        if kf.track_ids is None or kf.track_px is None:
            continue                      # RunBA skips these before submitting
        snaps.append((np.linalg.inv(kf.T_world_cam), kf.track_ids, kf.track_px,
                      kf.depth_m, kf.accel))
    return snaps


def _slam_snaps(kfs):
    return [(kf.T_world_cam, kf.gray_left, kf.depth_m, kf.seq) for kf in kfs]


def _run_inprocess(engine, snaps):
    out = []
    for s in snaps:
        engine.submit(s)
        out.append(engine.poll())
    engine.close()
    return out


def _run_subprocess(engine, snaps, expect_nonnull):
    """Feed every snapshot no-drop (blocking put); collect per-keyframe results."""
    engine.start()                        # spawn now (lazy: not done in __init__)
    out = []
    for i, s in enumerate(snaps):
        engine._in_q.put(s)               # blocking => no input drop, in order
        if expect_nonnull[i]:
            r, t0 = None, time.monotonic()
            while r is None and time.monotonic() - t0 < 15.0:
                r = engine.poll()
                if r is None:
                    time.sleep(0.003)
            out.append(r)
        else:
            out.append(None)
    engine.close()
    return out


def _cmp_ba(a, b) -> float:
    """Max abs diff between two BA results (4x4 or None); inf on shape mismatch."""
    if a is None and b is None:
        return 0.0
    if (a is None) != (b is None):
        return float("inf")
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _cmp_slam(a: SlamResult | None, b: SlamResult | None) -> float:
    if a is None and b is None:
        return 0.0
    if (a is None) != (b is None):
        return float("inf")
    if a.n_loops != b.n_loops or set(a.kf_poses) != set(b.kf_poses):
        return float("inf")
    return max((float(np.max(np.abs(a.kf_poses[k] - b.kf_poses[k])))
                for k in a.kf_poses), default=0.0)


def check_ba(session: str, kf_every: int, max_frames: int, tol: float) -> bool:
    K, kfs = collect_keyframes(session, kf_every=kf_every, max_frames=max_frames)
    snaps = _ba_snaps(kfs)
    cfg = WindowedConfig(window=6, ba=BAConfig(max_iters=5))   # == BackendFlow default
    ip = _run_inprocess(make_ba_engine(K, cfg, worker=False), snaps)
    expect = [r is not None for r in ip]
    sp = _run_subprocess(make_ba_engine(K, cfg, worker=True), snaps, expect)
    diffs = [_cmp_ba(a, b) for a, b in zip(ip, sp)]
    n_ref = sum(expect)
    worst = max(diffs) if diffs else 0.0
    ok = worst <= tol
    print(f"  BA   {Path(session).name:18s} kfs={len(snaps):3d} refined={n_ref:3d} "
          f"max|ip-sp|={worst:.2e}  [{'ok' if ok else 'FAIL'}]")
    return ok


def _ov_kf(ov):
    """Keyframe-position array of a SLAM overlay snapshot (kf_pos, n_loops, ...)."""
    return ov[0] if ov is not None else np.zeros((0, 3))


def _ov_nloops(ov):
    return int(ov[1]) if ov is not None else 0


def check_slam(session: str, kf_every: int, max_frames: int, tol: float) -> bool:
    """SLAM parity: every keyframe (image-bearing) crosses the process boundary
    no-drop and the child builds an identical map -- same per-keyframe correction
    (the pose-graph optimise output on each confirmed loop), same loop count, same
    keyframe positions. ``quick_motion`` revisits its view enough to fire dozens of
    closures, so this exercises the full ORB + pose-graph path AND the uint8 image
    pickling across the boundary, not just the no-op add."""
    K, kfs = collect_keyframes(session, kf_every=kf_every, max_frames=max_frames)
    snaps = _slam_snaps(kfs)
    cfg = SlamConfig(loop_max_odom_rot_deg=30.0)              # == SlamFlow default

    eip = make_slam_engine(K, cfg, worker=False)
    ip = []
    for s in snaps:
        eip.submit(s)
        ip.append(eip.poll())
    ov_ip = eip.poll_overlay()
    eip.close()

    expect = [r is not None for r in ip]
    esp = make_slam_engine(K, cfg, worker=True)
    esp.start()                           # spawn now (lazy: not done in __init__)
    sp = []
    for i, s in enumerate(snaps):
        esp._in_q.put(s)                  # blocking => no input drop, in order
        if expect[i]:
            r, t0 = None, time.monotonic()
            while r is None and time.monotonic() - t0 < 15.0:
                r = esp.poll()
                if r is None:
                    time.sleep(0.003)
            sp.append(r)
        else:
            sp.append(None)
    # Drain the freshest overlay once the child has caught up to the full map.
    ov_sp, t0 = None, time.monotonic()
    target = len(_ov_kf(ov_ip))
    while time.monotonic() - t0 < 15.0:
        o = esp.poll_overlay()
        if o is not None:
            ov_sp = o
        if ov_sp is not None and len(_ov_kf(ov_sp)) >= target:
            break
        time.sleep(0.02)
    esp.close()

    corr_diff = max((_cmp_slam(a, b) for a, b in zip(ip, sp)), default=0.0)
    kf_ip, kf_sp = _ov_kf(ov_ip), _ov_kf(ov_sp)
    ov_diff = (float(np.max(np.abs(kf_ip - kf_sp)))
               if kf_ip.shape == kf_sp.shape and kf_ip.size else
               (0.0 if kf_ip.size == kf_sp.size == 0 else float("inf")))
    ok = (corr_diff <= tol and kf_ip.shape == kf_sp.shape
          and _ov_nloops(ov_ip) == _ov_nloops(ov_sp) and ov_diff <= tol)
    print(f"  SLAM {Path(session).name:18s} kfs={len(snaps):3d} "
          f"map_kfs={len(kf_ip)}/{len(kf_sp)} loops={_ov_nloops(ov_ip)} "
          f"max|ip-sp|={max(corr_diff, ov_diff):.2e}  [{'ok' if ok else 'FAIL'}]")
    return ok


def main() -> int:
    print("engine_parity_selftest")
    tol = 1e-9
    ok = True
    # BA: a normal-motion gold session exercises the window warm-up + many solves.
    ok &= check_ba("sessions/gold/lab_loop_30s", kf_every=5, max_frames=150, tol=tol)
    # SLAM: quick_motion revisits its view enough to fire dozens of loop closures,
    # so this exercises the full pose-graph optimise path + image pickling.
    ok &= check_slam("sessions/gold/quick_motion_15s", kf_every=5, max_frames=0, tol=tol)
    print("ALL ENGINE PARITY SELFTESTS PASSED" if ok else "ENGINE PARITY SELFTEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
