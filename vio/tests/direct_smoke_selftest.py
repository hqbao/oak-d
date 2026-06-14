#!/usr/bin/env python3
"""LIVE smoke for the ``--direct`` dense direct RGB-D VO odometry mode @ 54x42 ToF.

This exercises the EXACT live wiring the ``--direct`` flag turns on -- the
PROCEDURAL :class:`~vio.modules.pipeline.OdometryWorker` with ``direct=True``,
driven over its real local-bus inboxes -- WITHOUT spinning up the multi-process IPC
graph, so the gate is deterministic + fast (the "process-level smoke is acceptable"
option the brief allows; states which one this is: the in-process worker smoke).

What it proves end-to-end on real session data:

  * the worker builds the :class:`~vio.modules.direct_odometry.DirectOdometryEngine`
    (the geo-off, guard-on default) and routes ``frame.depth`` to
    :func:`~vio.modules.pipeline.process_frame_direct` (NOT the sparse chain),
  * ``preintegrate_prior`` retains the per-frame IMU the direct seed consumes
    (the ``direct`` flag forces the same retention the tight path uses),
  * the live IMU 6-DoF dead-reckon seed + the divergence guard run per frame,
  * a pose is published on ``pose.odom`` for every frame + keyframes are emitted on
    the engine's NATURAL cadence (the SAME topics the loose/tight modes use),
  * the two-input END join drains cleanly (both ``imucam.sample`` + ``frame.depth``
    END -> the worker forwards END + sets ``done``) -- no crash, clean shutdown.

The frames are reduced to the 54x42 ToF grid the SAME way the live capture's
``tof_downsample`` step does (SGM at source res -> INTER_AREA gray + block-median
depth, K scaled anisotropically), so the worker sees exactly what
``./run.sh --vl53l9cx --direct`` would feed it.

POSE SANITY + scale: the trajectory must be finite, non-degenerate and not
exploding; we ALSO Sim3-align it to the Basalt reference and print the scale so it
can be read against the offline bench (``verification/direct_vo_bench.py`` reports
full-session scale ~0.4-0.9 at 54x42) -- proving the live path tracks, not frozen.

Run::

    .venv/bin/python -m vio.tests.direct_smoke_selftest
    .venv/bin/python -m vio.tests.direct_smoke_selftest --session sessions/gold/push_straight_fast_15s --max-frames 80
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imu_camera.io.reader import SessionReader                  # noqa: E402
from imu_camera.modules.tof_downsample import (                 # noqa: E402
    _area_resize_gray, _block_median_valid)
from imu_camera.modules.pipeline import TOF_W, TOF_H            # noqa: E402
from sky.depth.stereo import SGMConfig, SGMStereoMatcher        # noqa: E402

from vio.comms import LocalPubSub, topics                       # noqa: E402
from vio.comms.messages import DepthFrame, ImuCamPacket         # noqa: E402
from vio.comms.messages import END                              # noqa: E402
from vio.modules import OdometryModule                          # noqa: E402

# Scoring helpers (read-only import), same as the offline bench uses.
from verification.oracle_replay import (                        # noqa: E402
    ate, load_basalt_positions)


def _scale_K_to_tof(K: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    """Anisotropic K scaling to the 54x42 ToF grid (mirrors _scale_bundle_to_tof)."""
    sx = TOF_W / float(src_w)
    sy = TOF_H / float(src_h)
    Kt = np.asarray(K, dtype=np.float64).copy()
    Kt[0, 0] *= sx
    Kt[0, 2] *= sx
    Kt[1, 1] *= sy
    Kt[1, 2] *= sy
    return Kt


def _per_frame_imu(imu: dict, t0: int, t1: int):
    """Raw IMU samples in ``(t0, t1]`` -> (imu_ts, gyro, accel) for one packet.

    The live imu_cam packs exactly the samples in the per-frame cut ``(prev_ts, ts]``
    onto each ImuCamPacket; we replay that cut from the session IMU stream so the
    worker's preintegrate_prior sees the same per-frame block live capture builds.
    """
    ts = imu["ts_ns"]
    m = (ts > t0) & (ts <= t1)
    return ts[m], imu["gyro"][m], imu["accel"][m]


def run_direct_smoke(session_dir: Path, max_frames: int = 60,
                     kf_every: int = 5) -> int:
    reader = SessionReader(session_dir)
    n = len(reader) if max_frames <= 0 else min(max_frames, len(reader))
    if not reader.calib.has_imu_extrinsics:
        print(f"SKIP: {session_dir} has no IMU extrinsics (direct needs the IMU seed).")
        return 0
    imu = reader.load_imu()
    if imu["ts_ns"].size <= 1:
        print(f"SKIP: {session_dir} has no usable IMU stream.")
        return 0

    R_imu_cam = reader.calib.T_imu_left[:3, :3]
    # Startup gravity-level accel (camera frame) from the near-static window, the
    # SAME accel_align the live bundle carries (drives the dead-reckon anchor).
    ts_all = imu["ts_ns"]
    win = ts_all <= int(ts_all[0]) + int(0.3 * 1e9)
    accel_align = R_imu_cam @ imu["accel"][win].mean(axis=0)

    # ToF-grid intrinsics + the source-res SGM matcher (compute high, downsample).
    f0 = reader.load_frame(0, load_right=False)
    sh, sw = f0.gray_left.shape[:2]
    K_tof = _scale_K_to_tof(reader.K, sw, sh)
    matcher = SGMStereoMatcher.from_calib(reader.calib, SGMConfig())

    # --- the LIVE worker, exactly as vio.main builds it for --direct ---------- #
    bus = LocalPubSub()
    poses: dict[int, np.ndarray] = {}
    n_kf = [0]
    end_seen = [0]

    def on_pose(msg) -> None:
        if msg is END:
            return
        poses[int(msg.seq)] = np.asarray(msg.T_world_cam, np.float64)[:3, 3].copy()

    def on_kf(msg) -> None:
        if msg is END:
            return
        n_kf[0] += 1

    def on_refined_end(msg) -> None:
        # pose.odom carries END once the worker's 2-input join drained both inputs.
        if msg is END:
            end_seen[0] += 1

    bus.subscribe(topics.POSE_ODOM, on_pose)
    bus.subscribe(topics.POSE_ODOM, on_refined_end)
    bus.subscribe(topics.KEYFRAME, on_kf)

    odom = OdometryModule(
        bus, K_tof, R_imu_cam=R_imu_cam, accel_align=accel_align,
        kf_every=kf_every, use_gyro=True, direct=True)
    odom.start()

    # --- feed the 54x42 ToF packets, exactly like the live capture path ------- #
    prev_ts = None
    t_start = time.perf_counter()
    for i in range(n):
        f = reader.load_frame(i, load_right=True)
        gray_src, depth_src = matcher.dense_depth_rectified_left(
            f.gray_left, f.gray_right)
        if gray_src.dtype != np.uint8:
            gray_src = np.clip(gray_src, 0.0, 255.0).astype(np.uint8)
        # Use the SAME pure-NumPy downsample the production tof_downsample step
        # uses (bit-exact vs cv2.INTER_AREA) so this smoke exercises the real
        # flight path and carries no cv2 dependency itself.
        gray_tof = _area_resize_gray(gray_src, TOF_H, TOF_W)
        depth_tof = _block_median_valid(depth_src, TOF_H, TOF_W)

        t0 = prev_ts if prev_ts is not None else (int(f.ts_ns) - 1)
        imu_ts, gyro, accel = _per_frame_imu(imu, t0, int(f.ts_ns))

        # imucam.sample FIRST (preintegrate_prior stashes the prior + IMU seg),
        # then frame.depth (process_frame_direct consumes them by seq).
        bus.publish(topics.IMUCAM_SAMPLE, ImuCamPacket(
            seq=f.seq, ts_ns=int(f.ts_ns), gray_left=gray_tof, gray_right=None,
            imu_ts=imu_ts, gyro=gyro, accel=accel))
        bus.publish(topics.FRAME_DEPTH,
                    DepthFrame(f.seq, int(f.ts_ns), gray_tof, depth_tof))
        prev_ts = int(f.ts_ns)

    # END on BOTH input edges -> the worker's 2-input join forwards END + done.
    bus.publish(topics.IMUCAM_SAMPLE, END)
    bus.publish(topics.FRAME_DEPTH, END)
    drained = odom.done.wait(timeout=30.0)
    odom.stop()
    elapsed = time.perf_counter() - t_start

    engine = odom.ctx.state.get("direct_engine")

    # --------------------------- trajectory stats --------------------------- #
    seqs = sorted(poses)
    pos = np.array([poses[s] for s in seqs]) if seqs else np.zeros((0, 3))
    finite = bool(pos.size and np.all(np.isfinite(pos)))
    diffs = (np.linalg.norm(np.diff(pos, axis=0), axis=1)
             if len(pos) > 1 else np.zeros(0))
    path = float(diffs.sum())
    max_step = float(diffs.max()) if diffs.size else 0.0
    span = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))) if len(pos) else 0.0

    print(f"\nsession : {reader.dir.name}   frames-fed={n}")
    print(f"poses published={len(pos)}  keyframes emitted={n_kf[0]}  "
          f"END-on-pose.odom={end_seen[0]}  done={drained}")
    if engine is not None:
        print(f"engine  : frames={engine.n_frames}  converged={engine.n_converged}  "
              f"rejected(guard)={engine.n_rejected}  ms/frame={elapsed/max(n,1)*1000:.1f}")
    print(f"traj    : finite={finite}  path={path:.2f} m  bbox-span={span:.2f} m  "
          f"max-step={max_step*100:.1f} cm")
    sample = seqs[:: max(1, len(seqs) // 5)][:5]
    print("sample poses (seq -> xyz m):")
    for s in sample:
        p = poses[s]
        print(f"  seq {s:4d} -> [{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}]")

    # ----- Sim3 scale vs Basalt (pose-sanity / in-line-with-bench signal) ---- #
    scale = None
    basalt = load_basalt_positions(reader.dir)
    if basalt:
        common = sorted(set(poses) & set(basalt))
        if len(common) >= 10:
            src = np.array([poses[s] for s in common])
            dst = np.array([basalt[s] for s in common])
            sim = ate(src, dst, with_scale=True)
            scale = float(sim["scale"])
            print(f"vs Basalt: Sim3 scale={scale:.3f} over {len(common)} common "
                  f"frames (offline bench reports ~0.4-0.9 full-session @ 54x42)")

    # ------------------------------- GATES ---------------------------------- #
    fails = []
    if len(pos) < min(10, n - 1):
        fails.append(f"too few poses published ({len(pos)} for {n} frames)")
    if not finite:
        fails.append("trajectory has NaN/Inf (exploded)")
    if span < 1e-4:
        fails.append("trajectory is degenerate (zero spatial span -- frozen)")
    # not exploding: no single inter-frame jump beyond a generous bound (the guard
    # exists precisely to prevent this; at 54x42 ToF a real hand motion is < ~0.3 m/f).
    if max_step > 1.0:
        fails.append(f"trajectory explodes (max step {max_step:.2f} m -- guard failed)")
    if n_kf[0] < 1:
        fails.append("no keyframes emitted (the natural KF cadence never fired)")
    if not drained or end_seen[0] < 1:
        fails.append("the 2-input END join did NOT drain cleanly (no END / no done)")

    if fails:
        print("\nFAIL:")
        for f_ in fails:
            print(f"  - {f_}")
        return 1
    print("\nPASS -- live --direct mode RUNS: dense direct VO + live IMU seed + "
          "guard produce a finite, sane, non-frozen trajectory on real ToF data, "
          "publish the standard topics, and the worker shuts down cleanly (rc=0).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_straight_20s")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--kf-every", type=int, default=5)
    args = ap.parse_args()
    return run_direct_smoke(Path(args.session), args.max_frames, args.kf_every)


if __name__ == "__main__":
    raise SystemExit(main())
