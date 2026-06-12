#!/usr/bin/env python3
"""READ-ONLY diagnosis: reproduce the live-pose loop-correction TELEPORT/OSCILLATION.

Symptom (user, live --tight): pointing at one fixed spot, the body frame (the live
``pose.odom`` that drives the drone triad) jumps back-and-forth by a large distance
on SLAM loop closures -- a teleport that REPEATS, not a single snap.

This harness reuses the closed_loop_drift_selftest two-phase structure but instead
of a pass/fail proof it TRACES, per frame around each loop correction:

  * the live blended body->world position emitted on ``pose.odom`` (what the triad
    shows),
  * the pending loop-correction-delta magnitude in PropagateImu,
  * the stream of ``loop.correction`` messages SLAM emits (seq, n_loops, and the
    corrected position it claims for the revisited keyframe).

It then reports, per loop-closing keyframe, the corrected target SLAM emits each
time, and prints the live-pose position trace so the back-and-forth is visible in
real numbers. Nothing is changed; this only instruments the existing modules.

Run::

    .venv/bin/python -m verification.loop_teleport_diag
    .venv/bin/python -m verification.loop_teleport_diag --session sessions/gold/loop_closure_45s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imu_camera.io.reader import SessionReader                       # noqa: E402
from vio.comms import LocalPubSub, topics                            # noqa: E402
from vio.comms.messages import DepthFrame, END, ImuCamPacket         # noqa: E402
from vio.modules import OdometryModule                               # noqa: E402
from sky.front.odometry import OdometryConfig                        # noqa: E402
from slam.modules import SlamModule                                  # noqa: E402
from sky.slam.slam import SlamConfig                                 # noqa: E402
from vio.mathlib.backend.vio_window import T_cw_to_body_world        # noqa: E402


def _per_frame_imu(ts_all, gyro, accel, prev_ts, ts):
    m = ts_all <= ts if prev_ts is None else (ts_all > prev_ts) & (ts_all <= ts)
    return (ts_all[m].astype(np.int64), gyro[m].astype(np.float64),
            accel[m].astype(np.float64))


def _session_imu(session_dir: Path):
    reader = SessionReader(session_dir)
    R_imu_cam = reader.calib.T_imu_left[:3, :3]
    imu = reader.load_imu()
    ts_all = imu["ts_ns"].astype(np.int64)
    gyro = imu["gyro"].astype(np.float64)
    accel = imu["accel"].astype(np.float64)
    t0 = int(ts_all[0])
    win = ts_all <= t0 + int(0.3 * 1e9)
    accel_align = R_imu_cam @ accel[win].mean(axis=0)
    return reader, R_imu_cam, ts_all, gyro, accel, accel_align


def _collect_slam_corrections(session_dir: Path, n: int):
    """PHASE A: run the real OdometryModule + SlamModule; CAPTURE every emitted
    ``loop.correction`` (the WHOLE message, so we can read the corrected pose it
    claims for any keyframe), tagged with the seq it fired at."""
    reader, R_imu_cam, ts_all, gyro, accel, accel_align = _session_imu(session_dir)
    bus = LocalPubSub()
    odom = OdometryModule(
        bus, reader.K, R_imu_cam=R_imu_cam, accel_align=accel_align,
        odom_cfg=OdometryConfig(gyro_fuse=True), kf_every=5, use_gyro=True,
        latest_only=False, level_tilt=True, publish_vo=False,
        retain_imu=True, loop_correct=False)
    slam = SlamModule(bus, reader.K, SlamConfig(loop_max_odom_rot_deg=30.0),
                      latest_only=False, worker=False, publish_map=False)

    corrections: list[tuple[int, object]] = []
    bus.subscribe(topics.LOOP_CORRECTION,
                  lambda m: corrections.append((int(m.seq), m))
                  if m is not END and m is not None else None)
    odom.start()
    slam.start()
    prev_ts = None
    for i in range(n):
        f = reader.load_frame(i)
        its, ig, ia = _per_frame_imu(ts_all, gyro, accel, prev_ts, int(f.ts_ns))
        bus.publish(topics.IMUCAM_SAMPLE,
                    ImuCamPacket(f.seq, int(f.ts_ns), f.gray_left, None,
                                 its, ig, ia))
        bus.publish(topics.FRAME_DEPTH,
                    DepthFrame(f.seq, int(f.ts_ns), f.gray_left, f.depth_m))
        prev_ts = int(f.ts_ns)
    bus.publish(topics.IMUCAM_SAMPLE, END)
    bus.publish(topics.FRAME_DEPTH, END)
    odom.done.wait(timeout=180.0)
    slam.done.wait(timeout=180.0)
    odom.stop()
    slam.stop()
    return corrections


def _trace_closed_loop(session_dir: Path, n: int, corrections):
    """PHASE B: replay the OdometryModule alone with the REAL corrections fed back
    AT their emit seq (deterministic, the live timeline). TRACE per pose.odom frame:
    seq, live position, pending-delta magnitude (from nav['loop_delta'])."""
    reader, R_imu_cam, ts_all, gyro, accel, accel_align = _session_imu(session_dir)
    bus = LocalPubSub()
    odom = OdometryModule(
        bus, reader.K, R_imu_cam=R_imu_cam, accel_align=accel_align,
        odom_cfg=OdometryConfig(gyro_fuse=True), kf_every=5, use_gyro=True,
        latest_only=False, level_tilt=True, publish_vo=False,
        retain_imu=True, loop_correct=True)

    by_seq: dict[int, list] = {}
    for emit_seq, corr in corrections:
        by_seq.setdefault(int(emit_seq), []).append(corr)
    inbox = odom.ctx.state.get("loop_inbox")

    trace: list[dict] = []

    def _on_pose(m):
        if m is END:
            return
        pos = m.T_world_cam[:3, 3].copy()
        nav = odom.ctx.state.get("live_nav") or {}
        pend = nav.get("loop_delta")
        pend_mag = float(np.linalg.norm(pend[1])) if pend is not None else 0.0
        applied = nav.get("loop_applied")
        appl_mag = (float(np.linalg.norm(applied[:3, 3]))
                    if applied is not None else 0.0)
        trace.append({"seq": int(m.seq), "pos": pos,
                      "pend": pend_mag, "applied": appl_mag,
                      "fired": int(m.seq) in by_seq})
        if inbox is not None and int(m.seq) in by_seq:
            for corr in by_seq[int(m.seq)]:
                inbox.push(corr)
    bus.subscribe(topics.POSE_ODOM, _on_pose)

    odom.start()
    prev_ts = None
    for i in range(n):
        f = reader.load_frame(i)
        its, ig, ia = _per_frame_imu(ts_all, gyro, accel, prev_ts, int(f.ts_ns))
        bus.publish(topics.IMUCAM_SAMPLE,
                    ImuCamPacket(f.seq, int(f.ts_ns), f.gray_left, None,
                                 its, ig, ia))
        bus.publish(topics.FRAME_DEPTH,
                    DepthFrame(f.seq, int(f.ts_ns), f.gray_left, f.depth_m))
        prev_ts = int(f.ts_ns)
    bus.publish(topics.IMUCAM_SAMPLE, END)
    bus.publish(topics.FRAME_DEPTH, END)
    odom.done.wait(timeout=180.0)
    odom.stop()
    return trace


def run(session_dir: Path, max_frames: int) -> int:
    reader = SessionReader(session_dir)
    n = len(reader) if max_frames <= 0 else min(max_frames, len(reader))
    print(f"session: {session_dir.name}  frames={n}\n")

    print("phase A: real SLAM -> collect loop.correction stream ...")
    corrections = _collect_slam_corrections(session_dir, n)
    print(f"  SLAM emitted {len(corrections)} loop.correction message(s)\n")

    # --- Q: does SLAM emit a STREAM (not one), and do successive corrected poses
    #        for the SAME revisited keyframe shift / alternate? ------------------
    # For each emitted correction, read the corrected position of a FIXED probe
    # keyframe (the FIRST revisited 'old' kf the first correction targets) so we
    # can watch how the same physical keyframe's corrected pose moves solve-to-solve.
    print("=== loop.correction STREAM (per emitted correction) ===")
    print("  emit_seq  n_loops  #kf   corrected-pos[probe-kf]   delta-vs-prev")
    probe_seq = None
    prev_probe = None
    for emit_seq, corr in corrections:
        kf_poses = corr.kf_poses
        if probe_seq is None:
            # probe = the lowest-seq keyframe present (the revisited 'start' area,
            # most affected by a loop closure that pulls the end back to start).
            probe_seq = min(int(s) for s in kf_poses)
        if int(probe_seq) in kf_poses:
            _, ppos = T_cw_to_body_world(np.linalg.inv(kf_poses[int(probe_seq)]))
        else:
            ppos = None
        d = (float(np.linalg.norm(ppos - prev_probe))
             if ppos is not None and prev_probe is not None else 0.0)
        ps = "  n/a" if ppos is None else \
            f"({ppos[0]:+.3f},{ppos[1]:+.3f},{ppos[2]:+.3f})"
        print(f"  {emit_seq:7d}  {int(corr.n_loops):6d}  {len(kf_poses):4d}   "
              f"{ps}   {d*100:6.1f} cm")
        if ppos is not None:
            prev_probe = ppos

    # Also: the MOST-RECENT revisited keyframe per correction (what PropagateImu
    # actually targets -- max seq in store). Show its corrected pos solve-to-solve.
    print("\n=== corrected pose of the LATEST keyframe each solve (the blend target) ===")
    print("  emit_seq   target-kf   corrected-pos              jump-vs-prev-target")
    prev_tgt = None
    for emit_seq, corr in corrections:
        kf_poses = corr.kf_poses
        tgt = max(int(s) for s in kf_poses)
        _, tpos = T_cw_to_body_world(np.linalg.inv(kf_poses[tgt]))
        d = (float(np.linalg.norm(tpos - prev_tgt))
             if prev_tgt is not None else 0.0)
        print(f"  {emit_seq:7d}   {tgt:7d}     "
              f"({tpos[0]:+.3f},{tpos[1]:+.3f},{tpos[2]:+.3f})   {d*100:6.1f} cm")
        prev_tgt = tpos

    print("\nphase B: closed-loop replay -> TRACE live pose.odom position ...")
    trace = _trace_closed_loop(session_dir, n, corrections)

    pos = np.array([t["pos"] for t in trace])
    seqs = np.array([t["seq"] for t in trace])
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1) if len(pos) > 1 else np.zeros(0)

    fire_seqs = sorted({s for s, _ in corrections})
    first_fire = fire_seqs[0] if fire_seqs else seqs[-1]

    print(f"\n  total frames traced: {len(trace)};  first loop fires at seq {first_fire}")
    print("  largest single-frame live-pose steps (potential teleports):")
    if len(steps):
        order = np.argsort(steps)[::-1][:12]
        for k in sorted(order):
            print(f"    seq {seqs[k]:4d} -> {seqs[k+1]:4d}: step "
                  f"{steps[k]*100:7.1f} cm   "
                  f"(pend@{seqs[k+1]}={trace[k+1]['pend']*100:.1f}cm "
                  f"applied={trace[k+1]['applied']*100:.1f}cm"
                  f"{'  <FIRE>' if trace[k+1]['fired'] else ''})")

    # Per-frame live trace in the window around the loop activity, so the
    # back-and-forth (sign change of motion direction along the trajectory) shows.
    lo = max(0, int(np.searchsorted(seqs, first_fire)) - 5)
    print(f"\n=== live pose.odom trace from seq {seqs[lo] if lo < len(seqs) else first_fire} "
          "through end (watch the position oscillate) ===")
    print("  seq      pos (x,y,z)                 |step|   pend     applied  fire")
    prev = None
    for t in trace[lo:]:
        p = t["pos"]
        st = float(np.linalg.norm(p - prev)) if prev is not None else 0.0
        flag = "FIRE" if t["fired"] else ""
        print(f"  {t['seq']:4d}   ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})   "
              f"{st*100:6.1f}  {t['pend']*100:6.1f}  {t['applied']*100:7.1f}  {flag}")
        prev = p

    # Oscillation metric: count sign flips of the per-frame motion projected on the
    # dominant axis AFTER the first loop fires -- a clean settle has ~0-1 flips, an
    # oscillation has many.
    after = seqs >= first_fire
    pa = pos[after]
    if len(pa) > 3:
        d = np.diff(pa, axis=0)
        ax = int(np.argmax(np.abs(d).sum(axis=0)))   # dominant motion axis
        sign = np.sign(d[:, ax])
        flips = int(np.sum((sign[:-1] * sign[1:]) < 0))
        rng = float(np.ptp(pa[:, ax]))
        print(f"\n  OSCILLATION (post-loop, dominant axis {ax}): "
              f"{flips} direction reversals over {len(pa)} frames, "
              f"range {rng*100:.1f} cm")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    return run(Path(args.session), args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
