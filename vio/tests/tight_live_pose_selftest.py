#!/usr/bin/env python3
"""Functional self-test: the LIVE ``--tight`` ``pose.odom`` is IMU-propagated.

This drives the REAL :class:`~vio.modules.pipeline.OdometryModule` (the exact
front-end the live ``--tight`` graph runs, with ``retain_imu=True``) over a
:class:`~vio.comms.LocalPubSub` on a gold session, feeding the same per-frame
``ImuCamPacket`` + ``DepthFrame`` the imu_camera producer publishes, and captures
every ``pose.odom`` the module emits. It then asserts the two behaviours the user
asked for, on the live module (not just the primitives):

1. **IMU-propagated between keyframes.** On the ``--tight`` path the per-frame
   ``pose.odom`` advances on NON-keyframe frames too (the IMU forward-propagation
   in :class:`~vio.modules.propagate_imu.PropagateImu`), then is re-anchored to the
   vision pose at each keyframe -- a smooth, vision-corrected live trajectory, not a
   sparse one that only updates at keyframes.

2. **Covered camera + motion -> keeps moving (does NOT freeze).** A second run
   replays the SAME session but BLANKS the imagery for a mid-session window
   (constant gray + zero depth -> the KLT front-end starves, PnP fails / freezes,
   exactly the covered-camera symptom). On the ``--tight`` path the live pose still
   ADVANCES through the blanked window via the IMU, whereas the loose front-end's
   pose.odom freezes there. The delta proves the IMU is now driving the live output.

The module runs on its own thread; we publish all frames, send END on both inputs,
wait for ``odom.done`` (clean drain), then check the captured poses. Deterministic
because the inbox is an in-order FIFO per topic.

Run::

    .venv/bin/python -m vio.tests.tight_live_pose_selftest
    .venv/bin/python -m vio.tests.tight_live_pose_selftest --session sessions/gold/push_straight_fast_15s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imu_camera.io.reader import SessionReader                       # noqa: E402
from vio.comms import LocalPubSub, topics                            # noqa: E402
from vio.comms.messages import (                                     # noqa: E402
    DepthFrame, END, ImuCamPacket)
from vio.modules import OdometryModule                               # noqa: E402
from sky.front.odometry import OdometryConfig            # noqa: E402


def _per_frame_imu(ts_all, gyro, accel, prev_ts, ts):
    """IMU samples in the interval ``(prev_ts, ts]`` (the imu_camera packet cut)."""
    if prev_ts is None:
        m = ts_all <= ts
    else:
        m = (ts_all > prev_ts) & (ts_all <= ts)
    return (ts_all[m].astype(np.int64), gyro[m].astype(np.float64),
            accel[m].astype(np.float64))


def _run_module(session_dir: Path, n: int, *, tight: bool,
                blank_window=None) -> dict[int, np.ndarray]:
    """Run the real OdometryModule over a session; return ``{seq: position}``.

    ``tight`` toggles ``retain_imu`` (the --tight front-end behaviour: per-frame
    IMU retention + IMU forward-propagation of pose.odom). ``blank_window`` is an
    optional ``(lo, hi)`` frame-index range whose imagery is BLANKED (covered
    camera): constant gray + zero depth so the visual front-end starves.
    """
    reader = SessionReader(session_dir)
    R_imu_cam = reader.calib.T_imu_left[:3, :3]
    imu = reader.load_imu()
    ts_all = imu["ts_ns"].astype(np.int64)
    gyro = imu["gyro"].astype(np.float64)
    accel = imu["accel"].astype(np.float64)
    # static-startup accel for the one-shot gravity align (camera frame).
    t0 = int(ts_all[0])
    win = ts_all <= t0 + int(0.3 * 1e9)
    accel_align = R_imu_cam @ accel[win].mean(axis=0)

    bus = LocalPubSub()
    odom = OdometryModule(
        bus, reader.K, R_imu_cam=R_imu_cam, accel_align=accel_align,
        odom_cfg=OdometryConfig(gyro_fuse=True), kf_every=5, use_gyro=True,
        latest_only=False, level_tilt=True, publish_vo=False,
        retain_imu=tight)

    captured: dict[int, np.ndarray] = {}
    bus.subscribe(topics.POSE_ODOM,
                  lambda m: captured.__setitem__(
                      m.seq, m.T_world_cam[:3, 3].copy()) if m is not END
                  else None)
    odom.start()

    blank_gray = None
    prev_ts = None
    for i in range(n):
        f = reader.load_frame(i)
        gl = f.gray_left
        dm = f.depth_m
        if blank_window is not None and blank_window[0] <= i < blank_window[1]:
            # Covered camera: flat gray + no depth -> KLT/PnP cannot solve.
            if blank_gray is None:
                blank_gray = np.full_like(gl, 128)
            gl = blank_gray
            dm = np.zeros_like(dm)
        its, ig, ia = _per_frame_imu(ts_all, gyro, accel, prev_ts, int(f.ts_ns))
        # imu_camera publishes the synced packet first, then the depth frame.
        bus.publish(topics.IMUCAM_SAMPLE,
                    ImuCamPacket(f.seq, int(f.ts_ns), gl, None, its, ig, ia))
        bus.publish(topics.FRAME_DEPTH,
                    DepthFrame(f.seq, int(f.ts_ns), gl, dm))
        prev_ts = int(f.ts_ns)

    bus.publish(topics.IMUCAM_SAMPLE, END)
    bus.publish(topics.FRAME_DEPTH, END)
    if not odom.done.wait(timeout=120.0):
        odom.stop()
        raise RuntimeError("odometry module did not drain")
    odom.stop()
    return captured


def _nonkeyframe_motion(pos: dict[int, np.ndarray], kf_every: int) -> float:
    """Total path length over NON-keyframe frames (proves per-frame propagation).

    On the loose path the live pose.odom only meaningfully changes at vision
    updates; this measures how much the live pose moves on the in-between frames.
    """
    seqs = sorted(pos)
    total = 0.0
    for a, b in zip(seqs[:-1], seqs[1:]):
        if b % kf_every != 0:               # b is a non-keyframe frame
            total += float(np.linalg.norm(pos[b] - pos[a]))
    return total


def run(session_dir: Path, max_frames: int) -> int:
    reader = SessionReader(session_dir)
    n = len(reader) if max_frames <= 0 else min(max_frames, len(reader))
    print(f"session: {session_dir.name}  frames={n}\n")

    # --- (1) tight pose.odom is IMU-propagated between keyframes -------------
    tight = _run_module(session_dir, n, tight=False)        # warm import
    tight = _run_module(session_dir, n, tight=True)
    loose = _run_module(session_dir, n, tight=False)
    assert len(tight) >= n - 2 and len(loose) >= n - 2, "missing pose.odom frames"
    t_pos = np.array([tight[s] for s in sorted(tight)])
    l_pos = np.array([loose[s] for s in sorted(loose)])
    assert np.all(np.isfinite(t_pos)), "tight pose.odom has NaN/Inf"
    t_path = float(np.sum(np.linalg.norm(np.diff(t_pos, axis=0), axis=1)))
    l_path = float(np.sum(np.linalg.norm(np.diff(l_pos, axis=0), axis=1)))
    t_nonkf = _nonkeyframe_motion(tight, 5)
    print(f"[1] full-vision run   tight path={t_path:6.2f} m  loose path={l_path:6.2f} m")
    print(f"    non-keyframe live motion (tight, IMU-propagated) = {t_nonkf:.3f} m")
    fails = []
    if t_nonkf < 0.05:
        fails.append(f"tight pose.odom barely moves between keyframes "
                     f"({t_nonkf:.3f} m) -- not IMU-propagated")
    if not (0.3 <= t_path / max(l_path, 1e-6) <= 3.0):
        fails.append(f"tight path wildly off loose (ratio "
                     f"{t_path/max(l_path,1e-6):.2f}) -- propagation unstable")

    # --- (2) covered-camera: tight keeps moving, loose freezes --------------
    # Blank a 20-frame window mid-session (the camera is "covered").
    lo = n // 3
    hi = min(lo + 20, n - 2)
    tight_b = _run_module(session_dir, n, tight=True, blank_window=(lo, hi))
    loose_b = _run_module(session_dir, n, tight=False, blank_window=(lo, hi))

    def _window_motion(pos):
        seqs = [s for s in sorted(pos) if lo <= s < hi]
        if len(seqs) < 2:
            return 0.0
        p = np.array([pos[s] for s in seqs])
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))

    t_win = _window_motion(tight_b)
    l_win = _window_motion(loose_b)
    print(f"\n[2] covered-camera window [{lo},{hi})  "
          f"tight moves {t_win*100:5.1f} cm  |  loose moves {l_win*100:5.1f} cm")
    # Hard gate: through the covered window the TIGHT live pose advances via the
    # IMU (does NOT freeze) and moves materially more than the frozen loose pose.
    if t_win < 0.02:
        fails.append(f"tight FROZE through covered window ({t_win*100:.1f} cm)")
    if t_win <= l_win + 1e-4:
        fails.append(f"tight did not out-move loose in covered window "
                     f"(tight {t_win*100:.1f} cm <= loose {l_win*100:.1f} cm)")

    if fails:
        print("\nFAIL:")
        for f_ in fails:
            print(f"  - {f_}")
        return 1
    print("\nPASS -- live --tight pose.odom is IMU-propagated between keyframes "
          "and dead-reckons through a covered-camera window (does not freeze).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--max-frames", type=int, default=160)
    args = ap.parse_args()
    return run(Path(args.session), args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
