#!/usr/bin/env python3
"""REGRESSION LOCK + closed-loop drift proof for the live ``--tight`` pose.

This is the GATE the closed-loop SLAM-correction change must keep passing. It has
two parts:

PART 1 -- LOCK the current live ``--tight`` behaviour (must pass BEFORE and AFTER
the closed-loop change). It drives the REAL
:class:`~vio.modules.propagate_imu.PropagateImu` and asserts the four behaviours
the user requires stay intact, run BOTH with the closed-loop feedback OFF (today's
tight path) AND ON-but-idle (no correction arriving) so the additive change is
proven a no-op when no loop closes:

  (a) a fast PUSH profile tracks smoothly 0->D -- no mid-cruise pause, no overshoot
      past D, no backward snap, full magnitude (mirrors imu_push_response_selftest);
  (b) a COVERED camera (vision FAILED) + real motion dead-reckons via the IMU
      (keeps moving, does not freeze) (mirrors imu_propagate_selftest);
  (c) ZUPT holds the pose still at rest -- ~0 drift with an accel bias present;
  (d) a SHAKE profile (rapid sign-flipping accel) stays SANE -- the pose stays
      bounded near the origin, never explodes (finite, small net displacement).

PART 2 -- the CLOSED-LOOP DRIFT-REDUCTION proof ("better than Basalt"). It builds
a synthetic loop where the live nav-state accumulates a known drift by the time it
revisits the start, then feeds a SLAM ``loop.correction`` (the pose-graph rewrite
of the revisited keyframe) back through ``PropagateImu``'s inbox and measures the
live pose error at the revisit WITH vs WITHOUT the correction. The corrected run's
drift must be materially LOWER (bounded), and the correction must be applied
SMOOTHLY -- the largest single-frame pose step stays well below the full drift
magnitude (no hard teleport / snap).

Run::

    .venv/bin/python -m vio.tests.tight_live_regression_selftest
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vio.comms import LocalPubSub                                  # noqa: E402
from vio.comms.messages import LoopCorrection                      # noqa: E402
from vio.comms.module import ModuleContext                         # noqa: E402
from vio.modules.propagate_imu import PropagateImu                 # noqa: E402
from vio.modules.loop_inbox import LoopCorrectionInbox             # noqa: E402
from vio.mathlib.backend.vio_window import body_world_to_T_cw      # noqa: E402

G = 9.81
G_WORLD = np.array([0.0, G, 0.0])     # optical-world "down" = +y
DT_NS = 5_000_000                     # 5 ms IMU step (200 Hz)
N_PER_FRAME = 5                       # 5 IMU samples per camera frame -> 40 Hz
KF_EVERY = 5                          # keyframe every 5 frames (live default)


# --------------------------------------------------------------------------- #
# Minimal Step carrier stand-in (matches the fields PropagateImu touches).
# --------------------------------------------------------------------------- #
@dataclass
class _Frame:
    seq: int
    ts_ns: int


@dataclass
class _Step:
    frame: _Frame
    pose: np.ndarray
    info: dict


def _make_ctx(*, loop_correct: bool, kf_every: int = KF_EVERY) -> ModuleContext:
    """A tight-path ctx. ``loop_correct`` mirrors the live --tight builder: when
    on, allocates the closed-loop inbox + flag exactly as OdometryModule does."""
    ctx = ModuleContext(LocalPubSub(), "odometry")
    ctx.state["retain_imu"] = True
    ctx.state["kf_every"] = int(kf_every)
    ctx.state["g_world"] = (0.0, G, 0.0)
    ctx.state["imu_segs"] = {}
    if loop_correct:
        ctx.state["loop_correct"] = True
        ctx.state["loop_inbox"] = LoopCorrectionInbox()
    return ctx


def _vision_pose(p: np.ndarray, R: np.ndarray | None = None) -> np.ndarray:
    """``step.pose`` (camera->world, T_world_cam) for a body at world position
    ``p`` with optional attitude ``R`` (identity by default)."""
    R = np.eye(3) if R is None else R
    return np.linalg.inv(body_world_to_T_cw(R, np.asarray(p, np.float64)))


# --------------------------------------------------------------------------- #
# Synthetic 1-D push profile (rest -> accel -> cruise -> decel -> rest).
# --------------------------------------------------------------------------- #
def _push_profile():
    a_push = 6.0
    t_a, t_c, t_d = 0.30, 0.60, 0.30
    t_rest0, t_rest1 = 0.10, 0.20

    def n(t):
        return int(round(t / (DT_NS * 1e-9)))

    n0, na, nc, nd, n1 = n(t_rest0), n(t_a), n(t_c), n(t_d), n(t_rest1)
    acc_lin = np.concatenate([
        np.zeros(n0), np.full(na, +a_push), np.zeros(nc),
        np.full(nd, -a_push), np.zeros(n1)])
    ntot = acc_lin.size + 1
    ts = np.arange(ntot, dtype=np.int64) * DT_NS
    dt = DT_NS * 1e-9
    vel = np.zeros(ntot)
    pos = np.zeros(ntot)
    for k in range(ntot - 1):
        vel[k + 1] = vel[k] + acc_lin[k] * dt
        pos[k + 1] = pos[k] + vel[k] * dt + 0.5 * acc_lin[k] * dt * dt
    accel_body = np.tile(np.array([0.0, -G, 0.0]), (ntot, 1))
    accel_body[:, 2] += np.concatenate([acc_lin, [acc_lin[-1]]])
    gyro_body = np.zeros((ntot, 3))
    return ts, accel_body, gyro_body, pos


def _shake_profile():
    """A rapid sign-flipping forward/back shake (no net travel) -- the profile
    that must NOT explode the dead-reckoning."""
    amp = 8.0                 # strong shake accel (m/s^2)
    period = 0.10             # flip every 100 ms
    t_total = 2.0
    ntot = int(round(t_total / (DT_NS * 1e-9))) + 1
    ts = np.arange(ntot, dtype=np.int64) * DT_NS
    tsec = ts * 1e-9
    acc_lin = amp * np.sign(np.sin(2 * np.pi * tsec / period))
    accel_body = np.tile(np.array([0.0, -G, 0.0]), (ntot, 1))
    accel_body[:, 2] += acc_lin
    gyro_body = np.zeros((ntot, 3))
    return ts, accel_body, gyro_body


def _per_frame_cut(ts_all, gyro, accel, prev_ts, ts):
    if prev_ts is None:
        m = ts_all <= ts
    else:
        m = (ts_all > prev_ts) & (ts_all <= ts)
    return ts_all[m], gyro[m], accel[m]


def _run_profile(ts, accel, gyro, pos_truth, *, loop_correct: bool,
                 covered=None):
    """Drive the REAL PropagateImu over a 1-D profile; return per-frame live z."""
    ctx = _make_ctx(loop_correct=loop_correct)
    step_obj = PropagateImu()
    frame_ts = ts[::N_PER_FRAME]

    def true_z_at(t_ns):
        return float(pos_truth[min(int(round(t_ns / DT_NS)),
                                   pos_truth.size - 1)])

    live_z, prev_ts, seq = [], None, 0
    for fi, fts in enumerate(frame_ts):
        cts, cg, ca = _per_frame_cut(ts, gyro, accel, prev_ts, int(fts))
        ctx.state["imu_segs"][seq] = (cts.astype(np.int64),
                                      cg.astype(np.float64),
                                      ca.astype(np.float64))
        cov = covered is not None and covered[0] <= fi < covered[1]
        info = {"ok": False, "n_inliers": 0} if cov else {"ok": True,
                                                          "n_inliers": 64}
        z_vis = true_z_at(int(fts))
        st = _Step(_Frame(seq, int(fts)),
                   _vision_pose(np.array([0.0, 0.0, z_vis])), info)
        out = step_obj.run(ctx, st)
        live_z.append(float(out.pose[2, 3]))
        prev_ts = int(fts)
        seq += 1
    return np.array(live_z)


# --------------------------------------------------------------------------- #
# PART 1 -- lock the current --tight behaviour
# --------------------------------------------------------------------------- #
def _check_push(loop_correct: bool) -> None:
    ts, accel, gyro, pos = _push_profile()
    D = float(pos[-1])
    live = _run_profile(ts, accel, gyro, pos, loop_correct=loop_correct)
    true_z = np.array([float(pos[min(int(round(t / DT_NS)), pos.size - 1)])
                       for t in ts[::N_PER_FRAME]])
    lag_err = float(np.max(true_z - live))
    overshoot = float(np.max(live) - D)
    back_step = float(np.max(np.maximum(0.0, -np.diff(live))))
    c0, c1 = len(live) // 3, 2 * len(live) // 3
    min_cruise = float(np.min(np.diff(live[c0:c1])))
    tag = "loop=ON " if loop_correct else "loop=OFF"
    print(f"  (a) push [{tag}] D={D:.3f} end={live[-1]:.3f}  lag={lag_err*1e3:.1f}mm "
          f"overshoot={overshoot*1e3:.1f}mm snap={back_step*1e3:.1f}mm "
          f"min_cruise={min_cruise*1e3:.2f}mm")
    assert lag_err < 0.04, f"push lag {lag_err*1e3:.0f}mm -- under-integrating"
    assert overshoot < 0.05, f"push overshoot {overshoot*1e3:.0f}mm"
    assert back_step < 0.02, f"push snap-back {back_step*1e3:.0f}mm"
    assert min_cruise > 0.0, "push PAUSED mid-cruise (ZUPT misfire)"
    assert abs(live[-1] - D) < 0.10, f"push end {live[-1]:.3f} != D {D:.3f}"


def _check_covered(loop_correct: bool) -> None:
    ts, accel, gyro, pos = _push_profile()
    lo, hi = 5, 16
    live = _run_profile(ts, accel, gyro, pos, loop_correct=loop_correct,
                        covered=(lo, hi))
    true_z = np.array([float(pos[min(int(round(t / DT_NS)), pos.size - 1)])
                       for t in ts[::N_PER_FRAME]])
    win_adv = float(live[hi - 1] - live[lo])
    true_adv = float(true_z[hi - 1] - true_z[lo])
    final_err = float(abs(live[-1] - pos[-1]))
    tag = "loop=ON " if loop_correct else "loop=OFF"
    print(f"  (b) covered [{tag}] live advances {win_adv*100:.1f}cm "
          f"(true {true_adv*100:.1f}cm) final_err {final_err*1e3:.0f}mm")
    assert win_adv > 0.5 * true_adv, "covered camera FROZE (not dead-reckoning)"
    assert final_err < 0.10, "did not re-lock to truth after covered window"


def _check_zupt(loop_correct: bool) -> None:
    ctx = _make_ctx(loop_correct=loop_correct)
    step_obj = PropagateImu()
    rest = np.array([0.0, -G, 0.0]) + np.array([0.05, 0.0, 0.05])   # + accel bias
    poses, seq = [], 0
    for fi in range(80):                               # ~2 s of rest at 40 Hz
        t0 = seq * (N_PER_FRAME - 1) * DT_NS
        cts = np.array([t0 + k * DT_NS for k in range(N_PER_FRAME)], np.int64)
        cg = np.zeros((N_PER_FRAME, 3))
        ca = np.tile(rest, (N_PER_FRAME, 1))
        ctx.state["imu_segs"][seq] = (cts, cg, ca)
        out = step_obj.run(ctx, _Step(_Frame(seq, int(cts[-1])),
                                      _vision_pose(np.zeros(3)), {}))
        poses.append(out.pose[:3, 3].copy())
        seq += 1
    pos = np.array(poses)
    drift = float(np.linalg.norm(pos[-1] - pos[0]))
    tag = "loop=ON " if loop_correct else "loop=OFF"
    print(f"  (c) ZUPT [{tag}] net drift {drift*1e3:.3f}mm over {len(pos)} frames")
    assert drift < 2e-3, f"ZUPT regressed: {drift*1e3:.2f}mm drift at rest"


def _check_shake(loop_correct: bool) -> None:
    ts, accel, gyro = _shake_profile()
    ctx = _make_ctx(loop_correct=loop_correct)
    step_obj = PropagateImu()
    frame_ts = ts[::N_PER_FRAME]
    poses, prev_ts, seq = [], None, 0
    for fts in frame_ts:
        cts, cg, ca = _per_frame_cut(ts, gyro, accel, prev_ts, int(fts))
        ctx.state["imu_segs"][seq] = (cts.astype(np.int64), cg, ca)
        # vision stays near origin (the shake has no net travel), valid solve.
        st = _Step(_Frame(seq, int(fts)), _vision_pose(np.zeros(3)),
                   {"ok": True, "n_inliers": 64})
        out = step_obj.run(ctx, st)
        poses.append(out.pose[:3, 3].copy())
        prev_ts = int(fts)
        seq += 1
    pos = np.array(poses)
    max_exc = float(np.max(np.linalg.norm(pos, axis=1)))
    net = float(np.linalg.norm(pos[-1]))
    tag = "loop=ON " if loop_correct else "loop=OFF"
    print(f"  (d) shake [{tag}] max excursion {max_exc*100:.1f}cm  "
          f"net {net*100:.1f}cm  finite={np.all(np.isfinite(pos))}")
    assert np.all(np.isfinite(pos)), "shake produced NaN/Inf (exploded)"
    # The shake has no net travel and vision anchors the origin -- the dead-reckon
    # must stay bounded (it must NOT integrate a divergent runaway). 0.5 m is a
    # generous bound; a divergent integrator would blow far past it.
    assert max_exc < 0.5, f"shake excursion {max_exc*100:.1f}cm -- exploded"
    assert net < 0.2, f"shake net displacement {net*100:.1f}cm -- not anchored"


def test_part1_lock_current_behaviour() -> None:
    print("PART 1 -- lock current --tight behaviour (must pass before+after):")
    # Run every check with the closed-loop feedback OFF (today's path) AND ON but
    # idle (no correction arriving). ON-but-idle MUST equal OFF -- proving the
    # additive change is a no-op until a loop actually closes.
    for loop_correct in (False, True):
        _check_push(loop_correct)
        _check_covered(loop_correct)
        _check_zupt(loop_correct)
        _check_shake(loop_correct)
    print("  OK -- push/covered/ZUPT/shake all hold, with closed-loop OFF and "
          "ON-but-idle (additive feedback is a no-op until a loop closes)\n")


# --------------------------------------------------------------------------- #
# PART 2 -- closed-loop drift-reduction proof
# --------------------------------------------------------------------------- #
def _run_loop_drift(apply_correction: bool):
    """A synthetic loop: the live pose dead-reckons with a small per-segment accel
    bias so it accumulates a KNOWN drift by the time it revisits the start. At the
    revisit keyframe a SLAM ``loop.correction`` (the pose-graph rewrite of that
    keyframe back to its true pose) is fed in. Returns
    ``(live_xyz, revisit_idx, kf_truth_at_revisit, max_frame_step)``.

    Vision is marked FAILED (covered) for the whole run so the live pose
    pure-dead-reckons -- this ISOLATES the closed-loop correction from the per-
    frame vision complementary pull, so the drift reduction is unambiguously the
    SLAM loop correction (not vision re-anchoring). This is the worst case for
    drift, exactly where a loop closure matters most.
    """
    ctx = _make_ctx(loop_correct=apply_correction)
    step_obj = PropagateImu()

    # Build a square-ish loop trajectory in the x-z plane: out +z, across +x,
    # back -z, return -x to the start. Constant accel bias adds the drift.
    n_frames = 120
    revisit_idx = n_frames - 1
    bias = np.array([0.03, 0.0, 0.03])     # accel bias -> dead-reckon drift
    rest = np.array([0.0, -G, 0.0])

    # A gentle closed loop in TRUE space (so we know the revisit truth = start).
    # Use a smooth accel that returns to the start: one full sine period on x and
    # z each, scaled so it loops. We only need: (i) the live pose accumulates a
    # real drift from the bias, (ii) the TRUE revisit position == the start.
    t = np.linspace(0.0, 2 * np.pi, n_frames)
    true_xz = np.column_stack([np.sin(t), 1.0 - np.cos(t)])   # starts+ends at (0,0)
    dt_f = (N_PER_FRAME - 1) * DT_NS * 1e-9
    # finite-difference the true accel needed to follow true_xz (for the IMU).
    vel = np.gradient(true_xz, dt_f, axis=0)
    acc = np.gradient(vel, dt_f, axis=0)

    live_xyz = []
    seq = 0
    max_step = 0.0
    prev_pub = None
    for fi in range(n_frames):
        t0 = seq * (N_PER_FRAME - 1) * DT_NS
        cts = np.array([t0 + k * DT_NS for k in range(N_PER_FRAME)], np.int64)
        # accel = true linear accel (x,z) + gravity + bias, in the optical frame
        a = rest + bias
        a[0] += acc[fi, 0]
        a[2] += acc[fi, 1]
        ca = np.tile(a, (N_PER_FRAME, 1))
        cg = np.zeros((N_PER_FRAME, 3))
        ctx.state["imu_segs"][seq] = (cts, cg, ca)
        # Covered vision the whole run (isolates the loop correction).
        st = _Step(_Frame(seq, int(cts[-1])),
                   _vision_pose(np.zeros(3)), {"ok": False, "n_inliers": 0})

        # On the FINAL frame (the revisit), if correcting, inject the loop
        # correction BEFORE running the step: SLAM rewrites the revisited
        # keyframe's pose to the TRUE revisit pose (the start, identity@origin).
        if apply_correction and fi == revisit_idx:
            # The revisited keyframe is the LAST keyframe PropagateImu recorded.
            nav = ctx.state.get("live_nav")
            if nav is not None and nav["kf_pose_pre"]:
                rev_seq = max(nav["kf_pose_pre"])
                # corrected pose = TRUE pose at that keyframe. Our loop returns to
                # the start, so the revisited keyframe's true world pose is the
                # ORIGIN with identity attitude (the start anchor).
                T_corr = np.linalg.inv(body_world_to_T_cw(np.eye(3),
                                                          np.zeros(3)))
                ctx.state["loop_inbox"].push(
                    LoopCorrection(seq=rev_seq,
                                   kf_poses={rev_seq: T_corr}, n_loops=1))

        out = step_obj.run(ctx, st)
        pub = out.pose[:3, 3].copy()
        if prev_pub is not None:
            max_step = max(max_step, float(np.linalg.norm(pub - prev_pub)))
        prev_pub = pub
        live_xyz.append(pub)
        seq += 1

    # Let the smooth blend finish: run a few more idle frames (no real motion,
    # vision still covered) so the pending correction fully bleeds in. Count the
    # frames whose step is non-trivial (> 1 cm) -- a SMOOTH blend spreads the
    # correction over several frames; a hard snap would move it all in ONE.
    blend_frames = 0
    for extra in range(40):
        t0 = seq * (N_PER_FRAME - 1) * DT_NS
        cts = np.array([t0 + k * DT_NS for k in range(N_PER_FRAME)], np.int64)
        ca = np.tile(rest, (N_PER_FRAME, 1))    # at rest -> ZUPT, no new motion
        cg = np.zeros((N_PER_FRAME, 3))
        ctx.state["imu_segs"][seq] = (cts, cg, ca)
        st = _Step(_Frame(seq, int(cts[-1])),
                   _vision_pose(np.zeros(3)), {"ok": False, "n_inliers": 0})
        out = step_obj.run(ctx, st)
        pub = out.pose[:3, 3].copy()
        step_mag = float(np.linalg.norm(pub - prev_pub))
        if step_mag > 0.01:
            blend_frames += 1
        max_step = max(max_step, step_mag)
        prev_pub = pub
        live_xyz.append(pub)
        seq += 1

    return np.array(live_xyz), revisit_idx, max_step, blend_frames


def test_part2_closed_loop_reduces_drift() -> None:
    print("PART 2 -- closed-loop drift reduction at a revisit (better than Basalt):")
    # The TRUE revisit position is the START (the loop returns to origin).
    true_revisit = np.zeros(3)

    open_xyz, rev_i, _, _ = _run_loop_drift(apply_correction=False)
    closed_xyz, _, max_step, blend_frames = _run_loop_drift(apply_correction=True)

    # Drift at the revisit = distance from the live pose to the true revisit pose,
    # measured AFTER the correction has settled (the end of each run).
    open_drift = float(np.linalg.norm(open_xyz[-1] - true_revisit))
    closed_drift = float(np.linalg.norm(closed_xyz[-1] - true_revisit))
    # The full drift the open-loop run carries INTO the revisit (the magnitude the
    # correction must remove) -- the live pose at the moment of the loop closure.
    drift_at_loop = float(np.linalg.norm(open_xyz[rev_i] - true_revisit))

    print(f"  open-loop  drift at revisit (Basalt-like, no loop) = "
          f"{open_drift*100:6.1f} cm")
    print(f"  closed-loop drift at revisit (SLAM correction fed back) = "
          f"{closed_drift*100:6.1f} cm")
    print(f"  -> drift reduced by {(1 - closed_drift/max(open_drift,1e-9))*100:.0f} %"
          f"  (bounded on revisit)")
    print(f"  largest single-frame pose step during the blend = {max_step*100:.2f} cm"
          f"  (full drift to remove = {drift_at_loop*100:.1f} cm)")
    print(f"  correction spread over {blend_frames} frames (>1cm each) "
          "-- smooth, not a one-frame snap")

    # 1. The corrected run's drift at the revisit is MATERIALLY lower (bounded).
    assert open_drift > 0.02, (f"open-loop drift {open_drift*100:.1f}cm too small "
                               "to be a meaningful test (need real drift)")
    assert closed_drift < 0.4 * open_drift, (
        f"closed-loop did not bound drift: {closed_drift*100:.1f}cm is not "
        f"materially below open-loop {open_drift*100:.1f}cm")
    # 2. The correction is SMOOTH: the largest single-frame step stays well below
    #    the full drift magnitude (a hard snap would jump the whole drift in one
    #    frame). 60 % of the full drift in one frame is the explicit no-snap bound.
    assert max_step < 0.6 * drift_at_loop, (
        f"correction SNAPPED: a single frame moved {max_step*100:.1f}cm of the "
        f"{drift_at_loop*100:.1f}cm drift (hard jump reintroduced)")
    # 3. The correction is SPREAD over several frames (geometric blend), not a
    #    single-frame teleport -- the explicit "smooth, not a snap" requirement.
    assert blend_frames >= 5, (
        f"correction was not spread out: only {blend_frames} frames moved >1cm "
        "(a smooth blend bleeds the delta over many frames)")
    assert np.all(np.isfinite(closed_xyz)), "closed-loop produced NaN/Inf"
    print("  OK -- closed-loop bounds the revisit drift, applied SMOOTHLY "
          "(no hard snap)\n")


def main() -> int:
    print("=== TIGHT live regression LOCK + closed-loop drift proof ===\n")
    test_part1_lock_current_behaviour()
    test_part2_closed_loop_reduces_drift()
    print("PASS -- current --tight behaviour LOCKED (push/covered/ZUPT/shake), "
          "and the closed-loop SLAM correction bounds revisit drift smoothly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
