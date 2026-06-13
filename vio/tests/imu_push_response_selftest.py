#!/usr/bin/env python3
"""GATE self-test: the live ``--tight`` IMU pose tracks a fast push SMOOTHLY.

This reproduces the exact bug the user reported and proves the fix. It synthesises
a realistic 1-D forward push -- ``rest -> accel spike -> CONSTANT-VELOCITY cruise
-> decel -> rest`` -- as a 200 Hz IMU stream (specific force incl. gravity + gyro
+ device-clock timestamps), cuts it into per-frame packets exactly the way the live
imu_camera producer does (the ``(prev_ts, ts]`` window), and makes vision poses
available only at the keyframe cadence (optionally LAGGING the truth to model a
slow back-end). It then drives the REAL
:class:`~vio.modules.propagate_imu.PropagateImu` and asserts the published live
position tracks the true position with:

* **no mid-cruise pause** -- the position keeps advancing during the
  constant-velocity phase (the OLD accel/gyro-only ZUPT froze here);
* **no overshoot** past the true travel distance ``D`` beyond a small tolerance;
* **no snap-back** -- no large backward jump at a keyframe correction;
* **monotonic 0 -> D** forward tracking with bounded lag (the OLD per-block
  integration captured only ~50-65 % of ``D`` -- the dropped inter-block segment).

It ALSO pins the broken OLD behaviours directly (so the regression is explicit):

* a per-block integration that drops the boundary segment UNDER-integrates the
  push (proves root-cause #3);
* the accel/gyro-only at-rest gate misfires during the constant-velocity cruise
  (proves root-cause #1).

Run::

    .venv/bin/python -m vio.tests.imu_push_response_selftest
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vio.comms.module import ModuleContext                       # noqa: E402
from vio.comms import LocalPubSub                                # noqa: E402
from vio.modules.propagate_imu import propagate_imu             # noqa: E402
from sky.vio.imu import imu_at_rest, predict_state       # noqa: E402
from sky.vio.window import body_world_to_T_cw    # noqa: E402

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


# --------------------------------------------------------------------------- #
def _push_profile():
    """A 1-D forward (+z) push: rest -> accel -> cruise -> decel -> rest.

    Returns ``(ts, accel_body, gyro_body, true_pos_z, t_accel_end)`` sampled at
    200 Hz, where ``accel_body`` is the SPECIFIC FORCE the sensor reports (rest =
    +g up = optical -y) and ``true_pos_z`` is the analytic forward position at
    each sample (the ground truth the live pose must track).
    """
    a_push = 6.0          # m/s^2 forward accel during the spike (a brisk push)
    t_a = 0.30            # accel spike duration  (s)  -> v_cruise = 1.8 m/s
    t_c = 0.60            # constant-velocity cruise   (s)
    t_d = 0.30            # decel back to rest         (s)
    t_rest0 = 0.10        # initial rest               (s)
    t_rest1 = 0.20        # final rest                 (s)

    def n(t):
        return int(round(t / (DT_NS * 1e-9)))

    n0, na, nc, nd, n1 = n(t_rest0), n(t_a), n(t_c), n(t_d), n(t_rest1)
    # per-segment forward acceleration (linear, m/s^2)
    acc_lin = np.concatenate([
        np.zeros(n0),
        np.full(na, +a_push),
        np.zeros(nc),
        np.full(nd, -a_push),
        np.zeros(n1),
    ])
    ntot = acc_lin.size + 1
    ts = np.arange(ntot, dtype=np.int64) * DT_NS
    # analytic velocity / position by trapezoidal integration of acc_lin
    dt = DT_NS * 1e-9
    vel = np.zeros(ntot)
    pos = np.zeros(ntot)
    for k in range(ntot - 1):
        vel[k + 1] = vel[k] + acc_lin[k] * dt
        pos[k + 1] = pos[k] + vel[k] * dt + 0.5 * acc_lin[k] * dt * dt
    # specific force = rest (-g on optical y) + forward linear accel on +z.
    accel_body = np.tile(np.array([0.0, -G, 0.0]), (ntot, 1))
    accel_body[:, 2] += np.concatenate([acc_lin, [acc_lin[-1]]])
    gyro_body = np.zeros((ntot, 3))           # pure translation, no rotation
    t_accel_end_ns = int(ts[n0 + na])         # sample index where cruise begins
    return ts, accel_body, gyro_body, pos, t_accel_end_ns


def _per_frame_cut(ts_all, gyro, accel, prev_ts, ts):
    """IMU samples in ``(prev_ts, ts]`` -- the live imu_camera packet cut."""
    if prev_ts is None:
        m = ts_all <= ts
    else:
        m = (ts_all > prev_ts) & (ts_all <= ts)
    return ts_all[m], gyro[m], accel[m]


def _make_ctx() -> ModuleContext:
    ctx = ModuleContext(LocalPubSub(), "odometry")
    ctx.state["retain_imu"] = True
    ctx.state["kf_every"] = KF_EVERY
    ctx.state["g_world"] = (0.0, G, 0.0)
    ctx.state["imu_segs"] = {}
    return ctx


def _vision_pose(z: float) -> np.ndarray:
    """``step.pose`` (camera->world, T_world_cam) for a body at forward +z.

    PropagateImu reads the vision pose as
    ``T_cw_to_body_world(inv(step.pose))`` and publishes ``step.pose`` as the
    camera->world transform, so ``step.pose`` must be ``T_world_cam`` (the inverse
    of the world->camera ``body_world_to_T_cw`` matrix). Identity attitude.
    """
    return np.linalg.inv(
        body_world_to_T_cw(np.eye(3), np.array([0.0, 0.0, z])))


# --------------------------------------------------------------------------- #
def test_old_behaviours_were_broken() -> None:
    """Pin the two OLD failure modes directly (regression evidence)."""
    ts, accel, gyro, pos, t_accel_end = _push_profile()
    D = float(pos[-1])

    # (a) per-block integration that DROPS the inter-block boundary segment
    # under-integrates the push (root-cause #3). This is the naive per-frame
    # predict_state the live path used before the gap-free fix.
    R, p, v = np.eye(3), np.zeros(3), np.zeros(3)
    prev_ts = None
    frame_ts = ts[::N_PER_FRAME]
    for fts in frame_ts:
        cts, cg, ca = _per_frame_cut(ts, gyro, accel, prev_ts, int(fts))
        if cts.size >= 2:
            R, p, v = predict_state(R, p, v, cts, cg, ca,
                                    np.zeros(3), np.zeros(3), G_WORLD)
        prev_ts = int(fts)
    naive_z = float(p[2])
    frac = naive_z / D
    print(f"[OLD] naive per-block integration: z={naive_z:.3f} m of D={D:.3f} m "
          f"({frac*100:.0f} %)  -- dropped boundary segments under-integrate")
    assert frac < 0.9, ("expected the naive per-block path to under-integrate, "
                        f"got {frac*100:.0f} %")

    # (b) the accel/gyro-ONLY at-rest gate misfires during the CONSTANT-VELOCITY
    # cruise (root-cause #1): pick a block fully inside the cruise.
    cruise_mask = (ts > t_accel_end + 2 * DT_NS) & (ts < ts[-1] - int(0.4e9))
    cg, ca = gyro[cruise_mask], accel[cruise_mask]
    assert cg.shape[0] >= 2, "cruise block too short for the test"
    misfire = imu_at_rest(cg, ca, gravity=G)
    print(f"[OLD] accel/gyro-only at-rest gate during cruise -> {misfire} "
          "(True == would FREEZE mid-push: the PAUSE)")
    assert misfire, ("the cruise block should fool the accel/gyro-only gate "
                     "(that is the bug being fixed)")


# --------------------------------------------------------------------------- #
def _run_live(lag_frames: int = 0, covered_window=None):
    """Drive the REAL PropagateImu over the push profile; return tracking data.

    ``lag_frames`` delays the per-frame vision pose by N frames (a slow / stale
    vision fix). ``covered_window`` is an optional ``(lo, hi)`` frame-index range
    where vision is marked FAILED (``step.info["ok"] = False`` + 0 inliers, the
    covered-camera signal EstimateMotion emits) -- there the live pose must
    pure-dead-reckon from the IMU (no pull toward a stale pose).
    Returns ``(live_z, true_z_at_frame, kf_seqs, D)``.
    """
    ts, accel, gyro, pos, _ = _push_profile()
    D = float(pos[-1])
    ctx = _make_ctx()
    step_obj = propagate_imu

    # true forward position sampled at each FRAME boundary (last IMU sample).
    frame_ts = ts[::N_PER_FRAME]
    # map a device-clock ns -> analytic true z (nearest sample).
    def true_z_at(t_ns: int) -> float:
        return float(pos[min(int(round(t_ns / DT_NS)), pos.size - 1)])

    live_z = []
    true_z = []
    kf_seqs = []
    prev_ts = None
    seq = 0
    for fi, fts in enumerate(frame_ts):
        cts, cg, ca = _per_frame_cut(ts, gyro, accel, prev_ts, int(fts))
        ctx.state["imu_segs"][seq] = (cts.astype(np.int64),
                                      cg.astype(np.float64),
                                      ca.astype(np.float64))
        # Fresh per-frame vision pose, optionally lagged. A covered window marks
        # the solve as FAILED so PropagateImu skips the correction there.
        z_truth = true_z_at(int(fts))
        z_vis = true_z_at(int(frame_ts[max(fi - lag_frames, 0)]))
        covered = (covered_window is not None
                   and covered_window[0] <= fi < covered_window[1])
        info = {"ok": False, "n_inliers": 0} if covered \
            else {"ok": True, "n_inliers": 64}
        st = _Step(_Frame(seq, int(fts)), _vision_pose(z_vis), info)
        out = step_obj(ctx, st)
        # out.pose is T_world_cam (camera->world); body == camera, so the
        # body->world forward position is its translation column directly.
        live_z.append(float(out.pose[2, 3]))
        true_z.append(z_truth)
        if ctx.state.get("is_kf_frame"):
            kf_seqs.append(fi)
        prev_ts = int(fts)
        seq += 1
    return np.array(live_z), np.array(true_z), kf_seqs, D


def test_fixed_tracks_push_smoothly() -> None:
    """THE GATE: the FIXED live pose tracks the push 0->D with no pause / no
    overshoot / no snap, with vision available only at keyframes."""
    for lag in (0, 2):
        live_z, true_z, kf_seqs, D = _run_live(lag_frames=lag)

        # max forward lag (true ahead of live) -- the gap-free fix keeps this small.
        lag_err = float(np.max(true_z - live_z))
        # overshoot past D (live ahead of the final travel distance).
        overshoot = float(np.max(live_z) - D)
        # largest backward step in the live curve (the snap-back at a keyframe).
        back_step = float(np.max(np.maximum(0.0, -np.diff(live_z))))
        # mid-cruise pause check: the live position must keep ADVANCING through
        # the constant-velocity cruise (no flat / frozen run). Look at the
        # per-frame forward delta during the cruise frames and require it stays
        # clearly positive (no ZUPT freeze).
        # cruise frames ~ after the accel spike, before decel: middle third.
        c0, c1 = len(live_z) // 3, 2 * len(live_z) // 3
        cruise_steps = np.diff(live_z[c0:c1])
        min_cruise_step = float(np.min(cruise_steps))

        print(f"\n[FIXED lag={lag}f] D={D:.3f} m  live_end={live_z[-1]:.3f} m")
        print(f"    max forward lag      = {lag_err*1000:6.1f} mm")
        print(f"    overshoot past D     = {overshoot*1000:6.1f} mm")
        print(f"    max backward step    = {back_step*1000:6.1f} mm (snap-back)")
        print(f"    min cruise fwd step  = {min_cruise_step*1000:6.2f} mm "
              f"(>0 == no mid-cruise pause)")

        # Hard gates (the user's three requirements + full-magnitude tracking).
        # The lag budget scales with the DELIBERATE vision lag: with no vision lag
        # the gap-free IMU integration must track to a few mm (proves root-cause
        # #3 fixed); when vision is stale by N frames the complementary pull
        # toward the stale fix adds ~v*N*dt of expected, correct lag (vision-
        # anchored, not a defect) -- ~40 mm per lagged frame at the ~1.8 m/s peak.
        lag_budget = 0.04 + lag * 0.04
        assert lag_err < lag_budget, (
            f"live lags the true push too much ({lag_err*1000:.0f} mm > "
            f"{lag_budget*1000:.0f} mm budget) -- under-integrating")
        assert overshoot < 0.05, (f"live overshoots D by {overshoot*1000:.0f} mm "
                                  "-- bad velocity injection / hard re-anchor")
        assert back_step < 0.02, (f"live snaps back {back_step*1000:.0f} mm at a "
                                  "keyframe -- hard re-anchor not removed")
        assert min_cruise_step > 0.0, ("live pose PAUSED during the cruise "
                                       "(ZUPT misfired) -- the bug is not fixed")
        # End-to-end magnitude: the live pose reaches ~D (full double-integral),
        # not the ~60 % the dropped-boundary path produced.
        assert abs(live_z[-1] - D) < 0.10, (f"live end {live_z[-1]:.3f} m vs "
                                            f"D={D:.3f} m -- not full magnitude")
    print("\n  OK -- fixed live pose tracks the push 0->D smoothly: no pause, "
          "no overshoot, no snap, full magnitude")


def test_pure_rest_no_drift() -> None:
    """A pure-rest profile: the live pose must not walk off (ZUPT still works)."""
    ctx = _make_ctx()
    step_obj = propagate_imu
    rest = np.array([0.0, -G, 0.0]) + np.array([0.05, 0.0, 0.05])  # + accel bias
    poses = []
    seq = 0
    for fi in range(80):                          # ~2 s of rest at 40 Hz
        t0 = seq * (N_PER_FRAME - 1) * DT_NS
        cts = np.array([t0 + k * DT_NS for k in range(N_PER_FRAME)], np.int64)
        cg = np.zeros((N_PER_FRAME, 3))
        ca = np.tile(rest, (N_PER_FRAME, 1))
        ctx.state["imu_segs"][seq] = (cts, cg, ca)
        # vision sits at the origin (device truly still).
        st = _Step(_Frame(seq, int(cts[-1])), _vision_pose(0.0), {})
        out = step_obj(ctx, st)
        poses.append(out.pose[:3, 3].copy())
        seq += 1
    pos = np.array(poses)
    drift = float(np.linalg.norm(pos[-1] - pos[0]))
    print(f"\npure-rest: net drift = {drift*1000:.3f} mm over {len(pos)} frames "
          "(ZUPT + vision anchor hold it still)")
    assert drift < 2e-3, f"rest drift {drift*1000:.2f} mm -- ZUPT regressed"
    print("  OK -- pure rest: no drift (ZUPT still works)")


def test_cruise_after_spike_no_freeze() -> None:
    """A spike-then-cruise profile: the cruise frames must NOT freeze (the exact
    misfire), proving the velocity gate keeps the IMU integrating."""
    live_z, true_z, _, D = _run_live(lag_frames=0)
    live_d = np.diff(live_z)
    true_d = np.diff(true_z)
    # "Genuinely moving" = the TRUE trajectory is advancing this step (this
    # excludes the initial AND final rest plateaus, where a still pose is
    # CORRECT, not a freeze). During those frames the live pose must also advance
    # -- a zero/negative live step there is the mid-motion ZUPT misfire (the
    # PAUSE) the velocity gate removes.
    truly_moving = true_d > 1e-4
    frozen = int(np.sum(live_d[truly_moving] <= 1e-5))
    print(f"\ncruise-after-spike: frozen frames during true motion = {frozen} "
          f"of {int(truly_moving.sum())} moving frames (0 == no mid-motion freeze)")
    assert frozen == 0, (f"{frozen} frames froze while truly moving -- "
                         "velocity-gated ZUPT misfired")
    print("  OK -- cruise after spike: no mid-motion freeze")


def test_covered_window_keeps_moving() -> None:
    """Covered camera mid-push (vision FAILED): the live pose must keep advancing
    via the IMU through the blind window and NOT freeze, then re-lock to truth."""
    # cover the whole accel+cruise region so the device is genuinely moving while
    # vision is out (frames ~5..15 of ~28).
    lo, hi = 5, 16
    live_z, true_z, _, D = _run_live(lag_frames=0, covered_window=(lo, hi))
    # advance through the covered window (the live pose dead-reckons forward).
    win_adv = float(live_z[hi - 1] - live_z[lo])
    true_adv = float(true_z[hi - 1] - true_z[lo])
    # after re-lock, the live pose tracks truth again to within a small lag.
    final_err = float(abs(live_z[-1] - D))
    print(f"\ncovered window [{lo},{hi}): live advances {win_adv*100:.1f} cm "
          f"(true {true_adv*100:.1f} cm) -- dead-reckons, no freeze; "
          f"final err {final_err*1000:.0f} mm")
    assert win_adv > 0.5 * true_adv, (
        f"live froze in the covered window ({win_adv*100:.1f} cm of "
        f"{true_adv*100:.1f} cm true) -- not dead-reckoning")
    assert final_err < 0.10, (f"live did not re-lock to truth after the covered "
                              f"window (final err {final_err*1000:.0f} mm)")
    print("  OK -- covered window: live dead-reckons through it and re-locks")


def main() -> int:
    print("=== TIGHT push-response self-test (reproduce bug, prove fix) ===\n")
    test_old_behaviours_were_broken()
    test_fixed_tracks_push_smoothly()
    test_covered_window_keeps_moving()
    test_pure_rest_no_drift()
    test_cruise_after_spike_no_freeze()
    print("\nPASS -- fast push tracked smoothly 0->D (no pause/overshoot/snap, "
          "full magnitude); rest holds still; old failure modes reproduced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
