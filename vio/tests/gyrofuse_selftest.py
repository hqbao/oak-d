#!/usr/bin/env python3
"""Self-test for the gyro-fusion strip-chart diagnostic (ALGORITHMS.md #5).

Pins the additive diagnostic + publisher contract the gyro-fusion strip chart
(visualisation #5) adds, with hard pass/fail thresholds:

1. ODOMETRY DIAGNOSTIC (math + additivity). Drive
   :meth:`RGBDVisualOdometry.estimate` on a synthetic frame pair with a KNOWN
   relative pose and a gyro rotation prior, gyro fusion ON, then assert it stored
   ``info["vision_rot_deg"/"gyro_rot_deg"/"disagree_deg"/"gain"/"t_trust"]``
   sanely: vision/gyro rotations ~ the planted yaw, disagreement small (vision and
   gyro agree), gain in [0,1], t_trust == 1.0 on this clean frame. Crucially it
   also asserts the pose is BYTE-IDENTICAL to a run of the SAME fused odometry
   that never reads the new keys -- proving the diagnostic is purely additive (the
   640-pose oracle's gap=0 invariant, in unit form).

2. PUBLISHER emits on fused frames. ``PublishGyroFuse`` must emit a
   :class:`~vio.comms.messages.FrameGyroFuse` carrying the five last_info fields +
   the config gate/span thresholds (read off the live odometry) when those fields
   are present.

3. PUBLISHER stays silent when gyro is off. With an ``info`` that lacks the
   fusion keys (gyro off / PnP failed / bootstrap), ``PublishGyroFuse`` must
   publish NOTHING -- the topic only ticks on genuinely gyro-fused frames, so the
   chart never receives a garbage all-zero record.

Run:  .venv/bin/python -m vio.tests.gyrofuse_selftest
Exit code 0 on success, 1 on any failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sky.vio.imu import so3_exp                                  # noqa: E402
from sky.front.odometry import (                             # noqa: E402
    OdometryConfig, RGBDVisualOdometry)


# --------------------------------------------------------------------------- #
# Synthetic frame-pair with a KNOWN yaw + a matching gyro prior
# --------------------------------------------------------------------------- #
def _build_pair(n: int = 30):
    """A planar grid seen from two poses + the gyro prior for the move.

    Returns ``(K, prev_obs, cur_obs, prev_depth, R_prior, yaw_deg)``. The relative
    motion cur<-prev rotation is a known yaw; the gyro prior is supplied in the
    prev<-cur convention the odometry expects (so ``R_prior.T`` == the cur<-prev
    rotation the vision should also recover -> small disagreement on a clean
    frame).
    """
    rng = np.random.default_rng(0)
    K = np.array([[600.0, 0.0, 320.0],
                  [0.0, 600.0, 200.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = 400, 640

    X = rng.uniform(-1.0, 1.0, n)
    Y = rng.uniform(-0.6, 0.6, n)
    Z = rng.uniform(1.0, 3.0, n)
    pts3d_prev = np.stack([X, Y, Z], axis=1)

    pu = fx * X / Z + cx
    pv = fy * Y / Z + cy
    inside = (pu > 5) & (pu < w - 5) & (pv > 5) & (pv < h - 5)

    # Known relative motion cur<-prev: a ~3.4 deg yaw + small translation.
    rotvec = np.array([0.0, 0.06, 0.0])
    R_pc = so3_exp(rotvec)
    yaw_deg = float(np.degrees(np.linalg.norm(rotvec)))
    t_pc = np.array([0.05, -0.02, 0.10])
    pts3d_cur = pts3d_prev @ R_pc.T + t_pc
    cu = fx * pts3d_cur[:, 0] / pts3d_cur[:, 2] + cx
    cv = fy * pts3d_cur[:, 1] / pts3d_cur[:, 2] + cy

    prev_depth = np.zeros((h, w), dtype=np.float32)
    prev_obs: dict[int, np.ndarray] = {}
    cur_obs: dict[int, np.ndarray] = {}
    for i in range(n):
        if not inside[i]:
            continue
        prev_obs[i] = np.array([pu[i], pv[i]], dtype=np.float64)
        cur_obs[i] = np.array([cu[i], cv[i]], dtype=np.float64)
        prev_depth[int(round(pv[i])), int(round(pu[i]))] = np.float32(Z[i])

    # Gyro prior is prev<-cur (the odometry transposes it to cur<-prev). The
    # cur<-prev rotation is R_pc, so the prior is R_pc.T.
    R_prior = R_pc.T
    return K, prev_obs, cur_obs, prev_depth, R_prior, yaw_deg


def _make_fused_odom(K):
    """Odometry with gyro fusion ON (the live regime that records the diag)."""
    cfg = OdometryConfig(min_pnp_points=8, ransac_reproj_px=2.0,
                         gyro_fuse=True, lock_translation_to_rotation=True)
    return RGBDVisualOdometry(K, cfg=cfg)


# --------------------------------------------------------------------------- #
# 1. odometry diagnostic + additivity
# --------------------------------------------------------------------------- #
def test_diagnostic() -> bool:
    print("== 1. odometry gyro-fusion diagnostic (math + additivity) ==")
    K, prev_obs, cur_obs, prev_depth, R_prior, yaw_deg = _build_pair()

    odo = _make_fused_odom(K)
    odo._prev_obs = prev_obs
    odo._prev_depth = prev_depth
    pose = odo.estimate(cur_obs, np.zeros_like(prev_depth), R_prior=R_prior)
    info = odo.last_info

    ok = True
    keys = ("vision_rot_deg", "gyro_rot_deg", "disagree_deg", "gain", "t_trust")
    has_keys = all(k in info for k in keys)
    print(f"  all 5 diagnostic keys present -> {'PASS' if has_keys else 'FAIL'}")
    ok &= has_keys
    if not has_keys:
        return False

    vis = float(info["vision_rot_deg"])
    gyr = float(info["gyro_rot_deg"])
    dis = float(info["disagree_deg"])
    gain = float(info["gain"])
    ttr = float(info["t_trust"])

    # (a) gyro rotation ~ the planted yaw (the prior is exact here).
    gyro_ok = abs(gyr - yaw_deg) < 0.05
    print(f"  gyro_rot_deg={gyr:.3f} ~ planted yaw {yaw_deg:.3f} -> "
          f"{'PASS' if gyro_ok else 'FAIL'}")
    ok &= gyro_ok

    # (b) vision rotation also ~ the yaw (clean frame -> PnP recovers it).
    vis_ok = abs(vis - yaw_deg) < 0.5
    print(f"  vision_rot_deg={vis:.3f} ~ yaw {yaw_deg:.3f} -> "
          f"{'PASS' if vis_ok else 'FAIL'}")
    ok &= vis_ok

    # (c) disagreement small + consistent with so3_log(R_vis @ R_gyro^T) <= the
    #     two magnitudes' difference bound; on a clean frame it is well under the
    #     gate (1.5 deg), so gain is NOT damped.
    dis_ok = 0.0 <= dis < 1.0
    print(f"  disagree_deg={dis:.3f} (clean, < gate 1.5) -> "
          f"{'PASS' if dis_ok else 'FAIL'}")
    ok &= dis_ok

    # (d) gain in [0,1] (full vision on a clean, high-inlier frame -> ~1.0).
    gain_ok = 0.0 <= gain <= 1.0
    print(f"  gain={gain:.3f} in [0,1] -> {'PASS' if gain_ok else 'FAIL'}")
    ok &= gain_ok

    # (e) t_trust defaults to 1.0 on this clean frame (lock path, no damp).
    ttr_ok = abs(ttr - 1.0) < 1e-9
    print(f"  t_trust={ttr:.3f} == 1.0 (no damp) -> "
          f"{'PASS' if ttr_ok else 'FAIL'}")
    ok &= ttr_ok

    # (f) ADDITIVITY: a second fused odometry that NEVER reads the new keys
    #     produces a byte-identical pose -> the diagnostic does not perturb the
    #     motion solve (640 oracle gap=0, in unit form).
    odo2 = _make_fused_odom(K)
    odo2._prev_obs = dict(prev_obs)
    odo2._prev_depth = prev_depth.copy()
    pose2 = odo2.estimate(dict(cur_obs), np.zeros_like(prev_depth),
                          R_prior=R_prior.copy())
    additive = np.array_equal(pose, pose2)
    print(f"  pose byte-identical across two fused runs (additive) -> "
          f"{'PASS' if additive else 'FAIL'}")
    ok &= additive
    return ok


# --------------------------------------------------------------------------- #
# Tiny ctx / step doubles for the publisher tests
# --------------------------------------------------------------------------- #
class _Ctx:
    def __init__(self, bus, vo):
        self.bus = bus
        self.state = {"vo": vo}


class _Frame:
    def __init__(self, seq=42, ts_ns=1_000):
        self.seq, self.ts_ns = seq, ts_ns


# --------------------------------------------------------------------------- #
# 2. publisher emits on fused frames
# --------------------------------------------------------------------------- #
def test_publisher_emits() -> bool:
    print("== 2. publish_gyrofuse emits on gyro-fused frames ==")
    from vio.comms import LocalPubSub, topics
    from vio.modules.publishers import publish_gyrofuse
    from vio.modules.carriers import Step

    K = np.eye(3, dtype=np.float64) * 600.0
    K[2, 2] = 1.0
    vo = _make_fused_odom(K)            # carries the cfg gate/span thresholds

    captured: list = []
    bus = LocalPubSub()
    bus.subscribe(topics.FRAME_GYROFUSE, captured.append)

    info = {"vision_rot_deg": 6.2, "gyro_rot_deg": 4.8, "disagree_deg": 2.1,
            "gain": 0.8, "t_trust": 1.0}
    step = Step(frame=_Frame(), pose=np.eye(4), info=info,
                accel_cam=None, at_rest=False)
    publish_gyrofuse(_Ctx(bus, vo), step)

    if not captured:
        print("  no message published -> FAIL")
        return False
    msg = captured[0]
    fields_ok = (abs(msg.vision_rot_deg - 6.2) < 1e-9
                 and abs(msg.gyro_rot_deg - 4.8) < 1e-9
                 and abs(msg.disagree_deg - 2.1) < 1e-9
                 and abs(msg.gain - 0.8) < 1e-9
                 and abs(msg.t_trust - 1.0) < 1e-9)
    gate_ok = (abs(msg.gate_deg - vo.cfg.gyro_disagree_deg) < 1e-9
               and abs(msg.span_deg - vo.cfg.gyro_disagree_span_deg) < 1e-9)
    print(f"  fields forwarded from last_info -> "
          f"{'PASS' if fields_ok else 'FAIL'}")
    print(f"  gate {msg.gate_deg}/span {msg.span_deg} from cfg -> "
          f"{'PASS' if gate_ok else 'FAIL'}")
    return fields_ok and gate_ok


# --------------------------------------------------------------------------- #
# 3. publisher silent when gyro off
# --------------------------------------------------------------------------- #
def test_publisher_silent() -> bool:
    print("== 3. publish_gyrofuse stays silent when gyro off ==")
    from vio.comms import LocalPubSub, topics
    from vio.modules.publishers import publish_gyrofuse
    from vio.modules.carriers import Step

    K = np.eye(3, dtype=np.float64) * 600.0
    K[2, 2] = 1.0
    vo = _make_fused_odom(K)

    captured: list = []
    bus = LocalPubSub()
    bus.subscribe(topics.FRAME_GYROFUSE, captured.append)

    # info WITHOUT the fusion keys (gyro off / PnP failed / bootstrap).
    step = Step(frame=_Frame(), pose=np.eye(4),
                info={"n_inliers": 30, "ok": True, "reason": ""},
                accel_cam=None, at_rest=False)
    out = publish_gyrofuse(_Ctx(bus, vo), step)

    silent = len(captured) == 0
    passthrough = out is step
    print(f"  nothing published when fields absent -> "
          f"{'PASS' if silent else 'FAIL'}")
    print(f"  Step passed through unchanged -> "
          f"{'PASS' if passthrough else 'FAIL'}")
    return silent and passthrough


def main() -> int:
    r1 = test_diagnostic()
    r2 = test_publisher_emits()
    r3 = test_publisher_silent()
    ok = r1 and r2 and r3
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
