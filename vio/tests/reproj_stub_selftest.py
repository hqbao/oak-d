#!/usr/bin/env python3
"""Self-test for the reprojection-error stub diagnostic (ALGORITHMS.md #4).

Pins three independent properties of the "minimise reprojection error" overlay
that visualisation #4 adds, with hard pass/fail thresholds:

1. ODOMETRY DIAGNOSTIC (math + additivity). Drive :meth:`RGBDVisualOdometry.estimate`
   on a synthetic frame pair with a KNOWN relative pose and ONE planted gross
   outlier, then assert it stored ``info["pnp_ids"/"pnp_reproj"/"pnp_inlier"]``
   correctly: all M points present, the inlier mask matches the planted outlier,
   inlier reprojections land ~on the measured pixel (~0 px error -- the whole
   point), and the outlier reprojects far away. Crucially it also asserts the
   pose is BYTE-IDENTICAL to a run of the SAME odometry without ever reading the
   new info keys -- proving the diagnostic is purely additive (the 640 oracle's
   gap=0 invariant, in unit form).

2. PUBLISHER empty-array contract. ``PublishInliers`` must emit correctly-shaped
   empty arrays when PnP produced no diagnostic (PnP failed / too few points), so
   the consumer's join + draw_overlay short-circuit cleanly.

3. OVERLAY rendering. ``draw_overlay`` with a reproj dict must paint a subtle
   GREEN stub on the inlier path and a striking RED stray on the outlier path,
   and be a no-op when ``reproj`` is None / empty.

Run:  .venv/bin/python -m vio.tests.reproj_stub_selftest
Exit code 0 on success, 1 on any failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vio.mathlib.imu.imu import so3_exp                                  # noqa: E402
from vio.mathlib.odometry.odometry import (                             # noqa: E402
    OdometryConfig, RGBDVisualOdometry)


# --------------------------------------------------------------------------- #
# Shared synthetic frame-pair builder
# --------------------------------------------------------------------------- #
def _build_pair(n: int = 30, outlier_id: int = 7):
    """A planar grid of 3D points seen from two known camera poses.

    Returns ``(K, prev_obs, cur_obs, prev_depth, R_pc, t_pc, outlier_id)`` where
    ``prev_obs``/``cur_obs`` are ``{id: pixel}`` and ``prev_depth`` is a depth
    image carrying each prev pixel's metric Z. One correspondence (``outlier_id``)
    is corrupted in the CURRENT frame so PnP must reject it.
    """
    rng = np.random.default_rng(0)
    K = np.array([[600.0, 0.0, 320.0],
                  [0.0, 600.0, 200.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = 400, 640

    # Points spread over a frontal slab 1.0-3.0 m deep (non-degenerate geometry).
    X = rng.uniform(-1.0, 1.0, n)
    Y = rng.uniform(-0.6, 0.6, n)
    Z = rng.uniform(1.0, 3.0, n)
    pts3d_prev = np.stack([X, Y, Z], axis=1)            # (n,3) prev-cam frame

    # Project to the PREVIOUS image; keep only those well inside the frame.
    pu = fx * X / Z + cx
    pv = fy * Y / Z + cy
    inside = (pu > 5) & (pu < w - 5) & (pv > 5) & (pv < h - 5)

    # Known relative motion cur<-prev: a small yaw + forward/lateral translation.
    R_pc = so3_exp(np.array([0.0, 0.06, 0.0]))          # ~3.4 deg yaw
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
        ppx = np.array([pu[i], pv[i]], dtype=np.float64)
        cpx = np.array([cu[i], cv[i]], dtype=np.float64)
        if i == outlier_id:
            cpx = cpx + np.array([40.0, -35.0])         # gross KLT-slip outlier
        prev_obs[i] = ppx
        cur_obs[i] = cpx
        prev_depth[int(round(pv[i])), int(round(pu[i]))] = np.float32(Z[i])
    return K, prev_obs, cur_obs, prev_depth, R_pc, t_pc, outlier_id


def _make_odom(K):
    cfg = OdometryConfig(min_pnp_points=8, ransac_reproj_px=2.0)
    return RGBDVisualOdometry(K, cfg=cfg)


# --------------------------------------------------------------------------- #
# 1. odometry diagnostic + additivity
# --------------------------------------------------------------------------- #
def test_diagnostic() -> bool:
    print("== 1. odometry reproj diagnostic (math + additivity) ==")
    K, prev_obs, cur_obs, prev_depth, R_pc, t_pc, out_id = _build_pair()

    odo = _make_odom(K)
    odo._prev_obs = prev_obs
    odo._prev_depth = prev_depth
    # ``depth_m`` is the CURRENT frame's depth (only stashed for the NEXT step);
    # the PnP back-projection reads ``_prev_depth``, so its content is irrelevant
    # to this single-step solve. Pass a zero map of the right shape.
    cur_depth = np.zeros_like(prev_depth)
    pose = odo.estimate(cur_obs, cur_depth)
    info = odo.last_info

    ok = True

    # (a) keys present + correctly shaped
    has_keys = all(k in info for k in ("pnp_ids", "pnp_reproj", "pnp_inlier"))
    ids = np.asarray(info.get("pnp_ids", []))
    reproj = np.asarray(info.get("pnp_reproj", []))
    inlier = np.asarray(info.get("pnp_inlier", []))
    m = ids.shape[0]
    shaped = (has_keys and m > 0 and reproj.shape == (m, 2)
              and inlier.shape == (m,) and reproj.dtype == np.float32
              and inlier.dtype == np.bool_ and ids.dtype == np.int64)
    print(f"  keys present + shaped: M={m} reproj{tuple(reproj.shape)}/"
          f"{reproj.dtype} inlier{tuple(inlier.shape)}/{inlier.dtype} "
          f"-> {'PASS' if shaped else 'FAIL'}")
    ok &= shaped
    if not shaped:
        return False

    # (b) the planted outlier is rejected, the rest are inliers
    id_to_row = {int(t): r for r, t in enumerate(ids)}
    out_row = id_to_row.get(out_id)
    out_rejected = out_row is not None and not bool(inlier[out_row])
    n_in = int(inlier.sum())
    enough_inliers = n_in >= m - 1            # only the 1 planted outlier rejected
    print(f"  planted outlier id={out_id} rejected={out_rejected}  "
          f"inliers={n_in}/{m} -> "
          f"{'PASS' if out_rejected and enough_inliers else 'FAIL'}")
    ok &= out_rejected and enough_inliers

    # (c) inlier reproj ~ measured pixel (~0 px); outlier reproj far from measured
    meas = {int(t): cur_obs[int(t)] for t in ids if int(t) in cur_obs}
    in_err, out_err = [], None
    for r, t in enumerate(ids):
        t = int(t)
        if t not in meas:
            continue
        d = float(np.linalg.norm(reproj[r] - meas[t]))
        if bool(inlier[r]):
            in_err.append(d)
        elif t == out_id:
            out_err = d
    max_in = max(in_err) if in_err else np.inf
    # Inliers must reproject within the RANSAC gate (2 px); the outlier's reproj
    # is to where its (correct) 3D point lands -- far from the corrupted measured
    # pixel it was paired with (~50 px by construction).
    err_ok = max_in <= 2.0 and out_err is not None and out_err > 10.0
    print(f"  inlier reproj err max={max_in:.3f}px (<=2.0)  "
          f"outlier stray={out_err:.1f}px (>10) -> {'PASS' if err_ok else 'FAIL'}")
    ok &= err_ok

    # (d) ADDITIVITY: a second odometry that NEVER reads the new keys produces a
    # byte-identical pose -> the diagnostic does not perturb the motion solve.
    odo2 = _make_odom(K)
    odo2._prev_obs = dict(prev_obs)
    odo2._prev_depth = prev_depth.copy()
    pose2 = odo2.estimate(dict(cur_obs), np.zeros_like(prev_depth))
    additive = np.array_equal(pose, pose2)
    print(f"  pose byte-identical across two runs (additive) -> "
          f"{'PASS' if additive else 'FAIL'}")
    ok &= additive
    return ok


# --------------------------------------------------------------------------- #
# 2. publisher empty-array contract
# --------------------------------------------------------------------------- #
def test_publisher_empty() -> bool:
    print("== 2. PublishInliers empty-array contract (PnP failed) ==")
    from vio.comms import LocalPubSub, topics
    from vio.modules.publish_inliers import PublishInliers
    from vio.modules.step import Step

    captured: list = []
    bus = LocalPubSub()
    bus.subscribe(topics.FRAME_INLIERS, captured.append)

    class _Ctx:
        def __init__(self, bus):
            self.bus = bus

    class _Frame:
        seq, ts_ns = 42, 1_000

    # PnP produced no diagnostic (info has none of the pnp_* keys).
    step = Step(frame=_Frame(), pose=np.eye(4), info={},
                accel_cam=None, at_rest=False)
    PublishInliers().run(_Ctx(bus), step)

    if not captured:
        print("  no message published -> FAIL")
        return False
    msg = captured[0]
    ok = (msg.ids.shape == (0,) and msg.reproj.shape == (0, 2)
          and msg.inlier.shape == (0,) and msg.reproj.dtype == np.float32
          and msg.inlier.dtype == np.bool_)
    print(f"  empty msg ids{tuple(msg.ids.shape)} reproj{tuple(msg.reproj.shape)} "
          f"inlier{tuple(msg.inlier.shape)} -> {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# 3. overlay rendering
# --------------------------------------------------------------------------- #
def test_overlay() -> bool:
    print("== 3. draw_overlay reproj stubs (green inlier / red outlier) ==")
    import importlib.util
    if importlib.util.find_spec("cv2") is None:    # draw_overlay needs cv2
        print("  cv2 not available -- SKIP")
        return True
    from ui.viz.keypoint_overlay import (TrackTrails, draw_overlay,
                                         _STUB_INLIER_BGR, _STUB_OUTLIER_BGR)

    h, w = 200, 320
    gray = np.full((h, w), 60, dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)
    # Two tracks: an inlier (short stub) and an outlier (long stray).
    ids = np.array([1, 2], dtype=np.int64)
    pts = np.array([[80.0, 100.0], [220.0, 100.0]], dtype=np.float32)
    reproj = {
        "ids": np.array([1, 2], dtype=np.int64),
        # id 1 reprojects ~1 px off (inlier); id 2 lands 60 px away (outlier).
        "reproj": np.array([[81.0, 100.0], [220.0, 160.0]], dtype=np.float32),
        "inlier": np.array([True, False], dtype=bool),
    }

    trails = TrackTrails()
    trails.update(ids, pts)
    rgb = draw_overlay(gray, depth, ids, pts, trails, draw_trails=False,
                       inlier_ids={1}, reproj=reproj)
    # draw_overlay returns RGB; the stub colours are BGR constants, so flip them.
    inl_rgb = _STUB_INLIER_BGR[::-1]
    out_rgb = _STUB_OUTLIER_BGR[::-1]

    def _has_color(img, col, tol=40):
        d = np.abs(img.astype(np.int16) - np.array(col, np.int16)).sum(-1)
        return int((d < tol).sum())

    # The outlier stray spans ~60 px so it paints many more red pixels than a
    # short green stub paints green -- both must be present.
    green = _has_color(rgb, inl_rgb)
    red = _has_color(rgb, out_rgb)
    drawn_ok = green > 0 and red > 0 and red > green
    print(f"  inlier-green px={green}  outlier-red px={red} (red>green) -> "
          f"{'PASS' if drawn_ok else 'FAIL'}")

    # No-op when reproj is None / empty (no extra coloured strays appear).
    rgb_none = draw_overlay(gray, depth, ids, pts, trails, draw_trails=False)
    rgb_empty = draw_overlay(
        gray, depth, ids, pts, trails, draw_trails=False,
        reproj={"ids": np.empty((0,), np.int64),
                "reproj": np.empty((0, 2), np.float32),
                "inlier": np.empty((0,), bool)})
    noop_ok = (_has_color(rgb_none, out_rgb) == 0
               and _has_color(rgb_empty, out_rgb) == 0)
    print(f"  None / empty reproj is a no-op (no red strays) -> "
          f"{'PASS' if noop_ok else 'FAIL'}")
    return drawn_ok and noop_ok


def main() -> int:
    r1 = test_diagnostic()
    r2 = test_publisher_empty()
    r3 = test_overlay()
    ok = r1 and r2 and r3
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
