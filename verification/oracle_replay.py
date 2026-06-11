"""In-process byte-parity ORACLE driven by the NEW projects' verbatim-ported math.

WHAT THIS IS
------------
The pre-split reference oracle is ``ours/tools/vio_run.py`` (``score_session``):
it runs the from-scratch RGB-D VIO over a recorded session and scores the
trajectory against the recorded Basalt reference with a rigid-SE3 (and Sim3)
Umeyama alignment (ATE). Those numbers are the FROZEN baseline
(``verification/baseline_metrics.json``).

This module reproduces that EXACT scoring loop -- the same deterministic seeding,
task order and per-frame VO graph that ``ours.app.run_replay`` realises in-process
-- but it imports the MATH from the SPLIT projects instead of ``ours.lib``:

    * ``imu_camera`` -- session reader (the replay data source), SGM dense stereo,
      the gyro preintegrator's SO(3) helpers.
    * ``vio``        -- ``RGBDVisualOdometry`` (f2f), ``WindowedRGBDOdometry`` (ba),
      ``WindowedVIORGBDOdometry`` (vio), the gyro preintegrator + ``OdometryConfig``.
    * ``slam``       -- ``SlamMap`` / ``SlamConfig`` + ``se3_inv`` (loop closure +
      pose graph).

NO ``IPCPubSub``: this is the single-process oracle. The live multi-process IPC
pipeline (``*/main.py``) has nondeterministic timing and CANNOT give byte-parity;
its CONTRACT (the wire codec) is proved separately by
``verification/ipc_comms_selftest.py``.

WHY IT MUST MATCH BYTE-FOR-BYTE
-------------------------------
The per-component math was ported VERBATIM (``vio.tests.vio_ba_selftest``,
``slam.tests.loop_closure_selftest``, the stereo / imucam-sync self-tests already
proved byte-parity per module; this harness's own probe re-confirms the source is
identical modulo import roots + docstrings). So the END-TO-END oracle below MUST
reproduce the PRE-SPLIT reference's ATE/Sim3 scores exactly. ``oracle_replay_selftest``
asserts that against the FROZEN ``verification/baseline_metrics.json`` (the
pre-split reference tree itself has since been removed).

The ``ate`` / ``umeyama`` scoring below is COPIED VERBATIM from
``ours/tools/vio_run.py`` (no algorithm change) so the comparison is the SAME
estimator on both sides -- only the VO math source differs.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# --- NEW-project math (the split projects), NOT ours.lib --------------------- #
from imu_camera.io.reader import SessionReader
from sky.depth.stereo import SGMConfig, SGMStereoMatcher
from vio.mathlib.imu.imu import GyroPreintegrator
from vio.mathlib.odometry.odometry import OdometryConfig, RGBDVisualOdometry
from vio.mathlib.backend.windowed import WindowedConfig, WindowedRGBDOdometry
from vio.mathlib.backend.vio_window import WindowedVIORGBDOdometry
from slam.mathlib.loop.slam import SlamConfig, SlamMap
from sky.math import se3_inv


# --------------------------------------------------------------------------- #
# Basalt reference + ATE scoring -- COPIED VERBATIM from ours/tools/vio_run.py.
# (Same estimator on both sides; only the VO math source above differs.)
# --------------------------------------------------------------------------- #
def load_basalt_positions(session_dir: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    path = session_dir / "basalt" / "vio_pose.jsonl"
    import json
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[int(d["seq"])] = np.asarray(d["pos"], dtype=np.float64)
    return out


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    """Least-squares similarity/rigid transform mapping src onto dst.

    src, dst: (N,3). Returns (R, t, s) such that dst ~= s*R@src + t.
    """
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc.T @ sc) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_s = (sc ** 2).sum() / src.shape[0]
        s = np.trace(np.diag(D) @ S) / var_s
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def ate(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    R, t, s = umeyama(src, dst, with_scale)
    aligned = (s * (R @ src.T)).T + t
    err = np.linalg.norm(aligned - dst, axis=1)
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
        "scale": float(s),
    }


# --------------------------------------------------------------------------- #
# The in-process replay oracle: reproduces vio_run.score_session's per-frame VO
# graph (the same graph ours.app.run_replay builds, single LocalPubSub / FIFO /
# in-process engines) on the NEW-project math. Returns the rigid-SE3 ATE block
# (with the Sim3 scale folded in), byte-identical to score_session.
# --------------------------------------------------------------------------- #
def score_session_oracle(session_dir: Path, max_frames: int, verbose: bool = False,
                         quiet: bool = True, use_imu: bool = True,
                         backend: str = "f2f", slam_kf_every: int = 5,
                         slam_radius_m: float = 0.0, slam_kf_min_trans: float = 0.0,
                         slam_kf_min_rot: float = 0.0, slam_max_kf: int = 0,
                         use_gyro: bool = True, depth_source: str = "chip",
                         depth_fast: bool = False, sgm_cfg=None, marg: bool = False,
                         vo_trans_sigma: float = 0.0):
    """In-process oracle: NEW-project math through the EXACT vio_run scoring loop.

    Mirrors ``ours.tools.vio_run.score_session`` line-for-line (same deterministic
    seeding: gyro_bias / accel-align from the first ~0.3 s; KLT seeded by the
    frontend; FIFO frame order; in-process engines). The ONLY difference is that
    every algorithm class is imported from the split projects (see module head).
    Returns the same dict ``score_session`` returns (rigid ATE + scale + loops +
    drift), or ``None`` if there is no Basalt overlap.
    """
    reader = SessionReader(session_dir)
    n = len(reader) if max_frames <= 0 else min(max_frames, len(reader))

    # Depth source: 'chip' = recorded StereoDepth; 'ours' = our SGM rebuilt live.
    matcher = None
    if depth_source == "ours":
        cfg = sgm_cfg or (SGMConfig.live() if depth_fast else SGMConfig())
        matcher = SGMStereoMatcher.from_calib(reader.calib, cfg)

    odom_cfg = OdometryConfig(gyro_fuse=use_gyro,
                              use_own_pnp=os.environ.get("OAKD_OWN_PNP", "1") != "0")

    # Load the IMU stream once up front (vio backend needs it at construction;
    # the gyro rotation prior reuses the same arrays).
    imu = None
    R_imu_cam = None
    if reader.calib.has_imu_extrinsics:
        imu_raw = reader.load_imu()
        if imu_raw["ts_ns"].size > 1:
            imu = imu_raw
            R_imu_cam = reader.calib.T_imu_left[:3, :3]

    slam = None
    if backend in ("ba", "slam"):
        wcfg = WindowedConfig(use_marg=marg)
        if vo_trans_sigma > 0.0:
            wcfg.ba.use_vo_trans_prior = True
            wcfg.ba.vo_trans_sigma_m = vo_trans_sigma
        vo = WindowedRGBDOdometry(reader.K, cfg=wcfg, odom_cfg=odom_cfg)
        if backend == "slam":
            slam = SlamMap(reader.K, SlamConfig(
                loop_search_radius_m=slam_radius_m,
                loop_max_odom_rot_deg=30.0,
                kf_min_trans_m=slam_kf_min_trans,
                kf_min_rot_deg=slam_kf_min_rot,
                max_keyframes=slam_max_kf))
    elif backend == "vio":
        if imu is None:
            raise SystemExit(
                "backend 'vio' requires a session with IMU extrinsics "
                f"(none in {session_dir})")
        gyro_cam = (R_imu_cam @ imu["gyro"].T).T
        accel_cam = (R_imu_cam @ imu["accel"].T).T
        t0 = imu["ts_ns"][0]
        win = imu["ts_ns"] <= t0 + int(0.3 * 1e9)
        bg0 = gyro_cam[win].mean(axis=0) if win.any() else np.zeros(3)
        vo = WindowedVIORGBDOdometry(
            reader.K, imu["ts_ns"], gyro_cam, accel_cam,
            bg0=bg0, ba0=np.zeros(3), odom_cfg=odom_cfg)
    else:
        vo = RGBDVisualOdometry(reader.K, odom_cfg)

    # Gyro preintegrator + gravity-align the initial attitude (same as oracle).
    pre = None
    if use_gyro and imu is not None:
        pre = GyroPreintegrator(imu["ts_ns"], imu["gyro"],
                                reader.calib.T_imu_left)
        t0 = imu["ts_ns"][0]
        win = imu["ts_ns"] <= t0 + int(0.3 * 1e9)   # first ~0.3 s
        accel_imu = imu["accel"][win].mean(axis=0)
        vo.align_to_gravity(R_imu_cam @ accel_imu)

    if not quiet:
        print(f"session : {reader.dir}")
        print(f"frames  : {n}/{len(reader)}")
        print(f"imu     : {'gyro rotation prior ON' if pre else 'OFF (vision only)'}")
        depth_tag = ("ours SGM (" + ("live" if depth_fast else "full") + ")"
                     if matcher is not None else "chip StereoDepth")
        print(f"depth   : {depth_tag}")
        print("running VO ...")

    est: dict[int, np.ndarray] = {}
    n_ok = 0
    prev_ts = None
    frames_since_kf = 0
    last_kf_idx = -1
    anchors: dict[int, tuple[int, np.ndarray]] = {}
    for i in range(n):
        f = reader.load_frame(i, load_right=(matcher is not None))
        depth = f.depth_m
        if matcher is not None:
            depth = matcher.dense_depth(f.gray_left, f.gray_right)
        R_prior = None
        if pre is not None and prev_ts is not None:
            R_prior = pre.delta_rotation(prev_ts, f.ts_ns)
        if backend == "vio":
            pose = vo.process(f.gray_left, depth, f.ts_ns, R_prior=R_prior)
        else:
            pose = vo.process(f.gray_left, depth, R_prior=R_prior)
        prev_ts = f.ts_ns
        est[f.seq] = pose[:3, 3].copy()
        if vo.last_info.get("ok"):
            n_ok += 1

        if slam is not None:
            is_kf = (last_kf_idx < 0) or (frames_since_kf >= slam_kf_every)
            if is_kf:
                frames_since_kf = 0
                slam.add_keyframe(pose, f.gray_left, depth, seq=f.seq)
                last_kf_idx = len(slam.kf_orig) - 1
            else:
                frames_since_kf += 1
            rel = se3_inv(slam.kf_orig[last_kf_idx]) @ pose
            anchors[f.seq] = (last_kf_idx, rel)

        if verbose and i % 50 == 0:
            inf = vo.last_info
            print(f"  f{i:4d} tracks={inf.get('n_tracks', 0):3d} "
                  f"pnp={inf.get('n_pnp', 0):3d} "
                  f"inliers={inf.get('n_inliers', 0):3d} ok={inf.get('ok')} "
                  f"pos={pose[:3,3]}")

    # --- SLAM: close loops + pose-graph optimise, then rewrite the trajectory.
    n_loops = 0
    drift_before = drift_after = None
    if slam is not None:
        seqs_sorted = sorted(est)
        p0, p1 = est[seqs_sorted[0]], est[seqs_sorted[-1]]
        drift_before = float(np.linalg.norm(p1 - p0))
        n_loops = len(slam.loop_events)
        slam.optimize()
        for seq, (kidx, rel) in anchors.items():
            est[seq] = (slam.kf_pose[kidx] @ rel)[:3, 3].copy()
        p0, p1 = est[seqs_sorted[0]], est[seqs_sorted[-1]]
        drift_after = float(np.linalg.norm(p1 - p0))
        if not quiet:
            print(f"SLAM: {len(slam.kf_orig)} keyframes, {n_loops} loop closures")
            print(f"  end-start drift: {drift_before*100:.1f} cm "
                  f"-> {drift_after*100:.1f} cm")

    basalt = load_basalt_positions(reader.dir)
    common = sorted(set(est) & set(basalt))
    if len(common) < 10:
        if not quiet:
            print(f"!! only {len(common)} common poses with Basalt -- cannot score")
        return None

    src = np.array([est[s] for s in common])       # our optical-frame traj
    dst = np.array([basalt[s] for s in common])     # Basalt FLU-world traj

    rigid = ate(src, dst, with_scale=False)
    sim = ate(src, dst, with_scale=True)
    traj_len = float(np.linalg.norm(np.diff(dst, axis=0), axis=1).sum())
    rigid["path"] = traj_len
    rigid["scale"] = sim["scale"]
    rigid["loops"] = n_loops
    rigid["drift_before"] = drift_before
    rigid["drift_after"] = drift_after

    if not quiet:
        print(f"VO ok on {n_ok}/{n-1} motion steps")
        print()
        print(f"compared on {len(common)} poses | Basalt path length {traj_len:.2f} m")
        print("--- ATE vs Basalt (rigid SE3 align) ---")
        print(f"  RMSE   = {rigid['rmse']*1000:7.1f} mm")
        print(f"  median = {rigid['median']*1000:7.1f} mm")
        print(f"  max    = {rigid['max']*1000:7.1f} mm")
        print(f"  RMSE/path = {100*rigid['rmse']/traj_len:.2f}%")
        print("--- with scale (Sim3) ---")
        print(f"  RMSE   = {sim['rmse']*1000:7.1f} mm   "
              f"(our scale = {sim['scale']:.3f} of Basalt)")

    return rigid
