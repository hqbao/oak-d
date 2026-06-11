#!/usr/bin/env python3
"""Phase-3 LOOSE-vs-TIGHT ATE benchmark harness (docs/TIGHT_COUPLED_PLAN.md §6).

THE QUESTION THIS ANSWERS
-------------------------
"so sánh cái nào ngon hơn" -- is the tightly-coupled RGB-D VIO backend
(`WindowedVIORGBDOdometry`, IMU preintegration factors, covariance-weighted via
`imu_info_weight=True`) actually better than the existing LOOSE backend
(`WindowedRGBDOdometry`, vision-only windowed BA + a VO translation prior)?

It runs BOTH backends over the gold session suite at TWO resolutions
(full-res chip depth AND the 54x42 VL53L9CX ToF target) and reports, per session:

  * ATE RMSE after rigid-SE(3) Umeyama alignment (cm)            -- the field standard
  * Sim3 scale  (our path scale vs the Basalt reference)          -- scale collapse
  * end-vs-start drift (cm)                                       -- accumulated drift
  * max single-frame step (cm)                                    -- the fast-push "ì lại" symptom
  * phantom translation on in-place-yaw / shake sessions          -- the loose path's documented failure

then a per-session verdict (who wins + by how much) with the fast-push and
in-place-yaw cases highlighted, and a headline answer.

HARD SCOPE / SAFETY (PLAN §4 byte-parity rules)
-----------------------------------------------
This is a NEW, read-only harness. It does NOT modify the loose path, the comms,
`oracle_replay.py`, or any frozen baseline. It only IMPORTS the reusable,
side-effect-free scoring helpers from `verification.oracle_replay`
(`load_basalt_positions`, `umeyama`, `ate`) and the same math classes the live
projects use. The byte-parity oracle (`oracle_replay_selftest.py`) stays gap=0
because this file changes nothing it depends on.

The ONLY difference vs the frozen `score_session_oracle("vio")` path is that the
tight runs here build a `WindowedVIOConfig` with `imu_info_weight=True` -- the
covariance-weighted tight path the PLAN prescribes for the live `--tight`
backend (Phase 1). The frozen oracle's `backend="vio"` entries use the DEFAULT
config (`imu_info_weight=False`) and are untouched.

THE 54x42 ToF RESOLUTION
------------------------
Per the PLAN audit note, the 54x42 simulation is producer-side
(`imu_camera/modules/tof_downsample.py`: SGM at source res -> block-median to
54x42 -> K scaled anisotropically). The gold sessions are recorded at 640x400;
this harness replays them through the SAME producer-side reduction in-process
(SGM dense depth at source res, then `_block_median_valid` to 54x42, gray via
INTER_AREA, K via `_scale_K_to_tof`) so VIO consumes 54x42 transparently -- no
vio-side flag, exactly as the live ToF pipeline does.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reusable, side-effect-free scoring helpers (NOT modified -- read-only import).
from verification.oracle_replay import (  # noqa: E402
    ate,
    load_basalt_positions,
    umeyama,
)

# The same math classes the live projects build. Imported, never edited.
from imu_camera.io.reader import SessionReader  # noqa: E402
from sky.depth.stereo import (  # noqa: E402
    SGMConfig,
    SGMStereoMatcher,
)
from imu_camera.modules.tof_downsample import _block_median_valid  # noqa: E402
from imu_camera.modules.pipeline import TOF_W, TOF_H  # noqa: E402
from vio.mathlib.imu.imu import GyroPreintegrator  # noqa: E402
from sky.front.odometry import OdometryConfig  # noqa: E402
from vio.mathlib.backend.windowed import WindowedConfig, WindowedRGBDOdometry  # noqa: E402
from vio.mathlib.backend.vio_window import (  # noqa: E402
    WindowedVIOConfig,
    WindowedVIORGBDOdometry,
)

import cv2  # noqa: E402


# --------------------------------------------------------------------------- #
# Gold suite + which sessions exercise which documented failure mode.
# --------------------------------------------------------------------------- #
GOLD_DIR = Path("sessions/gold")

# A recorded Basalt trajectory is only a valid reference if it didn't diverge
# (verbatim threshold from verification/vio_oracle_runner.py so we drop the same
# broken refs the oracle drops).
_MAX_VALID_STEP_M = 1.0

# Sessions whose TRUE motion is (near-)rotation-in-place: any net translation the
# estimator reports there is PHANTOM translation -- the classic loose-coupling
# failure (slipped visual tracks read as motion). Highlighted in the verdict.
_INPLACE_YAW = {"yaw_inplace_15s", "push_shake_20s", "quick_motion_15s"}

# Sessions with a fast forward/back push -- the "ì lại" (stall / scale-collapse)
# regime where the loose VO-trans-prior is documented weak.
_FAST_PUSH = {"push_straight_fast_15s", "push_fwdback_20s"}


def basalt_ref_is_broken(positions: dict[int, np.ndarray]) -> bool:
    if len(positions) < 2:
        return True
    pos = np.array([positions[s] for s in sorted(positions)])
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return bool(steps.max() > _MAX_VALID_STEP_M)


# --------------------------------------------------------------------------- #
# 54x42 ToF reduction (producer-side, mirrors imu_camera/modules/tof_downsample).
# --------------------------------------------------------------------------- #
def _scale_K_to_tof(K: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    """K scaled ANISOTROPICALLY to the 54x42 ToF grid (matches main._scale_bundle_to_tof).

    fx, cx scale by TOF_W/src_w ; fy, cy by TOF_H/src_h. The world distance a
    pixel sees is unchanged (depth metres carry through); only the focal length
    in pixels changes with the non-uniform resize.
    """
    sx = TOF_W / float(src_w)
    sy = TOF_H / float(src_h)
    Kt = np.asarray(K, dtype=np.float64).copy()
    Kt[0, 0] *= sx   # fx
    Kt[0, 2] *= sx   # cx
    Kt[1, 1] *= sy   # fy
    Kt[1, 2] *= sy   # cy
    return Kt


def _tof_reduce(gray_src: np.ndarray, depth_src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a source-res (gray, metric depth) pair to the 54x42 ToF grid.

    GRAY  -> cv2.resize INTER_AREA (correct anti-aliasing reduction for intensity).
    DEPTH -> block-median of valid (>0) pixels (the honest ToF reduction; never
             linear, which would blend across depth edges + 0-holes).
    Identical to ToFDownsampleStep, just driven from recorded source-res arrays.
    """
    gray_tof = cv2.resize(gray_src, (TOF_W, TOF_H), interpolation=cv2.INTER_AREA)
    if gray_tof.dtype != np.uint8:
        gray_tof = gray_tof.astype(np.uint8)
    depth_tof = _block_median_valid(depth_src.astype(np.float32), TOF_H, TOF_W)
    return gray_tof, depth_tof


# --------------------------------------------------------------------------- #
# The replay: ONE per-frame loop, parameterised by backend + resolution. It
# mirrors verification.oracle_replay.score_session_oracle's VO graph EXACTLY
# (same deterministic seeding, gyro prior, FIFO order); the only knobs are the
# backend, the tight info-weight, and the 54x42 ToF reduction.
# --------------------------------------------------------------------------- #
def run_session(session_dir: Path, *, backend: str, resolution: str,
                imu_info_weight: bool = False, min_ba_views: int | None = None,
                max_frames: int = 0) -> dict | None:
    """Run one backend over one session at one resolution; return metric dict.

    backend     : "ba" (LOOSE windowed BA) or "vio" (TIGHT preintegration VIO).
    resolution  : "full" (chip depth, native res) or "tof54" (54x42 ToF sim).
    imu_info_weight : tight only -- True = covariance-weighted Omega_I path
                  (Phase 1), the live --tight weight. The frozen oracle uses False.
    min_ba_views: tight only -- lower it for the feature-starved 54x42 profile
                  (PLAN gate 3 allows this); None keeps the config default.

    Returns None if the Basalt reference is missing/broken or there is too little
    overlap to score (so the caller can print "--").
    """
    reader = SessionReader(session_dir)
    n = len(reader) if max_frames <= 0 else min(max_frames, len(reader))

    tof = (resolution == "tof54")

    # K the BACKEND solves with: native at full-res, anisotropically-scaled at ToF.
    if tof:
        # Source dims come from the first frame (gold sessions are 640x400).
        f0 = reader.load_frame(0, load_right=False)
        sh, sw = f0.gray_left.shape[:2]
        K_solve = _scale_K_to_tof(reader.K, sw, sh)
    else:
        K_solve = reader.K

    # Source-res SGM matcher is needed only for the ToF path (depth must be
    # recomputed at source res before the block-median reduction). At full res we
    # use the recorded chip StereoDepth, exactly like the oracle's depth_source="chip".
    matcher = None
    if tof:
        matcher = SGMStereoMatcher.from_calib(reader.calib, SGMConfig())

    odom_cfg = OdometryConfig(gyro_fuse=True,
                              use_own_pnp=os.environ.get("OAKD_OWN_PNP", "1") != "0")

    # IMU stream (tight backend needs it at construction; the gyro prior reuses it).
    imu = None
    R_imu_cam = None
    if reader.calib.has_imu_extrinsics:
        imu_raw = reader.load_imu()
        if imu_raw["ts_ns"].size > 1:
            imu = imu_raw
            R_imu_cam = reader.calib.T_imu_left[:3, :3]

    # --- build the selected backend (LOOSE ba vs TIGHT vio) --------------- #
    if backend == "ba":
        vo = WindowedRGBDOdometry(K_solve, cfg=WindowedConfig(), odom_cfg=odom_cfg)
    elif backend == "vio":
        if imu is None:
            return None  # tight needs IMU extrinsics; can't run -> caller prints "--"
        gyro_cam = (R_imu_cam @ imu["gyro"].T).T
        accel_cam = (R_imu_cam @ imu["accel"].T).T
        t0 = imu["ts_ns"][0]
        win = imu["ts_ns"] <= t0 + int(0.3 * 1e9)
        bg0 = gyro_cam[win].mean(axis=0) if win.any() else np.zeros(3)
        # Covariance-weighted tight path (Phase 1): override ONLY imu_info_weight
        # (+ optional min_ba_views for the 54x42 profile). Everything else is the
        # tuned WindowedVIOConfig default.
        wcfg = WindowedVIOConfig()
        wcfg.vio = replace(wcfg.vio, imu_info_weight=imu_info_weight)
        if min_ba_views is not None:
            wcfg.min_ba_views = int(min_ba_views)
        vo = WindowedVIORGBDOdometry(
            K_solve, imu["ts_ns"], gyro_cam, accel_cam,
            bg0=bg0, ba0=np.zeros(3), cfg=wcfg, odom_cfg=odom_cfg)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    # Gyro preintegrator + gravity-align initial attitude (same as the oracle).
    pre = None
    if imu is not None:
        pre = GyroPreintegrator(imu["ts_ns"], imu["gyro"], reader.calib.T_imu_left)
        t0 = imu["ts_ns"][0]
        win = imu["ts_ns"] <= t0 + int(0.3 * 1e9)
        accel_imu = imu["accel"][win].mean(axis=0)
        vo.align_to_gravity(R_imu_cam @ accel_imu)

    # --- per-frame VO loop (FIFO, gyro rotation prior between frames) ------ #
    est: dict[int, np.ndarray] = {}
    seqs_in_order: list[int] = []
    prev_ts = None
    n_kf = 0
    n_ba_ok = 0           # tight/loose: keyframe solves that returned non-None
    t_start = time.perf_counter()
    for i in range(n):
        f = reader.load_frame(i, load_right=tof)
        if tof:
            # ToF: SGM at source res, then block-median + INTER_AREA to 54x42.
            gray_src, depth_src = matcher.dense_depth_rectified_left(
                f.gray_left, f.gray_right)
            if gray_src.dtype != np.uint8:
                gray_src = np.clip(gray_src, 0.0, 255.0).astype(np.uint8)
            gray, depth = _tof_reduce(gray_src, depth_src)
        else:
            gray, depth = f.gray_left, f.depth_m

        R_prior = None
        if pre is not None and prev_ts is not None:
            R_prior = pre.delta_rotation(prev_ts, f.ts_ns)
        if backend == "vio":
            pose = vo.process(gray, depth, f.ts_ns, R_prior=R_prior)
        else:
            pose = vo.process(gray, depth, R_prior=R_prior)
        prev_ts = f.ts_ns
        est[f.seq] = pose[:3, 3].copy()
        seqs_in_order.append(f.seq)
        info = vo.last_info
        if info.get("is_kf"):
            n_kf += 1
            # a window solve happened iff the refined pose was applied; the engine
            # surfaces its solve via last_info -- 'vio_imu'/'vio_reproj_px' (tight)
            # or 'ba_reproj_px' (loose). Presence == a non-None run_ba.
            if any(k in info for k in ("vio_imu", "vio_reproj_px", "ba_reproj_px")):
                n_ba_ok += 1
    elapsed = time.perf_counter() - t_start

    # --- score against the Basalt reference -------------------------------- #
    basalt = load_basalt_positions(reader.dir)
    if not basalt or basalt_ref_is_broken(basalt):
        return None
    common = sorted(set(est) & set(basalt))
    if len(common) < 10:
        return None

    src = np.array([est[s] for s in common])     # our optical-frame trajectory
    dst = np.array([basalt[s] for s in common])  # Basalt FLU-world reference

    rigid = ate(src, dst, with_scale=False)      # ATE RMSE (no scale)
    sim = ate(src, dst, with_scale=True)         # Sim3 -> scale
    traj_len = float(np.linalg.norm(np.diff(dst, axis=0), axis=1).sum())

    # --- failure-case metrics --------------------------------------------- #
    # max single-frame step of OUR trajectory (the fast-push "ì lại" stalls show
    # up as a giant single step or, conversely, a near-zero crawl; we report the
    # peak step so the table shows whether the estimator jumped or held).
    est_ordered = np.array([est[s] for s in seqs_in_order])
    our_steps = (np.linalg.norm(np.diff(est_ordered, axis=0), axis=1)
                 if len(est_ordered) > 1 else np.zeros(1))
    max_step_m = float(our_steps.max())

    # phantom translation: net displacement of OUR path relative to the
    # reference's net displacement. On an in-place-yaw session the reference net
    # is ~0, so any net we report IS phantom. We give it after Sim3-aligning our
    # path onto the reference (so a scale-collapsed estimate isn't unfairly
    # flattered) -- this is the residual translation the IMU should kill.
    # (verification.oracle_replay.ate returns only summary scalars, so re-run the
    # same Umeyama here to get the aligned endpoints.)
    R_a, t_a, s_a = umeyama(src, dst, with_scale=True)
    aligned = (s_a * (R_a @ src.T)).T + t_a
    ref_net = float(np.linalg.norm(dst[-1] - dst[0]))
    our_net_aligned = float(np.linalg.norm(aligned[-1] - aligned[0]))
    phantom_m = abs(our_net_aligned - ref_net)

    # end-vs-start drift of OUR raw (un-aligned) path -- accumulated error.
    drift_m = float(np.linalg.norm(est_ordered[-1] - est_ordered[0]))

    return {
        "ate_cm": rigid["rmse"] * 100.0,
        "scale": sim["scale"],
        "drift_cm": drift_m * 100.0,
        "max_step_cm": max_step_m * 100.0,
        "phantom_cm": phantom_m * 100.0,
        "ref_net_m": ref_net,
        "path_m": traj_len,
        "n_common": len(common),
        "n_kf": n_kf,
        "n_ba_ok": n_ba_ok,
        "ba_ok_frac": (n_ba_ok / n_kf) if n_kf else 0.0,
        "ms_per_frame": (elapsed / max(n, 1)) * 1000.0,
    }


# --------------------------------------------------------------------------- #
# Comparison table + verdict.
# --------------------------------------------------------------------------- #
def _fmt(v, width, prec=2, dash="--"):
    if v is None:
        return f"{dash:>{width}}"
    return f"{v:>{width}.{prec}f}"


def _verdict(loose: dict | None, tight: dict | None, name: str) -> str:
    """One-line verdict: who wins on ATE + by how much, with the failure flag."""
    if loose is None and tight is None:
        return "both failed to score"
    if loose is None:
        return "TIGHT only (loose did not score)"
    if tight is None:
        return "LOOSE only (tight did not score)"
    la, ta = loose["ate_cm"], tight["ate_cm"]
    flag = ""
    if name in _INPLACE_YAW:
        flag = "  [in-place-yaw: lower phantom wins]"
    elif name in _FAST_PUSH:
        flag = "  [fast-push: watch scale + max-step]"
    if ta < la:
        pct = 100.0 * (la - ta) / la if la > 1e-9 else 0.0
        return f"TIGHT wins  ({la:.1f} -> {ta:.1f} cm, -{pct:.0f}%)" + flag
    elif la < ta:
        pct = 100.0 * (ta - la) / ta if ta > 1e-9 else 0.0
        return f"LOOSE wins  ({ta:.1f} -> {la:.1f} cm, -{pct:.0f}%)" + flag
    return "TIE" + flag


def run_benchmark(max_frames: int = 0, tof_min_ba_views: int | None = 1,
                  only: list[str] | None = None) -> int:
    """Run loose vs tight on every gold session, full-res + 54x42, print the table."""
    sessions = sorted(d for d in GOLD_DIR.iterdir()
                      if (d / "basalt" / "vio_pose.jsonl").exists())
    if only:
        sessions = [d for d in sessions if d.name in only]

    print("=" * 100)
    print("LOOSE (windowed BA, vision-only) vs TIGHT (preintegration VIO, "
          "imu_info_weight=True)")
    print("ATE = rigid-SE3 RMSE (cm) | scale = Sim3 vs Basalt | drift = end-start "
          "(cm) | max-step (cm) | phantom (cm)")
    if max_frames:
        print(f"(max_frames={max_frames} -- quick mode)")
    print("=" * 100)

    results: dict[str, dict] = {}
    for d in sessions:
        name = d.name
        results[name] = {}
        for res in ("full", "tof54"):
            mbv = tof_min_ba_views if res == "tof54" else None
            loose = run_session(d, backend="ba", resolution=res,
                                max_frames=max_frames)
            tight = run_session(d, backend="vio", resolution=res,
                                imu_info_weight=True, min_ba_views=mbv,
                                max_frames=max_frames)
            results[name][res] = (loose, tight)
        print(f"  scored {name}")

    # ----- the table ----- #
    for res, res_label in (("full", "FULL-RES (chip depth)"),
                           ("tof54", "54x42 ToF (VL53L9CX target)")):
        print()
        print("#" * 100)
        print(f"#  {res_label}")
        print("#" * 100)
        hdr = (f"{'session':22s} {'be':6s} {'ATE cm':>8s} {'scale':>7s} "
               f"{'drift':>8s} {'maxstep':>8s} {'phantom':>8s} {'kf':>4s} "
               f"{'okfrac':>7s} {'ms/f':>7s}")
        print(hdr)
        print("-" * len(hdr))
        for d in sessions:
            name = d.name
            loose, tight = results[name][res]
            tag = ""
            if name in _INPLACE_YAW:
                tag = " *yaw"
            elif name in _FAST_PUSH:
                tag = " *push"
            for be_label, m in (("LOOSE", loose), ("TIGHT", tight)):
                if m is None:
                    print(f"{name:22s} {be_label:6s} {'--':>8s} {'--':>7s} "
                          f"{'--':>8s} {'--':>8s} {'--':>8s} {'--':>4s} "
                          f"{'--':>7s} {'--':>7s}")
                    continue
                print(f"{name:22s} {be_label:6s} "
                      f"{_fmt(m['ate_cm'], 8)} {_fmt(m['scale'], 7, 3)} "
                      f"{_fmt(m['drift_cm'], 8)} {_fmt(m['max_step_cm'], 8)} "
                      f"{_fmt(m['phantom_cm'], 8)} {m['n_kf']:>4d} "
                      f"{_fmt(m['ba_ok_frac'], 7, 2)} {_fmt(m['ms_per_frame'], 7, 1)}")
            print(f"{'':22s} {'verdict:':6s} {_verdict(loose, tight, name)}{tag}")
            print()

    # ----- headline ----- #
    print("#" * 100)
    print("#  HEADLINE VERDICT")
    print("#" * 100)
    _headline(results, sessions)
    return 0


def _headline(results: dict, sessions: list[Path]) -> None:
    """Aggregate: where tight beats loose (count + the named failure cases)."""
    for res, res_label in (("full", "FULL-RES"), ("tof54", "54x42 ToF")):
        wins = ties = losses = scored = 0
        push_yaw_notes = []
        for d in sessions:
            name = d.name
            loose, tight = results[name][res]
            if loose is None or tight is None:
                continue
            scored += 1
            la, ta = loose["ate_cm"], tight["ate_cm"]
            if ta < la - 1e-6:
                wins += 1
            elif la < ta - 1e-6:
                losses += 1
            else:
                ties += 1
            if name in _FAST_PUSH:
                push_yaw_notes.append(
                    f"      {name}: ATE {la:.1f}->{ta:.1f} cm | "
                    f"scale {loose['scale']:.2f}->{tight['scale']:.2f} | "
                    f"max-step {loose['max_step_cm']:.0f}->{tight['max_step_cm']:.0f} cm")
            elif name in _INPLACE_YAW:
                push_yaw_notes.append(
                    f"      {name}: ATE {la:.1f}->{ta:.1f} cm | "
                    f"phantom {loose['phantom_cm']:.1f}->{tight['phantom_cm']:.1f} cm "
                    f"(ref net {loose['ref_net_m']:.2f} m)")
        print(f"  [{res_label}]  tight beats loose on {wins}/{scored} scored "
              f"sessions  (loose wins {losses}, tie {ties})")
        if push_yaw_notes:
            print("    failure-mode sessions (loose -> tight):")
            print("\n".join(push_yaw_notes))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = all frames (default); >0 for a quick smoke run")
    ap.add_argument("--tof-min-ba-views", type=int, default=1,
                    help="WindowedVIOConfig.min_ba_views for the 54x42 profile "
                         "(PLAN gate 3 allows lowering; default 1)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these session names (default: all gold)")
    args = ap.parse_args()
    return run_benchmark(max_frames=args.max_frames,
                         tof_min_ba_views=args.tof_min_ba_views,
                         only=args.only)


if __name__ == "__main__":
    raise SystemExit(main())
