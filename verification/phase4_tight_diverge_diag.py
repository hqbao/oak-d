#!/usr/bin/env python3
"""Phase-4 DIAGNOSTIC: pinpoint WHY the tight RGB-D VIO scale-collapses / explodes.

THIS IS READ-ONLY / ADDITIVE. It does NOT change ``vio_window.py``'s solve
behaviour, does NOT touch the loose path, comms, ``oracle_replay.py``, or any
frozen baseline. It only IMPORTS the same math classes and the same scoring
helpers (``load_basalt_positions``, ``umeyama``) the benchmark uses, drives the
EXACT tight path ``verification/loose_vs_tight_bench.py`` drives
(``WindowedVIORGBDOdometry`` with ``imu_info_weight=True``, full-res chip depth
AND the 54x42 ToF reduction), and READS the window state out of the live map
after each keyframe solve. The byte-parity oracle stays gap=0 because nothing it
depends on is modified.

WHAT IT CAPTURES, PER KEYFRAME (after each ``run_ba`` that ran)
--------------------------------------------------------------
  * running scale     -- Sim3 scale of OUR path vs the Basalt reference, computed
                         over the common keyframes seen SO FAR (when does scale
                         start collapsing from ~1.0?).
  * |v|               -- world-velocity norm of the latest in-window keyframe.
  * |ba|, |bg|        -- accel / gyro bias norms of the latest keyframe.
  * residual breakdown-- the post-solve cost of each factor family on the SAME
                         window state the optimiser converged to, re-evaluated
                         with the optimiser's own residual primitives
                         (``_imu_residual`` / reprojection / ``_bias_rw_residual``)
                         -> which term dominates (IMU vs reproj vs bias-walk).
  * window health     -- #keyframes in window, #landmarks used by the solve,
                         #observations, mean reproj px, whether ``run_ba``
                         returned None (feature starvation), and which keyframe
                         got dropped by marginalisation this step.

Output: a per-KF table to stdout AND a matplotlib PNG to
``/tmp/skyviz/phase4_<session>_<res>.png`` (scale / |v| / |ba|,|bg| / residual
breakdown over keyframe index), for each requested (session, resolution).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from verification.oracle_replay import load_basalt_positions, umeyama  # noqa: E402

from imu_camera.io.reader import SessionReader  # noqa: E402
from sky.depth.stereo import SGMConfig, SGMStereoMatcher  # noqa: E402
from imu_camera.modules.tof_downsample import _block_median_valid  # noqa: E402
from imu_camera.modules.pipeline import TOF_W, TOF_H  # noqa: E402

from vio.mathlib.imu.imu import GyroPreintegrator  # noqa: E402
from vio.mathlib.odometry.odometry import OdometryConfig  # noqa: E402
from vio.mathlib.backend.vio_window import (  # noqa: E402
    WindowedVIOConfig,
    WindowedVIORGBDOdometry,
    T_cw_to_body_world,
    _imu_residual,
    _bias_rw_residual,
    _project,
)

GOLD_DIR = Path("sessions/gold")
OUT_DIR = Path("/tmp/skyviz")


# --------------------------------------------------------------------------- #
# 54x42 ToF reduction -- VERBATIM from loose_vs_tight_bench (so we drive the
# identical pixels the benchmark scored as ATE 1554 cm / scale 0.03).
# --------------------------------------------------------------------------- #
def _scale_K_to_tof(K: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    sx = TOF_W / float(src_w)
    sy = TOF_H / float(src_h)
    Kt = np.asarray(K, dtype=np.float64).copy()
    Kt[0, 0] *= sx
    Kt[0, 2] *= sx
    Kt[1, 1] *= sy
    Kt[1, 2] *= sy
    return Kt


def _tof_reduce(gray_src: np.ndarray, depth_src: np.ndarray):
    gray_tof = cv2.resize(gray_src, (TOF_W, TOF_H), interpolation=cv2.INTER_AREA)
    if gray_tof.dtype != np.uint8:
        gray_tof = gray_tof.astype(np.uint8)
    depth_tof = _block_median_valid(depth_src.astype(np.float32), TOF_H, TOF_W)
    return gray_tof, depth_tof


# --------------------------------------------------------------------------- #
# Post-solve residual breakdown -- re-evaluate the optimiser's OWN factor
# residuals on the converged window state. This reads the same fields the solver
# wrote; it does NOT re-run or alter the solve.
# --------------------------------------------------------------------------- #
def _window_residual_breakdown(vmap) -> dict:
    """Per-family chi-squared (0.5 * r^T r) on the current converged window.

    Mirrors ``optimize_vio.total_cost`` exactly: same whitened residual
    primitives, same g_world, same cfg. Returns the IMU / reprojection / depth /
    bias-walk cost split + the raw (whitened) norms so we can see which term
    blows up.
    """
    kfs = vmap.keyframes
    cfg = vmap.cfg.vio
    K = vmap.K
    g_world = vmap.g_world
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # rebuild the (R,p,v,bg,ba) state arrays from the keyframes
    R = [T_cw_to_body_world(kf["T_cw"])[0] for kf in kfs]
    p = [T_cw_to_body_world(kf["T_cw"])[1] for kf in kfs]
    v = [kf["v"] for kf in kfs]
    bg = [kf["bg"] for kf in kfs]
    ba = [kf["ba"] for kf in kfs]

    # ---- IMU + bias-walk residual cost over consecutive in-window edges ----
    imu_cost = 0.0
    bias_cost = 0.0
    imu_norm_max = 0.0
    for ci in range(1, len(kfs)):
        edge = kfs[ci]["edge"]
        if edge is None or edge.pre is None:
            continue
        pre = edge.pre
        ri = _imu_residual(R[ci - 1], p[ci - 1], v[ci - 1], bg[ci - 1], ba[ci - 1],
                           R[ci], p[ci], v[ci], pre, g_world, cfg)
        rb = _bias_rw_residual(bg[ci - 1], ba[ci - 1], bg[ci], ba[ci], cfg)
        imu_cost += 0.5 * float(ri @ ri)
        bias_cost += 0.5 * float(rb @ rb)
        imu_norm_max = max(imu_norm_max, float(np.linalg.norm(ri)))

    # ---- reprojection + depth residual cost over the landmark observations --
    reproj_cost = 0.0
    depth_cost = 0.0
    n_obs = 0
    px_errs = []
    for ci, kf in enumerate(kfs):
        for tid, uvz in kf["obs"].items():
            lm = vmap.landmarks.get(tid)
            if lm is None:
                continue
            u_obs, v_obs, z_obs = float(uvz[0]), float(uvz[1]), float(uvz[2])
            _, Z, u, vv = _project(R[ci], p[ci], lm, fx, fy, cx, cy,
                                   cfg.min_view_z)
            ru = (u - u_obs) / cfg.sigma_px
            rv = (vv - v_obs) / cfg.sigma_px
            e_px = float(np.hypot(ru, rv) * cfg.sigma_px)
            px_errs.append(e_px)
            # huber on pixel (read-only mirror of total_cost)
            w = 1.0 if e_px <= cfg.huber_px else cfg.huber_px / max(e_px, 1e-300)
            reproj_cost += 0.5 * w * (ru * ru + rv * rv)
            if z_obs > 0:
                sz = cfg.depth_sigma_coeff * z_obs ** 2
                rz = (Z - z_obs) / sz
                az = abs(rz)
                thr = cfg.depth_huber / sz
                depth_cost += (0.5 * rz * rz if az <= thr
                               else thr * (az - 0.5 * thr))
            n_obs += 1

    return {
        "imu_cost": imu_cost,
        "bias_cost": bias_cost,
        "reproj_cost": reproj_cost,
        "depth_cost": depth_cost,
        "imu_norm_max": imu_norm_max,
        "mean_px": float(np.mean(px_errs)) if px_errs else float("nan"),
        "n_obs_resid": n_obs,
    }


# --------------------------------------------------------------------------- #
# Drive the EXACT tight path; capture per-keyframe diagnostics.
# --------------------------------------------------------------------------- #
def diagnose_session(session_dir: Path, *, resolution: str,
                     min_ba_views: int | None, max_frames: int = 0) -> dict | None:
    reader = SessionReader(session_dir)
    n = len(reader) if max_frames <= 0 else min(max_frames, len(reader))
    tof = (resolution == "tof54")

    if tof:
        f0 = reader.load_frame(0, load_right=False)
        sh, sw = f0.gray_left.shape[:2]
        K_solve = _scale_K_to_tof(reader.K, sw, sh)
    else:
        K_solve = reader.K

    matcher = SGMStereoMatcher.from_calib(reader.calib, SGMConfig()) if tof else None

    odom_cfg = OdometryConfig(gyro_fuse=True,
                              use_own_pnp=os.environ.get("OAKD_OWN_PNP", "1") != "0")

    imu = None
    R_imu_cam = None
    if reader.calib.has_imu_extrinsics:
        imu_raw = reader.load_imu()
        if imu_raw["ts_ns"].size > 1:
            imu = imu_raw
            R_imu_cam = reader.calib.T_imu_left[:3, :3]
    if imu is None:
        return None

    gyro_cam = (R_imu_cam @ imu["gyro"].T).T
    accel_cam = (R_imu_cam @ imu["accel"].T).T
    t0 = imu["ts_ns"][0]
    win = imu["ts_ns"] <= t0 + int(0.3 * 1e9)
    bg0 = gyro_cam[win].mean(axis=0) if win.any() else np.zeros(3)

    wcfg = WindowedVIOConfig()
    wcfg.vio = replace(wcfg.vio, imu_info_weight=True)  # the live --tight weight
    if min_ba_views is not None:
        wcfg.min_ba_views = int(min_ba_views)

    vo = WindowedVIORGBDOdometry(
        K_solve, imu["ts_ns"], gyro_cam, accel_cam,
        bg0=bg0, ba0=np.zeros(3), cfg=wcfg, odom_cfg=odom_cfg)

    pre = GyroPreintegrator(imu["ts_ns"], imu["gyro"], reader.calib.T_imu_left)
    accel_imu = imu["accel"][win].mean(axis=0)
    vo.align_to_gravity(R_imu_cam @ accel_imu)

    basalt = load_basalt_positions(reader.dir)

    est: dict[int, np.ndarray] = {}
    prev_ts = None
    rows = []                  # per-keyframe diagnostic rows
    kf_idx = -1
    n_kf = 0
    n_ba_none = 0
    prev_window_first_ts = None  # detect marginalisation drops

    for i in range(n):
        f = reader.load_frame(i, load_right=tof)
        if tof:
            gray_src, depth_src = matcher.dense_depth_rectified_left(
                f.gray_left, f.gray_right)
            if gray_src.dtype != np.uint8:
                gray_src = np.clip(gray_src, 0.0, 255.0).astype(np.uint8)
            gray, depth = _tof_reduce(gray_src, depth_src)
        else:
            gray, depth = f.gray_left, f.depth_m

        R_prior = pre.delta_rotation(prev_ts, f.ts_ns) if prev_ts is not None else None
        pose = vo.process(gray, depth, f.ts_ns, R_prior=R_prior)
        prev_ts = f.ts_ns
        est[f.seq] = pose[:3, 3].copy()

        info = vo.last_info
        if not info.get("is_kf"):
            continue
        kf_idx += 1
        n_kf += 1

        ran_ba = any(k in info for k in ("vio_imu", "vio_reproj_px"))
        if not ran_ba:
            n_ba_none += 1

        vmap = vo.map
        kfs = vmap.keyframes
        # latest in-window keyframe nav-state (the one whose pose is reported)
        last = kfs[-1]
        v_norm = float(np.linalg.norm(last["v"]))
        bg_norm = float(np.linalg.norm(last["bg"]))
        ba_norm = float(np.linalg.norm(last["ba"]))

        # marginalisation: did the window's first KF timestamp change (drop)?
        cur_first_ts = kfs[0]["ts_ns"] if kfs else None
        dropped = (prev_window_first_ts is not None
                   and cur_first_ts != prev_window_first_ts
                   and len(kfs) >= vmap.cfg.window)
        prev_window_first_ts = cur_first_ts

        # residual breakdown on the converged window (read-only)
        bd = _window_residual_breakdown(vmap) if ran_ba else {
            "imu_cost": float("nan"), "bias_cost": float("nan"),
            "reproj_cost": float("nan"), "depth_cost": float("nan"),
            "imu_norm_max": float("nan"), "mean_px": float("nan"),
            "n_obs_resid": 0}

        # running Sim3 scale of OUR path vs Basalt over the common KFs SO FAR
        common = sorted(set(est) & set(basalt))
        run_scale = float("nan")
        run_net_ratio = float("nan")
        if len(common) >= 4:
            src = np.array([est[s] for s in common])
            dst = np.array([basalt[s] for s in common])
            _, _, s = umeyama(src, dst, with_scale=True)
            run_scale = float(s)
            ref_net = float(np.linalg.norm(dst[-1] - dst[0]))
            our_net = float(np.linalg.norm(src[-1] - src[0]))
            run_net_ratio = our_net / ref_net if ref_net > 1e-6 else float("nan")

        rows.append({
            "kf": kf_idx,
            "frame": i,
            "seq": int(f.seq),
            "ran_ba": ran_ba,
            "win": len(kfs),
            "n_lms": info.get("vio_lms", 0),
            "n_obs": info.get("vio_obs", 0),
            "n_imu": info.get("vio_imu", 0),
            "iters": info.get("vio_iters", 0),
            "reproj_px": float(info.get("vio_reproj_px", float("nan"))),
            "v": v_norm,
            "bg": bg_norm,
            "ba": ba_norm,
            "scale": run_scale,
            "net_ratio": run_net_ratio,
            "dropped_kf": dropped,
            **bd,
        })

    return {
        "name": session_dir.name,
        "resolution": resolution,
        "rows": rows,
        "n_kf": n_kf,
        "n_ba_none": n_ba_none,
        "window": wcfg.window,
        "min_ba_views": wcfg.min_ba_views,
    }


# --------------------------------------------------------------------------- #
# Reporting: per-KF table + matplotlib PNG.
# --------------------------------------------------------------------------- #
def _print_table(d: dict) -> None:
    rows = d["rows"]
    print("=" * 124)
    print(f"SESSION {d['name']}  res={d['resolution']}  "
          f"min_ba_views={d['min_ba_views']}  window={d['window']}  "
          f"#kf={d['n_kf']}  #run_ba=None(starved)={d['n_ba_none']}")
    print("=" * 124)
    hdr = (f"{'kf':>3} {'frm':>4} {'ba?':>3} {'win':>3} {'lms':>4} {'obs':>5} "
           f"{'imu':>3} {'scale':>7} {'netR':>6} {'|v|':>7} {'|bg|':>7} "
           f"{'|ba|':>8} {'rpx':>6} {'C_imu':>9} {'C_rpj':>8} {'C_dep':>8} "
           f"{'C_bw':>8} {'rImx':>7} {'drop':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['kf']:>3} {r['frame']:>4} {('Y' if r['ran_ba'] else 'NONE'):>3} "
              f"{r['win']:>3} {r['n_lms']:>4} {r['n_obs']:>5} {r['n_imu']:>3} "
              f"{r['scale']:>7.3f} {r['net_ratio']:>6.2f} {r['v']:>7.2f} "
              f"{r['bg']:>7.3f} {r['ba']:>8.3f} {r['reproj_px']:>6.2f} "
              f"{r['imu_cost']:>9.1f} {r['reproj_cost']:>8.1f} "
              f"{r['depth_cost']:>8.1f} {r['bias_cost']:>8.1f} "
              f"{r['imu_norm_max']:>7.1f} {('*' if r['dropped_kf'] else ''):>4}")


def _plot(d: dict, path: Path) -> None:
    # matplotlib is an OPTIONAL convenience here (the per-KF table is the primary
    # evidence). If it is not installed, skip the PNG instead of failing -- this
    # keeps the diagnostic from forcing a new hard project dependency.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not installed -- skipping PNG "
              "(pip install matplotlib to enable)")
        return

    rows = d["rows"]
    if not rows:
        return
    kf = [r["kf"] for r in rows]
    scale = [r["scale"] for r in rows]
    vnorm = [r["v"] for r in rows]
    ba = [r["ba"] for r in rows]
    bg = [r["bg"] for r in rows]
    nlms = [r["n_lms"] for r in rows]
    c_imu = [r["imu_cost"] for r in rows]
    c_rpj = [r["reproj_cost"] for r in rows]
    c_dep = [r["depth_cost"] for r in rows]
    c_bw = [r["bias_cost"] for r in rows]

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"Phase-4 tight VIO divergence  --  {d['name']}  ({d['resolution']})",
                 fontsize=13)

    a = ax[0, 0]
    a.plot(kf, scale, "r.-", label="Sim3 scale (ours/basalt)")
    a.axhline(1.0, color="g", ls="--", lw=1, label="ideal scale=1")
    a.set_ylabel("scale"); a.set_xlabel("keyframe"); a.set_title("SCALE collapse")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(kf, vnorm, "b.-")
    a.set_ylabel("|v| (m/s)"); a.set_xlabel("keyframe")
    a.set_title("world velocity norm |v|"); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot(kf, ba, "m.-", label="|ba| accel bias")
    a.plot(kf, bg, "c.-", label="|bg| gyro bias")
    a2 = a.twinx()
    a2.plot(kf, nlms, "k.", alpha=0.4, ms=4, label="#landmarks")
    a2.set_ylabel("#landmarks", color="gray")
    a.set_ylabel("bias norm"); a.set_xlabel("keyframe")
    a.set_title("bias norms + landmark health"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(kf, c_imu, label="IMU cost")
    a.plot(kf, c_rpj, label="reproj cost")
    a.plot(kf, c_dep, label="depth cost")
    a.plot(kf, c_bw, label="bias-walk cost")
    a.set_yscale("symlog")
    a.set_ylabel("0.5 r^T r (symlog)"); a.set_xlabel("keyframe")
    a.set_title("per-family residual cost"); a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print(f"  [plot] {path}")


def _summarise(d: dict) -> None:
    """Print the divergence-onset summary line (when/which state)."""
    rows = [r for r in d["rows"] if r["ran_ba"]]
    if not rows:
        print("  (no run_ba solves -- fully feature-starved)")
        return
    # find the first KF where scale drops below 0.5 (collapse onset)
    onset = next((r for r in rows if not np.isnan(r["scale"]) and r["scale"] < 0.5),
                 None)
    last = rows[-1]
    print("  SUMMARY:")
    print(f"    final: scale={last['scale']:.3f}  |v|={last['v']:.2f}  "
          f"|ba|={last['ba']:.3f}  |bg|={last['bg']:.3f}  "
          f"C_imu={last['imu_cost']:.0f}  C_rpj={last['reproj_cost']:.0f}")
    lms = np.array([r["n_lms"] for r in rows])
    print(f"    landmarks used per solve: min={lms.min()} med={int(np.median(lms))} "
          f"max={lms.max()}  | starved(run_ba=None)={d['n_ba_none']}/{d['n_kf']}")
    if onset is not None:
        print(f"    scale<0.5 FIRST at kf={onset['kf']} (frame {onset['frame']}): "
              f"scale={onset['scale']:.3f} |v|={onset['v']:.2f} "
              f"|ba|={onset['ba']:.3f} #lms={onset['n_lms']} "
              f"C_imu={onset['imu_cost']:.0f} C_rpj={onset['reproj_cost']:.0f}")
    else:
        print("    scale never dropped below 0.5")
    # which residual family dominates at the end
    fam = {"IMU": last["imu_cost"], "reproj": last["reproj_cost"],
           "depth": last["depth_cost"], "bias-walk": last["bias_cost"]}
    dom = max(fam, key=lambda k: (fam[k] if not np.isnan(fam[k]) else -1))
    print(f"    dominant residual family at end: {dom} "
          f"({fam[dom]:.0f}); max single-edge whitened IMU norm "
          f"|r_imu|={last['imu_norm_max']:.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="(name, res) pairs as name:res ; res in {full,tof54}. "
                         "Default = the three failing cases from the benchmark.")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--tof-min-ba-views", type=int, default=1,
                    help="WindowedVIOConfig.min_ba_views for the 54x42 profile")
    args = ap.parse_args()

    if args.sessions:
        targets = []
        for s in args.sessions:
            name, _, res = s.partition(":")
            targets.append((name, res or "full"))
    else:
        targets = [
            ("push_shake_20s", "full"),
            ("push_shake_20s", "tof54"),
            ("push_straight_fast_15s", "tof54"),
        ]

    for name, res in targets:
        sd = GOLD_DIR / name
        if not sd.exists():
            print(f"!! missing session {name}")
            continue
        mbv = args.tof_min_ba_views if res == "tof54" else None
        d = diagnose_session(sd, resolution=res, min_ba_views=mbv,
                             max_frames=args.max_frames)
        if d is None:
            print(f"!! {name} ({res}) could not run (no IMU extrinsics?)")
            continue
        _print_table(d)
        _summarise(d)
        _plot(d, OUT_DIR / f"phase4_{name}_{res}.png")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
