#!/usr/bin/env python3
"""Phase-4 BISECTION DIAGNOSTIC: is the tight-VIO velocity SEED already wrong, or
does the OPTIMIZER inflate it?

READ-ONLY / ADDITIVE. Does NOT change ``vio_window.py`` solve behaviour, the loose
path, comms, ``oracle_replay.py``, or any frozen baseline. It drives the EXACT
tight path that ``loose_vs_tight_bench.py`` / ``phase4_tight_diverge_diag.py``
drive (``WindowedVIORGBDOdometry``, ``imu_info_weight=True``), and only OBSERVES
the window state by wrapping ``map.add_keyframe`` / ``map.run_ba`` with snapshot
shims (the originals run unmodified; the shims just copy ``keyframes[-1]["v"]``).

PER KEYFRAME (KF0..KF~8, the FIRST window) it captures three velocities:
  v_seed : keyframes[-1]["v"] right AFTER add_keyframe seeds it (the IMU
           dead-reckoning ``v_j = v_i + g.dt + R_i@dv``), BEFORE run_ba.
  v_opt  : keyframes[-1]["v"] right AFTER run_ba's optimize_vio (post-solve).
  v_gt   : ground-truth speed = finite-difference of the Basalt reference
           positions at the matching keyframe timestamps.  (Position-difference
           magnitude is rotation-invariant, so |v_gt| is the true camera speed
           even though Basalt lives in a different world frame.)

It also records, per keyframe, the SEED's two additive terms so the SEED-bug can
be drilled into:
  term_g  = g_world * pre.dt          (gravity contribution to the velocity step)
  term_Rdv = R_i @ dv                 (preintegrated accel contribution)
plus pre.dt and ba0, and prints g_world's actual value/sign once.

Output: per-(session,res) table + a one-line bisection verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from verification.oracle_replay import load_basalt_positions  # noqa: E402

from imu_camera.io.reader import SessionReader  # noqa: E402
from sky.depth.stereo import SGMConfig, SGMStereoMatcher  # noqa: E402
from imu_camera.modules.tof_downsample import _block_median_valid  # noqa: E402
from imu_camera.modules.pipeline import TOF_W, TOF_H  # noqa: E402

from vio.mathlib.imu.imu import GyroPreintegrator  # noqa: E402
from sky.front.odometry import OdometryConfig  # noqa: E402
from vio.mathlib.backend.vio_window import (  # noqa: E402
    WindowedVIOConfig,
    WindowedVIORGBDOdometry,
    T_cw_to_body_world,
)

GOLD_DIR = Path("sessions/gold")


# --------------------------------------------------------------------------- #
# 54x42 ToF reduction -- VERBATIM from the benchmark / existing diag.
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
# Basalt timestamps per seq (the loader returns only positions; we re-read the
# jsonl for ts so v_gt can be a true finite-difference over time).
# --------------------------------------------------------------------------- #
def _basalt_ts_ns(session_dir: Path) -> dict[int, int]:
    path = session_dir / "basalt" / "vio_pose.jsonl"
    out: dict[int, int] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[int(d["seq"])] = int(d["ts_ns"])
    return out


# --------------------------------------------------------------------------- #
# The seed/optimised snapshot shims. We wrap the bound methods of the live map.
# The ORIGINAL method runs untouched; the shim only copies state out afterwards.
# add_keyframe additionally re-derives the SEED's two additive terms from the
# same cached inputs the seed used (read-only: identical formula to line ~854).
# --------------------------------------------------------------------------- #
class _SeedRecorder:
    def __init__(self, vmap):
        self.vmap = vmap
        self.g_world = vmap.g_world
        self._orig_add = vmap.add_keyframe
        self._orig_ba = vmap.run_ba
        vmap.add_keyframe = self._add_keyframe  # type: ignore[assignment]
        vmap.run_ba = self._run_ba             # type: ignore[assignment]
        # filled per keyframe, consumed by the driver after each run_ba.
        self.last_seed = None       # v_seed norm vector
        self.last_term_g = None     # g_world * pre.dt
        self.last_term_Rdv = None   # R_i @ dv
        self.last_prev_v = None     # prev keyframe's (optimised) velocity inherited
        self.last_dt = None
        self.last_opt = None        # v_opt vector

    def _add_keyframe(self, T_cw, ids, pts, depth_m, ts_ns, imu_seg=None):
        # Snapshot the PREVIOUS keyframe state BEFORE the original mutates the
        # window, so we can recompute the seed's two terms with the same inputs
        # the original used (prev v / bias / pose).
        had_prev = bool(self.vmap.keyframes)
        prev = self.vmap.keyframes[-1] if had_prev else None
        prev_T = prev["T_cw"].copy() if had_prev else None
        prev_bg = prev["bg"].copy() if had_prev else None
        prev_ba = prev["ba"].copy() if had_prev else None
        self.last_prev_v = prev["v"].copy() if had_prev else np.zeros(3)

        self._orig_add(T_cw, ids, pts, depth_m, ts_ns, imu_seg=imu_seg)

        kf = self.vmap.keyframes[-1]
        self.last_seed = kf["v"].copy()
        # Per-term breakdown -- only meaningful when there is an inbound IMU edge.
        pre = kf.get("pre")
        if pre is not None and prev_T is not None:
            R_i, _ = T_cw_to_body_world(prev_T)
            _, dv, _ = pre.corrected(prev_bg, prev_ba)
            self.last_term_g = self.g_world * pre.dt
            self.last_term_Rdv = R_i @ dv
            self.last_dt = float(pre.dt)
        else:
            self.last_term_g = np.zeros(3)
            self.last_term_Rdv = np.zeros(3)
            self.last_dt = 0.0

    def _run_ba(self):
        out = self._orig_ba()
        kf = self.vmap.keyframes[-1] if self.vmap.keyframes else None
        self.last_opt = kf["v"].copy() if kf is not None else None
        return out


# --------------------------------------------------------------------------- #
# Drive the tight path; capture v_seed / v_opt / v_gt per keyframe.
# --------------------------------------------------------------------------- #
def bisect_session(session_dir: Path, *, resolution: str,
                   min_ba_views: int | None, max_kf: int = 9,
                   vel_cv_prior: bool = False, vel_zupt: bool = False,
                   sigma_vel_cv: float = 0.15,
                   sigma_vel_zupt: float = 0.5) -> dict | None:
    reader = SessionReader(session_dir)
    n = len(reader)
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

    if not reader.calib.has_imu_extrinsics:
        return None
    imu_raw = reader.load_imu()
    if imu_raw["ts_ns"].size <= 1:
        return None
    imu = imu_raw
    R_imu_cam = reader.calib.T_imu_left[:3, :3]

    gyro_cam = (R_imu_cam @ imu["gyro"].T).T
    accel_cam = (R_imu_cam @ imu["accel"].T).T
    t0 = imu["ts_ns"][0]
    win = imu["ts_ns"] <= t0 + int(0.3 * 1e9)
    bg0 = gyro_cam[win].mean(axis=0) if win.any() else np.zeros(3)
    ba0 = np.zeros(3)

    wcfg = WindowedVIOConfig()
    wcfg.vio = replace(wcfg.vio, imu_info_weight=True,
                       vel_cv_prior=vel_cv_prior, vel_zupt=vel_zupt,
                       sigma_vel_cv=sigma_vel_cv,
                       sigma_vel_zupt=sigma_vel_zupt)
    if min_ba_views is not None:
        wcfg.min_ba_views = int(min_ba_views)

    vo = WindowedVIORGBDOdometry(
        K_solve, imu["ts_ns"], gyro_cam, accel_cam,
        bg0=bg0, ba0=ba0, cfg=wcfg, odom_cfg=odom_cfg)

    # Install the read-only snapshot shims on the live map.
    rec = _SeedRecorder(vo.map)

    pre = GyroPreintegrator(imu["ts_ns"], imu["gyro"], reader.calib.T_imu_left)
    accel_imu = imu["accel"][win].mean(axis=0)
    vo.align_to_gravity(R_imu_cam @ accel_imu)

    basalt_pos = load_basalt_positions(reader.dir)
    basalt_ts = _basalt_ts_ns(reader.dir)

    prev_ts = None
    rows = []
    kf_idx = -1
    # Track each keyframe's (seq, ts_ns) so v_gt can finite-difference Basalt at
    # ADJACENT keyframe timestamps (true camera speed over the same interval).
    kf_meta: list[tuple[int, int]] = []   # (seq, ts_ns) per keyframe in order

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
        vo.process(gray, depth, f.ts_ns, R_prior=R_prior)
        prev_ts = f.ts_ns

        info = vo.last_info
        if not info.get("is_kf"):
            continue
        kf_idx += 1
        kf_meta.append((int(f.seq), int(f.ts_ns)))

        ran_ba = any(k in info for k in ("vio_imu", "vio_reproj_px"))

        v_seed = rec.last_seed
        v_opt = rec.last_opt if ran_ba else None
        term_g = rec.last_term_g
        term_Rdv = rec.last_term_Rdv
        prev_v = rec.last_prev_v
        dt = rec.last_dt
        # The pure IMU increment this interval added on top of the inherited
        # prev velocity (g*dt + R@dv). If this stays small but |v_seed| grows,
        # the inflation is INHERITED from prev (compounding optimiser output),
        # not produced fresh by the dead-reckoning step.
        incr = (term_g + term_Rdv) if (term_g is not None
                                       and term_Rdv is not None) else np.zeros(3)

        # ---- v_gt: finite-difference Basalt over the SAME keyframe interval --
        v_gt = float("nan")
        if kf_idx >= 1:
            s_prev, t_prev = kf_meta[kf_idx - 1]
            s_cur, t_cur = kf_meta[kf_idx]
            if (s_prev in basalt_pos and s_cur in basalt_pos
                    and s_prev in basalt_ts and s_cur in basalt_ts):
                dpos = basalt_pos[s_cur] - basalt_pos[s_prev]
                dt_gt = (basalt_ts[s_cur] - basalt_ts[s_prev]) * 1e-9
                if dt_gt > 1e-6:
                    v_gt = float(np.linalg.norm(dpos) / dt_gt)

        rows.append({
            "kf": kf_idx,
            "frame": i,
            "seq": int(f.seq),
            "ran_ba": ran_ba,
            "v_seed": float(np.linalg.norm(v_seed)) if v_seed is not None else float("nan"),
            "v_opt": float(np.linalg.norm(v_opt)) if v_opt is not None else float("nan"),
            "v_gt": v_gt,
            "term_g": float(np.linalg.norm(term_g)) if term_g is not None else float("nan"),
            "term_Rdv": float(np.linalg.norm(term_Rdv)) if term_Rdv is not None else float("nan"),
            "prev_v": float(np.linalg.norm(prev_v)) if prev_v is not None else float("nan"),
            "incr": float(np.linalg.norm(incr)),
            "term_g_vec": term_g.copy() if term_g is not None else None,
            "term_Rdv_vec": term_Rdv.copy() if term_Rdv is not None else None,
            "v_seed_vec": v_seed.copy() if v_seed is not None else None,
            "dt": dt,
        })
        if kf_idx >= max_kf:
            break

    return {
        "name": session_dir.name,
        "resolution": resolution,
        "rows": rows,
        "g_world": rec.g_world.copy(),
        "ba0": ba0.copy(),
        "min_ba_views": wcfg.min_ba_views,
        "window": wcfg.window,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_table(d: dict) -> None:
    rows = d["rows"]
    g = d["g_world"]
    print("=" * 104)
    print(f"SESSION {d['name']}  res={d['resolution']}  "
          f"min_ba_views={d['min_ba_views']}  window={d['window']}")
    print(f"g_world = [{g[0]:+.4f}, {g[1]:+.4f}, {g[2]:+.4f}] m/s^2   "
          f"|g|={np.linalg.norm(g):.4f}   ba0=[{d['ba0'][0]:+.3f},"
          f"{d['ba0'][1]:+.3f},{d['ba0'][2]:+.3f}]")
    print("=" * 104)
    hdr = (f"{'kf':>3} {'frm':>4} {'seq':>4} {'ba?':>4} {'dt':>6} "
           f"{'prev_v':>7} {'incr':>7} "
           f"{'|v_seed|':>9} {'|v_opt|':>9} {'|v_gt|':>8} "
           f"{'seed/gt':>8} {'opt/gt':>8} {'|g*dt|':>8} {'|R@dv|':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        seed_gt = (r["v_seed"] / r["v_gt"]) if r["v_gt"] > 1e-9 else float("nan")
        opt_gt = (r["v_opt"] / r["v_gt"]) if (r["v_gt"] > 1e-9
                                              and not np.isnan(r["v_opt"])) else float("nan")
        print(f"{r['kf']:>3} {r['frame']:>4} {r['seq']:>4} "
              f"{('Y' if r['ran_ba'] else 'NONE'):>4} {r['dt']:>6.3f} "
              f"{r['prev_v']:>7.3f} {r['incr']:>7.3f} "
              f"{r['v_seed']:>9.3f} {r['v_opt']:>9.3f} {r['v_gt']:>8.3f} "
              f"{seed_gt:>8.2f} {opt_gt:>8.2f} {r['term_g']:>8.3f} "
              f"{r['term_Rdv']:>8.3f}")

    # KF1 per-term vector dump (the first interval with an IMU edge).
    kf1 = next((r for r in rows if r["kf"] == 1), None)
    if kf1 is not None and kf1["term_g_vec"] is not None:
        tg, tr, vs = kf1["term_g_vec"], kf1["term_Rdv_vec"], kf1["v_seed_vec"]
        print("  KF1 SEED term vectors (v_seed = v0[=0] + g*dt + R@dv):")
        print(f"    g*dt   = [{tg[0]:+.4f}, {tg[1]:+.4f}, {tg[2]:+.4f}]  "
              f"|.|={np.linalg.norm(tg):.4f}")
        print(f"    R@dv   = [{tr[0]:+.4f}, {tr[1]:+.4f}, {tr[2]:+.4f}]  "
              f"|.|={np.linalg.norm(tr):.4f}")
        print(f"    v_seed = [{vs[0]:+.4f}, {vs[1]:+.4f}, {vs[2]:+.4f}]  "
              f"|.|={np.linalg.norm(vs):.4f}")


def _verdict(d: dict) -> None:
    rows = [r for r in d["rows"]
            if r["kf"] >= 1 and r["v_gt"] > 1e-6]
    if not rows:
        print("  VERDICT: insufficient ground-truth overlap to bisect.")
        return
    seed_ratios = [r["v_seed"] / r["v_gt"] for r in rows]
    opt_ratios = [r["v_opt"] / r["v_gt"] for r in rows
                  if not np.isnan(r["v_opt"])]
    med_seed = float(np.median(seed_ratios))
    med_opt = float(np.median(opt_ratios)) if opt_ratios else float("nan")
    print("  VERDICT:")
    print(f"    median |v_seed|/|v_gt| = {med_seed:5.2f}   "
          f"median |v_opt|/|v_gt| = {med_opt:5.2f}")
    if med_seed >= 3.0:
        # SEED already inflated -> drill the dominant term.
        kf1 = next((r for r in rows if r["kf"] == 1), rows[0])
        print(f"    -> SEED-BUG: the IMU dead-reckoning seed over-predicts "
              f"({med_seed:.1f}x gt) BEFORE the optimiser runs.")
        if kf1["term_g"] >= kf1["term_Rdv"]:
            print(f"       Dominant term = g_world*dt (|g*dt|={kf1['term_g']:.3f} "
                  f">= |R@dv|={kf1['term_Rdv']:.3f}) -> gravity term suspect "
                  f"(sign/axis double-count).")
        else:
            print(f"       Dominant term = R_i@dv (|R@dv|={kf1['term_Rdv']:.3f} "
                  f"> |g*dt|={kf1['term_g']:.3f}) -> accel/ba0/integration suspect.")
    elif not np.isnan(med_opt) and med_opt >= 3.0:
        print(f"    -> OPTIMIZER-BUG: seed ~= gt ({med_seed:.1f}x) but the solve "
              f"inflates it to {med_opt:.1f}x gt -> IMU factor over-weight.")
    else:
        print(f"    -> neither stage shows a 3x+ inflation in this window "
              f"(seed {med_seed:.1f}x, opt {med_opt:.1f}x).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="name:res pairs ; res in {full,tof54}.")
    ap.add_argument("--max-kf", type=int, default=8,
                    help="last keyframe index to capture (first window).")
    ap.add_argument("--tof-min-ba-views", type=int, default=1)
    ap.add_argument("--vel-cv-prior", action="store_true",
                    help="enable the Phase-4 constant-velocity smoothness prior")
    ap.add_argument("--vel-zupt", action="store_true",
                    help="enable the Phase-4 excitation-gated ZUPT")
    ap.add_argument("--sigma-vel-cv", type=float, default=0.15)
    ap.add_argument("--sigma-vel-zupt", type=float, default=0.5)
    args = ap.parse_args()

    if args.sessions:
        targets = []
        for s in args.sessions:
            name, _, res = s.partition(":")
            targets.append((name, res or "full"))
    else:
        targets = [
            ("push_straight_fast_15s", "full"),
            ("push_straight_fast_15s", "tof54"),
        ]

    for name, res in targets:
        sd = GOLD_DIR / name
        if not sd.exists():
            print(f"!! missing session {name}")
            continue
        mbv = args.tof_min_ba_views if res == "tof54" else None
        d = bisect_session(sd, resolution=res, min_ba_views=mbv,
                           max_kf=args.max_kf,
                           vel_cv_prior=args.vel_cv_prior,
                           vel_zupt=args.vel_zupt,
                           sigma_vel_cv=args.sigma_vel_cv,
                           sigma_vel_zupt=args.sigma_vel_zupt)
        if d is None:
            print(f"!! {name} ({res}) could not run (no IMU extrinsics?)")
            continue
        _print_table(d)
        _verdict(d)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
