#!/usr/bin/env python3
"""DENSE DIRECT RGB-D VO benchmark @ 54x42 ToF (Stage-1 hypothesis + Stage-2a IMU seed).

THE QUESTION THIS ANSWERS
-------------------------
Does dense **direct** photometric RGB-D odometry (:func:`sky.front.direct.
estimate_pose_direct`) -- which uses ALL gradient pixels and reads metric scale
straight from the accurate per-pixel ToF depth -- fix the **scale collapse** the
SPARSE corner/KLT VIO suffers at the 54x42 ToF target?

STAGE-2a -- the IMU TRANSLATION SEED (``--seed imu``)
-----------------------------------------------------
Stage-1 proved direct + ToF depth FIXES scale collapse on slow motion
(lab_straight ATE 98->18 cm, scale 0.63->0.95) but DIVERGES on fast motion
(push_fast 53->1160 cm): the inter-frame baseline exceeds the Gauss-Newton
convergence basin and the warp was seeded gyro-only (rotation but NO
forward-translation prior), so GN starts too far from the true pose.

Stage-2a seeds ``estimate_pose_direct``'s ``init_T`` with the **full 6-DoF
IMU dead-reckoned relative pose** between the keyframe and the current frame
(rotation AND translation), so GN STARTS near the truth and escapes divergence.
It REUSES the stack's tight IMU machinery verbatim:
:func:`sky.vio.imu.predict_state` (gravity-aware forward propagation, the same
``predictState`` the live tight path uses), :func:`sky.vio.imu.gravity_aligned_R0`
(level the first frame from accel) and :func:`sky.vio.imu.complementary_correct`
(pull the dead-reckoned nav-state toward the metric VO fix each frame). See
:func:`_imu_seed_state` for the velocity approximation and its honesty caveat.

The measured sparse baseline @ 54x42 (from loose_vs_tight_bench / the memory
notes) is feature-starved: Sim3 scale 0.23-0.63 (motion-insensitive) and ATE
50-98 cm. The hypothesis (Steinbrucker'11 / Kerl ICRA'13 / Whelan ICRA'13): with
dense direct + given depth, the pose is pure 6-DoF and scale is OBSERVED, so the
scale should snap back toward 1.0 and ATE should drop.

WHAT IT DOES
------------
Runs FRAME-TO-KEYFRAME direct VO over the gold sessions, reduced to 54x42 ToF
in-process via the SAME producer-side reduction the live ToF pipeline uses
(SGM dense depth at source res -> block-median depth + INTER_AREA gray to 54x42,
K scaled anisotropically). It integrates the per-frame relative poses into a
trajectory and reports, PER SESSION, the SAME columns as loose_vs_tight_bench so
the numbers are directly comparable:

  * ATE RMSE (cm, rigid-SE(3) Umeyama-aligned)   -- the field standard
  * Sim3 scale (our path vs Basalt)              -- THE scale-collapse signal
  * end-vs-start drift (cm)
  * max single-frame step (cm)
  * % frames converged                           -- direct-VO health
  * ms/frame

then prints each row SIDE-BY-SIDE with the hardcoded sparse baseline and an
HONEST verdict (does direct beat sparse + fix scale, per session).

HARD SCOPE / SAFETY
-------------------
NEW, read-only harness. It does NOT modify the loose/tight path, the comms,
``oracle_replay.py``, or any frozen baseline. It only IMPORTS the reusable,
side-effect-free helpers (``_tof_reduce``, ``_scale_K_to_tof``,
``_block_median_valid`` via the loose bench; ``ate`` / ``umeyama`` /
``load_basalt_positions`` via the oracle; ``SessionReader`` + ``SGMStereoMatcher``)
and the NEW ``sky.front.direct`` module. The byte-parity oracle stays gap=0
because nothing it depends on changes.

KEYFRAMING
----------
Direct photometric alignment degrades once the baseline grows (the linearisation
+ in-bounds overlap shrink). We therefore use a small frame-to-KEYFRAME scheme:
align each new frame to the current keyframe (seeded by the previous frame's
relative pose), and promote a new keyframe once the translation or rotation from
the keyframe exceeds a threshold (or convergence/overlap drops). Global pose is
the chain of keyframe-anchored relative poses -- the honest way to integrate a
relative-pose VO into an absolute trajectory.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reusable, side-effect-free scoring helpers (NOT modified -- read-only import).
from verification.oracle_replay import (  # noqa: E402
    ate,
    load_basalt_positions,
    umeyama,
)

# Reuse the loose bench's ToF reduction + Basalt-ref guard verbatim (read-only).
from verification.loose_vs_tight_bench import (  # noqa: E402
    GOLD_DIR,
    _scale_K_to_tof,
    _tof_reduce,
    basalt_ref_is_broken,
)

from imu_camera.io.reader import SessionReader  # noqa: E402
from sky.depth.stereo import SGMConfig, SGMStereoMatcher  # noqa: E402
from sky.math import se3_log  # noqa: E402
from sky.front.direct import DirectConfig, estimate_pose_direct  # noqa: E402
# Tight IMU dead-reckoning machinery (reused VERBATIM for the Stage-2a seed --
# the SAME predict_state / complementary_correct the live tight VIO path uses).
from sky.vio.imu import (  # noqa: E402
    GyroPreintegrator,
    complementary_correct,
    gravity_aligned_R0,
    predict_state,
)


# --------------------------------------------------------------------------- #
# The measured SPARSE baseline @ 54x42 (the bar to beat). From the task brief /
# loose_vs_tight_bench memory notes -- hardcoded so the table is self-contained
# and the comparison is explicit. (ate_cm, scale)
# --------------------------------------------------------------------------- #
SPARSE_BASELINE = {
    "lab_straight_20s":        {"ate_cm": 98.0, "scale": 0.63},
    "push_straight_fast_15s":  {"ate_cm": 53.0, "scale": 0.38},
    "push_shake_20s":          {"ate_cm": 91.0, "scale": 0.38},
    "quick_motion_15s":        {"ate_cm": 75.0, "scale": 0.23},
}

# The four sessions the brief mandates (others can be added via --only).
DEFAULT_SESSIONS = [
    "lab_straight_20s",
    "push_straight_fast_15s",
    "push_shake_20s",
    "quick_motion_15s",
]


# --------------------------------------------------------------------------- #
# Stage-2a -- IMU dead-reckoning for the 6-DoF direct-warp seed.
# --------------------------------------------------------------------------- #
class _ImuDeadReckoner:
    """Gravity-aware IMU dead-reckoning of a camera-optical world nav-state.

    This is the Stage-2a translation-seed engine. It REUSES the stack's tight
    machinery verbatim -- :func:`sky.vio.imu.predict_state` (the same Basalt-style
    ``predictState`` forward propagation the live tight VIO runs) and
    :func:`sky.vio.imu.complementary_correct` (the soft pull toward a fresh vision
    fix) -- to maintain a body==camera nav-state ``(R, p, v)`` in the camera-optical
    world frame and hand the per-frame relative pose to the direct estimator.

    THE VELOCITY APPROXIMATION (the honest weak link)
    -------------------------------------------------
    ``predict_state`` needs a starting world VELOCITY to integrate translation;
    the bench has no independent velocity state. We obtain it the SAME way the
    live tight path does -- bootstrap from rest (all four gold sessions start
    near-static, so ``v = 0`` at frame 0 is honest) and then keep it scaled by
    feeding the *metric VO position fix* back into velocity each frame via
    :func:`complementary_correct` (its ``k_vel`` term bleeds the position error
    ``p_vis - p_dr`` into ``v`` over the anchor interval). So the velocity that
    drives the next IMU translation prediction is anchored to the depth-true VO
    motion, NOT pure accel double-integration (which would drift unboundedly).

    Caveat: this makes the IMU seed a PREDICTOR seeded by the PREVIOUS VO fix --
    it carries genuine inter-frame forward translation (the whole point), but its
    metric scale is inherited from the VO it is correcting against. On a frame
    where the VO is mid-divergence the velocity feedback is poisoned; the
    complementary gains are bounded (< 1) so a single bad frame cannot snap the
    state, but a SUSTAINED divergence is the residual failure Stage-2b's
    geometric (point-to-plane) factor + divergence guard would address.
    """

    # Complementary-correction gains (bounded in [0,1]) -- the EXACT values the
    # live tight path uses (``vio/modules/propagate_imu.py`` _K_POS/_K_VEL/_K_ROT):
    # firmly vision-anchored (pos/att error half-life ~2.4 frames) so the
    # dead-reckoned drift cannot run away, with a deliberately SMALL k_vel that
    # only bleeds the phantom-drift velocity down -- NOT the destabilising full
    # ``v = displacement/dt`` injection. Reused verbatim (no per-bench tuning).
    K_POS = 0.25
    K_VEL = 0.05
    K_ROT = 0.25

    def __init__(self, imu_raw: dict, T_imu_cam: np.ndarray):
        # Rotate the raw IMU stream into the camera-optical frame ONCE (body==cam),
        # exactly as loose_vs_tight_bench builds gyro_cam / accel_cam.
        R_imu_cam = np.asarray(T_imu_cam, np.float64)[:3, :3]
        self._ts = np.asarray(imu_raw["ts_ns"], np.int64)
        self._gyro = (R_imu_cam @ np.asarray(imu_raw["gyro"], np.float64).T).T
        self._accel = (R_imu_cam @ np.asarray(imu_raw["accel"], np.float64).T).T
        # Gravity bias estimate from the near-static startup window (0.3 s), same
        # window the tight bench uses for bg0; accel bias left at zero (the live
        # tight path also seeds ba0 = 0 and lets the filter absorb it).
        t0 = int(self._ts[0])
        win = self._ts <= t0 + int(0.3 * 1e9)
        self._bg = (self._gyro[win].mean(axis=0) if win.any() else np.zeros(3))
        self._ba = np.zeros(3)
        # Gravity ACCELERATION vector in the optical world frame ("down" = +y),
        # the exact convention sky.vio.imu.predict_state / WindowedVIOConfig use.
        self._g_world = np.array([0.0, 9.81, 0.0])

        # Nav-state (body==cam -> world): rotation, position, velocity. Filled by
        # init_at(); the world frame is gravity-levelled there.
        self._R = np.eye(3)
        self._p = np.zeros(3)
        self._v = np.zeros(3)
        self._last_ts = t0      # last propagated timestamp
        self._anchor_ts = t0    # last VO-correction timestamp (for k_vel dt)

    def init_at(self, ts_ns: int, T_world_cam0: np.ndarray) -> None:
        """Anchor the nav-state to the first frame: gravity-level the attitude.

        ``T_world_cam0`` is the photometric world pose of frame 0 (identity by
        construction); we pin the dead-reckoned origin to it and level roll/pitch
        from the static-startup accelerometer via :func:`gravity_aligned_R0`, so
        the dead-reckoned and photometric world frames share an origin + gravity.
        Velocity starts at zero (the sessions begin near-static).
        """
        t0 = int(ts_ns)
        win = self._ts <= t0 + int(0.3 * 1e9)
        accel_cam0 = (self._accel[win].mean(axis=0) if win.any()
                      else np.array([0.0, -9.81, 0.0]))
        self._R = gravity_aligned_R0(accel_cam0) @ np.asarray(
            T_world_cam0, np.float64)[:3, :3]
        self._p = np.asarray(T_world_cam0, np.float64)[:3, 3].copy()
        self._v = np.zeros(3)
        self._last_ts = t0
        self._anchor_ts = t0

    def propagate_to(self, ts_ns: int) -> np.ndarray:
        """Forward-integrate the nav-state to ``ts_ns`` and return its world pose.

        Integrates the IMU samples in ``(last_ts, ts_ns]`` with the gravity-aware
        :func:`predict_state`, advancing ``(R, p, v)``. Returns the 4x4
        ``T_world_cam`` after propagation. ``last_ts`` is moved to ``ts_ns``.
        """
        ts_ns = int(ts_ns)
        lo = np.searchsorted(self._ts, self._last_ts, side="left")
        hi = np.searchsorted(self._ts, ts_ns, side="right")
        idx = np.arange(max(lo - 1, 0), min(hi + 1, len(self._ts)))
        if idx.size >= 2:
            seg_ts = self._ts[idx].copy()
            # Clamp the segment ends to the requested [last_ts, ts_ns] window so
            # the integrated dt matches the frame interval exactly.
            seg_ts[0] = max(int(seg_ts[0]), self._last_ts)
            seg_ts[-1] = min(int(seg_ts[-1]), ts_ns)
            self._R, self._p, self._v = predict_state(
                self._R, self._p, self._v,
                seg_ts, self._gyro[idx], self._accel[idx],
                self._bg, self._ba, self._g_world)
        self._last_ts = ts_ns
        return self.pose()

    def correct_toward(self, T_world_cam_vis: np.ndarray, ts_ns: int) -> None:
        """Soft-pull the nav-state toward a fresh metric VO fix (complementary).

        ``T_world_cam_vis`` is the photometric world pose just solved for this
        frame. We close a bounded fraction of the position/attitude error and
        bleed the position error into velocity over the inter-frame interval, so
        the velocity that drives the NEXT seed stays scaled by depth-true vision.
        """
        T = np.asarray(T_world_cam_vis, np.float64)
        dt_anchor = max((int(ts_ns) - self._anchor_ts) * 1e-9, 0.0)
        self._R, self._p, self._v = complementary_correct(
            self._R, self._p, self._v, T[:3, :3], T[:3, 3],
            dt_anchor, self.K_POS, self.K_VEL, self.K_ROT)
        self._anchor_ts = int(ts_ns)

    def pose(self) -> np.ndarray:
        """Current 4x4 ``T_world_cam`` nav-pose."""
        out = np.eye(4)
        out[:3, :3] = self._R
        out[:3, 3] = self._p
        return out


# --------------------------------------------------------------------------- #
# Frame-to-keyframe direct VO over one session @ 54x42 ToF.
# --------------------------------------------------------------------------- #
def run_session_direct(
    session_dir: Path,
    *,
    cfg: DirectConfig,
    kf_trans_m: float = 0.10,
    kf_rot_deg: float = 6.0,
    kf_max_gap: int = 12,
    converged_overlap_min: float = 0.30,
    max_frames: int = 0,
    seed: str = "gyro",
) -> dict | None:
    """Run frame-to-keyframe direct VO over a session reduced to 54x42 ToF.

    Returns the metric dict (same columns as loose_vs_tight_bench) or None if the
    Basalt reference is missing/broken or there is too little overlap to score.

    Keyframing: align frame i to the current keyframe. Promote frame i to a NEW
    keyframe when the relative translation/rotation from the keyframe exceeds the
    thresholds, the alignment failed to converge, or ``kf_max_gap`` frames have
    elapsed -- whichever first. The global camera pose chains the keyframe anchors.

    ``seed`` selects the Gauss-Newton ``init_T`` strategy (the Stage-2a lever):

    * ``"none"`` -- the previous frame's relative pose only (pure photometric).
    * ``"gyro"`` -- Stage-1: previous relative pose, ROTATION refreshed with the
      gyro-preintegrated rotation between prev and cur (NO translation prior).
    * ``"imu"``  -- Stage-2a: the full 6-DoF IMU dead-reckoned relative pose
      (keyframe->cur, rotation AND translation) from :func:`sky.vio.imu.
      predict_state`, the same gravity-aware ``predictState`` the tight path uses.
      See :func:`_imu_seed_state` for how the velocity is obtained.
    """
    reader = SessionReader(session_dir)
    n = len(reader) if max_frames <= 0 else min(max_frames, len(reader))

    # Source dims + anisotropic K for the 54x42 grid (gold sessions are 640x400).
    f0 = reader.load_frame(0, load_right=False)
    sh, sw = f0.gray_left.shape[:2]
    K_tof = _scale_K_to_tof(reader.K, sw, sh)

    matcher = SGMStereoMatcher.from_calib(reader.calib, SGMConfig())

    # --- IMU machinery, shared by the gyro + imu seeds -------------------- #
    # ``pre``      : gyro rotation prior (Stage-1 / "gyro" seed).
    # ``dr``       : full dead-reckoning nav-state holder (Stage-2a / "imu" seed),
    #                or None when IMU is unavailable / the seed doesn't need it.
    pre = None
    dr = None
    if seed in ("gyro", "imu") and reader.calib.has_imu_extrinsics:
        imu_raw = reader.load_imu()
        if imu_raw["ts_ns"].size > 1:
            pre = GyroPreintegrator(imu_raw["ts_ns"], imu_raw["gyro"],
                                    reader.calib.T_imu_left)
            if seed == "imu":
                dr = _ImuDeadReckoner(imu_raw, reader.calib.T_imu_left)

    def reduce_frame(i: int):
        f = reader.load_frame(i, load_right=True)
        gray_src, depth_src = matcher.dense_depth_rectified_left(
            f.gray_left, f.gray_right)
        if gray_src.dtype != np.uint8:
            gray_src = np.clip(gray_src, 0.0, 255.0).astype(np.uint8)
        gray, depth = _tof_reduce(gray_src, depth_src)
        return f, gray, depth

    # --- per-frame loop --------------------------------------------------- #
    est: dict[int, np.ndarray] = {}
    seqs_in_order: list[int] = []

    # World pose convention: T_world_cam (camera-to-world). The keyframe holds its
    # world pose; each frame's world pose = T_world_kf @ inv(T_cur_kf).
    T_world_kf = np.eye(4)
    f_kf, gray_kf, depth_kf = reduce_frame(0)
    T_world_cur = T_world_kf.copy()
    est[f_kf.seq] = T_world_cur[:3, 3].copy()
    seqs_in_order.append(f_kf.seq)

    # The IMU dead-reckoner is anchored to the FIRST frame: gravity-level its
    # attitude and pin its world pose to the camera-trajectory origin (identity),
    # so the dead-reckoned and the photometric world frames coincide.
    T_world_kf_dr = np.eye(4)   # IMU dead-reckoned world pose of the keyframe
    if dr is not None:
        dr.init_at(f_kf.ts_ns, T_world_cur)

    kf_index = 0
    prev_ts = f_kf.ts_ns
    T_prev_kf = np.eye(4)        # previous frame -> keyframe (for seeding)
    n_aligned = 0
    n_converged = 0
    t_start = time.perf_counter()

    for i in range(1, n):
        f, gray, depth = reduce_frame(i)

        # ---- Gauss-Newton seed (the Stage-2a lever) ----------------------- #
        if dr is not None:
            # STAGE-2a: dead-reckon the IMU nav-state forward to THIS frame, then
            # the 6-DoF seed is the relative pose keyframe->cur expressed as the
            # direct estimator's T_cur_kf (point ref-cam -> cur-cam):
            #   T_cur_kf = inv(T_world_cur_dr) @ T_world_kf_dr
            T_world_cur_dr = dr.propagate_to(f.ts_ns)
            init_T = _inv_se3(T_world_cur_dr) @ T_world_kf_dr
        else:
            # STAGE-1 ("gyro") / "none": previous relative pose, optionally with
            # the gyro-integrated rotation refreshed (translation untouched -- the
            # photometric solve must recover it from depth on its own).
            init_T = T_prev_kf.copy()
            if pre is not None:
                R_pp = pre.delta_rotation(prev_ts, f.ts_ns)  # prev->cur rotation
                init_T[:3, :3] = R_pp @ T_prev_kf[:3, :3]

        # Align CURRENT frame to the KEYFRAME (T_cur_kf).
        T_cur_kf, info = estimate_pose_direct(
            gray_kf, depth_kf, gray, K_tof, init_T=init_T, cfg=cfg)
        n_aligned += 1
        if info["converged"]:
            n_converged += 1

        # Global pose of the current frame.
        T_world_cur = T_world_kf @ _inv_se3(T_cur_kf)
        est[f.seq] = T_world_cur[:3, 3].copy()
        seqs_in_order.append(f.seq)

        # Pull the IMU dead-reckoned nav-state toward this metric VO fix so the
        # velocity it feeds the NEXT seed stays scaled by depth-true vision (the
        # same complementary correction the live tight path applies). This is what
        # keeps the accel-only velocity from drifting over the session.
        if dr is not None:
            dr.correct_toward(T_world_cur, f.ts_ns)

        # --- keyframe promotion test --------------------------------------- #
        xi = se3_log(T_cur_kf)
        trans = float(np.linalg.norm(xi[:3]))
        rot_deg = float(np.degrees(np.linalg.norm(xi[3:])))
        overlap = float(info["valid_frac"])
        gap = i - kf_index
        promote = (
            trans >= kf_trans_m
            or rot_deg >= kf_rot_deg
            or gap >= kf_max_gap
            or (not info["converged"])
            or overlap < converged_overlap_min
        )
        if promote:
            # The new keyframe is THIS frame; anchor its world pose (photometric
            # AND -- when dead-reckoning -- the IMU nav-state's world pose, so the
            # next seed measures keyframe->cur from a fresh anchor).
            T_world_kf = T_world_cur.copy()
            if dr is not None:
                T_world_kf_dr = dr.pose()
            gray_kf, depth_kf = gray, depth
            kf_index = i
            T_prev_kf = np.eye(4)   # we are AT the keyframe now
        else:
            T_prev_kf = T_cur_kf    # seed next frame from this relative pose
        prev_ts = f.ts_ns

    elapsed = time.perf_counter() - t_start

    # --- score against the Basalt reference -------------------------------- #
    basalt = load_basalt_positions(reader.dir)
    if not basalt or basalt_ref_is_broken(basalt):
        return None
    common = sorted(set(est) & set(basalt))
    if len(common) < 10:
        return None

    src = np.array([est[s] for s in common])
    dst = np.array([basalt[s] for s in common])

    rigid = ate(src, dst, with_scale=False)
    sim = ate(src, dst, with_scale=True)
    traj_len = float(np.linalg.norm(np.diff(dst, axis=0), axis=1).sum())

    est_ordered = np.array([est[s] for s in seqs_in_order])
    our_steps = (np.linalg.norm(np.diff(est_ordered, axis=0), axis=1)
                 if len(est_ordered) > 1 else np.zeros(1))
    max_step_m = float(our_steps.max())

    R_a, t_a, s_a = umeyama(src, dst, with_scale=True)
    aligned = (s_a * (R_a @ src.T)).T + t_a
    ref_net = float(np.linalg.norm(dst[-1] - dst[0]))
    drift_m = float(np.linalg.norm(est_ordered[-1] - est_ordered[0]))

    return {
        "ate_cm": rigid["rmse"] * 100.0,
        "scale": sim["scale"],
        "drift_cm": drift_m * 100.0,
        "max_step_cm": max_step_m * 100.0,
        "ref_net_m": ref_net,
        "path_m": traj_len,
        "n_common": len(common),
        "n_frames": len(seqs_in_order),
        "n_aligned": n_aligned,
        "n_converged": n_converged,
        "conv_frac": (n_converged / n_aligned) if n_aligned else 0.0,
        "ms_per_frame": (elapsed / max(n, 1)) * 1000.0,
        "est_aligned": aligned,
        "gt": dst,
    }


def _inv_se3(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 SE(3) (R^T, -R^T t) -- local copy to avoid an extra import."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


# --------------------------------------------------------------------------- #
# Table + honest verdict
# --------------------------------------------------------------------------- #
def _fmt(v, width, prec=2, dash="--"):
    if v is None:
        return f"{dash:>{width}}"
    return f"{v:>{width}.{prec}f}"


def _verdict(name: str, direct: dict | None) -> str:
    base = SPARSE_BASELINE.get(name)
    if direct is None:
        return "DIRECT did not score"
    if base is None:
        return "(no sparse baseline on record)"
    da, ds = direct["ate_cm"], direct["scale"]
    ba, bs = base["ate_cm"], base["scale"]
    ate_better = da < ba
    # "scale closer to 1.0" -- the scale-collapse fix is the key signal.
    scale_better = abs(1.0 - ds) < abs(1.0 - bs)
    parts = []
    if ate_better:
        pct = 100.0 * (ba - da) / ba if ba > 1e-9 else 0.0
        parts.append(f"ATE {ba:.0f}->{da:.0f} cm (-{pct:.0f}%) WIN")
    else:
        pct = 100.0 * (da - ba) / ba if ba > 1e-9 else 0.0
        parts.append(f"ATE {ba:.0f}->{da:.0f} cm (+{pct:.0f}%) lose")
    parts.append(
        f"scale {bs:.2f}->{ds:.2f} "
        + ("CLOSER-to-1" if scale_better else "not closer"))
    return "  |  ".join(parts)


def run_benchmark(*, cfg: DirectConfig, only: list[str] | None,
                  max_frames: int, kf_trans_m: float, kf_rot_deg: float,
                  seed: str) -> int:
    names = only if only else DEFAULT_SESSIONS
    sessions = [GOLD_DIR / nm for nm in names]

    stage = "STAGE-2a (IMU 6-DoF seed)" if seed == "imu" else "STAGE-1"
    print("=" * 104)
    print(f"{stage} DENSE DIRECT RGB-D VO @ 54x42 ToF  vs  the measured SPARSE baseline")
    print("ATE = rigid-SE3 RMSE (cm) | scale = Sim3 vs Basalt | drift = end-start "
          "(cm) | conv% = frames converged")
    print(f"DirectConfig: levels={cfg.levels} max_iters={cfg.max_iters} "
          f"robust={cfg.robust} min_grad={cfg.min_grad} | "
          f"KF: trans>={kf_trans_m}m rot>={kf_rot_deg}deg seed={seed}")
    if max_frames:
        print(f"(max_frames={max_frames} -- quick mode)")
    print("=" * 104)

    results: dict[str, dict | None] = {}
    for d in sessions:
        if not (d / "basalt" / "vio_pose.jsonl").exists():
            print(f"  SKIP {d.name}: no Basalt reference")
            results[d.name] = None
            continue
        results[d.name] = run_session_direct(
            d, cfg=cfg, kf_trans_m=kf_trans_m, kf_rot_deg=kf_rot_deg,
            max_frames=max_frames, seed=seed)
        print(f"  scored {d.name}")

    # ----- the table (direct row + sparse baseline row, per session) ----- #
    print()
    print("#" * 104)
    print("#  54x42 ToF  --  DIRECT (this prototype)  vs  SPARSE (measured baseline)")
    print("#" * 104)
    hdr = (f"{'session':24s} {'method':8s} {'ATE cm':>8s} {'scale':>7s} "
           f"{'drift':>8s} {'maxstep':>9s} {'conv%':>7s} {'frames':>7s} "
           f"{'ms/f':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for nm in names:
        m = results.get(nm)
        base = SPARSE_BASELINE.get(nm)
        # DIRECT row.
        if m is None:
            print(f"{nm:24s} {'DIRECT':8s} {'--':>8s} {'--':>7s} {'--':>8s} "
                  f"{'--':>9s} {'--':>7s} {'--':>7s} {'--':>7s}")
        else:
            print(f"{nm:24s} {'DIRECT':8s} "
                  f"{_fmt(m['ate_cm'], 8)} {_fmt(m['scale'], 7, 3)} "
                  f"{_fmt(m['drift_cm'], 8)} {_fmt(m['max_step_cm'], 9)} "
                  f"{_fmt(m['conv_frac'] * 100.0, 7, 1)} {m['n_frames']:>7d} "
                  f"{_fmt(m['ms_per_frame'], 7, 1)}")
        # SPARSE baseline row (only the two recorded columns are known).
        if base is not None:
            print(f"{'':24s} {'SPARSE':8s} "
                  f"{_fmt(base['ate_cm'], 8)} {_fmt(base['scale'], 7, 3)} "
                  f"{'--':>8s} {'--':>9s} {'--':>7s} {'--':>7s} {'--':>7s}")
        print(f"{'':24s} {'verdict:':8s} {_verdict(nm, m)}")
        print()

    # ----- honest headline ----- #
    print("#" * 104)
    print("#  HONEST HEADLINE -- does dense direct + ToF depth fix scale collapse?")
    print("#" * 104)
    scored = [(nm, results[nm]) for nm in names if results.get(nm) is not None]
    if not scored:
        print("  no sessions scored -- cannot judge the hypothesis")
        return 0
    n_ate_win = n_scale_better = 0
    for nm, m in scored:
        base = SPARSE_BASELINE.get(nm)
        if base is None:
            continue
        if m["ate_cm"] < base["ate_cm"]:
            n_ate_win += 1
        if abs(1.0 - m["scale"]) < abs(1.0 - base["scale"]):
            n_scale_better += 1
    n_base = sum(1 for nm, _ in scored if nm in SPARSE_BASELINE)
    print(f"  scored {len(scored)} sessions ({n_base} with a sparse baseline)")
    print(f"  ATE beats sparse on   {n_ate_win}/{n_base} sessions")
    print(f"  scale closer to 1.0 on {n_scale_better}/{n_base} sessions "
          "(THE scale-collapse signal)")
    mean_scale = float(np.mean([m["scale"] for _, m in scored]))
    print(f"  mean direct Sim3 scale = {mean_scale:.3f} "
          f"(sparse baseline mean = "
          f"{np.mean([b['scale'] for b in SPARSE_BASELINE.values()]):.3f})")
    print("  NOTE: a null/partial result is a valid outcome -- read the per-session")
    print("        verdicts above for where direct wins vs fails.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = all frames (default); >0 for a quick smoke run")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these gold session names "
                         "(default: the 4 brief-mandated sessions)")
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--robust", choices=("t", "huber"), default="t")
    ap.add_argument("--min-grad", type=float, default=4.0)
    ap.add_argument("--kf-trans-m", type=float, default=0.10)
    ap.add_argument("--kf-rot-deg", type=float, default=6.0)
    ap.add_argument("--seed", choices=("none", "gyro", "imu"), default="gyro",
                    help="Gauss-Newton init_T seed: 'none' (pure photometric), "
                         "'gyro' (Stage-1: gyro rotation prior, default), or "
                         "'imu' (Stage-2a: full 6-DoF IMU dead-reckoned seed)")
    ap.add_argument("--no-gyro-seed", action="store_true",
                    help="alias for --seed none (kept for backward compat)")
    args = ap.parse_args()

    seed = "none" if args.no_gyro_seed else args.seed
    cfg = DirectConfig(levels=args.levels, max_iters=args.max_iters,
                       robust=args.robust, min_grad=args.min_grad)
    return run_benchmark(
        cfg=cfg, only=args.only, max_frames=args.max_frames,
        kf_trans_m=args.kf_trans_m, kf_rot_deg=args.kf_rot_deg, seed=seed)


if __name__ == "__main__":
    raise SystemExit(main())
