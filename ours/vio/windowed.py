"""Windowed RGB-D visual odometry: frame-to-frame tracking + keyframe BA.

This is the second odometry backend, kept deliberately separate from the plain
:class:`oakd.vio.odometry.RGBDVisualOdometry` so we can compare the two
stage-by-stage on identical input.

Pipeline
--------
1. **Per-frame tracking** reuses the proven frame-to-frame PnP VO
   (:class:`RGBDVisualOdometry`) — this gives a dense, smooth pose for *every*
   frame and shares the exact same KLT frontend, so any quality difference is
   purely the BA refinement, nothing else.
2. **Keyframes** are inserted every ``kf_every`` frames. At each keyframe we
   - initialise a 3D landmark (in the world frame) for every live track that
     does not have one yet, back-projected from that keyframe's metric depth;
   - record the pixel observation of every landmark visible in the keyframe.
3. **Sliding-window BA** (:func:`oakd.vio.bundle.optimize`) then jointly
   refines the last ``window`` keyframe poses + their landmarks by minimising
   reprojection error, holding the oldest keyframe fixed as the gauge anchor.
4. The BA correction to the *latest* keyframe (which is the current frame) is
   injected back into the tracker, so subsequent frame-to-frame motion
   continues from the optimised pose.

Honesty notes
-------------
- Marginalisation is a **plain drop** of the oldest keyframe (no Schur
  marginalisation prior is carried forward). This is the simple, well-understood
  choice; it loses a little information vs a full marginalisation prior but
  introduces no fake constraints.
- Only landmarks with >= 2 keyframe observations enter BA (a single view cannot
  constrain a 3D point); single-view tracks still drive per-frame PnP.
- This backend has **no loop closure / global map** — that is SLAM, a later
  stage. Windowed BA reduces *local* drift and scale error only.

Measured benefit (ATE %path vs Basalt, gold sessions, vs frame-to-frame VO)
---------------------------------------------------------------------------
    lab_loop_30s     1.18 -> 0.55%   (halved — looping motion benefits most)
    quick_motion_15s 2.08 -> 1.97%   (better)
    lab_straight_20s 1.09 -> 1.11%   (tie)
    corridor_60s     0.66 -> 0.82%   (slightly worse; long low-parallax hall)
Scale lands at 0.98-1.07 (vs 0.99-1.12 for f2f) — the metric depth residual in
the BA is what anchors scale; plain reprojection BA is scale-free and was
strictly worse. Net: a real win on varied/looping motion, no per-session tuning.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .bundle import BAConfig, optimize
from .frontend import FrontendConfig, KLTFrontend
from .odometry import OdometryConfig, RGBDVisualOdometry


@dataclass
class WindowedConfig:
    kf_every: int = 4            # insert a keyframe every N frames
    window: int = 8             # keyframes kept in the BA window
    min_depth_m: float = 0.2
    max_depth_m: float = 8.0
    min_ba_views: int = 2       # landmark needs >= this many KF views for BA
    ba: BAConfig = field(default_factory=lambda: BAConfig(max_iters=8, huber_px=2.0))


class WindowedBAMap:
    """Sliding-window keyframe map + bundle adjustment, tracker-agnostic.

    This owns *only* the keyframe/landmark bookkeeping and the BA solve. It does
    not run any visual odometry itself: the caller supplies, at each keyframe,
    the current ``T_cw`` pose (from whatever front-end) plus a snapshot of the
    live tracks and the depth map. Keeping it decoupled lets the same backend
    run **synchronously** inside :class:`WindowedRGBDOdometry` (offline) and on a
    **background thread** in the live source, with no duplicated map logic.
    """

    def __init__(self, K: np.ndarray, cfg: WindowedConfig | None = None):
        self.K = np.asarray(K, dtype=np.float64)
        self.cfg = cfg or WindowedConfig()
        self.landmarks: dict[int, np.ndarray] = {}
        self.keyframes: list[dict] = []
        self.last_info: dict = {}

    def _backproject_world(self, T_cw: np.ndarray, u: float, v: float,
                           z: float) -> np.ndarray:
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        Xc = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
        R, t = T_cw[:3, :3], T_cw[:3, 3]
        return R.T @ (Xc - t)   # inv(T_cw) applied to Xc

    def add_keyframe(self, T_cw: np.ndarray, ids: np.ndarray,
                     pts: np.ndarray, depth_m: np.ndarray,
                     accel_cam: np.ndarray | None = None) -> None:
        """Register a keyframe from a track snapshot + its depth map.

        ``accel_cam`` (optional) is the accelerometer vector in the camera
        optical frame at this keyframe, supplied only when the camera was at
        rest. When present (and ``cfg.ba.use_gravity``) it adds the gravity
        prior that keeps the keyframe's roll/pitch leveled inside BA.
        """
        h, w = depth_m.shape
        obs: dict[int, np.ndarray] = {}
        for tid, px in zip(ids, pts):
            tid = int(tid)
            u, v = float(px[0]), float(px[1])
            pu, pv = int(round(u)), int(round(v))
            if not (0 <= pu < w and 0 <= pv < h):
                continue
            z = float(depth_m[pv, pu])
            z_ok = self.cfg.min_depth_m <= z <= self.cfg.max_depth_m
            if tid not in self.landmarks:
                if not z_ok:
                    continue
                self.landmarks[tid] = self._backproject_world(T_cw, u, v, z)
            obs[tid] = np.array([u, v, z if z_ok else 0.0])
        kf = {"T_cw": np.asarray(T_cw, float).copy(), "obs": obs}
        kf["accel"] = (None if accel_cam is None
                       else np.asarray(accel_cam, float).copy())
        self.keyframes.append(kf)
        self._marginalize()

    def _marginalize(self) -> None:
        """Drop oldest keyframes beyond the window; prune orphan landmarks."""
        while len(self.keyframes) > self.cfg.window:
            self.keyframes.pop(0)
        live = set()
        for kf in self.keyframes:
            live.update(kf["obs"].keys())
        for tid in list(self.landmarks.keys()):
            if tid not in live:
                del self.landmarks[tid]

    def run_ba(self) -> np.ndarray | None:
        """Optimise the window; return the refined latest ``T_cw`` (or None)."""
        kfs = self.keyframes
        if len(kfs) < 2:
            return None
        cnt = Counter()
        for kf in kfs:
            for tid in kf["obs"]:
                if tid in self.landmarks:
                    cnt[tid] += 1
        ba_tids = [t for t, c in cnt.items() if c >= self.cfg.min_ba_views]
        if len(ba_tids) < 6:
            return None
        lm_index = {t: j for j, t in enumerate(ba_tids)}
        landmarks_arr = np.array([self.landmarks[t] for t in ba_tids])

        poses = [kf["T_cw"] for kf in kfs]
        fixed = [i == 0 for i in range(len(kfs))]
        obs_cam, obs_lm, obs_uv, obs_depth = [], [], [], []
        for ci, kf in enumerate(kfs):
            for tid, uvz in kf["obs"].items():
                j = lm_index.get(tid)
                if j is None:
                    continue
                obs_cam.append(ci)
                obs_lm.append(j)
                obs_uv.append(uvz[:2])
                obs_depth.append(uvz[2])
        if len(obs_cam) < 12:
            return None

        grav_meas = None
        if self.cfg.ba.use_gravity:
            grav_meas = np.full((len(kfs), 3), np.nan)
            for i, kf in enumerate(kfs):
                a = kf.get("accel")
                if a is not None:
                    grav_meas[i] = a

        res = optimize(
            self.K, poses, fixed, landmarks_arr,
            np.array(obs_cam), np.array(obs_lm), np.array(obs_uv),
            obs_depth=np.array(obs_depth),
            grav_meas=grav_meas, grav_world=np.array([0.0, 1.0, 0.0]),
            cfg=self.cfg.ba,
        )
        for kf, P in zip(kfs, res.poses):
            kf["T_cw"] = P
        for t, j in lm_index.items():
            self.landmarks[t] = res.landmarks[j]

        self.last_info = {
            "ba_kfs": len(kfs), "ba_lms": len(ba_tids),
            "ba_obs": len(obs_cam), "ba_reproj_px": res.mean_reproj_px,
        }
        return kfs[-1]["T_cw"].copy()


class WindowedRGBDOdometry:
    """Frame-to-frame tracking with sliding-window keyframe bundle adjustment."""

    def __init__(self, K: np.ndarray, cfg: WindowedConfig | None = None,
                 frontend: KLTFrontend | None = None,
                 odom_cfg: OdometryConfig | None = None):
        self.K = np.asarray(K, dtype=np.float64)
        self.cfg = cfg or WindowedConfig()
        fe = frontend or KLTFrontend(FrontendConfig())
        self.vo = RGBDVisualOdometry(self.K, odom_cfg or OdometryConfig(), frontend=fe)
        self.frontend = self.vo.frontend
        self.map = WindowedBAMap(self.K, self.cfg)
        self._frames_since_kf = 0
        self._frame_idx = -1
        self.pose = np.eye(4)         # T_world_cur (camera->world)
        self.last_info: dict = {}

    def align_to_gravity(self, accel_cam: np.ndarray) -> None:
        """Seed the initial attitude from gravity (call before the first frame).

        Delegates to the inner frame-to-frame tracker and mirrors its pose so
        the keyframe map is built in the same gravity-leveled world frame.
        """
        self.vo.align_to_gravity(accel_cam)
        self.pose = self.vo.pose.copy()

    def correct_tilt(self, accel_cam: np.ndarray,
                     alpha: float = 0.01, alpha_max: float = 0.5,
                     g_tol: float = 0.12) -> bool:
        """Per-frame gravity leveling of attitude (delegates to the f2f core).

        Mirrors the corrected rotation onto our own pose so the keyframe map
        stays in the same continuously-leveled world frame.
        """
        used = self.vo.correct_tilt(accel_cam, alpha=alpha,
                                    alpha_max=alpha_max, g_tol=g_tol)
        if used:
            self.pose[:3, :3] = self.vo.pose[:3, :3]
        return used

    # convenience pass-throughs (some tools inspect these)
    @property
    def landmarks(self) -> dict[int, np.ndarray]:
        return self.map.landmarks

    @property
    def keyframes(self) -> list[dict]:
        return self.map.keyframes

    # --------------------------------------------------------------------- #
    def process(self, gray: np.ndarray, depth_m: np.ndarray,
                R_prior: np.ndarray | None = None) -> np.ndarray:
        """Advance one frame; return the current 4x4 world pose (camera->world)."""
        self._frame_idx += 1
        # 1) frame-to-frame tracking (also advances the shared frontend).
        self.pose = self.vo.process(gray, depth_m, R_prior=R_prior).copy()
        self.last_info = dict(self.vo.last_info)
        self._frames_since_kf += 1

        # 2) keyframe? (always make the very first frame a keyframe)
        is_kf = (not self.keyframes) or (self._frames_since_kf >= self.cfg.kf_every)
        if is_kf:
            self._frames_since_kf = 0
            state = self.frontend.tracks
            self.map.add_keyframe(np.linalg.inv(self.pose),
                                  state.ids, state.points, depth_m)
            # 3) sliding-window BA, then inject the correction.
            post = self.map.run_ba()
            if post is not None:
                self.pose = np.linalg.inv(post)
                self.vo.pose = self.pose.copy()  # keep tracker consistent
                self.last_info.update(self.map.last_info)
            self.last_info["is_kf"] = True
        else:
            self.last_info["is_kf"] = False
        return self.pose
