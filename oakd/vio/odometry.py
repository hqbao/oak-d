"""RGB-D visual odometry: track-based 3D-2D PnP between consecutive frames.

For each frame pair we already have KLT tracks (persistent ids) from
:mod:`oakd.vio.frontend`. The motion estimate works like a minimal feature-based
VO:

1. Take tracks seen in *both* the previous and current frame.
2. Back-project the previous-frame observations to 3D using the previous depth
   map (metric, from the recorded stereo depth).
3. Solve ``cv2.solvePnPRansac`` for the rigid transform that reprojects those 3D
   points onto their current-frame pixels -> ``T_prev->cur``.
4. Compose into the world pose: ``T_w_cur = T_w_prev @ inv(T_prev->cur)``.

Everything is metric because depth is metric, so the trajectory has true scale
(no monocular scale ambiguity). Output poses are in the camera optical frame
(+x right, +y down, +z forward); the world frame is the first camera.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .frontend import FrontendConfig, KLTFrontend


def level_attitude(R: np.ndarray, accel_cam: np.ndarray,
                   g_ref: float | None = None,
                   alpha: float = 0.03, alpha_max: float = 0.5,
                   g_tol: float = 0.25) -> tuple[np.ndarray, bool, float]:
    """Gravity-level a camera->world rotation's roll/pitch from accelerometer.

    Complementary-filter step: returns a rotation that has been nudged so the
    gravity direction *implied* by ``R`` lines up with the measured
    accelerometer ``accel_cam`` (camera optical frame, x right, y down, z fwd).
    The correction axis is horizontal (perpendicular to gravity) so only
    roll/pitch move -- yaw is untouched (no magnetometer => absolute yaw is
    unobservable, same as Basalt).

    The gain is **adaptive**: a small base ``alpha`` for steady-state (smooth,
    trusts vision) ramped toward ``alpha_max`` as the tilt error grows, so a big
    discrepancy (e.g. after the camera was flipped and vision drifted) snaps back
    within a few frames instead of crawling. It corrects only the *error*, so a
    genuinely pitched camera keeps its true pitch (measured gravity already
    matches ``R`` => ~zero correction).

    Samples taken during strong linear acceleration (``|accel|`` more than
    ``g_tol`` away from ``g_ref``) are rejected so thrust/translation does not
    corrupt tilt. ``g_ref`` defaults to ``|accel_cam|`` when not given.

    Returns ``(R_corrected, used, tilt_err_deg)``. ``used`` is ``False`` when the
    sample was rejected (out of the gravity band); ``R_corrected`` is then ``R``
    unchanged. ``tilt_err_deg`` is the angle between implied and true gravity
    *before* correction (useful for diagnostics).
    """
    a = np.asarray(accel_cam, dtype=np.float64)
    na = float(np.linalg.norm(a))
    if na < 1e-6:
        return R, False, 0.0
    gr = g_ref or na
    down_cam = -a / na                      # gravity dir in camera frame
    g_est = R @ down_cam                    # world-down implied by attitude
    target = np.array([0.0, 1.0, 0.0])      # true world-down (optical +y)
    s_full = float(np.linalg.norm(np.cross(g_est, target)))
    tilt_deg = float(np.degrees(np.arctan2(
        s_full, float(np.dot(g_est, target)))))
    if abs(na - gr) > g_tol * gr:           # not ~1g: linear acceleration
        return R, False, tilt_deg
    v = np.cross(g_est, target)
    s = float(np.linalg.norm(v))
    if s < 1e-9:                            # already aligned (or anti-aligned)
        return R, True, tilt_deg
    ang = float(np.arctan2(s, float(np.dot(g_est, target))))
    axis = v / s
    gain = alpha + (alpha_max - alpha) * min(ang / np.deg2rad(30.0), 1.0)
    th = gain * ang
    Kx = np.array([[0.0, -axis[2], axis[1]],
                   [axis[2], 0.0, -axis[0]],
                   [-axis[1], axis[0], 0.0]])
    R_corr = np.eye(3) + np.sin(th) * Kx + (1.0 - np.cos(th)) * (Kx @ Kx)
    return R_corr @ R, True, tilt_deg


@dataclass
class OdometryConfig:
    min_depth_m: float = 0.2
    max_depth_m: float = 8.0  # far stereo depth is noisy -> cap it
    min_pnp_points: int = 8
    ransac_reproj_px: float = 2.0
    ransac_iters: int = 200
    ransac_conf: float = 0.999


class RGBDVisualOdometry:
    """Frame-to-frame RGB-D PnP odometry over a grayscale+depth stream."""

    def __init__(self, K: np.ndarray, cfg: OdometryConfig | None = None,
                 frontend: KLTFrontend | None = None):
        self.K = np.asarray(K, dtype=np.float64)
        self.cfg = cfg or OdometryConfig()
        self.frontend = frontend or KLTFrontend(FrontendConfig())
        self.pose = np.eye(4)  # T_world_cur
        self._prev_obs: dict[int, np.ndarray] = {}
        self._prev_depth: np.ndarray | None = None
        self.last_info: dict = {}
        self._g_ref: float | None = None      # |gravity| from startup accel

    def align_to_gravity(self, accel_cam: np.ndarray) -> None:
        """Seed the initial attitude from gravity (call before the first frame).

        ``accel_cam`` is the static-startup accelerometer reading rotated into
        the camera optical frame. This sets the world frame so its "down" axis
        is aligned with real gravity instead of the (arbitrary) starting camera
        tilt -- the trajectory is otherwise unchanged (ATE is Umeyama-aligned, so
        a global rotation of the world frame does not affect it).
        """
        from .imu import gravity_aligned_R0
        self.pose = np.eye(4)
        self.pose[:3, :3] = gravity_aligned_R0(accel_cam)
        self._g_ref = float(np.linalg.norm(accel_cam))

    def correct_tilt(self, accel_cam: np.ndarray,
                     alpha: float = 0.03, alpha_max: float = 0.5,
                     g_tol: float = 0.25) -> bool:
        """Continuously level the attitude roll/pitch from gravity (per frame).

        Thin wrapper around :func:`level_attitude` that mutates ``self.pose`` in
        place and remembers the reference gravity magnitude. See that function
        for the full description. Returns ``True`` if the sample was usable.
        """
        R_new, used, _ = level_attitude(
            self.pose[:3, :3], accel_cam, g_ref=self._g_ref,
            alpha=alpha, alpha_max=alpha_max, g_tol=g_tol)
        if used:
            self.pose[:3, :3] = R_new
        return used


    def _backproject_px(self, u: float, v: float, z: float) -> np.ndarray:
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])

    def process(self, gray: np.ndarray, depth_m: np.ndarray,
                R_prior: np.ndarray | None = None) -> np.ndarray:
        """Advance odometry by one frame; returns the current 4x4 world pose.

        ``R_prior`` (optional) is the predicted previous->current camera rotation
        (e.g. from gyro preintegration). When given it seeds the PnP solver, which
        helps a lot during fast rotation where KLT correspondences are sparse.
        """
        state = self.frontend.process(gray)
        cur_obs = {int(i): p for i, p in zip(state.ids, state.points)}

        info = {"n_tracks": len(cur_obs), "n_pnp": 0, "n_inliers": 0,
                "ok": False, "reason": ""}

        if self._prev_depth is not None and self._prev_obs:
            obj_pts, img_pts = [], []
            h, w = self._prev_depth.shape
            for tid, cur_px in cur_obs.items():
                prev_px = self._prev_obs.get(tid)
                if prev_px is None:
                    continue
                pu, pv = int(round(prev_px[0])), int(round(prev_px[1]))
                if not (0 <= pu < w and 0 <= pv < h):
                    continue
                z = float(self._prev_depth[pv, pu])
                if not (self.cfg.min_depth_m <= z <= self.cfg.max_depth_m):
                    continue
                obj_pts.append(self._backproject_px(prev_px[0], prev_px[1], z))
                img_pts.append(cur_px)

            info["n_pnp"] = len(obj_pts)
            if len(obj_pts) >= self.cfg.min_pnp_points:
                obj = np.asarray(obj_pts, dtype=np.float64)
                img = np.asarray(img_pts, dtype=np.float64)

                use_guess = False
                rvec0 = None
                tvec0 = None
                if R_prior is not None:
                    rvec0, _ = cv2.Rodrigues(np.asarray(R_prior, dtype=np.float64))
                    tvec0 = np.zeros((3, 1))
                    use_guess = True

                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj, img, self.K, None,
                    rvec=rvec0, tvec=tvec0, useExtrinsicGuess=use_guess,
                    iterationsCount=self.cfg.ransac_iters,
                    reprojectionError=self.cfg.ransac_reproj_px,
                    confidence=self.cfg.ransac_conf,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok and inliers is not None and len(inliers) >= self.cfg.min_pnp_points:
                    R, _ = cv2.Rodrigues(rvec)
                    T_pc = np.eye(4)
                    T_pc[:3, :3] = R
                    T_pc[:3, 3] = tvec.reshape(3)
                    self.pose = self.pose @ np.linalg.inv(T_pc)
                    info["n_inliers"] = int(len(inliers))
                    info["ok"] = True
                else:
                    info["reason"] = "pnp_failed"
            else:
                info["reason"] = "too_few_points"
        else:
            info["reason"] = "bootstrap"

        self._prev_obs = cur_obs
        self._prev_depth = depth_m
        self.last_info = info
        return self.pose
