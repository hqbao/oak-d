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
                   alpha: float = 0.01, alpha_max: float = 0.5,
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
    dot = float(np.dot(g_est, target))
    if s < 1e-9:
        if dot > 0.0:                       # already aligned -> nothing to do
            return R, True, tilt_deg
        # Anti-aligned (implied gravity points straight UP, ~180 deg off). The
        # rotation axis is undefined here, so pick any horizontal axis (perp to
        # the world-down target) and rotate toward gravity. Without this the
        # filter sits stuck at the singularity until the camera is perturbed --
        # the "I have to shake it before it levels" symptom.
        axis = np.array([1.0, 0.0, 0.0])
        ang = np.pi
    else:
        ang = float(np.arctan2(s, dot))
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
    # --- gyro complementary fusion (loosely-coupled VIO) --------------------
    # When True and a gyro rotation prior is supplied to ``process``, the
    # inter-frame ROTATION is owned by the gyro and merely *corrected* toward the
    # vision (PnP) estimate, with the correction strength scaled by vision
    # confidence (inlier count). Pure-vision PnP under-rotates when KLT loses
    # correspondences during fast turns (especially yaw, which gravity cannot
    # recover), so trusting the gyro through those frames keeps the attitude --
    # and hence every keyframe heading -- correct. Translation is then
    # re-estimated by least squares with that rotation held fixed, which removes
    # the phantom translation a wrong rotation would otherwise inject during a
    # pure rotation. On a PnP failure the rotation is still propagated by the
    # gyro (translation = 0) so the body frame tracks the turn through the blind
    # spell instead of freezing. Default False keeps the pure-vision behaviour
    # (and all existing tests) unchanged.
    gyro_fuse: bool = False
    # Max weight given to the vision rotation correction (0..1). At full vision
    # confidence the fused rotation equals the PnP estimate (gyro fully
    # corrected away), so healthy frames are pure vision and unchanged; the gyro
    # only retains influence as confidence drops below ``gyro_trust_inliers``.
    gyro_corr_gain_max: float = 1.0
    # Inlier count at which vision is considered fully trustworthy. Below it the
    # correction gain scales down linearly, so the gyro dominates exactly when
    # vision is weak (the fast-rotation / feature-starved regime).
    gyro_trust_inliers: int = 40
    # Rotation disagreement gate (degrees per frame). The inlier count alone is a
    # poor proxy for a TRUSTWORTHY vision rotation: during a fast yaw the KLT
    # tracker loses the high-parallax features at the image edge but keeps plenty
    # of low-parallax ones near the centre, so PnP still reports many inliers yet
    # UNDER-ROTATES. When the vision rotation disagrees with the gyro by more than
    # this many degrees, we treat the vision rotation as suspect and let the gyro
    # take over (gain ramps to 0 over ``gyro_disagree_span_deg`` beyond the gate).
    # The gyro is reliable over a single ~50 ms inter-frame interval, so this is
    # exactly the fast-rotation regime where it should win. On gold the
    # disagreement never exceeds ~0.75 deg, so a 1.5 deg gate leaves all
    # well-tracked motion (and the regression suite) byte-for-byte unchanged.
    # Over a single ~50 ms inter-frame interval the gyro is effectively
    # ground-truth for rotation, so once vision disagrees we want the gyro to
    # win decisively: a 3 deg span means any frame whose vision rotation is off
    # by >= gate+span (4.5 deg) is full gyro. Live fast-yaw frames showed vision
    # under-rotating by 5-7 deg at 80-165 inliers -- exactly the frames this
    # rescues (the inlier count alone called them healthy).
    gyro_disagree_deg: float = 1.5
    gyro_disagree_span_deg: float = 3.0


def _scale_rotation(R: np.ndarray, s: float) -> np.ndarray:
    """Return the rotation that applies a fraction ``s`` of ``R`` (axis-angle)."""
    rvec, _ = cv2.Rodrigues(R)
    R_s, _ = cv2.Rodrigues(rvec * float(s))
    return R_s


def _translation_given_rotation(obj: np.ndarray, img: np.ndarray,
                                R: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Least-squares camera translation with the rotation held fixed.

    Each correspondence constrains ``R @ X_prev + t`` to lie along the viewing
    ray of its current pixel: ``ray × (R X_prev + t) = 0``. Stacking the two
    independent rows per point over all matches gives ``A t = b`` solved for the
    3-vector ``t`` -- the metric translation that best explains the pixels for
    the given rotation (depth gives true scale, so this is well posed).
    """
    Kinv = np.linalg.inv(K)
    P = (R @ obj.T).T                                   # rotated 3D points
    ones = np.ones((img.shape[0], 1))
    rays = (Kinv @ np.hstack([img, ones]).T).T          # viewing directions
    A = np.zeros((2 * len(P), 3))
    b = np.zeros(2 * len(P))
    for i, (n, p) in enumerate(zip(rays, P)):
        # skew(n): two independent rows suffice (use rows 0 and 1).
        Sx = np.array([[0.0, -n[2], n[1]],
                       [n[2], 0.0, -n[0]],
                       [-n[1], n[0], 0.0]])
        A[2 * i:2 * i + 2] = Sx[:2]
        b[2 * i:2 * i + 2] = -(Sx @ p)[:2]
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return t



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
                     alpha: float = 0.01, alpha_max: float = 0.5,
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
                    ninl = int(len(inliers))
                    t_use = tvec.reshape(3)
                    if self.cfg.gyro_fuse and R_prior is not None:
                        # Gyro owns rotation; vision corrects it, weighted by
                        # confidence (inlier count). Then re-estimate translation
                        # with the fused rotation held fixed.
                        # The gyro prior comes in the prev<-cur convention, so
                        # transpose it into PnP's cur<-prev point-rotation frame.
                        R_gyro = np.asarray(R_prior, dtype=np.float64).T
                        gain = self.cfg.gyro_corr_gain_max * min(
                            1.0, ninl / max(self.cfg.gyro_trust_inliers, 1))
                        R_corr = R @ R_gyro.T          # vision relative to gyro
                        disagree_deg = float(np.degrees(
                            np.linalg.norm(cv2.Rodrigues(R_corr)[0])))
                        info["gyro_corr_deg"] = disagree_deg
                        # When vision disagrees strongly with the gyro, the
                        # vision rotation is the suspect one (fast-rotation KLT
                        # slip that PnP still calls "inliers"). Ramp the vision
                        # correction down so the gyro takes over -- regardless of
                        # how many inliers PnP claims. Below the gate this factor
                        # is 1.0, so gold / well-tracked motion is unaffected.
                        if disagree_deg > self.cfg.gyro_disagree_deg:
                            span = max(self.cfg.gyro_disagree_span_deg, 1e-6)
                            damp = 1.0 - (disagree_deg
                                          - self.cfg.gyro_disagree_deg) / span
                            gain *= max(0.0, damp)
                        R_fused = _scale_rotation(R_corr, gain) @ R_gyro
                        # Only re-estimate translation when the fused rotation
                        # actually departs from PnP's (i.e. the gyro overrode a
                        # weak vision rotation). On healthy frames R_fused == the
                        # vision rotation, so we keep PnP's robust RANSAC tvec and
                        # the result is byte-for-byte the pure-vision pose -- no
                        # accuracy regression where vision is already good.
                        dev = float(np.degrees(np.linalg.norm(
                            cv2.Rodrigues(R_fused @ R.T)[0])))
                        if dev > 0.5:
                            idx = inliers.reshape(-1)
                            t_use = _translation_given_rotation(
                                obj[idx], img[idx], R_fused, self.K)
                        R = R_fused
                        # Translation trust: the SAME slipped tracks that made
                        # the vision ROTATION untrustworthy also corrupt the
                        # vision TRANSLATION. During a fast in-place yaw the true
                        # translation is ~0, but depth + track slip makes the
                        # solver explain the rotational image flow as a
                        # translation -> a phantom linear drift on every spin
                        # (the symptom: yaw in place and the position walks off).
                        # Basalt avoids this by constraining translation with the
                        # accelerometer; we have no such term, so we instead trust
                        # the gyro's "this is rotation" and damp the translation
                        # by the same disagreement factor. Unchanged below the
                        # gate, so gold / well-tracked motion keeps full vision
                        # translation (gold disagreement stays < 0.75 deg).
                        if disagree_deg > self.cfg.gyro_disagree_deg:
                            span = max(self.cfg.gyro_disagree_span_deg, 1e-6)
                            t_trust = max(0.0, 1.0 - (disagree_deg
                                          - self.cfg.gyro_disagree_deg) / span)
                            t_use = t_use * t_trust
                            info["t_trust"] = t_trust
                    T_pc = np.eye(4)
                    T_pc[:3, :3] = R
                    T_pc[:3, 3] = t_use
                    self.pose = self.pose @ np.linalg.inv(T_pc)
                    info["n_inliers"] = ninl
                    info["ok"] = True
                else:
                    info["reason"] = self._gyro_propagate(R_prior, "pnp_failed")
            else:
                info["reason"] = self._gyro_propagate(R_prior, "too_few_points")
        else:
            info["reason"] = "bootstrap"

        self._prev_obs = cur_obs
        self._prev_depth = depth_m
        self.last_info = info
        return self.pose

    # ------------------------------------------------------------------ #
    def _gyro_propagate(self, R_prior, fail_reason: str) -> str:
        """Advance the pose by the gyro rotation alone when vision is unusable.

        Covers BOTH vision-failure regimes -- a PnP that ran but did not solve
        (``pnp_failed``) and the harder one where a fast rotation left too few
        tracked points to even attempt PnP (``too_few_points``). The latter is
        exactly the fast-yaw spike the user sees: without this branch the pose
        froze precisely when the camera was turning hardest. Translation is left
        at zero (the gyro says nothing about it); rotation is propagated so the
        body frame keeps tracking the turn through the blind frames. Returns the
        reason string to record (annotated when the gyro took over).
        """
        if self.cfg.gyro_fuse and R_prior is not None:
            T_pc = np.eye(4)
            # prev<-cur gyro prior -> PnP's cur<-prev point-rotation frame.
            T_pc[:3, :3] = np.asarray(R_prior, dtype=np.float64).T
            self.pose = self.pose @ np.linalg.inv(T_pc)
            return fail_reason + "_gyro_propagated"
        return fail_reason
