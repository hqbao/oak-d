"""RGB-D visual odometry: track-based 3D-2D PnP between consecutive frames.

For each frame pair we already have KLT tracks (persistent ids) from
:mod:`ours.vio.frontend`. The motion estimate works like a minimal feature-based
VO:

1. Take tracks seen in *both* the previous and current frame.
2. Back-project the previous-frame observations to 3D using the previous depth
   map (metric, from the recorded stereo depth).
3. Solve our own ``pnp.solve_pnp_ransac`` (library-free RANSAC + LM) for the
   rigid transform that reprojects those 3D points onto their current-frame
   pixels -> ``T_prev->cur``.
4. Compose into the world pose: ``T_w_cur = T_w_prev @ inv(T_prev->cur)``.

Everything is metric because depth is metric, so the trajectory has true scale
(no monocular scale ambiguity). Output poses are in the camera optical frame
(+x right, +y down, +z forward); the world frame is the first camera.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frontend.frontend import FrontendConfig, KLTFrontend
from ..imu.imu import so3_exp, so3_log
from .pnp import solve_pnp_ransac


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
    # --- rotation-gated translation damping (opt-in, default OFF) -----------
    # The disagreement gate above only fires when the VISION rotation is wrong.
    # But during a smooth in-place yaw the KLT tracker follows the rotational
    # image flow well (vision AGREES with the gyro, disagreement < gate), yet
    # solvePnP still explains that rotational flow + depth as a phantom
    # TRANSLATION -- so the position walks off on every spin even though the
    # rotation is tracked correctly. There is no vision signal that separates
    # this phantom from a real translation; the only honest discriminator is
    # the gyro: a large per-frame rotation means most of the apparent motion is
    # rotational, so the co-occurring vision translation is suspect. When
    # enabled, the translation is scaled down as the gyro's per-frame rotation
    # grows past ``rot_damp_gate_deg`` (reaching 0 at +``rot_damp_span_deg``),
    # INDEPENDENTLY of the disagreement gate. Default OFF so every offline score
    # (f2f/ba/slam) and the regression suite stay byte-for-byte unchanged; the
    # live tight-coupled ``vio`` backend turns it on. A pure translation (no
    # rotation) is never damped; a pure fast yaw is fully suppressed.
    rot_translation_damp: bool = False
    rot_damp_gate_deg: float = 1.5
    rot_damp_span_deg: float = 4.0
    # --- constant-velocity translation prediction (opt-in, default OFF) -----
    # On fast motion the KLT tracker loses correspondences (motion blur + large
    # inter-frame flow at a low host frame rate), so PnP either fails or sees too
    # few points; the fallback ``_gyro_propagate`` then advances rotation but
    # leaves TRANSLATION at zero. During a fast straight-line move (gyro ~0) that
    # freezes the pose entirely -- the camera flies forward but the trajectory
    # sits still until vision re-locks (the "it lags then jumps" / "it just
    # stalls" symptom; Basalt avoids it by predicting the pose from the IMU).
    # When enabled we keep the last successful inter-frame translation as a
    # velocity and, on a vision-failure frame, propagate the pose by that
    # velocity (decayed each consecutive miss, capped at ``predict_max_frames``)
    # instead of zero -- so a brief tracking dropout during fast motion coasts
    # smoothly through instead of freezing. A genuine stop is unaffected: vision
    # keeps succeeding with t~0 there, so prediction only ever fills failure
    # frames. Default OFF so offline gold (f2f/ba/slam) stays byte-for-byte
    # unchanged; the live source turns it on for every backend.
    predict_translation: bool = False
    predict_decay: float = 0.85
    predict_max_frames: int = 8
    # --- IMU-locked translation solve (opt-in, default OFF) -----------------
    # The honest, Basalt-aligned separation of rotation and translation: the IMU
    # (gyro) owns rotation, so we solve the per-frame translation with that fused
    # rotation HELD FIXED (``_translation_given_rotation``) on EVERY frame,
    # instead of trusting solvePnP's joint rotation+translation ``tvec``. In a
    # joint solve a small rotation error is absorbed as a spurious translation,
    # so a pure in-place yaw produces a phantom linear drift; with the rotation
    # locked to the accurate gyro, the rotated 3D points already match the
    # current viewing rays and the least-squares translation comes out ~0 -- the
    # phantom is removed AT THE SOURCE, with no rotation-magnitude heuristic. A
    # genuine translate-while-turning still recovers its true translation (depth
    # gives metric scale). When this is on, the disagreement/rotation translation
    # damping below is bypassed (it is no longer needed). Default OFF so every
    # offline score (f2f/ba/slam) and the regression suite stay byte-for-byte
    # unchanged; the live source turns it on for all backends.
    lock_translation_to_rotation: bool = False
    # --- re-solve translation when vision disagrees with gyro (opt-in) -------
    # The honest middle ground between full ``lock_translation_to_rotation``
    # (re-solve EVERY frame -> injects lateral error on a straight push when the
    # gyro carries a small bias/extrinsic error) and the legacy disagreement
    # damping (multiply the translation by ``t_trust`` -> ZEROES real forward
    # motion whenever vision under-rotates, the "move + shake and it freezes"
    # symptom). When enabled, on a frame where the vision rotation disagrees with
    # the gyro by more than ``gyro_disagree_deg`` (KLT slipped under shake), we
    # RE-ESTIMATE the translation with the trusted gyro-fused rotation held fixed
    # (``_translation_given_rotation``) and KEEP it -- this removes the
    # rotational-flow phantom at the source (a pure yaw -> t~0) while preserving a
    # real forward push (its translation survives the rotation-locked solve). The
    # legacy ``t_trust`` multiply is then skipped (it would re-zero the recovered
    # translation). HEALTHY frames (disagreement below the gate, i.e. all of
    # gold) are untouched -> joint PnP ``tvec`` as before, so the regression suite
    # stays byte-for-byte unchanged. This is the loosely-coupled analog of what
    # Basalt does (drop slipped tracks via the forward-backward check, then let
    # the IMU own rotation while translation is still estimated). Default OFF; the
    # live source turns it on.
    resolve_translation_on_disagree: bool = False
    # --- physical per-frame translation speed clamp (opt-in, default OFF) ----
    # Under a hard/fast hand shake or a very fast in-place yaw, the surviving KLT
    # tracks are low-parallax and PnP turns the rotational image flow into a
    # spurious translation -- a per-frame "jump" much larger than any real hand
    # motion. Integrated, these jumps make the displayed path wobble like a
    # roller-coaster even though the net stays put (the roller-coaster symptom).
    # There is no vision-only signal that separates a phantom jump from a real
    # one (their magnitudes overlap), but a PHYSICAL upper bound does exist: a
    # hand cannot translate a camera faster than a few m/s. When this is > 0 and
    # the per-frame interval ``dt_s`` is known, the solved translation is clamped
    # so its implied speed never exceeds ``max_translation_speed`` (m/s) -- this
    # caps only the non-physical spikes (the visible wobble) and leaves every
    # real, in-budget motion untouched. fps-independent (scales with dt). Default
    # 0.0 (off) so offline gold / the regression suite are byte-for-byte
    # unchanged (gold per-frame motion is well under any sane bound); the live
    # source sets a generous value.
    max_translation_speed: float = 0.0
    # --- freeze translation on untrustworthy vision (opt-in, default OFF) ----
    # Pointing at a textureless surface (white wall / blank screen) the KLT
    # tracker still fills its corner budget with garbage corners, so
    # ``n_tracks`` stays high and is NOT a usable signal. But those garbage
    # tracks have no consistent depth+geometry, so PnP's RANSAC keeps only a
    # handful of inliers (measured: white-wall ``n_inliers`` median 0, p95 11;
    # a real fast push has median ~140). solvePnP still "succeeds" on those few
    # garbage inliers and returns a meaningless translation that walks the body
    # off in an undefined direction (the "white wall + move -> drifts randomly"
    # symptom). When this is > 0 and PnP returns fewer than this many inliers,
    # the translation is FROZEN: the rotation is still advanced by the gyro (so
    # the body tracks any turn) but the position is held put -- the honest
    # behaviour when vision cannot be trusted, consistent with staying still on
    # a covered camera. Unlike a fast-motion dropout this is NOT coasted: a
    # white wall carries no real motion to coast. The gate is well below the
    # inlier count of any real motion (fast-push p25 = 33, still ~36), so normal
    # use is untouched; the ~8% of fast-push frames that do dip below it are
    # frames that genuinely lost tracking, where freezing for one frame is
    # correct anyway. Default 0 (off) -> offline gold byte-for-byte unchanged;
    # the live source enables it.
    min_inliers_for_translation: int = 0
    # --- library-free PnP (default ON) --------------------------------------
    # Use the pure-NumPy ``pnp.solve_pnp_ransac`` (RANSAC DLT + robust LM seed +
    # plain-LS refine) instead of ``cv2.solvePnPRansac``, so the production path
    # carries no cv2 dependency. Measured vs cv2 on gold: better on the genuine
    # forward-motion sessions (corridor, lab_straight, push_straight_fast) and a
    # wash through windowed BA. Set False (dev only, OAKD_OWN_PNP=0) to A/B
    # against the cv2 oracle, which is then lazily imported.
    use_own_pnp: bool = True



def _scale_rotation(R: np.ndarray, s: float) -> np.ndarray:
    """Return the rotation that applies a fraction ``s`` of ``R`` (axis-angle)."""
    return so3_exp(so3_log(R) * float(s))


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
        # Constant-velocity translation prediction state (see OdometryConfig):
        # the last successful inter-frame translation (cur<-prev, PnP T_pc
        # frame), reused to coast the pose through vision-failure frames during
        # fast motion. ``_predict_count`` caps consecutive predicted frames.
        self._vel_t: np.ndarray | None = None
        self._predict_count = 0

    def align_to_gravity(self, accel_cam: np.ndarray) -> None:
        """Seed the initial attitude from gravity (call before the first frame).

        ``accel_cam`` is the static-startup accelerometer reading rotated into
        the camera optical frame. This sets the world frame so its "down" axis
        is aligned with real gravity instead of the (arbitrary) starting camera
        tilt -- the trajectory is otherwise unchanged (ATE is Umeyama-aligned, so
        a global rotation of the world frame does not affect it).
        """
        from ..imu.imu import gravity_aligned_R0
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

    def track(self, gray: np.ndarray) -> dict[int, np.ndarray]:
        """Run the KLT frontend on one frame; return the live ``{id: pixel}``.

        Split out of :meth:`process` so the flow pipeline can run feature
        TRACKING as its own task (``TrackFeatures``). This is the ONLY ``numba
        parallel=True`` section on the odometry thread, so only it needs the
        process-wide parallel lock; the downstream :meth:`estimate` is pure NumPy
        and runs lock-free (overlapping the depth matcher on its own thread).
        """
        state = self.frontend.process(gray)
        return {int(i): p for i, p in zip(state.ids, state.points)}

    def process(self, gray: np.ndarray, depth_m: np.ndarray,
                R_prior: np.ndarray | None = None,
                dt_s: float | None = None,
                imu_moving: bool = False) -> np.ndarray:
        """Advance odometry by one frame; returns the current 4x4 world pose.

        Thin wrapper: :meth:`track` then :meth:`estimate`. Kept so every existing
        caller (the offline tools + the regression tests) stays byte-for-byte
        unchanged; the live flow runs the two halves as separate tasks instead.

        ``R_prior`` (optional) is the predicted previous->current camera rotation
        (e.g. from gyro preintegration). When given it seeds the PnP solver, which
        helps a lot during fast rotation where KLT correspondences are sparse.

        ``dt_s`` (optional) is the time since the previous processed frame in
        seconds; when given together with ``max_translation_speed`` it bounds the
        per-frame translation to a physically plausible hand speed (clamps the
        non-physical phantom jumps that make the path wobble under shake/yaw).

        ``imu_moving`` (optional) tells the low-inlier translation freeze that the
        IMU sees real motion this frame, so a sparse-inlier solve is motion blur
        (keep translating) NOT a textureless wall (freeze). See :meth:`estimate`.
        """
        return self.estimate(self.track(gray), depth_m, R_prior, dt_s, imu_moving)

    def estimate(self, cur_obs: dict[int, np.ndarray], depth_m: np.ndarray,
                 R_prior: np.ndarray | None = None,
                 dt_s: float | None = None,
                 imu_moving: bool = False) -> np.ndarray:
        """Estimate motion from already-tracked features; returns the world pose.

        ``cur_obs`` is the live ``{track_id: current_pixel}`` produced by
        :meth:`track`. This is the pure-NumPy half (build correspondences ->
        RGB-D PnP -> optional gyro fusion / translation handling -> compose pose);
        it never enters a numba parallel region, so the flow ``EstimateMotion``
        task runs it WITHOUT the parallel lock. See :meth:`process` for the
        ``R_prior`` / ``dt_s`` semantics.
        """
        info = {"n_tracks": len(cur_obs), "n_pnp": 0, "n_inliers": 0,
                "ok": False, "reason": "",
                "inlier_ids": np.empty((0,), dtype=np.int64)}

        if self._prev_depth is not None and self._prev_obs:
            obj_pts, img_pts, id_list = [], [], []
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
                id_list.append(int(tid))

            info["n_pnp"] = len(obj_pts)
            if len(obj_pts) >= self.cfg.min_pnp_points:
                obj = np.asarray(obj_pts, dtype=np.float64)
                img = np.asarray(img_pts, dtype=np.float64)

                if self.cfg.use_own_pnp:
                    # cur<-prev rotation prior (R_prior is prev<-cur).
                    R_seed = (np.asarray(R_prior, np.float64).T
                              if R_prior is not None else None)
                    ok, R_own, t_own, inliers = solve_pnp_ransac(
                        obj, img, self.K, R_init=R_seed,
                        reproj_px=self.cfg.ransac_reproj_px,
                        iters=self.cfg.ransac_iters,
                        conf=self.cfg.ransac_conf,
                        min_points=self.cfg.min_pnp_points)
                else:
                    # Dev-only A/B path against the cv2 oracle (OAKD_OWN_PNP=0).
                    # cv2 is imported lazily here so the production path (own
                    # PnP) carries no cv2 runtime dependency.
                    import cv2
                    use_guess = False
                    rvec0 = tvec0 = None
                    if R_prior is not None:
                        rvec0, _ = cv2.Rodrigues(
                            np.asarray(R_prior, dtype=np.float64))
                        # Warm-start the translation with the predicted velocity
                        # (if enabled) so the iterative refinement converges on
                        # large fast-motion steps instead of a zero-motion guess.
                        if (self.cfg.predict_translation
                                and self._vel_t is not None):
                            tvec0 = self._vel_t.reshape(3, 1).astype(np.float64)
                        else:
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
                    if self.cfg.use_own_pnp:
                        R = R_own
                    else:
                        import cv2
                        R, _ = cv2.Rodrigues(rvec)
                    ninl = int(len(inliers))
                    # Track ids PnP kept as inliers (the clean subset the motion
                    # solve actually used) -> exposed for the keypoint visualiser.
                    inl_idx = np.asarray(inliers, dtype=np.int64).reshape(-1)
                    info["inlier_ids"] = np.asarray(id_list, dtype=np.int64)[inl_idx]
                    t_use = t_own if self.cfg.use_own_pnp else tvec.reshape(3)
                    # Freeze translation when vision is untrustworthy (too few
                    # inliers -> textureless / white wall). Advance rotation by
                    # the gyro but hold the position put; do NOT coast (no real
                    # motion to coast on a blank wall). Off when the gate is 0.
                    #
                    # ``imu_moving`` vetoes the freeze: a motion-blurred shake
                    # ALSO starves PnP of inliers, but there the camera IS moving
                    # and freezing would pin the marker through real motion (the
                    # "move + shake -> ours-ba/slam freezes in place" symptom). The
                    # accelerometer is the honest discriminator -- at rest the
                    # sparse inliers mean a blank wall (freeze), under motion they
                    # mean blur (keep the vision translation we did solve). Only
                    # the still+textureless case freezes.
                    if (self.cfg.min_inliers_for_translation > 0
                            and ninl < self.cfg.min_inliers_for_translation
                            and not imu_moving):
                        info["n_inliers"] = ninl
                        info["reason"] = "low_inliers_frozen"
                        if self.cfg.gyro_fuse and R_prior is not None:
                            T_pc = np.eye(4)
                            T_pc[:3, :3] = np.asarray(R_prior, dtype=np.float64).T
                            self.pose = self.pose @ np.linalg.inv(T_pc)
                        self._prev_obs = cur_obs
                        self._prev_depth = depth_m
                        self.last_info = info
                        return self.pose
                    if self.cfg.gyro_fuse and R_prior is not None:
                        # Gyro owns rotation; vision corrects it, weighted by
                        # confidence (inlier count). Then re-estimate translation
                        # with the fused rotation held fixed.
                        # The gyro prior comes in the prev<-cur convention, so
                        # transpose it into PnP's cur<-prev point-rotation frame.
                        R_gyro = np.asarray(R_prior, dtype=np.float64).T
                        rot_deg = float(np.degrees(np.linalg.norm(
                            so3_log(R_gyro))))
                        info["rot_deg"] = rot_deg
                        gain = self.cfg.gyro_corr_gain_max * min(
                            1.0, ninl / max(self.cfg.gyro_trust_inliers, 1))
                        R_corr = R @ R_gyro.T          # vision relative to gyro
                        disagree_deg = float(np.degrees(
                            np.linalg.norm(so3_log(R_corr))))
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
                        # Translation solve. With ``lock_translation_to_rotation``
                        # we ALWAYS re-estimate the translation with the fused
                        # (gyro-owned) rotation held fixed -- the honest
                        # Basalt-style separation that keeps rotational image flow
                        # out of the translation (a pure yaw -> t~0). Otherwise
                        # (offline gold) we keep solvePnP's joint ``tvec`` on
                        # healthy frames and only re-solve when the gyro overrode
                        # a weak vision rotation (``dev > 0.5``), so those scores
                        # stay byte-for-byte unchanged.
                        idx = inliers.reshape(-1)
                        if self.cfg.lock_translation_to_rotation:
                            t_use = _translation_given_rotation(
                                obj[idx], img[idx], R_fused, self.K)
                        elif (self.cfg.resolve_translation_on_disagree
                                and disagree_deg > self.cfg.gyro_disagree_deg):
                            # Vision rotation is suspect (KLT slip under shake):
                            # recover the REAL translation with the gyro rotation
                            # held fixed instead of zeroing it. A pure yaw -> ~0;
                            # a real forward push survives.
                            t_use = _translation_given_rotation(
                                obj[idx], img[idx], R_fused, self.K)
                        else:
                            dev = float(np.degrees(np.linalg.norm(
                                so3_log(R_fused @ R.T))))
                            if dev > 0.5:
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
                        # (Skipped entirely when the translation is already solved
                        # with the rotation locked -- the phantom is gone at the
                        # source, so no damping is needed.)
                        if (not self.cfg.lock_translation_to_rotation
                                and not self.cfg.resolve_translation_on_disagree
                                and disagree_deg > self.cfg.gyro_disagree_deg):
                            span = max(self.cfg.gyro_disagree_span_deg, 1e-6)
                            t_trust = max(0.0, 1.0 - (disagree_deg
                                          - self.cfg.gyro_disagree_deg) / span)
                            t_use = t_use * t_trust
                            info["t_trust"] = t_trust
                        # Rotation-gated translation handling (opt-in). Even
                        # when vision AGREES with the gyro on the rotation, a
                        # large per-frame yaw makes the co-occurring vision
                        # translation a likely phantom (rotational image flow
                        # read as a translation). But simply zeroing it also
                        # kills a REAL forward push that happens to carry some
                        # hand-rotation jitter (the "forward move doesn't track"
                        # symptom). So instead of damping toward ZERO we blend
                        # toward the constant-velocity motion model ``_vel_t``
                        # (the trusted velocity, only ever refreshed on
                        # low-rotation = clean frames below): a true forward push
                        # keeps moving (prediction is forward), while an in-place
                        # yaw from rest blends toward ~0 (no recent clean
                        # velocity) -- killing the phantom without freezing real
                        # motion. With prediction off (offline gold) ``_vel_t``
                        # is None, so this falls back to the original damp-to-0.
                        if (self.cfg.rot_translation_damp
                                and not self.cfg.lock_translation_to_rotation
                                and rot_deg > self.cfg.rot_damp_gate_deg):
                            span = max(self.cfg.rot_damp_span_deg, 1e-6)
                            r_trust = max(0.0, 1.0 - (rot_deg
                                          - self.cfg.rot_damp_gate_deg) / span)
                            vel = (self._vel_t if self._vel_t is not None
                                   else np.zeros(3))
                            t_use = r_trust * t_use + (1.0 - r_trust) * vel
                            info["rot_t_trust"] = r_trust

                    # Physical per-frame translation speed clamp (opt-in): cap
                    # only the non-physical phantom jumps (a hand cannot move the
                    # camera faster than a few m/s). Real, in-budget motion is
                    # left untouched. Off when speed<=0 or dt unknown -> gold
                    # byte-identical.
                    if (self.cfg.max_translation_speed > 0.0
                            and dt_s is not None and dt_s > 0.0):
                        cap = self.cfg.max_translation_speed * dt_s
                        tmag = float(np.linalg.norm(t_use))
                        if tmag > cap:
                            t_use = t_use * (cap / tmag)
                            info["t_clamped"] = tmag

                    T_pc = np.eye(4)
                    T_pc[:3, :3] = R
                    T_pc[:3, 3] = t_use
                    self.pose = self.pose @ np.linalg.inv(T_pc)
                    info["n_inliers"] = ninl
                    info["ok"] = True
                    # Refresh the trusted velocity for coasting / rotation-gated
                    # blending -- but ONLY from clean (low-rotation) frames, so a
                    # phantom translation produced during a fast yaw never
                    # pollutes it. On a high-rotation frame the velocity is left
                    # holding the last clean value (which decays through the
                    # coast path on vision failures). ``rot_deg`` is 0 here when
                    # there is no gyro prior (pure-vision), so every frame counts
                    # as clean in that case.
                    if self.cfg.predict_translation:
                        rot_deg = float(info.get("rot_deg", 0.0))
                        if rot_deg < self.cfg.rot_damp_gate_deg:
                            self._vel_t = np.asarray(t_use, float).copy()
                            self._predict_count = 0
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
        froze precisely when the camera was turning hardest. Rotation is
        propagated so the body frame keeps tracking the turn through the blind
        frames. Translation is normally left at zero (the gyro says nothing about
        it), but when ``predict_translation`` is on we coast it with the last
        successful inter-frame velocity (decayed, capped) so a fast STRAIGHT
        move -- where the freeze-at-zero is most visible -- keeps advancing
        through the dropout instead of stalling. Returns the reason string to
        record (annotated when the gyro / velocity took over).
        """
        if not (self.cfg.gyro_fuse and R_prior is not None):
            # No rotation prior: optionally still coast translation so a pure
            # fast forward move (no usable vision) does not freeze.
            if (self.cfg.predict_translation and self._vel_t is not None
                    and self._predict_count < self.cfg.predict_max_frames):
                self._predict_count += 1
                T_pc = np.eye(4)
                T_pc[:3, 3] = self._vel_t           # coast the last velocity
                self.pose = self.pose @ np.linalg.inv(T_pc)
                self._vel_t = self._vel_t * self.cfg.predict_decay  # then decay
                return fail_reason + "_vel_predicted"
            return fail_reason

        T_pc = np.eye(4)
        # prev<-cur gyro prior -> PnP's cur<-prev point-rotation frame.
        R_gyro_pc = np.asarray(R_prior, dtype=np.float64).T
        T_pc[:3, :3] = R_gyro_pc
        tag = "_gyro_propagated"
        # Only coast translation when the gyro reports little rotation: during a
        # fast yaw the true translation is ~0 (the same reason the rotation-gated
        # damping zeroes it on success frames), so coasting a stale forward
        # velocity through a yaw would re-introduce the phantom drift. Slow/no
        # rotation is exactly the fast straight-line regime we want to bridge.
        rot_deg = float(np.degrees(np.linalg.norm(so3_log(R_gyro_pc))))
        if (self.cfg.predict_translation and self._vel_t is not None
                and rot_deg < self.cfg.rot_damp_gate_deg
                and self._predict_count < self.cfg.predict_max_frames):
            self._predict_count += 1
            T_pc[:3, 3] = self._vel_t              # coast the last velocity
            self._vel_t = self._vel_t * self.cfg.predict_decay  # then decay
            tag = "_gyro_vel_propagated"
        self.pose = self.pose @ np.linalg.inv(T_pc)
        return fail_reason + tag


