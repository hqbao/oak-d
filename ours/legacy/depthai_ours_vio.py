"""Live RGB-D VIO from the OAK-D using *our* from-scratch odometry.

Unlike :mod:`baseline.depthai_vio` (which reads poses out of DepthAI's
built-in ``BasaltVIO`` node), this source runs **our own** frame-to-frame
RGB-D PnP odometry (:class:`ours.vio.RGBDVisualOdometry`) with depth from our
**own** SGM stereo matcher. It exists so we can watch our VIO drive the 3D
viewer in real time and eyeball its quality against Basalt *before* we add
sliding-window bundle adjustment.

Fully portable pipeline (NO VPU / depth library): we tap the two RAW mono
camera frames directly and do everything ourselves --

    Camera CAM_B (raw left) + CAM_C (raw right)
        -> our Left/RightRectifier (library-free rectification)
        -> our SGM matcher -> metric depth (float32 m)

There is no ``StereoDepth`` node: the chip's stereo/depth engine is never used,
so this front-end ports unchanged to any platform with two cameras + a CPU.


Our odometry produces poses in the standard OpenCV **camera optical** frame
(x right, y down, z forward), world = first frame (assumed level at start, since
vision-only VO has no gravity reference). For the NED 3D viewer we remap optical
-> NED with the textbook mapping::

    North = +z_opt (forward),  East = +x_opt (right),  Down = +y_opt (down)

i.e. ``M_opt->ned = [[0,0,1],[1,0,0],[0,1,0]]``. Combined with the attitude
column reorder ``P`` below this gives an *identity* startup attitude
(roll=pitch=yaw=0) and a physically self-consistent display: moving the camera
up moves the marker up, moving it forward moves it North, moving it right moves
it East. (An earlier flipped mapping ``[[0,0,-1],[1,0,0],[0,-1,0]]`` baked in a
spurious 180 roll -- the symptom was the green 'right' arrow pointing left and
upward camera motion showing as downward marker motion.)

Note: absolute North/Down here are *not* gravity-aligned (no IMU leveling); the
frame is anchored to the first camera pose. Trajectory accuracy (ATE) is
Umeyama-aligned so it is unaffected by this convention choice.

Startup attitude: at launch we average the accelerometer over a short static
window and gravity-level the initial pose (``RGBDVisualOdometry.align_to_gravity``)
so the world "down" is real gravity and the reported roll/pitch reflect the
camera's actual tilt -- not an assumed-level identity start. Yaw stays at the
starting heading (no magnetometer). If the device reports no IMU, we fall back to
an identity start.

Note: the gyro rotation prior is a measured no-op on well-synced data (see
``ours/vio/imu.py``), so this live source runs pure vision for simplicity.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ours.lib.misc.pose import Pose
from ours.lib.misc.frames import quat_to_rpy
from ours.lib import (
    InertialTranslationFilter, KLTFrontend,
    RGBDVisualOdometry, gravity_aligned_R0, level_attitude,
)
from ours.ui.source import PoseSource


# Standard OpenCV optical (x right, y down, z forward) -> world NED.
# North = +z (forward), East = +x (right), Down = +y (down). With the column
# reorder P below this yields an identity startup attitude (roll=pitch=yaw=0)
# and a self-consistent display (up->up, forward->N, right->E).
_M_OPT_TO_NED = np.array(
    [[0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]]
)

# Measured mounted-pose attitude (camera->world), used only as a FALLBACK when
# the device reports no accelerometer. Captured 2026-06-02 with the camera in
# its drone-mount pose via ``ours/tools/measure_mount_attitude.py`` (avg of 760 raw
# accel samples, per-axis std ~0.02 m/s^2 -> tilt good to <0.05 deg): the mount
# is essentially level (roll +0.1, pitch -0.5, lens looking horizontally
# forward). The live path still prefers a fresh accel measurement at startup, so
# this only matters on IMU-less devices. Re-run the tool and update this if the
# physical mount changes.
_MOUNT_R0 = np.array(
    [[+0.999998, -0.002003, +0.000000],
     [+0.002003, +0.999956, +0.009146],
     [-0.000018, -0.009146, +0.999958]]
)

# Accel leveling only runs when the camera is at rest, detected from the residual
# of the raw accelerometer against its EMA (recent motion energy, m/s^2). Below
# this threshold the camera is still and the accel reads pure gravity; above it
# there is translation/rotation whose lateral linear acceleration would bias the
# gravity DIRECTION (a magnitude gate cannot catch that), so leveling is skipped
# and vision holds the attitude. ~0.35 m/s^2 sits above the sensor noise floor
# (~0.02-0.15) and below the accel of deliberate handheld motion.
_REST_MOTION_THRESH = 0.35

# --- live tight-coupled VIO correction gating (backend='vio') -------------
# The background VIO worker double-integrates the device accelerometer to tie
# keyframe translations. On the OAK-D the accel is noisy, so a diverged optimiser
# can publish a huge correction that would jerk the displayed trajectory. We
# reject any single correction whose translation exceeds ``_VIO_CORR_MAX_T`` as
# such a blow-up; whatever survives is eased onto the display so it never snaps.
# (An earlier rotation-freeze was removed: it held the correction at identity
# during all hand motion, so the accelerometer path never reached the screen --
# the "linear acceleration has no effect" symptom.)
_VIO_CORR_MAX_T = 0.25           # m; reject corrections larger than this

# --- live BA / loop-closure correction rate limit ------------------------
# The BA (ours-ba) and pose-graph loop-closure (ours-slam) corrections are eased
# onto the LIVE marker. A loop closure can produce a large, sudden world-frame
# correction; without a rate limit the fractional ease applies ~15% of it in one
# frame, which teleports the marker. Measured on device (ours-slam in a loopy
# lab): right after each loop the displayed step jumped ~0.2 m in a single frame
# (disp/filt up to 99x) while the camera was nearly still, so the marker stopped
# tracking the live push ("đi một đoạn rồi ì lại"). We cap the per-frame
# correction STEP to a bounded velocity so the correction bleeds in smoothly
# (still converges within ~1 s) instead of yanking. 0.015 m/frame ~= 0.3 m/s at
# 20 fps -- well below a hand push, so live motion always stays visible.
_CORR_MAX_STEP_T = 0.015                 # m per frame   (~0.3 m/s @ 20 fps)
_CORR_MAX_STEP_R = float(np.deg2rad(0.5))  # rad per frame (~10 deg/s @ 20 fps)

# Speed gate for the loose BA / loop-closure correction slew. The live BA map
# can diverge from the frame-to-frame filter pose; slewing that (large)
# correction onto the displayed tip WHILE the camera is pushing fast drags the
# marker backward -- measured on device as ``disp/filt`` falling to 0.70-0.84
# during a fast push (the "đẩy nhanh rồi ì lại" stall), even though the filter
# pose itself tracks the motion faithfully (``filt/vo ~ 1.0``). So we FREEZE the
# correction (no slew) whenever the filter speed is above this threshold: a
# frozen correction is a rigid transform, which preserves the path length of the
# live motion (``disp/filt = 1`` -> full-distance tracking, like ``ours``). Below
# the threshold (slow / looping) the correction slews normally so BA drift and
# SLAM loop closures still fold in. 1.0 m/s is chosen from the gold + fast_push
# Basalt speed profiles: a loop / gentle motion stays almost entirely below it
# (lab_loop p90=0.80 m/s -> the loop-closure correction is untouched, ATE
# identical offline), while a fast hand push spends ~36% of its time above it
# (fast_push p90=2.2 m/s) -> the fastest part of the push always freezes and
# tracks the full distance. Verified offline: freeze@1.0 leaves every gold
# session's Sim3 scale/ATE unchanged (the correction there is already small).
_CORR_FREEZE_SPEED = 1.0                 # m/s; above this, freeze the loose corr

# Column reorder optical (right, down, fwd) -> body FRD (fwd, right, down).
# The viewer triad expects the attitude columns to be [forward, right, down],
# but our VO's rotation columns are the optical axes [right, down, fwd]. The
# body attitude in NED is therefore the camera axes mapped to NED with the
# columns picked as [optical_z, optical_x, optical_y] -> M @ R_opt @ P. Using
# the naive conjugation M @ R_opt @ M.T leaves the forward+down arrows 180
# off (only the right axis happens to line up). Verified vs Basalt (all body
# axes +0.97..+1.0 cos).
_P_OPT_TO_FRD = np.array(
    [[0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0]]
)


def _ease_se3(C_cur: np.ndarray, C_tgt: np.ndarray, alpha: float,
              max_t: float = 0.0, max_ang: float = 0.0) -> np.ndarray:
    """Move ``C_cur`` a fraction ``alpha`` toward ``C_tgt`` (smooth correction).

    Rotation eases along the geodesic (scaled axis-angle); translation eases
    linearly. Keeps the applied correction continuous so BA updates never snap
    the displayed trajectory.

    ``max_t`` / ``max_ang`` (when > 0) additionally clamp the per-call STEP to a
    bounded translation (m) and rotation (rad). This rate-limits the correction
    so that a large pose-graph jump after a loop closure bleeds in over several
    frames instead of teleporting the live marker: measured on device a single
    loop closure moved the displayed pose ~0.2 m in one frame (disp/filt up to
    99x) while the camera was still, so the marker stopped following the live
    push. With the clamp the correction still fully converges (within ~1 s) but
    never moves faster than the cap, so live motion stays visible. The clamp is a
    no-op (exact old behaviour) when both limits are 0.
    """
    R_cur, R_tgt = C_cur[:3, :3], C_tgt[:3, :3]
    dR = R_tgt @ R_cur.T
    ang = np.arccos(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
    a = alpha * ang
    if max_ang > 0.0:
        a = min(a, max_ang)            # cap the rotation STEP (rad), not fraction
    out = np.eye(4)
    if ang < 1e-8:
        out[:3, :3] = R_tgt
    else:
        axis = np.array([dR[2, 1] - dR[1, 2],
                         dR[0, 2] - dR[2, 0],
                         dR[1, 0] - dR[0, 1]]) / (2.0 * np.sin(ang))
        K_ = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
        R_step = np.eye(3) + np.sin(a) * K_ + (1.0 - np.cos(a)) * (K_ @ K_)
        out[:3, :3] = R_step @ R_cur
    dt = alpha * (C_tgt[:3, 3] - C_cur[:3, 3])
    if max_t > 0.0:
        n = float(np.linalg.norm(dt))
        if n > max_t:
            dt *= max_t / n            # cap the translation STEP (m)
    out[:3, 3] = C_cur[:3, 3] + dt
    return out


def _rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) unit quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


class OakOursVioSource(PoseSource):
    """OAK-D + *our* RGB-D odometry -> NED pose stream.

    ``backend='f2f'`` runs the plain frame-to-frame PnP VO; ``backend='ba'``
    runs the sliding-window bundle-adjustment VO. Both share the same KLT
    frontend and depth, so switching backends isolates exactly what BA adds.

    ``backend='slam'`` runs the f2f VO for display and a **background loop-closure
    SLAM** thread (:class:`ours.vio.SlamMap`): every few frames a keyframe (the
    raw f2f pose + its image + depth) is handed to the SLAM map, which recognises
    revisited places (ORB + fundamental-matrix + PnP geometric verification) and
    runs SE(3) pose-graph optimisation. The resulting world-frame correction is
    eased onto the displayed trajectory exactly like the BA correction, so loop
    closures snap out the accumulated drift smoothly. The SLAM map is fed the
    *raw* f2f poses (a fixed world frame), so its odometry edges stay
    self-consistent over the whole trajectory; gravity leveling is still applied
    as the final display step (the ordering rule: loop closure owns position+yaw,
    accel re-levels tilt last).

    **Gyro complementary fusion:** the gyroscope is integrated each frame into an
    inter-frame rotation prior that is handed to the odometry (``gyro_fuse``).
    Vision (PnP) corrects this rotation weighted by its inlier confidence, so a
    fast yaw turn that makes the KLT tracker lose features no longer under-rotates
    the pose. On a healthy frame the fusion collapses to pure vision (no accuracy
    cost on good data). Translation stays vision-only.
    """

    def __init__(self, width: int = 640, height: int = 400, fps: int = 20,
                 backend: str = "f2f", slam_kf_every: int = 5,
                 slam_radius_m: float = 0.0, ba_window: int = 6,
                 ba_kf_every: int = 5, ba_iters: int = 5, ba_marg: bool = False,
                 slam_kf_min_trans: float = 0.0,
                 slam_kf_min_rot: float = 0.0, slam_max_kf: int = 0,
                 depth_fast: bool = True,
                 max_corners: int | None = None,
                 min_distance: float | None = None,
                 klt_win: int | None = None, klt_levels: int | None = None,
                 reproj_px: float | None = None,
                 num_disparities: int | None = None,
                 orb_features: int | None = None) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.cam_fps = int(fps)
        self.backend = backend
        # Resolution-aware vision tuning. Every pixel-unit threshold in the
        # pipeline was tuned at 640x400; running at a lower resolution to save
        # CPU shrinks all of them, so they are auto-scaled from that baseline to
        # the live (width, height). Any of the seven knobs can be overridden at
        # runtime (None = keep the auto-scaled value) -- this is the set we
        # co-tune per resolution (see docs/RESOLUTION_TUNING.md).
        from ours.lib.config.resolution import ResolutionProfile
        self.res = ResolutionProfile.for_resolution(
            self.width, self.height,
            max_corners=max_corners, min_distance=min_distance,
            klt_win=klt_win, klt_levels=klt_levels, reproj_px=reproj_px,
            num_disparities=num_disparities, orb_features=orb_features)
        # Depth feeding the VIO is ALWAYS our own from-scratch SGM matcher
        # (ours.vio.stereo) run live on the rectified left + the RAW syncedRight
        # frame. The chip StereoDepth map is deliberately NOT used here: this is
        # the portable pipeline that must run on a target platform with no VPU /
        # depth library. (The chip-depth path lives only in the Basalt sources
        # ``depthai_vio`` / ``depthai_slam`` and in the offline oracle
        # ``ours/tools/vio_run.py --depth chip`` for A/B measurement.)
        # ``depth_fast`` uses the half-res SGMConfig.live() preset that fits the
        # live per-frame budget (full-res SGM is too slow for real time).
        self.depth_fast = bool(depth_fast)

        # SLAM update cadence: insert a keyframe (and run loop detection) every
        # ``slam_kf_every`` frames. This is the main lever for the SLAM update
        # rate -- fewer keyframes = more responsive loop closure AND a smaller
        # pose graph (cheaper PGO). ``slam_radius_m`` optionally spatially gates
        # loop candidates (0 = check all, the default): measured to help little
        # at ~200 keyframes because the ORB appearance gate already rejects
        # distant keyframes cheaply, but it bounds cost on very long runs.
        self.slam_kf_every = int(slam_kf_every)
        self.slam_radius_m = float(slam_radius_m)
        # Keyframe budget for long runs. The motion gate (min translation /
        # rotation since the last keyframe) makes the map grow with TRAJECTORY
        # length instead of run TIME -- a hovering/stationary drone stops piling
        # up redundant keyframes, the main cause of unbounded memory + PGO cost.
        # ``slam_max_kf`` is an absolute safety cap (drops the oldest keyframe
        # when exceeded; 0 = unlimited). Both default to off so behaviour is
        # unchanged unless requested.
        self.slam_kf_min_trans = float(slam_kf_min_trans)
        self.slam_kf_min_rot = float(slam_kf_min_rot)
        self.slam_max_kf = int(slam_max_kf)

        # Sliding-window BA tuning (backend='ba'): window size, keyframe cadence,
        # and BA iterations per solve. Smaller = cheaper/faster, larger = more
        # accurate but heavier on the background thread.
        self.ba_window = int(ba_window)
        self.ba_kf_every = int(ba_kf_every)
        self.ba_iters = int(ba_iters)
        # Marginalization prior (opt-in): Schur-marginalize the dropped keyframe
        # into a pose prior over the survivors instead of plain-dropping it.
        # Tightens metric scale; trades a little local ATE (see ba_marg_selftest
        # + vio_run --marg). Off by default.
        self.ba_marg = bool(ba_marg)

        # --- live SLAM overlay (read by the 3D viewer) ----------------------
        # Thread-safe snapshot of the SLAM map for the UI: keyframe dots, the
        # matched (revisited) keyframes, and the loop-closure links. All in NED
        # so the viewer only has to apply its NED->ENU display transform. These
        # are REAL SlamMap outputs (corrected keyframe poses + confirmed loop
        # events), not a parallel/derived pipeline. Empty for non-SLAM backends.
        self._slam_lock = threading.Lock()
        self._slam_kf_ned = np.zeros((0, 3), dtype=np.float32)
        self._slam_match_ned = np.zeros((0, 3), dtype=np.float32)
        self._slam_loop_ned: list[np.ndarray] = []
        # Flash counter: bumped each time a NEW loop closes, so the viewer can
        # detect a fresh teleport and play a short fade-out (instead of drawing
        # the whole accumulated loop history, which turns into a magenta mess).
        self._slam_flash_id = 0

        # Set by the UI "clear keyframes" button; the read loop picks it up,
        # wipes the SLAM map + the loop-closure correction + the overlay, so a
        # test run can be restarted without relaunching the pipeline.
        self._slam_reset = threading.Event()

        # Gyro zero-rate bias (rad/s, IMU frame), measured over the static
        # startup window by ``_collect_startup_accel``. None until startup runs
        # (or if the device has no IMU), in which case gyro integration uses a
        # zero bias.
        self._gyro_bias: np.ndarray | None = None

    def slam_overlay_snapshot(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], int]:
        """Latest SLAM overlay for the viewer (all positions in NED).

        Returns ``(kf_ned, match_ned, loop_segs, flash_id)`` where ``kf_ned`` is
        every keyframe position (Nx3), ``match_ned`` the keyframes revisited by
        the MOST RECENT loop closure (Mx3), ``loop_segs`` its ``[cur, old]``
        teleport segments, and ``flash_id`` a counter the viewer watches to know
        a new loop just closed (so it can flash then fade the link). The match /
        loop fields hold only the latest closure, not the full history.
        """
        with self._slam_lock:
            return (self._slam_kf_ned.copy(),
                    self._slam_match_ned.copy(),
                    [s.copy() for s in self._slam_loop_ned],
                    self._slam_flash_id)

    def clear_slam_map(self) -> None:
        """Forget every SLAM keyframe (UI "clear keyframes" button).

        Signals the read loop, which wipes the map worker-side and resets the
        loop-closure correction + overlay. Safe to call from the UI thread; a
        no-op for non-SLAM backends. Does NOT touch the displayed trajectory or
        the f2f/gyro odometry — only the SLAM keyframe map and its corrections.
        """
        self._slam_reset.set()

    def _run(self) -> None:
        import depthai as dai  # lazy: --source fake works without depthai/device

        left_socket = dai.CameraBoardSocket.CAM_B
        right_socket = dai.CameraBoardSocket.CAM_C

        with dai.Pipeline() as p:
            left = p.create(dai.node.Camera).build(left_socket, sensorFps=self.cam_fps)
            right = p.create(dai.node.Camera).build(right_socket, sensorFps=self.cam_fps)
            imu = p.create(dai.node.IMU)

            # NO StereoDepth node: this is the fully portable pipeline. We pull
            # the RAW left + RAW right frames straight from the two cameras and
            # rectify BOTH ourselves (ours.vio.stereo Left/RightRectifier), then
            # run our SGM. Nothing here touches the VPU's stereo/depth engine.

            # Accelerometer (gravity leveling) + gyroscope (the inter-frame
            # rotation prior for the complementary fusion). The gyro is what
            # keeps yaw correct through fast turns where vision under-rotates;
            # accel cannot recover yaw. 200 Hz gyro >> the ~20 fps frame rate so
            # each frame integrates ~10 samples.
            imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW,
                                 dai.IMUSensor.GYROSCOPE_RAW], 200)
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)

            # Raw camera outputs (CAM_B/CAM_C are hardware frame-synced on the
            # OAK-D, so their sequence numbers increment together; we still pair
            # strictly by sequence below). These are unrectified -- our matcher
            # rectifies both internally.
            left_out = left.requestOutput((self.width, self.height))
            right_out = right.requestOutput((self.width, self.height))

            # Non-blocking queues so a slow consumer never stalls the device
            # (a stalled XLink read is what triggers X_LINK_ERROR). We keep a
            # small buffer and always consume the *latest* frame below.
            q_left = left_out.createOutputQueue(maxSize=4, blocking=False)
            q_right = right_out.createOutputQueue(maxSize=4, blocking=False)
            q_imu = imu.out.createOutputQueue(maxSize=50, blocking=False)

            p.start()

            # Pull rectified-left intrinsics for the metric back-projection.
            ch = p.getDefaultDevice().readCalibration()

            K = np.array(
                ch.getCameraIntrinsics(left_socket, self.width, self.height),
                dtype=np.float64,
            )

            # Build our SGM matcher from the live device calibration. We
            # assemble the same JSON shape the recorder writes (so
            # StereoCalib.from_json applies the identical cm->m extrinsic
            # convention) and hand it to SGMStereoMatcher, which precomputes BOTH
            # rectification maps (left + right) once here. ``rectify_left=True``
            # makes the matcher rectify the raw left frame too, so the whole
            # depth path is VPU-free: nothing reads the chip's rectifiedLeft or
            # StereoDepth output.
            from ours.lib.io.reader import StereoCalib
            from ours.lib.stereo.stereo import SGMStereoMatcher

            def _intr(sock):
                Ki = np.array(ch.getCameraIntrinsics(
                    sock, self.width, self.height), dtype=np.float64)
                dist = list(ch.getDistortionCoefficients(sock))
                return {"fx": float(Ki[0, 0]), "fy": float(Ki[1, 1]),
                        "cx": float(Ki[0, 2]), "cy": float(Ki[1, 2]),
                        "dist": [float(x) for x in dist],
                        "width": int(self.width), "height": int(self.height)}

            T_lr = np.array(
                ch.getCameraExtrinsics(left_socket, right_socket),
                dtype=np.float64).reshape(4, 4)
            calib = StereoCalib.from_json({
                "intrinsics_left": _intr(left_socket),
                "intrinsics_right": _intr(right_socket),
                "T_left_right": T_lr.tolist(),
            })
            sgm_cfg = self.res.sgm(fast=self.depth_fast)
            matcher = SGMStereoMatcher.from_calib(calib, sgm_cfg,
                                                  rectify_left=True)
            print(f"[ours-vio] depth source: OURS SGM "
                  f"({'live/half-res' if self.depth_fast else 'full'})")
            print(f"[ours-vio] resolution profile: {self.res.describe()}")

            # IMU->left-camera rotation (for bringing accel into the optical
            # frame). depthai returns the extrinsic with translation in cm; we
            # only need the 3x3 rotation here.
            try:
                R_imu_cam = np.array(
                    ch.getImuToCameraExtrinsics(left_socket), dtype=np.float64
                )[:3, :3]
            except Exception:
                R_imu_cam = np.eye(3)

            # The displayed pose is ALWAYS produced by the fast frame-to-frame
            # VO, so the read loop never blocks on BA and the UI stays smooth.
            # The live frontend is ALWAYS our own library-free KLT + Shi-Tomasi
            # (no cv2). Pick the config by whether Numba is available:
            #   * with Numba, the JIT core tracks the FULL-quality config in
            #     ~15 ms/frame (well under the 50 ms budget at 20 fps), so use it.
            #   * without Numba the pure-NumPy path costs ~140 ms/frame, so fall
            #     back to the lighter ``live_own`` preset (~38-58 ms) to stay
            #     roughly real time.
            from ours.lib.frontend.klt_numba import HAVE_NUMBA
            fe_cfg = self.res.frontend(numba=HAVE_NUMBA)

            # Translation is owned by vision (frame-to-frame RGB-D PnP) with the
            # gyro FUSED softly: the gyro corrects a slipped vision rotation via
            # the disagreement gate (``gyro_disagree_deg``) and damps the
            # co-occurring phantom translation, WITHOUT hard-locking translation
            # to the gyro rotation. Measured on the gold sessions
            # (``ours/tools/live_replay.py`` + ``ours/tools/lateral_analysis.py``), the hard
            # lock (``lock_translation_to_rotation``) made the FAST-MOTION case
            # markedly worse -- it forced every small gyro error (bias / timing /
            # extrinsic) into a SIDEWAYS translation, raising path jitter
            # 20.3 -> 27.0 and the lateral/longitudinal ratio 0.19 -> 0.23 on the
            # fwd/back push (the "veers sideways while pushing forward" symptom).
            # Joint PnP + the disagreement gate beats it on motion (ATE 0.99% vs
            # 1.28%, jitter below Basalt's own) at a small cost on pure-still
            # drift (107 vs 71 mm), so the lock stays OFF. The
            # ``InertialTranslationFilter`` then owns the displayed position
            # (accel feed-forward off; see ``use_accel_prediction``).
            #
            # ``resolve_translation_on_disagree``: when the camera is shaken
            # while moving, KLT slips and the vision rotation under-rotates vs the
            # gyro. The legacy disagreement handling multiplied the translation by
            # ``t_trust`` -> 0, FREEZING real forward motion (the "move + shake
            # and it doesn't move at all" symptom). MEASURED on the
            # ``push_shake_20s`` gold session this re-solve was INEFFECTIVE (the
            # disagreement gate fires on only ~8% of frames there and never
            # zeroed the translation), so it is left OFF -- the freeze under hard
            # shake is the missing tight-accel translation factor, not this gate.
            #
            # ``max_translation_speed``: a hard/fast shake or a very fast
            # in-place yaw makes PnP read the rotational image flow as a spurious
            # per-frame translation jump far bigger than any real hand motion;
            # integrated, the path wobbles (the "đi tàu lượn" / "cong vẹo"
            # symptom). No vision-only signal separates a phantom jump from a real
            # one, but a physical bound does: a hand can't move the camera faster
            # than a few m/s. 4.0 m/s caps only the non-physical spikes and leaves
            # every real in-budget motion untouched (gold per-frame motion is far
            # below this, so offline scoring is byte-identical). Needs the
            # per-frame ``dt_s`` passed to ``vo.process`` (below).
            #
            # ``min_inliers_for_translation``: pointing at a textureless surface
            # (white wall / blank screen) KLT still fills its corner budget with
            # garbage corners (n_tracks stays high), but those have no consistent
            # depth+geometry so PnP keeps only a handful of inliers (measured:
            # white-wall median 0, p95 11; a real fast push median ~140). solvePnP
            # still "succeeds" on the garbage and returns a meaningless
            # translation that walks the body off randomly (the "white wall +
            # move -> drifts in an undefined direction" symptom). 12 inliers
            # freezes translation in exactly that regime (rotation still tracked
            # by the gyro, position held put) while leaving all real motion
            # untouched -- fast-push p25 is 33 inliers, and the ~8% of its frames
            # that dip below 12 are genuine tracking losses where freezing for one
            # frame is correct anyway (measured: ATE 2.14% -> 1.82%).
            od_cfg = self.res.odometry(gyro_fuse=True,
                                       max_translation_speed=4.0)
            vo = RGBDVisualOdometry(
                K, od_cfg, frontend=KLTFrontend(fe_cfg))



            # Gravity-level the initial attitude: average the accelerometer over
            # a short static startup window, rotate it into the camera optical
            # frame, and seed the VO world frame so its "down" is real gravity.
            # If the device has no accelerometer, fall back to the measured
            # mounted pose ``_MOUNT_R0`` so we still start from a known attitude.
            accel_cam = self._collect_startup_accel(q_imu, R_imu_cam)
            if accel_cam is not None:
                vo.align_to_gravity(accel_cam)
                # Sanity-log how the live measurement compares to the recorded
                # mount baseline. The per-frame ``correct_tilt`` below keeps the
                # attitude pinned to gravity at any orientation, so a startup
                # offset (or even an upside-down hold) self-corrects within a few
                # frames -- no need to gate on it.
                dR = vo.pose[:3, :3] @ _MOUNT_R0.T
                ang = np.degrees(np.arccos(
                    np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)))
                print(f"[ours-vio] gravity-leveled startup; "
                      f"{ang:.1f} deg from recorded mount baseline")
            else:
                vo.pose = np.eye(4)
                vo.pose[:3, :3] = _MOUNT_R0
                print("[ours-vio] no accelerometer; using recorded mount R0")

            # In BA mode, a background *process* refines a sliding window of
            # keyframes and publishes a world-frame correction ``C``. We ease
            # the applied correction toward the latest ``C`` so updates never
            # snap the trajectory, and apply it as P_disp = C @ P_f2f. The BA
            # map is fed the (already gravity-aligned) f2f poses, so its
            # correction is frame-consistent without extra bookkeeping.
            ba_state = None
            C_applied = np.eye(4)
            C_target = np.eye(4)
            if self.backend == "ba":
                ba_state = self._start_ba_worker(K)

            # In SLAM mode, a background thread keeps a persistent keyframe map,
            # recognises revisited places and runs pose-graph optimisation; it
            # publishes a world-frame loop-closure correction that we ease onto
            # the displayed f2f trajectory the same way as the BA correction.
            slam_state = None
            if self.backend == "slam":
                slam_state = self._start_slam_worker(K)

            # In tight-coupled VIO mode, a background thread runs the joint
            # visual+IMU sliding-window solve (WindowedVIOMap). Each keyframe
            # carries its raw IMU segment (cam frame) since the previous one;
            # the worker publishes a correction eased onto the f2f display the
            # same way as BA. ``vio_imu_*`` accumulate the per-sample cam-frame
            # IMU between keyframe submits.
            vio_state = None
            vio_imu_ts: list[int] = []
            vio_imu_gyro: list[np.ndarray] = []
            vio_imu_accel: list[np.ndarray] = []
            if self.backend == "vio":
                vio_state = self._start_vio_worker(K, R_imu_cam)

            t0 = time.monotonic()
            prev_pos_ned = np.zeros(3)
            prev_t: float | None = None
            frames = 0
            kf_count = 0
            last_fps_t = t0
            # --- live throughput diagnostics (test #10: fast-push undershoot) ---
            # Answers the open question: is the loop dropping frames the camera
            # delivered (recoverable backlog -> process it) or is the camera
            # simply delivering few frames (raise sensorFps)? Per 0.5 s window:
            #   recv  = frames the camera handed us (processed + discarded)
            #   drop  = frames we DISCARDED by draining to latest
            #   vo/emit/loop ms = where the per-iteration time goes
            diag_backlog_sum = 0
            diag_recv = 0
            diag_sgm_ms = 0.0
            diag_vo_ms = 0.0
            diag_emit_ms = 0.0
            diag_loop_ms = 0.0
            diag_iter = 0
            # Translation accounting (test #10b: "đẩy nhanh chỉ ăn 1/2 đoạn").
            # Per 0.5 s window, sum the per-frame step magnitude at three stages
            # so the device tells us WHERE motion is lost: raw VO -> after the
            # inertial filter -> after the BA/VIO/SLAM correction (displayed).
            diag_vo_path = 0.0     # sum |raw vo step|  (camera-optical)
            diag_filt_path = 0.0   # sum |filter step|  (pre-correction)
            diag_disp_path = 0.0   # sum |displayed step| (post-correction+grav)
            diag_prev_vo = None
            diag_prev_filt = None
            diag_prev_disp = None
            diag_vo_fail = 0       # frames where f2f VO did not solve motion
            diag_vo_n = 0          # frames counted for the fail rate
            diag_corr_frz = 0      # frames the loose correction was FROZEN (fast)
            diag_corr_mag = 0.0    # latest |C_applied translation| (m)
            accel_n = 0
            accel_used = 0
            last_tilt_log = t0
            accel_ema: np.ndarray | None = None
            grav_corr = np.eye(3)   # stateful display-frame gravity correction

            # Per-frame inertial translation filter (predict with the
            # accelerometer, correct with the vision displacement). It owns the
            # displayed translation for ALL live backends. ``prev_vo_t`` is the
            # previous f2f world position used to form the vision displacement
            # measurement; ``prev_frame_ts`` is the device-clock timestamp of the
            # previous frame used for the integration dt.
            tfilt = InertialTranslationFilter()
            tfilt.reset(vo.pose[:3, 3].copy())
            prev_vo_t: np.ndarray | None = None
            prev_frame_ts: float | None = None

            # --- yaw-in-place phantom-translation diagnostics (vio backend) ---
            # Accumulate, over a ~1 s window: mean gyro angular speed, the f2f
            # per-frame translation that occurs WHILE rotating fast (the phantom
            # drift proxy), and the min t_trust / max vision-gyro disagreement so
            # we can see on the device whether the translation damping is even
            # firing during a spin.
            last_vio_log = t0
            vio_gyro_deg = 0.0      # sum of per-frame gyro rotation (deg)
            vio_t_all = 0.0         # sum of f2f per-frame translation (m)
            vio_t_rot = 0.0         # ditto, but only on fast-rotation frames
            vio_disagree_max = 0.0
            vio_ttrust_min = 1.0
            vio_rtrust_min = 1.0
            vio_n = 0
            prev_vo_pos: np.ndarray | None = None
            prev_disp_pos: np.ndarray | None = None
            vio_disp_rot = 0.0      # displayed (post-correction) t during rot
            vio_C_t = 0.0           # VIO worker correction translation magnitude
            vio_fail_n = 0          # vision-failure frames per window
            vio_coast_n = 0         # of those, velocity-coasted frames

            # Gyro complementary-fusion state. ``gyro_bias`` is the mean gyro over
            # the static startup window (rad/s, IMU frame); each frame we
            # integrate the gyro samples drained from the IMU queue into an
            # inter-frame rotation and hand it to ``vo.process`` as the rotation
            # prior. ``so3_exp`` is the same exponential map the offline
            # GyroPreintegrator uses, so live and offline share one convention.
            from ours.lib.imu.imu import so3_exp, so3_log
            gyro_bias = (self._gyro_bias if self._gyro_bias is not None
                         else np.zeros(3))
            gyro_last_ts: float | None = None

            # Sequence-number pairing buffers for our SGM depth: the raw left
            # and raw right come from SEPARATE camera outputs, so draining each
            # queue independently to "latest" can hand SGM a MISMATCHED pair
            # (left frame N vs right frame N-1/2) -> the disparity is then
            # garbage and PnP loses tracking. The cameras are hardware
            # frame-synced (shared sequence numbers), so we pair strictly by
            # sequence: stash by seq, only match a left+right that share one.
            pend_l: dict[int, object] = {}
            pend_r: dict[int, object] = {}

            def _seq(msg) -> int:
                try:
                    return int(msg.getSequenceNum())
                except Exception:
                    return -1

            while not self._stop.is_set() and p.isRunning():
                # Drain each queue to its most recent frame; drop the backlog so
                # that if anything briefly stalls we skip stale frames instead of
                # falling progressively further behind (stays real time).
                ld = q_left.tryGet()
                _backlog = 0          # how many left frames we DISCARD this iter
                while True:
                    nxt = q_left.tryGet()
                    if nxt is None:
                        break
                    ld = nxt
                    _backlog += 1
                # Pair left+right by sequence number (see pend_l/pend_r). We
                # already drained q_left above; stash every drained left and
                # every available right by seq, then take the NEWEST seq that
                # exists in both. Older partials are dropped so the buffers
                # never grow. This guarantees SGM always gets a true stereo
                # pair from the same capture instant.
                rd = None
                if ld is not None:
                    pend_l[_seq(ld)] = ld
                while True:
                    nxt = q_right.tryGet()
                    if nxt is None:
                        break
                    pend_r[_seq(nxt)] = nxt
                common = pend_l.keys() & pend_r.keys()
                if common:
                    seq = max(common)
                    ld = pend_l.pop(seq)
                    rd = pend_r.pop(seq)
                    # Drop any partials older than the one we just consumed.
                    for k in [k for k in pend_l if k < seq]:
                        pend_l.pop(k, None)
                    for k in [k for k in pend_r if k < seq]:
                        pend_r.pop(k, None)
                else:
                    # Bound the buffers if one stream stalls (keep newest 8).
                    for buf in (pend_l, pend_r):
                        if len(buf) > 8:
                            for k in sorted(buf)[:-8]:
                                buf.pop(k, None)
                    ld = None       # no matched pair yet this iteration
                if ld is None or rd is None:
                    # No matched stereo pair this iteration; wait for one.
                    time.sleep(0.002)
                    continue
                _iter_t0 = time.monotonic()
                diag_backlog_sum += _backlog
                diag_recv += 1 + _backlog

                gray = ld.getCvFrame()
                if gray.ndim == 3:
                    # BGR -> luminance (Rec.601), library-free; live display only
                    gray = (gray[..., 0] * 0.114 + gray[..., 1] * 0.587
                            + gray[..., 2] * 0.299).astype(np.uint8)
                # Our from-scratch depth: SGM on the RAW left + RAW right. The
                # matcher rectifies BOTH frames internally and returns metric
                # depth (float32 m, 0 = invalid). No chip StereoDepth involved.
                # CRITICAL: the depth is on the RECTIFIED-left grid, so we must
                # also TRACK on the rectified left (returned here) -- feeding the
                # raw left to vo.process while depth is rectified reads every
                # feature's depth at the wrong pixel (rectification warp, several
                # px near the edges) and degrades PnP. dense_depth_rectified_left
                # rectifies the left exactly once and reuses it, so it costs the
                # same as dense_depth.
                right = rd.getCvFrame()
                if right.ndim == 3:
                    right = (right[..., 0] * 0.114 + right[..., 1] * 0.587
                             + right[..., 2] * 0.299).astype(np.uint8)
                _sgm_t0 = time.monotonic()
                gray, depth_m = matcher.dense_depth_rectified_left(gray, right)
                diag_sgm_ms += (time.monotonic() - _sgm_t0) * 1e3


                # Drain the IMU queue ONCE per frame, BEFORE odometry, so we have
                # this frame's gyro rotation prior ready for PnP. We both (a)
                # average the accelerometer (gravity leveling, below) and (b)
                # integrate the gyro into an inter-frame rotation. Averaging the
                # ~10 accel samples/frame rejects per-sample noise; integrating
                # every gyro sample with its own dt preserves fast rotation.
                acc_sum = np.zeros(3)
                acc_cnt = 0
                R_imu_accum = np.eye(3)
                gyro_cnt = 0
                imsg = q_imu.tryGet()
                while imsg is not None:
                    for pkt in imsg.packets:
                        a = pkt.acceleroMeter
                        v = (a.x, a.y, a.z)
                        # Reject NaN/inf sentinel packets: a single bad sample
                        # would poison the EMA permanently (it never recovers from
                        # NaN), which then corrupts the pose into NaN forever.
                        if np.all(np.isfinite(v)):
                            acc_sum += v
                            acc_cnt += 1
                        g = pkt.gyroscope
                        w = np.array([g.x, g.y, g.z], dtype=np.float64)
                        if np.all(np.isfinite(w)):
                            try:
                                ts = g.getTimestampDevice().total_seconds()
                            except Exception:
                                ts = None
                            if ts is not None:
                                if gyro_last_ts is not None:
                                    dt = ts - gyro_last_ts
                                    if 0.0 < dt < 0.1:   # skip gaps/duplicates
                                        R_imu_accum = R_imu_accum @ so3_exp(
                                            (w - gyro_bias) * dt)
                                        gyro_cnt += 1
                                gyro_last_ts = ts
                                # Tight-coupled VIO keeps every raw sample
                                # (cam frame, no bias subtraction — the bias is
                                # handled inside preintegration) timestamped on
                                # the same device clock as the keyframes.
                                if (vio_state is not None
                                        and np.all(np.isfinite(v))):
                                    vio_imu_ts.append(int(ts * 1e9))
                                    vio_imu_gyro.append(R_imu_cam @ w)
                                    vio_imu_accel.append(
                                        R_imu_cam @ np.asarray(v, float))
                    imsg = q_imu.tryGet()
                accel_raw = None if acc_cnt == 0 else R_imu_cam @ (acc_sum / acc_cnt)

                # Inter-frame gyro rotation in the camera frame (prev<-cur
                # convention, matching GyroPreintegrator). None until we have a
                # spanned interval, so the very first frame stays pure vision.
                R_prior = (R_imu_cam @ R_imu_accum @ R_imu_cam.T
                           if gyro_cnt > 0 else None)

                # Per-frame interval (device clock) — needed BEFORE odometry so
                # the physical translation-speed clamp can bound this frame's
                # solved translation. Falls back to ~1/30 s on the first frame.
                try:
                    frame_ts = ld.getTimestampDevice().total_seconds()
                except Exception:
                    frame_ts = time.monotonic()
                dt_f = (frame_ts - prev_frame_ts
                        if prev_frame_ts is not None else 1.0 / 30.0)
                prev_frame_ts = frame_ts

                # IMU motion gate for the odometry's low-inlier translation
                # freeze. A textureless wall and a motion-blurred shake both
                # starve PnP of inliers, but only the wall should freeze (a shake
                # is REAL motion -- freezing it pins ours-ba/slam in place while
                # ours keeps moving, the user's "rung lắc -> đứng ì" symptom). The
                # accelerometer residual vs its EMA separates them: ~0 at rest
                # (wall), large under shake. Computed from the PREVIOUS frame's
                # EMA (updated post-pose below); fine since the EMA is slow.
                imu_moving = bool(
                    accel_raw is not None and accel_ema is not None
                    and float(np.linalg.norm(accel_raw - accel_ema))
                    > _REST_MOTION_THRESH)

                _vo_t0 = time.monotonic()
                vo.process(gray, depth_m, R_prior=R_prior,  # camera-optical world
                           dt_s=dt_f, imu_moving=imu_moving)
                diag_vo_ms += (time.monotonic() - _vo_t0) * 1e3


                # --- accelerometer leveling, gated on the camera being at rest --
                # We only trust accel for leveling when the camera is actually at
                # rest, because a magnitude gate cannot reject lateral linear
                # acceleration (a sideways push barely changes |accel| yet tilts
                # the gravity direction). The motion signal is the residual of the
                # batch-averaged sample against its EMA: tiny at rest, large during
                # any translation/rotation. When moving we skip leveling and let
                # vision hold the attitude; when still, accel pulls roll/pitch back
                # to true gravity.
                accel_cam = None
                at_rest = False
                motion = 0.0
                if accel_raw is not None:
                    if accel_ema is None:
                        accel_ema = accel_raw.copy()
                    else:
                        accel_ema += 0.2 * (accel_raw - accel_ema)
                    accel_cam = accel_ema
                    motion = float(np.linalg.norm(accel_raw - accel_ema))
                    at_rest = motion < _REST_MOTION_THRESH
                    # Track the true gravity magnitude from the at-rest samples.
                    # The startup g_ref can be captured during motion (it read
                    # 10.15 vs a real ~8.9 here), which would skew the inner
                    # magnitude gate; refresh it whenever we are actually still.
                    if at_rest:
                        na = float(np.linalg.norm(accel_cam))
                        if vo._g_ref is None:
                            vo._g_ref = na
                        else:
                            vo._g_ref += 0.05 * (na - vo._g_ref)

                # Level the f2f world frame too (only at rest), so the BA map is
                # fed gravity-consistent poses over the long term.
                if accel_cam is not None and at_rest:
                    vo.correct_tilt(accel_cam)

                pose = vo.pose.copy()  # camera-optical world

                # --- per-frame inertial translation filter (predict+correct) ---
                # Replace the f2f-accumulated translation with the inertial
                # filter's: predict the world position with the accelerometer,
                # correct it with the vision displacement (down-weighted while
                # rotating fast). This owns translation for every live backend;
                # the rotation below is still the gyro-fused VO attitude. The
                # filter runs in the SAME camera-optical world as ``vo.pose``, so
                # the BA/SLAM/VIO world-frame correction and ``grav_corr``
                # leveling apply downstream exactly as before.
                R_wc = vo.pose[:3, :3]
                gyro_deg = (float(np.degrees(np.linalg.norm(
                    so3_log(R_imu_accum)))) if gyro_cnt > 0 else 0.0)
                vo_t_now = vo.pose[:3, 3].copy()
                vis_ok = bool(vo.last_info.get("ok", False))
                diag_vo_n += 1
                if not vis_ok:
                    diag_vo_fail += 1
                if prev_vo_t is not None and vis_ok:
                    dp_vis = vo_t_now - prev_vo_t
                else:
                    dp_vis = None
                prev_vo_t = vo_t_now
                filt_p = tfilt.step(dt_f, R_wc, accel_cam, dp_vis, gyro_deg)
                # accounting: raw VO step + filter step (pre-correction)
                if diag_prev_vo is not None:
                    diag_vo_path += float(np.linalg.norm(vo_t_now - diag_prev_vo))
                diag_prev_vo = vo_t_now.copy()
                if diag_prev_filt is not None:
                    diag_filt_path += float(np.linalg.norm(filt_p - diag_prev_filt))
                diag_prev_filt = filt_p.copy()
                pose[:3, 3] = filt_p

                # Set True when a loop-closure correction slews the pose this
                # frame (a teleport while the camera is ~still). Coloured
                # distinctly in the path so the jump reads as a map correction.
                teleport = False


                if ba_state is not None:
                    # Submit a keyframe snapshot every kf_every frames. Drop if
                    # the worker is busy (non-blocking) — never stall the loop.
                    kf_count += 1
                    if kf_count >= ba_state["kf_every"]:
                        kf_count = 0
                        st = vo.frontend.tracks
                        ba_state["submit"](
                            np.linalg.inv(pose),          # T_cw (world->cam)
                            st.ids.copy(), st.points.copy(),
                            depth_m.copy(),
                            # Only hand BA a gravity measurement when the camera
                            # is at rest; during motion lateral acceleration would
                            # bias the gravity direction, so the keyframe carries
                            # no gravity prior (None) and BA levels it from its
                            # at-rest neighbours in the window.
                            accel_cam.copy() if (accel_cam is not None
                                                 and at_rest) else None,
                        )
                    # Pull the latest correction from the worker (drain to last).
                    newC = ba_state["poll"]()
                    if newC is not None:
                        C_target = newC
                    # Ease toward the target so the correction never snaps, and
                    # rate-limit the step so a big BA jump cannot yank the marker.
                    # Speed-gate the slew: freeze the correction while pushing fast
                    # so the marker rides the rigid correction and tracks the full
                    # distance (disp/filt=1), and only fold in BA drift when slow
                    # (see _CORR_FREEZE_SPEED).
                    slew = (0.15 if float(np.linalg.norm(tfilt.v))
                            < _CORR_FREEZE_SPEED else 0.0)
                    if slew == 0.0:
                        diag_corr_frz += 1
                    C_applied = _ease_se3(C_applied, C_target, slew,
                                          _CORR_MAX_STEP_T, _CORR_MAX_STEP_R)
                    diag_corr_mag = float(np.linalg.norm(C_applied[:3, 3]))
                    pose = C_applied @ pose

                if vio_state is not None:
                    # Same cadence as BA, but each keyframe also carries its raw
                    # IMU segment (cam frame) since the previous submit. Snapshot
                    # the accumulated buffers and clear them so the next segment
                    # spans exactly this->next keyframe.
                    kf_count += 1
                    if kf_count >= vio_state["kf_every"]:
                        kf_count = 0
                        st = vo.frontend.tracks
                        try:
                            frame_ts_ns = int(
                                ld.getTimestampDevice().total_seconds() * 1e9)
                        except Exception:
                            frame_ts_ns = vio_imu_ts[-1] if vio_imu_ts else 0
                        seg = (np.asarray(vio_imu_ts, np.int64),
                               np.asarray(vio_imu_gyro, np.float64),
                               np.asarray(vio_imu_accel, np.float64))
                        vio_state["submit"](
                            np.linalg.inv(pose),          # T_cw (world->cam)
                            st.ids.copy(), st.points.copy(),
                            depth_m.copy(), frame_ts_ns, seg)
                        vio_imu_ts.clear()
                        vio_imu_gyro.clear()
                        vio_imu_accel.clear()
                    # Pull + clamp + ease the tight-coupled correction every
                    # frame so the accelerometer-tied VIO refinement actually
                    # reaches the display (it must move during MOTION -- that is
                    # when it matters). A blown-up correction (optimiser diverged
                    # on noisy accel) is rejected by the translation clamp; the
                    # ease then smooths whatever survives so it never snaps.
                    newC = vio_state["poll"]()
                    if (newC is not None and
                            np.linalg.norm(newC[:3, 3]) < _VIO_CORR_MAX_T):
                        C_target = newC
                    C_applied = _ease_se3(C_applied, C_target, 0.15,
                                          _CORR_MAX_STEP_T, _CORR_MAX_STEP_R)
                    pose = C_applied @ pose

                if slam_state is not None:
                    # UI "clear keyframes": wipe the map worker-side and snap our
                    # local loop-closure correction + overlay buffers back to
                    # empty so the displayed path detaches from the old map at
                    # once (no eased slew back through a stale correction).
                    if self._slam_reset.is_set():
                        self._slam_reset.clear()
                        slam_state["reset_map"]()
                        C_target = np.eye(4)
                        C_applied = np.eye(4)
                        kf_count = 0
                        with self._slam_lock:
                            self._slam_kf_ned = np.zeros((0, 3), np.float32)
                            self._slam_match_ned = np.zeros((0, 3), np.float32)
                            self._slam_loop_ned = []
                            self._slam_flash_id += 1
                    # Hand a keyframe (raw f2f pose + image + depth) to the SLAM
                    # map every kf_every frames; drop if the worker is still busy
                    # on the previous one (non-blocking) — never stall the loop.
                    kf_count += 1
                    if kf_count >= slam_state["kf_every"]:
                        kf_count = 0
                        slam_state["submit"](pose.copy(), gray.copy(),
                                             depth_m.copy())
                    # Pull the latest loop-closure correction (drain to last) and
                    # ease it on, exactly like the BA correction. Rate-limited so
                    # a big pose-graph jump after a loop bleeds in smoothly
                    # instead of teleporting the live marker.
                    newC = slam_state["poll"]()
                    if newC is not None:
                        C_target = newC
                    C_prev = C_applied
                    slew = (0.15 if float(np.linalg.norm(tfilt.v))
                            < _CORR_FREEZE_SPEED else 0.0)
                    C_applied = _ease_se3(C_applied, C_target, slew,
                                          _CORR_MAX_STEP_T, _CORR_MAX_STEP_R)
                    # Teleport displacement = how far THIS pose moves purely from
                    # the correction slewing (same vo pose, only the correction
                    # changed). Real camera motion never enters this delta, so a
                    # non-trivial value means the loop-closure correction is
                    # dragging the displayed point back onto the remembered place.
                    corr_step = float(np.linalg.norm(
                        (C_applied @ pose)[:3, 3] - (C_prev @ pose)[:3, 3]))
                    teleport = corr_step > 0.01   # 1 cm/frame from correction
                    pose = C_applied @ pose

                # FINAL display leveling -- accel is the "trum cuoi" (last word)
                # on the shown roll/pitch. The Phase-4 in-BA gravity prior keeps
                # the keyframe MAP from tilt-drifting, but the BA correction
                # ``C_applied`` can still leave a residual tilt on the displayed
                # ``C_applied @ vo.pose`` (reprojection in a low-parallax view is
                # blind to absolute tilt, so the latest keyframe attitude need not
                # be perfectly level). So we keep a STATEFUL world-frame correction
                # ``grav_corr`` and apply ``pose = grav_corr @ pose`` as the last
                # step. At rest we nudge ``grav_corr`` by a small adaptive gain
                # toward cancelling the residual tilt (stateful => a partial gain
                # converges, unlike a partial gain on the freshly-rebuilt pose);
                # when moving we freeze it so nothing jumps at the rest-gate
                # boundary. Because BA now also levels the map, grav_corr stays
                # small. Re-orthonormalise (SVD project onto SO(3)) each update so
                # the repeated matrix products never drift out of SO(3) into NaN.
                # yaw is untouched (level_attitude only rotates about horizontal).
                R_pre = pose[:3, :3]
                R_disp = grav_corr @ R_pre
                tilt_deg = 0.0
                if accel_cam is not None and at_rest:
                    R_lvl, used, tilt_deg = level_attitude(
                        R_disp, accel_cam, g_ref=vo._g_ref,
                        alpha=0.05, alpha_max=0.25)
                    if used:
                        grav_corr = (R_lvl @ R_disp.T) @ grav_corr
                        U, _, Vt = np.linalg.svd(grav_corr)
                        grav_corr = U @ Vt
                        if np.linalg.det(grav_corr) < 0:
                            U[:, -1] *= -1.0
                            grav_corr = U @ Vt
                        R_disp = grav_corr @ R_pre
                # ``grav_corr`` is a world-frame rotation, so it MUST be applied
                # to BOTH the attitude and the position trajectory -- otherwise
                # the triad rotates by the leveling angle but the path does not,
                # and camera motion stops lining up with the body axes (moving
                # "forward" no longer tracks the red arrow; the symptom of
                # only-rotate-attitude). Rotating position by the same grav_corr
                # keeps displacement and triad consistent.
                pose[:3, :3] = R_disp
                pose[:3, 3] = grav_corr @ pose[:3, 3]

                # Refresh the SLAM overlay for the viewer when the worker has a
                # new map snapshot. Apply the SAME world-frame transform as the
                # displayed path (grav_corr in optical, then optical->NED) so the
                # keyframe dots and loop links line up with the trajectory.
                if slam_state is not None:
                    ov = slam_state["overlay"]()
                    if ov is not None:
                        kf_opt, match_opt, loop_pairs, has_new = ov

                        def _to_ned(p):
                            return (_M_OPT_TO_NED @ (grav_corr @ p)).astype(
                                np.float32)

                        kf_ned = (np.array([_to_ned(p) for p in kf_opt],
                                           dtype=np.float32)
                                  if len(kf_opt) else
                                  np.zeros((0, 3), np.float32))
                        with self._slam_lock:
                            self._slam_kf_ned = kf_ned
                            # Only refresh the flash (matched dot + teleport
                            # link) when a NEW loop actually closed; otherwise
                            # leave the previous flash to fade out in the viewer.
                            if has_new:
                                self._slam_match_ned = (
                                    np.array([_to_ned(p) for p in match_opt],
                                             dtype=np.float32)
                                    if len(match_opt) else
                                    np.zeros((0, 3), np.float32))
                                self._slam_loop_ned = [
                                    np.stack([_to_ned(a), _to_ned(b)])
                                    for a, b in loop_pairs]
                                self._slam_flash_id += 1

                # Accelerometer-ONLY attitude (gravity-leveled, yaw=0) for live
                # side-by-side comparison in the UI -- computed every frame
                # regardless of the rest gate, so we can see what accel "wants"
                # even when leveling is being withheld.
                accel_q_ned = None
                if accel_cam is not None:
                    R0_opt = gravity_aligned_R0(accel_cam)        # cam->world, yaw=0
                    R_acc_ned = _M_OPT_TO_NED @ R0_opt @ _P_OPT_TO_FRD
                    accel_q_ned = _rot_to_quat_wxyz(R_acc_ned)

                # Rate-limited diagnostics so we can see, on the device, whether
                # the rest gate is firing and what accel is reporting.
                if accel_cam is not None:
                    accel_n += 1
                    if at_rest:
                        accel_used += 1
                    if time.monotonic() - last_tilt_log >= 1.0:
                        rate = accel_used / max(accel_n, 1)
                        ar, ap, _ = (np.degrees(
                            quat_to_rpy(accel_q_ned)) if accel_q_ned is not None
                            else (0.0, 0.0, 0.0))
                        print(f"[ours-vio] accel r/p={ar:+5.1f}/{ap:+5.1f} "
                              f"tilt={tilt_deg:4.1f} at_rest={100*rate:3.0f}% "
                              f"motion={motion:.2f} n={acc_cnt} "
                              f"|a|={np.linalg.norm(accel_cam):.2f} "
                              f"g_ref={vo._g_ref or 0:.2f}")
                        accel_n = 0
                        accel_used = 0
                        last_tilt_log = time.monotonic()

                # Inertial-filter diagnostics (vio backend only). Reports the
                # honest separation: the RAW vision displacement during fast
                # rotation (the phantom the front-end emits) versus the DISPLAYED
                # (filter-driven) displacement during the same frames -- the
                # filter should reject the phantom -- plus the filter speed and
                # vision-failure count. All quantities trace to real outputs.
                if self.backend == "vio":
                    vo_pos = vo.pose[:3, 3]
                    disp_pos = pose[:3, 3]            # filter-driven position
                    if prev_vo_pos is not None:
                        tstep = float(np.linalg.norm(vo_pos - prev_vo_pos))
                        vio_t_all += tstep
                        if gyro_deg > 0.3:      # fast-rotation frame
                            vio_t_rot += tstep
                    if prev_disp_pos is not None and gyro_deg > 0.3:
                        vio_disp_rot += float(np.linalg.norm(
                            disp_pos - prev_disp_pos))
                    prev_vo_pos = vo_pos.copy()
                    prev_disp_pos = disp_pos.copy()
                    vio_C_t = max(vio_C_t,
                                  float(np.linalg.norm(C_applied[:3, 3])))
                    vio_gyro_deg += gyro_deg
                    if not vis_ok:
                        vio_fail_n += 1
                    # filter speed (m/s) max over the window
                    vio_rtrust_min = min(
                        vio_rtrust_min, float(np.linalg.norm(tfilt.v)))
                    vio_n += 1
                    now_v = time.monotonic()
                    win = now_v - last_vio_log
                    if win >= 1.0:
                        rate = vio_gyro_deg / max(win, 1e-6)   # deg/s
                        print(f"[ours-vio] filt-diag gyro={rate:5.1f}deg/s "
                              f"vis_t@rot={vio_t_rot*100:5.1f}cm "
                              f"disp_t@rot={vio_disp_rot*100:5.1f}cm "
                              f"C_t={vio_C_t*100:5.1f}cm "
                              f"fail={vio_fail_n} "
                              f"v={np.linalg.norm(tfilt.v):.2f}m/s")
                        vio_gyro_deg = vio_t_all = vio_t_rot = 0.0
                        vio_disp_rot = 0.0
                        vio_C_t = 0.0
                        vio_fail_n = 0
                        vio_coast_n = 0
                        vio_disagree_max = 0.0
                        vio_ttrust_min = 1.0
                        vio_rtrust_min = 1.0
                        last_vio_log = now_v




                pos_opt = pose[:3, 3]
                R_opt = pose[:3, :3]
                if diag_prev_disp is not None:
                    diag_disp_path += float(np.linalg.norm(pos_opt - diag_prev_disp))
                diag_prev_disp = pos_opt.copy()

                pos_ned = _M_OPT_TO_NED @ pos_opt
                # Body axes [forward, right, down] in NED for the viewer triad.
                R_ned = _M_OPT_TO_NED @ R_opt @ _P_OPT_TO_FRD
                q_ned = _rot_to_quat_wxyz(R_ned)

                now = time.monotonic()
                t = now - t0
                if prev_t is None:
                    vel_ned = np.zeros(3)
                else:
                    dt = max(now - prev_t, 1e-6)
                    vel_ned = (pos_ned - prev_pos_ned) / dt
                prev_pos_ned = pos_ned
                prev_t = now

                ok = bool(vo.last_info.get("ok", False))
                _emit_t0 = time.monotonic()
                self._emit(Pose(
                    t=t,
                    pos_ned=pos_ned,
                    vel_ned=vel_ned,
                    quat_wxyz=q_ned,
                    tracking_ok=ok,
                    teleport=teleport,
                    accel_quat_wxyz=accel_q_ned,
                ))
                diag_emit_ms += (time.monotonic() - _emit_t0) * 1e3
                diag_loop_ms += (time.monotonic() - _iter_t0) * 1e3
                diag_iter += 1

                frames += 1
                if now - last_fps_t >= 0.5:
                    self.fps = frames / (now - last_fps_t)
                    win = now - last_fps_t
                    n = max(diag_iter, 1)
                    tag = f"ours-{self.backend}"
                    print(f"[{tag}] thru recv={diag_recv/win:4.1f}fps "
                          f"proc={frames/win:4.1f}fps drop={diag_backlog_sum} "
                          f"sgm={diag_sgm_ms/n:4.1f}ms vo={diag_vo_ms/n:4.1f}ms "
                          f"emit={diag_emit_ms/n:4.1f}ms "
                          f"loop={diag_loop_ms/n:4.1f}ms")
                    # Translation accounting: where motion is lost between the raw
                    # VO, the inertial filter and the displayed (post-correction)
                    # path, plus the current filter speed. filt/vo<1 -> the filter
                    # is undershooting (the "đẩy nhanh rồi ì lại" symptom); |v| is
                    # the live speed so a stall reads as |v| collapsing to ~0 while
                    # the camera is still moving.
                    print(f"[{tag}] path  vo={diag_vo_path*1000:6.0f}mm "
                          f"filt={diag_filt_path*1000:6.0f}mm "
                          f"disp={diag_disp_path*1000:6.0f}mm "
                          f"(filt/vo={diag_filt_path/max(diag_vo_path,1e-6):.2f} "
                          f"disp/filt={diag_disp_path/max(diag_filt_path,1e-6):.2f}) "
                          f"|v|={np.linalg.norm(tfilt.v):.2f}m/s "
                          f"klt={fe_cfg.win_size}/{fe_cfg.max_level}"
                          f"{'+jit' if HAVE_NUMBA else ''} "
                          f"vofail={100*diag_vo_fail/max(diag_vo_n,1):2.0f}% "
                          f"corr={diag_corr_mag*1000:4.0f}mm "
                          f"frz={100*diag_corr_frz/max(diag_iter,1):2.0f}%")
                    frames = 0
                    last_fps_t = now
                    diag_backlog_sum = diag_recv = 0
                    diag_sgm_ms = diag_vo_ms = diag_emit_ms = diag_loop_ms = 0.0
                    diag_iter = 0
                    diag_vo_path = diag_filt_path = diag_disp_path = 0.0
                    diag_vo_fail = diag_vo_n = 0
                    diag_corr_frz = 0

            if ba_state is not None:
                ba_state["stop"].set()
                ba_state["event"].set()
                ba_state["thread"].join(timeout=1.0)

            if slam_state is not None:
                slam_state["stop"].set()
                slam_state["event"].set()
                slam_state["thread"].join(timeout=1.0)

    # ----------------------------------------------------------------------- #
    def _collect_startup_accel(self, q_imu, R_imu_cam: np.ndarray,
                               window_s: float = 0.4,
                               timeout_s: float = 2.0) -> np.ndarray | None:
        """Average the accelerometer over a short static startup window.

        Returns the mean specific-force vector rotated into the camera optical
        frame (ready for :meth:`RGBDVisualOdometry.align_to_gravity`), or ``None``
        if no IMU samples arrived within ``timeout_s`` (older device / no IMU) —
        in which case the caller falls back to an identity (unleveled) start.

        As a side effect, the mean gyro over the same static window is stored in
        ``self._gyro_bias`` (rad/s, IMU frame). The device is assumed motionless
        at startup, so this is the gyro zero-rate offset; subtracting it before
        integration keeps the rotation prior from drifting when the camera is
        still. Left ``None`` if no IMU samples arrived.
        """
        samples: list[np.ndarray] = []
        gyro: list[np.ndarray] = []
        t_start = time.monotonic()
        t_first: float | None = None
        while time.monotonic() - t_start < timeout_s:
            msg = q_imu.tryGet()
            if msg is None:
                time.sleep(0.005)
                continue
            for pkt in msg.packets:
                a = pkt.acceleroMeter
                samples.append(np.array([a.x, a.y, a.z], dtype=np.float64))
                g = pkt.gyroscope
                w = np.array([g.x, g.y, g.z], dtype=np.float64)
                if np.all(np.isfinite(w)):
                    gyro.append(w)
            if t_first is None:
                t_first = time.monotonic()
            elif time.monotonic() - t_first >= window_s:
                break
        if gyro:
            self._gyro_bias = np.mean(gyro, axis=0)
        if not samples:
            return None
        accel_imu = np.mean(samples, axis=0)
        return R_imu_cam @ accel_imu

    # ----------------------------------------------------------------------- #
    def _start_ba_worker(self, K: np.ndarray) -> dict:
        """Spawn the background sliding-window BA worker (separate PROCESS).

        Returns a state dict with ``submit(T_cw, ids, pts, depth_m, accel)`` and
        ``poll() -> C | None``. The worker keeps the BA map in the *raw f2f*
        world frame (it is always fed raw f2f poses), so the published
        correction ``C = inv(T_ba) @ T_cw`` maps that frame onto the BA-refined
        one. The BA refine is mostly pure-Python (only the inner solve releases
        the GIL), so an in-thread worker stole ~17-30% of the read loop's GIL and
        starved the frame reader on a fast push (dropped frames -> the
        "đẩy nhanh rồi ì lại" undershoot). It therefore runs out-of-process; see
        :mod:`ours.legacy.ba_worker_proc` for the rationale and the identical
        ``submit``/``poll``/``stop``/``event``/``thread`` interface this keeps.
        """
        from ours.lib import WindowedConfig
        from ours.lib.backend.bundle import BAConfig
        from .ba_worker_proc import start_ba_process

        # use_gravity=True adds the accelerometer leveling prior INSIDE the
        # sliding-window BA, so the optimised map keeps its roll/pitch pinned to
        # real gravity (no display-side correction needed). Only at-rest accel
        # samples are submitted per keyframe (see the read loop), so a moving
        # keyframe simply carries no gravity constraint.
        # use_vo_trans_prior=True feeds the frame-to-frame PnP relative
        # translation back as a soft scale anchor: it stops the windowed BA from
        # collapsing the forward baseline on a low-parallax push (the "đi một
        # đoạn rồi ì lại" undershoot). Measured on the gold suite (live SGM
        # depth): push_straight Sim3 scale 0.39->0.97, push_fwdback 0.30->0.78,
        # ATE on looping/straight motion unchanged. See BAConfig.use_vo_trans_prior.
        cfg = WindowedConfig(window=self.ba_window, kf_every=self.ba_kf_every,
                             use_marg=self.ba_marg,
                             ba=BAConfig(max_iters=self.ba_iters,
                                         huber_px=self.res.ba_huber_px(),
                                         use_gravity=True,
                                         use_vo_trans_prior=True))
        return start_ba_process(K, cfg)

    # ----------------------------------------------------------------------- #
    def _start_vio_worker(self, K: np.ndarray, R_imu_cam: np.ndarray) -> dict:
        """Spawn the background tight-coupled VIO thread.

        Returns a state dict with ``submit(T_cw, ids, pts, depth, ts_ns, seg)``
        and ``poll() -> C | None``. Mirrors :meth:`_start_ba_worker` but the map
        is a :class:`WindowedVIOMap`, so every keyframe also carries its raw IMU
        segment (``seg = (ts_ns, gyro_cam, accel_cam)``, already rotated into the
        camera frame and spanning the interval since the previous keyframe). The
        joint visual+IMU solve pins translation to real linear acceleration, so
        an in-place yaw can no longer be explained away as drift by slipped
        tracks. The published correction ``C = inv(T_vio) @ T_cw`` maps the raw
        f2f frame onto the VIO-refined one, eased onto the display.

        The dense finite-difference solver is ~100x heavier than the BA core, so
        a small window + few iterations keep each background solve short enough
        that corrections stay reasonably fresh; if a solve is still running when
        the next keyframe arrives, the single-slot pending queue simply drops the
        older request (corrections update less often, never blocking the loop).
        """
        import threading

        from ours.lib.backend.vio_window import (VioConfig, WindowedVIOConfig,
                                      WindowedVIOMap)

        # startup gyro bias is collected in the IMU frame; rotate it into the
        # camera frame to match the cam-frame gyro samples fed to the map.
        bg0 = None
        if getattr(self, "_gyro_bias", None) is not None:
            bg0 = R_imu_cam @ self._gyro_bias

        # Real-time budget for the background solve. The dense finite-difference
        # VIO solve cost scales ~linearly with both the window length and the LM
        # iteration count. Measured on gold (ours/tools/vio_run.py --backend vio,
        # per-solve run_ba timing): window=5/iters=6 costs ~320 ms median, which
        # at the kf_every=5 submit cadence (~350-420 ms between keyframes on this
        # host) leaves the worker busy ~90% of the time -- it saturates the CPU
        # almost continuously and starves the main-loop SGM+KLT (the live
        # ours-vio lag: sgm/vo inflated 2-5x, loop 45-150 ms). Trimming to
        # window=4/iters=3 drops the solve to ~125 ms (~30% duty) and breaks the
        # starvation, with ATE essentially unchanged across the gold sessions
        # (lab_loop 184->190 mm, straight 212->236, quick 112->117; all <0.1%
        # of path). The numbers are documented in
        # /memories/repo and the sweep in the lag investigation.
        cfg = WindowedVIOConfig(window=min(self.ba_window, 4),
                                kf_every=self.ba_kf_every,
                                vio=VioConfig(max_iters=3, lock_tilt=True))
        vio_map = WindowedVIOMap(K, bg0=bg0, cfg=cfg)

        snap_lock = threading.Lock()
        out_lock = threading.Lock()
        event = threading.Event()
        stop = threading.Event()
        state = {
            "event": event,
            "stop": stop,
            "kf_every": cfg.kf_every,
            "_pending": None,
            "_corr": None,
        }

        def submit(T_cw, ids, pts, depth_m, ts_ns, seg):
            with snap_lock:
                state["_pending"] = (T_cw, ids, pts, depth_m, ts_ns, seg)
            event.set()

        def poll():
            with out_lock:
                C = state["_corr"]
                state["_corr"] = None
            return C

        def worker():
            while not stop.is_set():
                event.wait()
                event.clear()
                if stop.is_set():
                    break
                with snap_lock:
                    snap = state["_pending"]
                    state["_pending"] = None
                if snap is None:
                    continue
                T_cw, ids, pts, depth_m, ts_ns, seg = snap
                vio_map.add_keyframe(T_cw, ids, pts, depth_m, ts_ns,
                                     imu_seg=seg)
                post = vio_map.run_ba()
                if post is not None:
                    with out_lock:
                        state["_corr"] = np.linalg.inv(post) @ T_cw

        th = threading.Thread(target=worker, name="OursVIOWorker", daemon=True)
        th.start()
        state["thread"] = th
        state["submit"] = submit
        state["poll"] = poll
        return state

    # ----------------------------------------------------------------------- #
    def _start_slam_worker(self, K: np.ndarray) -> dict:
        """Spawn the background loop-closure SLAM thread.

        Returns a state dict with ``submit(T_wc, gray, depth_m)`` and
        ``poll() -> C | None``. The worker keeps a persistent
        :class:`ours.vio.SlamMap` of every keyframe (in the *raw f2f* world
        frame, so its odometry edges stay self-consistent), recognises revisited
        places and runs pose-graph optimisation when a loop is confirmed. After
        each keyframe it publishes the world-frame correction for the **latest**
        keyframe, ``C = kf_pose[last] · inv(kf_orig[last])``; the read loop eases
        that onto the current f2f pose (the current frame hangs off the latest
        keyframe, so this maps it into the loop-corrected world). Until the first
        loop closes the correction is identity, so the display is plain f2f.

        ORB detection + matching against the growing keyframe set is the slow
        part, which is exactly why it lives on this background thread; the device
        read loop keeps running fast f2f for the display.
        """
        import threading

        from ours.lib import SlamMap
        from ours.lib.loop.slam import SlamConfig

        # Spatial gating keeps loop detection bounded as the map grows so the
        # configured keyframe cadence stays sustainable on the background thread.
        slam = SlamMap(K, SlamConfig(
            loop=self.res.loop(),
            loop_search_radius_m=self.slam_radius_m,
            loop_max_odom_rot_deg=30.0,
            kf_min_trans_m=self.slam_kf_min_trans,
            kf_min_rot_deg=self.slam_kf_min_rot,
            max_keyframes=self.slam_max_kf))

        snap_lock = threading.Lock()
        out_lock = threading.Lock()
        event = threading.Event()
        stop = threading.Event()
        reset = threading.Event()
        state = {
            "event": event,
            "stop": stop,
            "reset": reset,
            "kf_every": self.slam_kf_every,
            "_pending": None,
            "_corr": None,
            "_overlay": None,
        }

        def submit(T_wc, gray, depth_m):
            with snap_lock:
                state["_pending"] = (T_wc, gray, depth_m)
            event.set()

        def reset_map():
            # Ask the worker to forget every keyframe on its next wake.
            reset.set()
            event.set()

        def poll():
            with out_lock:
                C = state["_corr"]
                state["_corr"] = None
            return C

        def overlay():
            with out_lock:
                ov = state["_overlay"]
                state["_overlay"] = None
            return ov

        def worker():
            from ours.lib.loop.posegraph import se3_inv

            while not stop.is_set():
                event.wait()
                event.clear()
                if stop.is_set():
                    break
                if reset.is_set():
                    reset.clear()
                    slam.reset()
                    with snap_lock:
                        state["_pending"] = None
                    with out_lock:
                        # Identity correction (no loop) + an empty overlay marked
                        # "new" so the read loop drops the keyframe dots and the
                        # loop flash on its next refresh.
                        state["_corr"] = np.eye(4)
                        state["_overlay"] = (np.zeros((0, 3)),
                                             np.zeros((0, 3)), [], True)
                    continue
                with snap_lock:
                    snap = state["_pending"]
                    state["_pending"] = None
                if snap is None:
                    continue
                T_wc, gray, depth_m = snap
                events = slam.add_keyframe(T_wc, gray, depth_m)
                if events:
                    slam.optimize()
                    for ev in events:
                        print(f"[ours-slam] loop closed: kf {ev['cur']} <-> "
                              f"{ev['old']} ({ev['inliers']} inliers)")
                # Publish the correction for the latest keyframe (identity until
                # a loop has closed). The current display pose hangs off it.
                last = len(slam.kf_orig) - 1
                C = None
                if last >= 0:
                    C = slam.kf_pose[last] @ se3_inv(slam.kf_orig[last])
                # Snapshot the map for the UI overlay: every keyframe position,
                # the revisited (matched) keyframes to highlight, and the loop
                # links as [cur, old] segments. Positions are the CURRENT
                # (PGO-corrected) keyframe poses in the camera-optical world
                # frame; the read loop maps them to NED. These are real SlamMap
                # outputs, so the dots/links always reflect the actual graph.
                kf_opt = (
                    np.array([p[:3, 3] for p in slam.kf_pose], dtype=np.float64)
                    if slam.kf_pose else np.zeros((0, 3)))
                # Flash ONLY the loops confirmed at THIS keyframe (``events``),
                # i.e. the teleport that just happened -- not the whole history.
                match_opt = (
                    np.array([slam.kf_pose[ev["old"]][:3, 3] for ev in events],
                             dtype=np.float64)
                    if events else np.zeros((0, 3)))
                loop_pairs = [
                    (slam.kf_pose[ev["cur"]][:3, 3].copy(),
                     slam.kf_pose[ev["old"]][:3, 3].copy())
                    for ev in events]
                with out_lock:
                    state["_corr"] = C
                    state["_overlay"] = (kf_opt, match_opt, loop_pairs,
                                         bool(events))

        th = threading.Thread(target=worker, name="OursSlamWorker", daemon=True)
        th.start()
        state["thread"] = th
        state["submit"] = submit
        state["poll"] = poll
        state["overlay"] = overlay
        state["reset_map"] = reset_map
        return state

