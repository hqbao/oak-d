"""Live RGB-D VIO from the OAK-D using *our* from-scratch odometry.

Unlike :mod:`oakd.sources.depthai_vio` (which reads poses out of DepthAI's
built-in ``BasaltVIO`` node), this source runs **our own** frame-to-frame
RGB-D PnP odometry (:class:`oakd.vio.RGBDVisualOdometry`) on the live
rectified-left + depth stream. It exists so we can watch our VIO drive the 3D
viewer in real time and eyeball its quality against Basalt *before* we add
sliding-window bundle adjustment.

Pipeline (same front-end as the recorder):
    Camera CAM_B/CAM_C -> StereoDepth (depthAlign=left)
        -> rectifiedLeft (uint8 gray) + depth (uint16 mm)

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
``oakd/vio/imu.py``), so this live source runs pure vision for simplicity.
"""
from __future__ import annotations

import time

import numpy as np

from ..pose import Pose
from ..vio import OdometryConfig, RGBDVisualOdometry, level_attitude
from .base import PoseSource


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
# its drone-mount pose via ``tools/measure_mount_attitude.py`` (avg of 760 raw
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


def _ease_se3(C_cur: np.ndarray, C_tgt: np.ndarray, alpha: float) -> np.ndarray:
    """Move ``C_cur`` a fraction ``alpha`` toward ``C_tgt`` (smooth correction).

    Rotation eases along the geodesic (scaled axis-angle); translation eases
    linearly. Keeps the applied correction continuous so BA updates never snap
    the displayed trajectory.
    """
    R_cur, R_tgt = C_cur[:3, :3], C_tgt[:3, :3]
    dR = R_tgt @ R_cur.T
    ang = np.arccos(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
    out = np.eye(4)
    if ang < 1e-8:
        out[:3, :3] = R_tgt
    else:
        axis = np.array([dR[2, 1] - dR[1, 2],
                         dR[0, 2] - dR[2, 0],
                         dR[1, 0] - dR[0, 1]]) / (2.0 * np.sin(ang))
        a = alpha * ang
        K_ = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
        R_step = np.eye(3) + np.sin(a) * K_ + (1.0 - np.cos(a)) * (K_ @ K_)
        out[:3, :3] = R_step @ R_cur
    out[:3, 3] = (1.0 - alpha) * C_cur[:3, 3] + alpha * C_tgt[:3, 3]
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
    """

    def __init__(self, width: int = 640, height: int = 400, fps: int = 20,
                 backend: str = "f2f") -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.cam_fps = int(fps)
        self.backend = backend

    def _run(self) -> None:
        import cv2
        import depthai as dai  # lazy: --source fake works without depthai/device

        left_socket = dai.CameraBoardSocket.CAM_B
        right_socket = dai.CameraBoardSocket.CAM_C

        with dai.Pipeline() as p:
            left = p.create(dai.node.Camera).build(left_socket, sensorFps=self.cam_fps)
            right = p.create(dai.node.Camera).build(right_socket, sensorFps=self.cam_fps)
            stereo = p.create(dai.node.StereoDepth)
            imu = p.create(dai.node.IMU)

            stereo.setExtendedDisparity(False)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(False)
            stereo.setRectifyEdgeFillColor(0)
            stereo.enableDistortionCorrection(True)
            stereo.initialConfig.setLeftRightCheckThreshold(10)
            stereo.setDepthAlign(left_socket)

            # Accelerometer only: used once at startup to gravity-level the
            # initial attitude (so "down" is real gravity, not the arbitrary
            # camera tilt at launch). We do not fuse it per-frame.
            imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW], 100)
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)

            left.requestOutput((self.width, self.height)).link(stereo.left)
            right.requestOutput((self.width, self.height)).link(stereo.right)

            # Non-blocking queues so a slow consumer never stalls the device
            # (a stalled XLink read is what triggers X_LINK_ERROR). We keep a
            # small buffer and always consume the *latest* frame below.
            q_left = stereo.rectifiedLeft.createOutputQueue(maxSize=4, blocking=False)
            q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
            q_imu = imu.out.createOutputQueue(maxSize=50, blocking=False)

            p.start()

            # Pull rectified-left intrinsics for the metric back-projection.
            ch = p.getDefaultDevice().readCalibration()
            K = np.array(
                ch.getCameraIntrinsics(left_socket, self.width, self.height),
                dtype=np.float64,
            )
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
            vo = RGBDVisualOdometry(K, OdometryConfig())

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



            t0 = time.monotonic()
            prev_pos_ned = np.zeros(3)
            prev_t: float | None = None
            frames = 0
            kf_count = 0
            last_fps_t = t0
            accel_n = 0
            accel_used = 0
            last_tilt_log = t0

            while not self._stop.is_set() and p.isRunning():
                # Drain each queue to its most recent frame; drop the backlog so
                # that if anything briefly stalls we skip stale frames instead of
                # falling progressively further behind (stays real time).
                ld = q_left.tryGet()
                while True:
                    nxt = q_left.tryGet()
                    if nxt is None:
                        break
                    ld = nxt
                dd = q_depth.tryGet()
                while True:
                    nxt = q_depth.tryGet()
                    if nxt is None:
                        break
                    dd = nxt
                if ld is None or dd is None:
                    time.sleep(0.002)
                    continue

                gray = ld.getCvFrame()
                if gray.ndim == 3:
                    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                depth_mm = dd.getCvFrame()
                depth_m = depth_mm.astype(np.float32) / 1000.0

                vo.process(gray, depth_m)  # advance camera-optical world pose

                # Drain the IMU queue to the newest accelerometer sample (camera
                # optical frame). We level the *displayed* attitude with it below.
                latest_a = None
                imsg = q_imu.tryGet()
                while imsg is not None:
                    for pkt in imsg.packets:
                        a = pkt.acceleroMeter
                        latest_a = np.array([a.x, a.y, a.z], dtype=np.float64)
                    imsg = q_imu.tryGet()
                accel_cam = None if latest_a is None else R_imu_cam @ latest_a

                # Level the f2f world frame too, so the BA map is fed gravity-
                # consistent poses (keeps the tracker frame sane long term).
                if accel_cam is not None:
                    vo.correct_tilt(accel_cam)

                pose = vo.pose.copy()  # camera-optical world

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
                        )
                    # Pull the latest correction from the worker (drain to last).
                    newC = ba_state["poll"]()
                    if newC is not None:
                        C_target = newC
                    # Ease toward the target so the correction never snaps.
                    C_applied = _ease_se3(C_applied, C_target, 0.15)
                    pose = C_applied @ pose

                # Gravity-level the FINAL displayed attitude from accel. Doing it
                # here (after any BA correction) guarantees the body frame the
                # user sees always tracks gravity -- the BA correction carries the
                # drifted map attitude and would otherwise re-tilt it. roll/pitch
                # follow the IMU; yaw is left to vision (no magnetometer).
                if accel_cam is not None:
                    R_lvl, used, tilt_deg = level_attitude(
                        pose[:3, :3], accel_cam, g_ref=vo._g_ref)
                    if used:
                        pose[:3, :3] = R_lvl
                    # Rate-limited diagnostics so we can see, on the device,
                    # whether samples are being accepted and the tilt is closing.
                    accel_n += 1
                    if used:
                        accel_used += 1
                    if time.monotonic() - last_tilt_log >= 1.0:
                        rate = accel_used / max(accel_n, 1)
                        print(f"[ours-vio] tilt={tilt_deg:5.1f}deg "
                              f"accel_used={100*rate:3.0f}% "
                              f"|a|={np.linalg.norm(accel_cam):.2f} "
                              f"g_ref={vo._g_ref or 0:.2f}")
                        accel_n = 0
                        accel_used = 0
                        last_tilt_log = time.monotonic()

                pos_opt = pose[:3, 3]
                R_opt = pose[:3, :3]

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
                self._emit(Pose(
                    t=t,
                    pos_ned=pos_ned,
                    vel_ned=vel_ned,
                    quat_wxyz=q_ned,
                    tracking_ok=ok,
                ))

                frames += 1
                if now - last_fps_t >= 0.5:
                    self.fps = frames / (now - last_fps_t)
                    frames = 0
                    last_fps_t = now

            if ba_state is not None:
                ba_state["stop"].set()
                ba_state["event"].set()
                ba_state["thread"].join(timeout=1.0)

    # ----------------------------------------------------------------------- #
    def _collect_startup_accel(self, q_imu, R_imu_cam: np.ndarray,
                               window_s: float = 0.4,
                               timeout_s: float = 2.0) -> np.ndarray | None:
        """Average the accelerometer over a short static startup window.

        Returns the mean specific-force vector rotated into the camera optical
        frame (ready for :meth:`RGBDVisualOdometry.align_to_gravity`), or ``None``
        if no IMU samples arrived within ``timeout_s`` (older device / no IMU) —
        in which case the caller falls back to an identity (unleveled) start.
        """
        samples: list[np.ndarray] = []
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
            if t_first is None:
                t_first = time.monotonic()
            elif time.monotonic() - t_first >= window_s:
                break
        if not samples:
            return None
        accel_imu = np.mean(samples, axis=0)
        return R_imu_cam @ accel_imu

    # ----------------------------------------------------------------------- #
    def _start_ba_worker(self, K: np.ndarray) -> dict:
        """Spawn the background sliding-window BA thread.

        Returns a state dict with ``submit(T_cw, ids, pts, depth)`` and
        ``poll() -> C | None``. The worker keeps the BA map in the *raw f2f*
        world frame (it is always fed raw f2f poses), so the published
        correction ``C = inv(T_ba) @ T_cw`` maps that frame onto the BA-refined
        one. Because the BA core is vectorised NumPy (which releases the GIL on
        its heavy linear-algebra), a thread is enough to keep the device read
        loop responsive — no separate process needed.
        """
        import threading

        from ..vio import WindowedBAMap, WindowedConfig
        from ..vio.bundle import BAConfig

        cfg = WindowedConfig(window=6, kf_every=5,
                             ba=BAConfig(max_iters=5, huber_px=2.0))
        ba_map = WindowedBAMap(K, cfg)

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

        def submit(T_cw, ids, pts, depth_m):
            with snap_lock:
                state["_pending"] = (T_cw, ids, pts, depth_m)
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
                T_cw, ids, pts, depth_m = snap
                ba_map.add_keyframe(T_cw, ids, pts, depth_m)
                post = ba_map.run_ba()
                if post is not None:
                    with out_lock:
                        state["_corr"] = np.linalg.inv(post) @ T_cw

        th = threading.Thread(target=worker, name="OursBAWorker", daemon=True)
        th.start()
        state["thread"] = th
        state["submit"] = submit
        state["poll"] = poll
        return state
