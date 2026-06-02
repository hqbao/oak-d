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

Our odometry produces poses in the **camera optical** frame
(+x right, +y down, +z forward), world = first frame. For the NED 3D viewer we
remap optical -> NED with::

    NED = [ +z_opt (forward=North), +x_opt (right=East), +y_opt (down=Down) ]

i.e. ``M_opt->ned = [[0,0,1],[1,0,0],[0,1,0]]``.

Note: the gyro rotation prior is a measured no-op on well-synced data (see
``oakd/vio/imu.py``), so this live source runs pure vision for simplicity.
"""
from __future__ import annotations

import time

import numpy as np

from ..pose import Pose
from ..vio import OdometryConfig, RGBDVisualOdometry
from .base import PoseSource


# Camera optical (x right, y down, z forward) -> world NED (North, East, Down).
_M_OPT_TO_NED = np.array(
    [[0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]]
)


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

            stereo.setExtendedDisparity(False)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(False)
            stereo.setRectifyEdgeFillColor(0)
            stereo.enableDistortionCorrection(True)
            stereo.initialConfig.setLeftRightCheckThreshold(10)
            stereo.setDepthAlign(left_socket)

            left.requestOutput((self.width, self.height)).link(stereo.left)
            right.requestOutput((self.width, self.height)).link(stereo.right)

            # Non-blocking queues so a slow consumer never stalls the device
            # (a stalled XLink read is what triggers X_LINK_ERROR). We keep a
            # small buffer and always consume the *latest* frame below.
            q_left = stereo.rectifiedLeft.createOutputQueue(maxSize=4, blocking=False)
            q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

            p.start()

            # Pull rectified-left intrinsics for the metric back-projection.
            ch = p.getDefaultDevice().readCalibration()
            K = np.array(
                ch.getCameraIntrinsics(left_socket, self.width, self.height),
                dtype=np.float64,
            )

            # The displayed pose is ALWAYS produced by the fast frame-to-frame
            # VO, so the read loop never blocks on BA and the UI stays smooth.
            vo = RGBDVisualOdometry(K, OdometryConfig())

            # In BA mode, a background thread refines a sliding window of
            # keyframes and publishes a world-frame correction ``C`` that we
            # left-multiply onto the f2f pose: P_disp = C @ P_f2f. ``C`` updates
            # only at keyframe cadence with small steps, so the trajectory stays
            # smooth while still getting BA's drift/scale correction.
            ba_state = None
            if self.backend == "ba":
                ba_state = self._start_ba_worker(K)

            t0 = time.monotonic()
            prev_pos_ned = np.zeros(3)
            prev_t: float | None = None
            frames = 0
            kf_count = 0
            last_fps_t = t0

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

                pose = vo.process(gray, depth_m).copy()  # camera-optical world

                if ba_state is not None:
                    # Hand a keyframe snapshot to the BA worker every kf_every
                    # frames (non-blocking: overwrite any pending one).
                    kf_count += 1
                    if kf_count >= ba_state["kf_every"]:
                        kf_count = 0
                        st = vo.frontend.tracks
                        ba_state["submit"](
                            np.linalg.inv(pose),          # T_cw (world->cam)
                            st.ids.copy(), st.points.copy(),
                            depth_m.copy(),
                        )
                    # Apply the latest BA correction to the displayed pose.
                    with ba_state["lock"]:
                        C = ba_state["corr"]
                    pose = C @ pose

                pos_opt = pose[:3, 3]
                R_opt = pose[:3, :3]

                pos_ned = _M_OPT_TO_NED @ pos_opt
                R_ned = _M_OPT_TO_NED @ R_opt @ _M_OPT_TO_NED.T
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
    def _start_ba_worker(self, K: np.ndarray) -> dict:
        """Spawn the background sliding-window BA thread.

        Returns a small state dict the read loop uses to submit keyframe
        snapshots and read the current correction. The worker keeps the entire
        BA map in the *raw f2f* world frame (it is always fed raw f2f poses), so
        the published correction ``C = inv(T_ba) @ T_cw`` maps that frame onto
        the BA-refined one and is small (the window anchor is a raw f2f pose).
        """
        import threading

        from ..vio import WindowedBAMap, WindowedConfig
        from ..vio.bundle import BAConfig

        cfg = WindowedConfig(window=6, kf_every=5,
                             ba=BAConfig(max_iters=5, huber_px=2.0))
        ba_map = WindowedBAMap(K, cfg)

        lock = threading.Lock()
        snap_lock = threading.Lock()
        event = threading.Event()
        stop = threading.Event()
        state = {
            "corr": np.eye(4),
            "lock": lock,
            "event": event,
            "stop": stop,
            "kf_every": cfg.kf_every,
            "_pending": None,
        }

        def submit(T_cw, ids, pts, depth_m):
            with snap_lock:
                state["_pending"] = (T_cw, ids, pts, depth_m)
            event.set()

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
                    C = np.linalg.inv(post) @ T_cw
                    with lock:
                        state["corr"] = C

        th = threading.Thread(target=worker, name="OursBAWorker", daemon=True)
        th.start()
        state["thread"] = th
        state["submit"] = submit
        return state
