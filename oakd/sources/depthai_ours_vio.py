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
from ..vio import (
    OdometryConfig,
    RGBDVisualOdometry,
    WindowedRGBDOdometry,
)
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

            q_left = stereo.rectifiedLeft.createOutputQueue()
            q_depth = stereo.depth.createOutputQueue()

            p.start()

            # Pull rectified-left intrinsics for the metric back-projection.
            ch = p.getDefaultDevice().readCalibration()
            K = np.array(
                ch.getCameraIntrinsics(left_socket, self.width, self.height),
                dtype=np.float64,
            )

            vo = (WindowedRGBDOdometry(K) if self.backend == "ba"
                  else RGBDVisualOdometry(K, OdometryConfig()))

            t0 = time.monotonic()
            prev_pos_ned = np.zeros(3)
            prev_t: float | None = None
            frames = 0
            last_fps_t = t0

            while not self._stop.is_set() and p.isRunning():
                ld = q_left.tryGet()
                dd = q_depth.tryGet()
                if ld is None or dd is None:
                    time.sleep(0.002)
                    continue

                gray = ld.getCvFrame()
                if gray.ndim == 3:
                    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                depth_mm = dd.getCvFrame()
                depth_m = depth_mm.astype(np.float32) / 1000.0

                pose = vo.process(gray, depth_m)  # 4x4, camera-optical world
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
