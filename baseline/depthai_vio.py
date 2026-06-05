"""Real visual-inertial odometry from the OAK-D using DepthAI's BasaltVIO.

DepthAI 3.x ships the Basalt open-source stereo-inertial VIO as a native node
(``dai.node.BasaltVIO``). We wire two ``Camera`` nodes (CAM_B/CAM_C, the
stereo pair on the OAK-D W) plus the onboard ``IMU`` into it and read its
``transform`` output from the host.

Frame plumbing
--------------
Basalt emits poses in **FLU world** (X=Forward, Y=Left, Z=Up). On this drone
the camera is mounted with its optical axis along body Forward and the USB
connector pointing up, so at startup body-Forward aligns with world-North.

We map FLU -> NED with M = diag(+1, -1, -1):

    NED translation = (+x_flu, -y_flu, -z_flu)
    NED attitude    R_ned = M @ R_flu @ M.T   (M is self-inverse)

The camera mount is identity to the body frame, so R_ned is also the
body-attitude in NED.
"""
from __future__ import annotations

import time

import numpy as np

from oakd.frames import quat_to_rot
from oakd.pose import Pose
from oakd.sources.base import PoseSource


# FLU world -> NED world: flip Y (left->east) and Z (up->down).
_M_FLU_TO_NED = np.diag([1.0, -1.0, -1.0])


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


class OakBasaltVioSource(PoseSource):
    """OAK-D + Basalt VIO -> NED pose stream."""

    def __init__(self, width: int = 640, height: int = 400, fps: int = 60,
                 imu_rate_hz: int = 200) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.cam_fps = int(fps)
        self.imu_rate_hz = int(imu_rate_hz)

    def _run(self) -> None:
        import depthai as dai  # lazy: --source fake works without depthai/device

        with dai.Pipeline() as p:
            left = p.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_B, sensorFps=self.cam_fps,
            )
            right = p.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_C, sensorFps=self.cam_fps,
            )
            imu = p.create(dai.node.IMU)
            vio = p.create(dai.node.BasaltVIO)

            imu.enableIMUSensor(
                [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
                self.imu_rate_hz,
            )
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)

            vio.setImuUpdateRate(self.imu_rate_hz)

            left.requestOutput((self.width, self.height)).link(vio.left)
            right.requestOutput((self.width, self.height)).link(vio.right)
            imu.out.link(vio.imu)

            transform_q = vio.transform.createOutputQueue()

            p.start()

            t0 = time.monotonic()
            prev_pos = np.zeros(3)
            prev_t: float | None = None
            frames = 0
            last_fps_t = t0

            while not self._stop.is_set() and p.isRunning():
                td = transform_q.tryGet()
                if td is None:
                    time.sleep(0.002)
                    continue

                # Basalt translation: FLU world metres; quaternion: FLU world.
                tr = td.getTranslation()
                qf = td.getQuaternion()
                pos_flu = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
                q_flu_wxyz = np.array(
                    [qf.qw, qf.qx, qf.qy, qf.qz], dtype=np.float64
                )

                # FLU -> NED
                pos_ned = _M_FLU_TO_NED @ pos_flu
                R_flu = quat_to_rot(q_flu_wxyz)
                R_ned = _M_FLU_TO_NED @ R_flu @ _M_FLU_TO_NED.T
                q_ned = _rot_to_quat_wxyz(R_ned)

                now = time.monotonic()
                t = now - t0
                if prev_t is None:
                    vel_ned = np.zeros(3)
                else:
                    dt = max(now - prev_t, 1e-6)
                    vel_ned = (pos_ned - prev_pos) / dt
                prev_pos = pos_ned
                prev_t = now

                self._emit(Pose(
                    t=t,
                    pos_ned=pos_ned,
                    vel_ned=vel_ned,
                    quat_wxyz=q_ned,
                    tracking_ok=True,
                ))

                frames += 1
                if now - last_fps_t >= 0.5:
                    self.fps = frames / (now - last_fps_t)
                    frames = 0
                    last_fps_t = now
