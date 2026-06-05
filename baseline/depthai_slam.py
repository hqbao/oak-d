"""Loop-closing SLAM from the OAK-D: BasaltVIO odometry + RTABMapSLAM.

Pipeline
--------
  Camera(B) ─┐                 ┌─> BasaltVIO ─(odom transform)──┐
             ├─> StereoDepth ──┤                                 ├─> RTABMapSLAM
  Camera(C) ─┘                 └─> depth + rectifiedLeft ────────┘   .transform
                                                                     (loop-closed)
  IMU ──────────────────────────> BasaltVIO.imu

BasaltVIO gives the high-rate, low-latency odometry; RTABMapSLAM corrects
drift via loop closure on the rectified+depth pair. We consume
``slam.transform`` as the final pose stream — it is the loop-corrected pose
in the same FLU world frame as BasaltVIO, so the same FLU->NED conversion
applies.
"""
from __future__ import annotations

import time

import numpy as np

from oakd.frames import quat_to_rot
from oakd.pose import Pose
from oakd.sources.base import PoseSource
from .depthai_vio import _M_FLU_TO_NED, _rot_to_quat_wxyz


class OakBasaltSlamSource(PoseSource):
    """OAK-D + BasaltVIO + RTABMapSLAM -> loop-closed NED pose stream."""

    def __init__(self, width: int = 640, height: int = 400, fps: int = 20,
                 imu_rate_hz: int = 200,
                 database_path: str | None = None,
                 load_database: bool = False) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.cam_fps = int(fps)
        self.imu_rate_hz = int(imu_rate_hz)
        self.database_path = database_path
        self.load_database = bool(load_database)

    def _run(self) -> None:
        import depthai as dai  # lazy

        with dai.Pipeline() as p:
            left = p.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_B, sensorFps=self.cam_fps,
            )
            right = p.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_C, sensorFps=self.cam_fps,
            )
            imu = p.create(dai.node.IMU)
            stereo = p.create(dai.node.StereoDepth)
            vio = p.create(dai.node.BasaltVIO)
            slam = p.create(dai.node.RTABMapSLAM)

            # IMU @ raw 200 Hz feeds Basalt
            imu.enableIMUSensor(
                [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
                self.imu_rate_hz,
            )
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)
            vio.setImuUpdateRate(self.imu_rate_hz)

            # Stereo: rectified-left aligned depth for SLAM.
            # NOTE: setSubpixel(True) doubles VPU load and pushed the OAK-D W
            # into firmware crashes when combined with 4 image streams + IMU.
            # 1-pixel disparity is plenty for RTABMap loop closure.
            stereo.setExtendedDisparity(False)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(False)
            stereo.setRectifyEdgeFillColor(0)
            stereo.enableDistortionCorrection(True)
            stereo.initialConfig.setLeftRightCheckThreshold(10)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)

            # RTABMap params: enable loop closure + (optional) persistent DB.
            # NOTE: even when we don't render the occupancy grid, RTABMap
            # internally constructs LocalGrid cells from the sensor data and
            # ASSERTs cellSize > 0. The only combo proven to avoid the
            # assertion on first frame is to enable occupancy-grid creation
            # (which forces RTABMap to populate Grid/CellSize from defaults).
            # We just don't link the occupancyGridMap output and tell the
            # node not to publish it.
            slam_params = {
                "RGBD/CreateOccupancyGrid": "true",
                "Grid/3D": "true",
                "Rtabmap/DetectionRate": "1",
                "Rtabmap/SaveWMState": "true",
                "Mem/IncrementalMemory": "true",
            }
            slam.setParams(slam_params)
            slam.setPublishGrid(False)
            slam.setPublishObstacleCloud(False)
            slam.setPublishGroundCloud(False)
            if self.database_path:
                slam.setDatabasePath(self.database_path)
                slam.setLoadDatabaseOnStart(self.load_database)

            # Linking
            left.requestOutput((self.width, self.height)).link(stereo.left)
            right.requestOutput((self.width, self.height)).link(stereo.right)
            stereo.syncedLeft.link(vio.left)
            stereo.syncedRight.link(vio.right)
            imu.out.link(vio.imu)
            stereo.depth.link(slam.depth)
            stereo.rectifiedLeft.link(slam.rect)
            vio.transform.link(slam.odom)

            transform_q = slam.transform.createOutputQueue()

            p.start()

            t0 = time.monotonic()
            prev_pos = np.zeros(3)
            prev_t: float | None = None
            last_pose_t = t0
            frames = 0
            last_fps_t = t0

            while not self._stop.is_set() and p.isRunning():
                td = transform_q.tryGet()
                if td is None:
                    # mark LOST after 1 s without an updated pose
                    if time.monotonic() - last_pose_t > 1.0 and prev_t is not None:
                        # emit a stale pose with tracking_ok=False so the UI
                        # can show LOST without blanking the trail
                        self._emit(Pose(
                            t=time.monotonic() - t0,
                            pos_ned=prev_pos,
                            vel_ned=np.zeros(3),
                            quat_wxyz=np.array([1.0, 0, 0, 0]),
                            tracking_ok=False,
                        ))
                        last_pose_t = time.monotonic()  # throttle to 1 Hz
                    time.sleep(0.005)
                    continue

                tr = td.getTranslation()
                qf = td.getQuaternion()
                pos_flu = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
                q_flu_wxyz = np.array(
                    [qf.qw, qf.qx, qf.qy, qf.qz], dtype=np.float64
                )

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
                last_pose_t = now

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
