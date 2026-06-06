"""FlowPoseSource: drive the Qt 3D viewer from the live flow pipeline.

This is the architecture-aligned live source: it wires the flow graph
(:func:`ours.app.build_live`) with a :class:`~ours.flows.ui.render.UiRenderFlow`
whose callback converts each streamed ``pose.odom`` (camera-optical world) into a
viewer :class:`~ours.lib.misc.pose.Pose` in NED and pushes it through the
:class:`~ours.ui.source.PoseSource` callback the viewer already understands.

It replaces the monolithic ``OakOursVioSource`` as the default live source while
keeping the same UI contract (start / stop / fps / error). The displayed
trajectory is the real-time frame-to-frame VO (``pose.odom``).

It builds the live graph **realtime-bounded**: ``with_backend_slam=False`` (the
windowed-BA back-end + loop-closing SLAM are NOT built — this source displays
``pose.odom`` only, so running them would burn CPU and, worse, pile keyframe
images in their UNBOUNDED FIFO inboxes until host memory pressure starves the
depthai XLink and the OAK-D firmware watchdog crashes the device) and
``realtime_latest=True`` (the cam/imu_cam/odometry inboxes coalesce to the newest
frame, so a momentarily slow consumer drops stale frames instead of growing an
unbounded backlog). This mirrors the already-stable keypoint-depth live view,
which runs the same device with the same two flags. Applying BA/SLAM corrections
to the displayed path is a follow-up that must first bound the keyframe path.

Only exercisable on real hardware (it opens the OAK-D device).
"""
from __future__ import annotations

import time

import numpy as np

from ..lib.misc.frames import rot_to_quat
from ..lib.flow.messages import PoseMsg
from ..lib.misc.pose import Pose
from ..lib.flow.pubsub import Bus
from .source import PoseSource

# Camera optical (x right, y down, z forward) -> world NED, and the column
# reorder that maps the optical attitude columns to the body [forward, right,
# down] triad the viewer expects. Identical convention to the legacy source.
_M_OPT_TO_NED = np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
_P_OPT_TO_FRD = np.array([[0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0]])


class FlowPoseSource(PoseSource):
    def __init__(self, width: int = 640, height: int = 400, fps: int = 20,
                 kf_every: int = 5, use_gyro: bool = True,
                 depth_fast: bool = True,
                 recalibrate_bias: bool = False) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.cam_fps = int(fps)
        self.kf_every = int(kf_every)
        self.use_gyro = bool(use_gyro)
        self.depth_fast = bool(depth_fast)
        self.recalibrate_bias = bool(recalibrate_bias)
        self._t0 = 0.0
        self._prev_pos: np.ndarray | None = None
        self._prev_t: float | None = None
        self._frames = 0
        self._last_fps_t = 0.0

    def _on_pose(self, msg: PoseMsg) -> None:
        T = msg.T_world_cam
        pos_opt = T[:3, 3]
        R_opt = T[:3, :3]
        pos_ned = _M_OPT_TO_NED @ pos_opt
        R_ned = _M_OPT_TO_NED @ R_opt @ _P_OPT_TO_FRD
        q_ned = rot_to_quat(R_ned)

        now = time.monotonic()
        if self._prev_t is None:
            vel_ned = np.zeros(3)
        else:
            dt = max(now - self._prev_t, 1e-6)
            vel_ned = (pos_ned - self._prev_pos) / dt
        self._prev_pos = pos_ned
        self._prev_t = now

        ok = bool(msg.info.get("ok", True))
        self._emit(Pose(t=now - self._t0, pos_ned=pos_ned, vel_ned=vel_ned,
                        quat_wxyz=q_ned, tracking_ok=ok))

        self._frames += 1
        if now - self._last_fps_t >= 0.5:
            self.fps = self._frames / (now - self._last_fps_t)
            self._frames = 0
            self._last_fps_t = now

    def _run(self) -> None:
        from ..app import build_live           # lazy: pulls depthai only here
        from ..flows.ui import UiRenderFlow

        self._t0 = self._last_fps_t = time.monotonic()
        bus = Bus()
        ui = UiRenderFlow(bus, self._on_pose)
        try:
            device, (cam_flow, imu_flow), flows, _ = build_live(
                bus, width=self.width, height=self.height, fps=self.cam_fps,
                kf_every=self.kf_every, use_gyro=self.use_gyro,
                depth_fast=self.depth_fast,
                recalibrate_bias=self.recalibrate_bias, ui=ui,
                with_backend_slam=False, realtime_latest=True)
        except Exception as e:                                    # noqa: BLE001
            self._fail(f"device open failed: {e}")
            return

        for f in flows:
            f.start()
        imu_flow.start()
        cam_flow.start()
        try:
            while not self._stop.is_set() and cam_flow.is_alive():
                time.sleep(0.05)
        finally:
            cam_flow.stop()
            imu_flow.stop()
            ui.done.wait(timeout=5.0)
            for f in flows:
                f.stop()
            device.release()
