"""FlowPoseSource: drive the Qt 3D viewer from the live flow pipeline.

This is the architecture-aligned live source: it wires the flow graph
(:func:`ours.app.build_live`) with a :class:`~ours.flows.ui.render.UiRenderFlow`
whose callback converts each streamed ``pose.odom`` (camera-optical world) into a
viewer :class:`~ours.lib.misc.pose.Pose` in NED and pushes it through the
:class:`~ours.ui.source.PoseSource` callback the viewer already understands.

It replaces the monolithic ``OakOursVioSource`` as the live source for every
mode while keeping the same UI contract (start / stop / fps / error). The
displayed MARKER is always the real-time frame-to-frame VO (``pose.odom``) -- the
responsive tip that tracks the camera at full distance.

``mode`` selects which heavy optimiser flow runs in the BACKGROUND to refine the
map behind that marker:

* ``"odom"`` -- none (bare ``ours``).
* ``"ba"``   -- the windowed-BA back-end (``ours-ba``).
* ``"slam"`` -- the loop-closing SLAM flow (``ours-slam``).

Crucially the marker is **decoupled** from the heavy flow: the BA/SLAM output
(``pose.refined`` / ``loop.correction``) feeds the map overlay, never the marker,
so an async correction can never drag or stall the live tip -- the failure mode
that made the legacy source "ì lại" under fast / shaky motion. Offline-verified:
``pose.odom`` is byte-identical with and without BA running.

The graph is built **realtime-bounded**: the heavy flow is ``latest_only=True``
(its keyframe inbox coalesces, so a slow BA / loop solve drops backlog instead of
piling keyframe images in an unbounded FIFO until the depthai XLink starves and
the OAK-D watchdog crashes the device), and the cam/imu_cam/odometry inboxes are
``realtime_latest=True`` (newest-frame-wins, bounded latency). This mirrors the
already-stable keypoint-depth live view.

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
                 recalibrate_bias: bool = False,
                 mode: str = "odom") -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.cam_fps = int(fps)
        self.kf_every = int(kf_every)
        self.use_gyro = bool(use_gyro)
        self.depth_fast = bool(depth_fast)
        self.recalibrate_bias = bool(recalibrate_bias)
        # Display mode (the live MARKER is always the realtime pose.odom -- the
        # responsive f2f tip that tracks the camera at full distance, exactly
        # like the bare ``ours`` source). ``mode`` only selects which heavy
        # optimiser flow runs in the background to refine the MAP behind it:
        #   "odom" -> none (bare ours);  "ba" -> BackendFlow (windowed BA);
        #   "slam" -> SlamFlow (loop closure). The heavy flow is built latest-only
        # so it can never backlog the marker. Its refined output (pose.refined /
        # loop.correction) feeds the map overlay, NOT the marker -- so the marker
        # can never be dragged/stalled by an async correction (the failure mode of
        # the legacy OakOursVioSource).
        if mode not in ("odom", "ba", "slam"):
            raise ValueError(f"FlowPoseSource mode must be odom|ba|slam, got {mode!r}")
        self.mode = mode
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
                with_backend_slam=False, realtime_latest=True,
                backend=(self.mode == "ba"), slam=(self.mode == "slam"))
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
