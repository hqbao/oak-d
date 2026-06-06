"""FlowPoseSource: drive the Qt 3D viewer from the live flow pipeline.

This is the architecture-aligned live source: it wires the flow graph
(:func:`ours.app.build_live`) with a :class:`~ours.flows.ui.render.UiRenderFlow`
whose callback converts each streamed ``pose.odom`` (camera-optical world) into a
viewer :class:`~ours.lib.misc.pose.Pose` in NED and pushes it through the
:class:`~ours.ui.source.PoseSource` callback the viewer already understands.

It is the live source for every mode, keeping the UI contract the viewer expects
(start / stop / fps / error). The
displayed MARKER is always the real-time frame-to-frame VO (``pose.odom``) -- the
responsive tip that tracks the camera at full distance.

``mode`` selects which heavy optimiser flow runs in the BACKGROUND to refine the
map behind that marker:

* ``"odom"`` -- none (bare ``ours``).
* ``"ba"``   -- the windowed-BA back-end (``ours-ba``).
* ``"slam"`` -- the loop-closing SLAM flow (``ours-slam``).

Crucially the marker is **decoupled** from the heavy flow: the BA/SLAM output
(``pose.refined`` / ``loop.correction``) feeds the map overlay, never the marker,
so an async correction can never drag or stall the live tip -- the stall /
undershoot failure mode under fast / shaky motion. Offline-verified:
``pose.odom`` is byte-identical with and without BA running.

The graph is built **realtime-bounded**: the heavy flow is ``latest_only=True``
(its keyframe inbox coalesces, so a slow BA / loop solve drops backlog instead of
piling keyframe images in an unbounded FIFO until the depthai XLink starves and
the OAK-D watchdog crashes the device), and the cam/imu_cam/odometry inboxes are
``realtime_latest=True`` (newest-frame-wins, bounded latency). This mirrors the
already-stable keypoint-depth live view.

Crucially the heavy flow also runs ``worker=True``: the BA / SLAM solve executes
**out-of-process** (see :mod:`ours.lib.engine.subprocess`), so its mostly-pure-Python
work never holds the camera read loop's GIL. That is the actual fix for the
fast-push stall / undershoot -- a marker that is merely *data*-decoupled from the
correction still lagged because an in-thread solve starved the read loop and
dropped frames; out-of-process removes that contention entirely.

Only exercisable on real hardware (it opens the OAK-D device).
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ..lib.misc.frames import rot_to_quat
from ..lib.flow.messages import PoseMsg
from ..lib.misc.pose import Pose
from ..lib.flow.pubsub import Bus
from .source import PoseSource

# Camera optical (x right, y down, z forward) -> world NED, and the column
# reorder that maps the optical attitude columns to the body [forward, right,
# down] triad the viewer expects (the optical->NED display convention).
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
        # can never be dragged/stalled by an async correction (the earlier stall /
        # undershoot failure mode).
        if mode not in ("odom", "ba", "slam"):
            raise ValueError(f"FlowPoseSource mode must be odom|ba|slam, got {mode!r}")
        self.mode = mode
        self._t0 = 0.0
        self._prev_pos: np.ndarray | None = None
        self._prev_t: float | None = None
        self._frames = 0
        self._last_fps_t = 0.0

        # ---- live MAP overlay (read by the 3D viewer, written by _run) -------
        # The heavy optimiser's refined map, mirrored here so the Qt viewer can
        # read a consistent snapshot under a lock while the source thread updates
        # it from the (out-of-process) engine. All in NED so the viewer applies
        # only its NED->ENU display transform. REAL engine outputs: ours-ba ->
        # refined keyframe trajectory; ours-slam -> corrected keyframe dots +
        # loop-closure flash. Empty for bare ours (no engine).
        self._engine = None                          # heavy flow's Engine (or None)
        self._ov_lock = threading.Lock()
        self._ba_pts: dict[int, np.ndarray] = {}     # ba: {kf_id: world pos (opt)}
        self._refined_ned = np.zeros((0, 3), np.float32)   # refined trajectory line
        self._slam_kf_ned = np.zeros((0, 3), np.float32)   # slam keyframe dots
        self._slam_match_ned = np.zeros((0, 3), np.float32)  # last revisit dots
        self._slam_n_loops = 0
        self._slam_flash_id = 0

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

    # ---- live map overlay (engine -> mirror -> viewer) -------------------

    def _poll_overlay(self) -> None:
        """Drain the heavy engine's latest map snapshot into the locked mirror.

        Runs on the source thread (the ``_run`` loop). Reads a REAL engine output
        (ours-ba: ``{kf_id: refined world pos}``; ours-slam:
        ``(kf_pos, n_loops, match_pos)``), transforms optical->NED with the same
        matrix the marker uses, and updates the buffers the viewer reads.
        """
        if self._engine is None:
            return
        ov = self._engine.poll_overlay()
        if ov is None:
            return
        if self.mode == "ba":
            self._ba_pts.update(ov)              # finalise window kfs by id
            ids = sorted(self._ba_pts)
            pos = np.array([self._ba_pts[i] for i in ids], dtype=np.float64)
            ned = (pos @ _M_OPT_TO_NED.T).astype(np.float32) if len(pos) else \
                np.zeros((0, 3), np.float32)
            with self._ov_lock:
                self._refined_ned = ned
        else:                                    # slam
            kf_pos, n_loops, match_pos = ov
            kf_ned = ((np.asarray(kf_pos) @ _M_OPT_TO_NED.T).astype(np.float32)
                      if len(kf_pos) else np.zeros((0, 3), np.float32))
            with self._ov_lock:
                self._refined_ned = kf_ned
                self._slam_kf_ned = kf_ned
                if n_loops > self._slam_n_loops:     # a NEW loop just closed
                    self._slam_n_loops = int(n_loops)
                    self._slam_flash_id += 1
                    self._slam_match_ned = (
                        (np.asarray(match_pos) @ _M_OPT_TO_NED.T).astype(np.float32)
                        if len(match_pos) else np.zeros((0, 3), np.float32))

    def refined_path_snapshot(self) -> np.ndarray:
        """The BA/SLAM-refined trajectory in NED (read by the viewer)."""
        with self._ov_lock:
            return self._refined_ned.copy()

    def slam_overlay_snapshot(self):
        """``(kf_ned, match_ned, loop_segs, flash_id)`` for the SLAM map overlay."""
        with self._ov_lock:
            return (self._slam_kf_ned.copy(), self._slam_match_ned.copy(),
                    [], self._slam_flash_id)

    def clear_slam_map(self) -> None:
        """UI "clear keyframes": forget the heavy map + wipe the overlay buffers."""
        if self._engine is not None:
            self._engine.reset()
        with self._ov_lock:
            self._ba_pts.clear()
            self._refined_ned = np.zeros((0, 3), np.float32)
            self._slam_kf_ned = np.zeros((0, 3), np.float32)
            self._slam_match_ned = np.zeros((0, 3), np.float32)
            self._slam_n_loops = 0
            self._slam_flash_id += 1

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
                backend=(self.mode == "ba"), slam=(self.mode == "slam"),
                worker=(self.mode in ("ba", "slam")))
        except Exception as e:                                    # noqa: BLE001
            self._fail(f"device open failed: {e}")
            return

        # The heavy flow (BackendFlow/SlamFlow) exposes its Engine; poll its map
        # overlay here on the source thread (never the marker's path -- that stays
        # pose.odom). None for bare ours (no heavy flow), so the overlay is empty.
        self._engine = next((f.engine for f in flows if hasattr(f, "engine")), None)

        for f in flows:
            f.start()
        imu_flow.start()
        cam_flow.start()
        try:
            while not self._stop.is_set() and cam_flow.is_alive():
                self._poll_overlay()
                time.sleep(0.05)
        finally:
            self._engine = None
            cam_flow.stop()
            imu_flow.stop()
            ui.done.wait(timeout=5.0)
            for f in flows:
                f.stop()
            # Join so each heavy flow's run()-finally has reaped its subprocess
            # engine (sentinel -> join -> terminate) BEFORE we release the device;
            # otherwise a lingering worker could outlive the parent's teardown.
            for f in flows:
                f.join(timeout=3.0)
            device.release()
