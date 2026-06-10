"""FlowPoseSource: the single-process live source (NOT used by the proc4 UI).

This is the single-process architecture-aligned live source: it wired the
in-process flow graph (``build_live``) with a ``UiRenderModule`` whose callback
converts each streamed ``pose.odom`` (camera-optical world) into a viewer
:class:`~ui.comms.lib.misc.pose.Pose` in NED and pushed it through the
:class:`~ui.qt.source.PoseSource` callback the viewer understands.

In the 4-process proc4 UI the device-free contract holds: capture owns the OAK-D,
so this UI never opens it. The live marker is driven by
:class:`ui.main.IpcPoseSource` (VIO's ``pose.odom`` over IPC) instead. This class
is ported for the viewer's historical contract (it keeps the same start / stop /
fps / error surface + the overlay snapshot getters), but its in-process graph
lives in the single-process codebase, NOT in this ``ui`` project, so :meth:`_run`
surfaces a clear reason instead of opening a device. Importing this module pulls
neither depthai nor any in-process graph (the device path was lazy and is now a
guard).

The overlay snapshot getters (:meth:`refined_path_snapshot` /
:meth:`slam_overlay_snapshot` / :meth:`clear_slam_map`) are kept so a viewer that
duck-types on them still works; they return empty buffers since no in-process
engine runs here.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ui.comms.lib.misc.frames import rot_to_quat
from ui.comms.messages import PoseMsg
from ui.comms.lib.misc.pose import Pose
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
        # Inertial dead-reckoning flag (TIGHT path): vision lost but the IMU is
        # still propagating -> amber "INERTIAL DR" badge vs the red "TRACKING
        # LOST" one. Defaults False when absent (loose path / no IMU fallback).
        dr = bool(msg.info.get("inertial_dr", False))
        self._emit(Pose(t=now - self._t0, pos_ned=pos_ned, vel_ned=vel_ned,
                        quat_wxyz=q_ned, tracking_ok=ok, inertial_dr=dr))

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
        ``(kf_seq, kf_pos, n_loops, match_pos)``), transforms optical->NED with
        the same matrix the marker uses, and updates the buffers the viewer reads.
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
            # kf_seq aligns each keyframe to its source frame seq; the
            # single-process overlay draws dots by POSITION only, so it is
            # unpacked (tuple shape must match slam_overlay) but not used here.
            _kf_seq, kf_pos, n_loops, match_pos = ov
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

    def stop(self, timeout: float = 10.0) -> None:
        """Block until the live teardown (in ``_run``'s finally) has fully released
        the OAK-D, BEFORE returning to the caller (the Qt close handler).

        The base default of 1 s is far too short: the live teardown drains the
        flows, reaps the subprocess engine, joins the cam/imu readers and destroys
        the depthai pipeline -- a couple seconds. If ``stop`` returned early the GUI
        would proceed to exit while ``_run`` is still destroying the pipeline on its
        own thread, and that concurrent teardown aborts the process with
        ``mutex lock failed: Invalid argument``. Waiting here serialises it: the
        device is fully released first, then the process exits cleanly.
        """
        super().stop(timeout=timeout)

    def _run(self) -> None:
        # The in-process live graph (``build_live``) lives in the single-process
        # codebase, NOT in this device-free ``ui`` project (capture owns the
        # device; the UI never opens it). proc4 drives the live marker from
        # :class:`ui.main.IpcPoseSource` (VIO's ``pose.odom`` over IPC) instead of
        # this source, so ``_run`` is never reached here. Surface a clear reason
        # via the source's error contract (the UI polls ``.error``) rather than a
        # raw ImportError, keeping this module's import device-free.
        self._fail(
            "FlowPoseSource (in-process live graph) is not available in the "
            "device-free proc4 UI; use ui.main.IpcPoseSource instead.")
