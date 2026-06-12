"""ui process: PyQt6 viewer with ONE single view of 5 trajectories (IPC subscriber).

Connects (over IPC) to the ``vio`` and ``slam`` endpoints and renders ALL of the
pipeline's trajectory streams in a SINGLE :class:`~ui.qt.viewer3d.Viewer3D` (no
tabs). The live marker is ``pose.odom`` (VIO). The five trajectory lines, each
from its own source, are (back-to-front / drift order):

1. **VO** (``pose.vo``, pure vision) -- dim grey; drifts most.
2. **VIO** (``pose.odom``) -- NVG green; the responsive frame-to-frame trail +
   the live marker (never lags -- it never waits on a back-end).
3. **VIO-BA** (``pose.refined``, windowed-BA keyframes) -- violet-blue; sparse.
4. **SLAM-corrected VIO** -- warm orange; the dense VIO trail rubber-sheeted by
   SLAM's per-keyframe loop corrections, with teleport vertices (where SLAM
   pulled the path far) flashed red.
5. **SLAM** (``slam.map``) -- HUD cyan keyframe line + amber keyframe dots, with
   the just-revisited keyframes flashed on each loop closure. ``slam.map``
   supersedes the old ``loop.correction`` overlay (which only fired ON a loop --
   no dots along the path until the first loop closed).

A single :class:`SlamMapTracker` subscribes every stream across two IPC clients
(SLAM on the slam endpoint; odom/vo/refined on the vio endpoint) and exposes one
snapshot getter per line. A row of 5 checkable toggle buttons on the Controls
toolbar shows/hides each line independently. The IPC subscribers live on
background threads (one :class:`~ui.comms.IPCPubSub` client per source) and
marshal pose samples onto the GUI thread via the existing :class:`PoseSource`
callback contract (the viewer already handles the cross-thread hop via
signal/slot).

Calibration handshake
---------------------
Like VIO and SLAM, the UI waits for the retained ``calib.bundle`` on EACH
endpoint it cares about before building views. The bundle's
``width`` / ``height`` / ``device_id`` come from there so the
Visualize / Calibration menus know the capture resolution + device key without
needing a CLI flag.

Visualize / Calibration menus
-----------------------------
These menus reuse the EXISTING ``ui.qt`` windows + calib dialogs UNCHANGED,
fed entirely over IPC by the adapters in :mod:`ui.modules.ipc_sources` -- the UI
never opens a device (capture owns it). The triplet view reads capture's
``frame.depth`` + ``imucam.sample``; the keypoint tracker reads capture's
``frame.depth`` plus VIO's ``frame.tracks`` + ``frame.inliers``; the gyro / accel
dialogs read capture's ``imu.raw``. Calibration saves are keyed per device
(``device_id`` from the bundle), so a saved bias/scale takes effect on the NEXT
capture start (capture loads it via ``load_gyro_bias`` / ``load_accel_calib``); it
does not retro-fit the running pipeline.

Run::

    python -m ui.main                                  # default endpoints
    python -m ui.main --vio-endpoint oak.vio.test --slam-endpoint oak.slam.test
"""
from __future__ import annotations

import argparse
import logging
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.comms import topics                                          # noqa: E402
from ui.comms import IPCPubSub                                       # noqa: E402
from ui.comms.messages import END                                   # noqa: E402
from ui.comms.wire import (                                         # noqa: E402
    WireCalibBundle, WireEnd, WirePoseMsg, WireSlamMap,
)
from ui.comms.lib.misc.frames import rot_to_quat                     # noqa: E402
from ui.comms.lib.misc.pose import Pose, PoseHistory                 # noqa: E402
from ui.qt.source import PoseSource                                  # noqa: E402

LOG = logging.getLogger("ui.main")

DEFAULT_VIO_ENDPOINT = "oak.vio"
DEFAULT_SLAM_ENDPOINT = "oak.slam"
DEFAULT_CAPTURE_ENDPOINT = "oak.capture"

# Sentinel return code the UI uses to tell the launcher "respawn the whole
# pipeline from scratch" (the Restart toolbar button). The launcher loops on it;
# any other code is a normal exit. 42 is outside the usual 0/1/130/143 range so
# it can't collide with a clean close, a crash, or a signal-induced exit.
RESTART_EXIT_CODE = 42


# Camera optical (x right, y down, z forward) -> NED, matching the
# single-process FlowPoseSource so both code paths display in the same
# convention. _P_OPT_TO_FRD reorders attitude columns from optical to FRD body.
_M_OPT_TO_NED = np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
_P_OPT_TO_FRD = np.array([[0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0]])


def _wire_pose_to_ned(wm: WirePoseMsg, prev_pos: np.ndarray | None,
                      prev_t: float | None) -> tuple[Pose, np.ndarray, float]:
    """Convert a wire pose (T_world_cam, optical) into a viewer NED Pose.

    Returns ``(pose, new_prev_pos, new_prev_t)`` so the caller can carry the
    velocity finite-difference state forward.
    """
    T = wm.T_world_cam
    pos_opt = T[:3, 3]
    R_opt = T[:3, :3]
    pos_ned = _M_OPT_TO_NED @ pos_opt
    R_ned = _M_OPT_TO_NED @ R_opt @ _P_OPT_TO_FRD
    q_ned = rot_to_quat(R_ned)
    now = time.monotonic()
    if prev_t is None:
        vel_ned = np.zeros(3)
    else:
        dt = max(now - prev_t, 1e-6)
        vel_ned = (pos_ned - prev_pos) / dt
    ok = bool(wm.info.get("ok", True))
    # Inertial dead-reckoning flag (TIGHT path): vision lost but the IMU is
    # still propagating a valid pose -> the viewer shows the amber "INERTIAL DR"
    # badge instead of the red "TRACKING LOST" one. Defaults False on the loose
    # path / when the flag is absent.
    dr = bool(wm.info.get("inertial_dr", False))
    pose = Pose(t=now, pos_ned=pos_ned, vel_ned=vel_ned,
                quat_wxyz=q_ned, tracking_ok=ok, inertial_dr=dr)
    return pose, pos_ned, now


# --------------------------------------------------------------------------- #
# IPC-driven pose source: subscribe to one endpoint's pose.odom topic + emit
# `Pose` samples to the standard `PoseSource` callback the viewer consumes.
# --------------------------------------------------------------------------- #
class IpcPoseSource(PoseSource):
    """A :class:`PoseSource` that drains ``pose.odom`` off an IPC endpoint.

    Unlike the single-process ``FlowPoseSource``, this does NOT open the camera
    or run any flow graph: VIO does that in its own process. The UI just
    subscribes to VIO's ``pose.odom`` (the realtime frame-to-frame marker) and
    pushes each pose to the viewer via the standard callback.
    """

    def __init__(self, endpoint: str, *, label: str = "ipc",
                 connect_timeout_s: float = 30.0) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.label = label
        self._connect_timeout_s = float(connect_timeout_s)
        self._client: IPCPubSub | None = None
        self._prev_pos: np.ndarray | None = None
        self._prev_t: float | None = None
        self._frames = 0
        self._last_fps_t = 0.0

    # ------------------------------------------------------------------ #
    def _on_pose(self, wm: WirePoseMsg | object) -> None:
        # IPCPubSub delivers the wire-level END as a `WireEnd` instance, not
        # the local `END` sentinel, so guard both (else the shutdown END is
        # mistaken for a pose and raises on `.T_world_cam`).
        if wm is END or isinstance(wm, WireEnd):
            return                                # source closed; viewer stays
        pose, prev_pos, prev_t = _wire_pose_to_ned(
            wm, self._prev_pos, self._prev_t)     # type: ignore[arg-type]
        self._prev_pos = prev_pos
        self._prev_t = prev_t
        self._emit(pose)
        # Cheap FPS estimate: rolling 0.5 s window.
        self._frames += 1
        now = time.monotonic()
        if self._last_fps_t == 0.0:
            self._last_fps_t = now
        if now - self._last_fps_t >= 0.5:
            self.fps = self._frames / (now - self._last_fps_t)
            self._frames = 0
            self._last_fps_t = now

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        """Block until stop: the IPC client thread does all the real work."""
        client = IPCPubSub(self.endpoint, role="client",
                           connect_timeout_s=self._connect_timeout_s)
        client.subscribe("pose.odom", self._on_pose)
        try:
            client.start()
        except (TimeoutError, ConnectionError) as e:
            self._fail(f"connect to {self.endpoint}: {e}")
            return
        self._client = client
        # Wait for stop -- the IPC client's recv thread fires `_on_pose` for
        # every incoming wire pose; the work is done on that thread, not here.
        try:
            while not self._stop.is_set():
                time.sleep(0.1)
        finally:
            try:
                client.stop()
            except Exception:                                      # noqa: BLE001
                pass
            self._client = None


# --------------------------------------------------------------------------- #
# Single-view trajectory model: one subscriber across two IPC clients that feeds
# the VO / VIO-BA / SLAM-corrected VIO / SLAM lines behind the live VIO marker.
# --------------------------------------------------------------------------- #
# Hard cap on each dense pose ring (VIO odom + VO + BA) the tracker records.
# At ~20 Hz this is ~2.8 h of trajectory -- far longer than any session -- but
# bounds memory so a runaway pipeline can never grow a buffer without limit.
_VIO_BUF_CAP = 200_000

# Max allowed jump (m) between consecutive points on the VO / VIO-BA trajectory
# trails. Keyframes/poses arrive ~0.25 s apart (kf_every=5 @ 20 fps for BA;
# per-frame for VO), so a step beyond this implies >~20 m/s -- physically
# impossible for a hand-held indoor rig and therefore a DIVERGED solve (the
# windowed BA can shoot a single keyframe far off on degenerate / low-parallax
# geometry; pure-vision VO can too). Such a point is dropped at the DISPLAY so
# one bad solve never draws a huge line across the view. This is viewer
# robustness only -- it does not touch the VIO/BA math or the byte-parity oracle.
_MAX_TRAIL_JUMP_M = 5.0


def _append_trail(buf: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Append ``pos`` (NED, (3,)) to a trajectory trail ``buf`` ((N,3) float32),
    dropping it when it is non-finite or jumps > :data:`_MAX_TRAIL_JUMP_M` from
    the last kept point (a diverged BA/VO solve). Returns the new buffer, capped
    at :data:`_VIO_BUF_CAP`. Display-side guard only."""
    if not np.isfinite(pos).all():
        return buf
    if len(buf) and float(np.linalg.norm(
            pos.astype(np.float64) - buf[-1].astype(np.float64))) > _MAX_TRAIL_JUMP_M:
        return buf
    out = np.vstack((buf, pos[None, :].astype(np.float32)))
    return out[-_VIO_BUF_CAP:] if len(out) > _VIO_BUF_CAP else out

# Teleport threshold (metres): a corrected-VIO vertex is flagged "teleport" when
# the SLAM correction pulled it more than this from the raw VIO position. A loop
# closure that snaps the path back will exceed it; ordinary drift correction
# stays under. v1 = a simple magnitude threshold (see corrected_vio_snapshot).
TELEPORT_M = 0.15


class SlamMapTracker(threading.Thread):
    """Single trajectory model for the proc4 single-view UI.

    Subscribes -- across two IPC clients -- to every stream the 5 trajectory
    lines need and exposes one snapshot getter per line:

    * on ``endpoint`` (SLAM): ``slam.map`` -> the CONTINUOUS keyframe map (every
      keyframe), driving the cyan SLAM line + amber kf dots + loop flash, and the
      per-keyframe corrected positions used to deform VIO.
    * on ``vio_endpoint`` (VIO): ``pose.odom`` -> the dense per-frame VIO trail
      WITH its frame seqs (for the deform); ``pose.vo`` -> the pure-vision VO
      trail; ``pose.refined`` -> the windowed-BA keyframe trail.

    Drives the keyframe dots from the CONTINUOUS ``slam.map`` stream (every
    keyframe), not from ``loop.correction`` (which the SLAM process emits ONLY
    when a loop closes -- the old bug: no dots along the path until the first
    loop). ``loop.correction`` is untouched; this is a parallel live-only overlay.

    Combining the dense VIO trail with the keyframe seqs SLAM publishes lets it
    compute the SLAM-corrected VIO line: the dense VIO trail rubber-sheeted
    (piecewise-linearly in seq) so each keyframe anchor lands on its loop-
    corrected SLAM position -- see :meth:`corrected_vio_snapshot`.

    The viewer reads :meth:`slam_overlay_snapshot` (kf dots + loop flash),
    :meth:`refined_path_snapshot` (the SLAM kf line), :meth:`vo_snapshot`,
    :meth:`ba_snapshot`, and :meth:`corrected_vio_snapshot` (the deformed dense
    line + teleport flags). All snapshots are cheap (lock + copy / vectorised
    interp) so they can be polled at GUI rate (10-30 Hz). Mirrors the conversion
    in the single-process ``FlowPoseSource._poll_overlay``.
    """

    def __init__(self, endpoint: str, *, vio_endpoint: str,
                 connect_timeout_s: float = 30.0) -> None:
        super().__init__(name=f"slam-map-{endpoint}", daemon=True)
        self.endpoint = endpoint
        self.vio_endpoint = vio_endpoint
        self._connect_timeout_s = float(connect_timeout_s)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._client: IPCPubSub | None = None
        self._vio_client: IPCPubSub | None = None
        # SLAM keyframe overlay (corrected keyframe positions + flash state).
        self._kf_ned: np.ndarray = np.zeros((0, 3), np.float32)
        self._match_ned: np.ndarray = np.zeros((0, 3), np.float32)
        self._n_loops = 0
        self._flash_id = 0
        # Corrected SLAM keyframes keyed by source frame seq (for the deform).
        self._kf_seqs: np.ndarray = np.zeros(0, np.int64)
        self._kf_corr_ned: np.ndarray = np.zeros((0, 3), np.float64)
        # Dense VIO trail (parallel seq + NED-position arrays, bounded ring).
        self._vio_seqs: np.ndarray = np.zeros(0, np.int64)
        self._vio_ned: np.ndarray = np.zeros((0, 3), np.float64)
        # Pure-vision VO trail + windowed-BA trail (NED, bounded rings). Plain
        # position rings -- they are drawn as-is, with no seq-keyed deform.
        self._vo_ned: np.ndarray = np.zeros((0, 3), np.float32)
        self._ba_ned: np.ndarray = np.zeros((0, 3), np.float32)

    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        """Wipe EVERY display buffer feeding the 4 tracker-driven lines.

        Resets, under the lock, all internal buffers so vo_snapshot /
        ba_snapshot / corrected_vio_snapshot / refined_path_snapshot /
        slam_overlay_snapshot all return empty until new data arrives.

        DISPLAY-ONLY: the vio/slam processes keep their full state, so the
        lines rebuild from new incoming data -- the SLAM line/dots repopulate on
        the next ``slam.map`` (which always carries the full current map), and
        the VO / VIO-BA / SLAM-corrected trails restart from the next poses.
        That immediate-rebuild behaviour is intended (paired with the live
        PoseHistory.clear() for the green VIO trail).
        """
        with self._lock:
            # SLAM keyframe overlay (cyan line, amber kf dots, loop flash).
            self._kf_ned = np.zeros((0, 3), np.float32)
            self._match_ned = np.zeros((0, 3), np.float32)
            self._n_loops = 0
            self._flash_id = 0
            # Corrected-keyframe anchors used by the seq-keyed deform.
            self._kf_seqs = np.zeros(0, np.int64)
            self._kf_corr_ned = np.zeros((0, 3), np.float64)
            # Dense VIO trail (seq + pos) used to build the corrected line.
            self._vio_seqs = np.zeros(0, np.int64)
            self._vio_ned = np.zeros((0, 3), np.float64)
            # Pure-vision VO trail + windowed-BA trail.
            self._vo_ned = np.zeros((0, 3), np.float32)
            self._ba_ned = np.zeros((0, 3), np.float32)

    # ------------------------------------------------------------------ #
    def _on_slammap(self, wm: WireSlamMap | object) -> None:
        # Guard the wire-level END (a `WireEnd` instance from IPCPubSub, not
        # the local `END` sentinel) so the shutdown END isn't read as a map.
        if wm is END or isinstance(wm, WireEnd):
            return
        # `slam.map` carries the current (corrected) CAMERA-OPTICAL keyframe
        # positions every keyframe. Convert to NED for the viewer overlay.
        kf_pos = np.asarray(wm.kf_positions, dtype=np.float64)  # type: ignore[attr-defined]
        kf_ned64 = (kf_pos @ _M_OPT_TO_NED.T if len(kf_pos)
                    else np.zeros((0, 3), np.float64))
        kf_ned = kf_ned64.astype(np.float32)
        # The REAL per-keyframe source seqs (carried in kf_ids) -- aligned with
        # kf_positions -- so the deform can match each keyframe to a VIO pose.
        kf_seqs = np.asarray(wm.kf_ids, dtype=np.int64).reshape(-1)  # type: ignore[attr-defined]
        if len(kf_seqs) != len(kf_ned64):         # malformed -> drop the seqs
            kf_seqs = np.zeros(0, np.int64)
            kf_corr = np.zeros((0, 3), np.float64)
        else:
            kf_corr = kf_ned64
        with self._lock:
            self._kf_ned = kf_ned
            self._kf_seqs = kf_seqs
            self._kf_corr_ned = kf_corr
            if int(wm.n_loops) > self._n_loops:   # type: ignore[attr-defined]
                # A NEW loop just closed -> bump the flash id and latch the
                # just-revisited keyframes so the viewer can blink them.
                self._n_loops = int(wm.n_loops)   # type: ignore[attr-defined]
                self._flash_id += 1
                match = wm.last_match              # type: ignore[attr-defined]
                self._match_ned = (
                    (np.asarray(match, dtype=np.float64) @ _M_OPT_TO_NED.T)
                    .astype(np.float32)
                    if match is not None and len(match)
                    else np.zeros((0, 3), np.float32))

    def _on_pose(self, wm: WirePoseMsg | object) -> None:
        # Record the dense VIO trail (seq + NED position) for the corrected line.
        # Guard the wire-level END exactly like IpcPoseSource._on_pose does.
        if wm is END or isinstance(wm, WireEnd):
            return
        pos_ned = _M_OPT_TO_NED @ wm.T_world_cam[:3, 3]   # type: ignore[attr-defined]
        seq = int(wm.seq)                                 # type: ignore[attr-defined]
        with self._lock:
            self._vio_seqs = np.append(self._vio_seqs, seq)
            self._vio_ned = np.vstack((self._vio_ned, pos_ned[None, :]))
            # Bound the ring: drop the oldest samples once over the cap.
            if len(self._vio_seqs) > _VIO_BUF_CAP:
                self._vio_seqs = self._vio_seqs[-_VIO_BUF_CAP:]
                self._vio_ned = self._vio_ned[-_VIO_BUF_CAP:]

    def _on_vo(self, wm: WirePoseMsg | object) -> None:
        # Pure-vision ``pose.vo`` trail: convert optical->NED, append to the ring.
        # No seq needed (the VO line is drawn as-is, never deformed).
        if wm is END or isinstance(wm, WireEnd):
            return
        pos_ned = (_M_OPT_TO_NED @ wm.T_world_cam[:3, 3]).astype(np.float32)  # type: ignore[attr-defined]
        with self._lock:
            self._vo_ned = _append_trail(self._vo_ned, pos_ned)

    def _on_refined(self, wm: WirePoseMsg | object) -> None:
        # Windowed-BA ``pose.refined`` keyframe trail: optical->NED, append.
        if wm is END or isinstance(wm, WireEnd):
            return
        pos_ned = (_M_OPT_TO_NED @ wm.T_world_cam[:3, 3]).astype(np.float32)  # type: ignore[attr-defined]
        with self._lock:
            self._ba_ned = _append_trail(self._ba_ned, pos_ned)

    # ------------------------------------------------------------------ #
    def vo_snapshot(self) -> np.ndarray:
        """The pure-vision VO trail (NED, (N, 3) float32)."""
        with self._lock:
            return self._vo_ned.copy()

    def ba_snapshot(self) -> np.ndarray:
        """The windowed-BA (``pose.refined``) trail (NED, (N, 3) float32)."""
        with self._lock:
            return self._ba_ned.copy()

    def refined_path_snapshot(self) -> np.ndarray:
        with self._lock:
            return self._kf_ned.copy()

    def slam_overlay_snapshot(self):
        # 4-tuple Viewer3D._refresh_overlay expects:
        # (kf_ned, match_ned, loop_segs, flash_id). loop_segs is unused (no
        # polyline segments in proc4) -> always empty.
        with self._lock:
            return (self._kf_ned.copy(), self._match_ned.copy(),
                    [], self._flash_id)

    def corrected_vio_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """The dense VIO trail DEFORMED by SLAM's loop corrections (NED).

        Rubber-sheet: for every keyframe whose source seq is still in the dense
        VIO ring, the correction delta is ``kf_corrected - vio_pos_at(kf_seq)``.
        Each dense pose's delta is then interpolated PIECEWISE-LINEARLY by seq
        between its two bracketing keyframes (flat-clamped before the first /
        after the last keyframe), and added to the raw VIO position. With no
        usable keyframe the raw VIO trail is returned unchanged (delta = 0).

        ``np.interp`` per axis gives exactly that behaviour: it is linear between
        the (sorted, seq-keyed) keyframe deltas and constant outside their range.

        Returns ``(positions (M, 3) float32, teleport (M,) bool)``. A vertex is
        flagged ``teleport`` where the interpolated correction-delta magnitude
        exceeds :data:`TELEPORT_M` -- i.e. where SLAM pulled the path far (a loop
        closure snap), so the viewer can recolour that segment. With no usable
        correction the flags are all ``False`` (delta = 0 everywhere).
        """
        # Snapshot under the lock, then compute lock-free (the GUI poll thread
        # must not block the IPC recv threads on the interp work).
        with self._lock:
            vio_seqs = self._vio_seqs.copy()
            vio_ned = self._vio_ned.copy()
            kf_seqs = self._kf_seqs.copy()
            kf_corr = self._kf_corr_ned.copy()

        if len(vio_seqs) == 0:
            return (np.zeros((0, 3), np.float32), np.zeros(0, bool))
        no_corr = (vio_ned.astype(np.float32),
                   np.zeros(len(vio_ned), bool))    # raw VIO, nothing flagged
        if len(kf_seqs) == 0:
            return no_corr                           # no corrections yet

        # Map seq -> dense VIO position so we can look up each keyframe's anchor.
        # (np.searchsorted on the sorted VIO seqs; VIO seqs are monotonic in
        # arrival, but sort defensively to be robust to any out-of-order push.)
        order = np.argsort(vio_seqs, kind="stable")
        vs = vio_seqs[order]
        vp = vio_ned[order]
        # Keep only keyframes whose seq actually exists in the VIO ring (others
        # rolled off the bounded buffer and have no anchor to deform around).
        pos = np.searchsorted(vs, kf_seqs)
        in_range = pos < len(vs)
        hit = np.zeros(len(kf_seqs), dtype=bool)
        hit[in_range] = vs[pos[in_range]] == kf_seqs[in_range]
        if not hit.any():
            return no_corr                           # no anchor in range

        kf_seq_hit = kf_seqs[hit]
        delta = kf_corr[hit] - vp[pos[hit]]       # (Kh, 3) correction per keyframe

        # Sort the keyframe anchors by seq for np.interp (requires increasing x).
        ks_order = np.argsort(kf_seq_hit, kind="stable")
        ks = kf_seq_hit[ks_order].astype(np.float64)
        kd = delta[ks_order]
        dense_seq = vio_seqs.astype(np.float64)
        # Build the per-vertex correction delta (the SAME values we add to the
        # raw VIO), keep it so the teleport flag is its magnitude -- no second
        # interp pass.
        corr_delta = np.empty_like(vio_ned)
        for axis in range(3):
            # np.interp: linear between anchors, flat-clamped at the ends -- the
            # exact piecewise-linear-with-flat-ends deform the spec describes.
            corr_delta[:, axis] = np.interp(dense_seq, ks, kd[:, axis])
        corrected = vio_ned + corr_delta
        # Teleport = a large per-vertex correction magnitude (loop-closure snap).
        teleport = np.linalg.norm(corr_delta, axis=1) > TELEPORT_M
        return (corrected.astype(np.float32), teleport)

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        client = IPCPubSub(self.endpoint, role="client",
                           connect_timeout_s=self._connect_timeout_s)
        client.subscribe(topics.SLAM_MAP, self._on_slammap)
        # SECOND client: VIO's pose.odom feeds the dense trail for the deform;
        # pose.vo + pose.refined feed the VO + VIO-BA trajectory lines.
        vio_client = IPCPubSub(self.vio_endpoint, role="client",
                               connect_timeout_s=self._connect_timeout_s)
        vio_client.subscribe(topics.POSE_ODOM, self._on_pose)
        vio_client.subscribe(topics.POSE_VO, self._on_vo)
        vio_client.subscribe(topics.POSE_REFINED, self._on_refined)
        try:
            client.start()
            vio_client.start()
        except (TimeoutError, ConnectionError) as e:
            LOG.error("slam-map-tracker: connect failed: %s", e)
            try:
                client.stop()
            except Exception:                                      # noqa: BLE001
                pass
            try:
                vio_client.stop()
            except Exception:                                      # noqa: BLE001
                pass
            return
        self._client = client
        self._vio_client = vio_client
        try:
            self._stop.wait()
        finally:
            for c in (client, vio_client):
                try:
                    c.stop()
                except Exception:                                  # noqa: BLE001
                    pass
            self._client = None
            self._vio_client = None

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------- #
# Calibration handshake -- mirrors the one used in vio.main / slam.main
# --------------------------------------------------------------------------- #
def _await_calib_bundle(endpoint: str, timeout_s: float) -> WireCalibBundle:
    bundle: list[WireCalibBundle | None] = [None]
    got = threading.Event()

    def on_calib(wm: WireCalibBundle) -> None:
        bundle[0] = wm
        got.set()

    client = IPCPubSub(endpoint, role="client", connect_timeout_s=timeout_s)
    client.subscribe("calib.bundle", on_calib)
    client.start()
    try:
        if not got.wait(timeout=timeout_s):
            raise TimeoutError(
                f"ui: no calib.bundle from {endpoint!r} in {timeout_s}s")
    finally:
        client.stop()
    assert bundle[0] is not None
    return bundle[0]


# --------------------------------------------------------------------------- #
# Startup calibration nag (factored out so it's unit-testable offscreen).
# --------------------------------------------------------------------------- #
def install_calib_nag(win, toolbar, status: dict, open_dialog) -> object | None:
    """Surface the startup calibration notification on ``win`` -- NON-BLOCKING.

    Given the unified ``status`` dict (from
    :func:`~imu_camera.mathlib.device.calib_status.calibration_status`):

    * If anything is missing: show a prominent (but auto-fading, dismissible)
      status-bar message naming the missing items + the accuracy risk, AND add a
      PERSISTENT clickable "⚠ CALIB INCOMPLETE" indicator to ``toolbar`` that calls
      ``open_dialog`` (opens the status dialog). Returns the indicator button so the
      caller / a test can find it; also stashed on ``win._calib_nag_btn``.
    * If everything is calibrated: show a brief "calibration OK" confirmation, add NO
      indicator (no nag), and return ``None``.

    No blocking modal is ever shown on launch -- the indicator is the durable signal,
    the status-bar line is the immediate (fading) one. Qt is imported lazily so the
    module stays importable without PyQt6 (mirroring :func:`run_ui`).
    """
    from PyQt6.QtWidgets import QPushButton
    from ui.qt import theme

    if status["all_calibrated"]:
        win.statusBar().showMessage("Calibration OK — gyro, accel & camera.", 4000)
        win._calib_nag_btn = None
        return None

    missing = ", ".join(status["missing"])
    win.statusBar().showMessage(
        f"⚠ Calibration incomplete: {missing} not calibrated — flying "
        f"uncalibrated is inaccurate. Open Calibration ▸ Calibration status…",
        12000)
    # Persistent, clickable indicator. Red BAD-coloured so it reads as a warning;
    # clicking it opens the unified status dialog. Kept on `win` as a lifetime
    # anchor + test hook.
    nag = QPushButton("⚠ CALIB INCOMPLETE")
    nag.setObjectName("CalibNag")
    nag.setStyleSheet(
        f"QPushButton#CalibNag {{ color: {theme.BAD}; font-weight: bold; "
        f"border-color: {theme.BAD}; }}")
    nag.setToolTip(f"Not calibrated: {missing}. Click to open the calibration status.")
    nag.clicked.connect(lambda _c=False: open_dialog())
    toolbar.addSeparator()
    toolbar.addWidget(nag)
    win._calib_nag_btn = nag
    return nag


# --------------------------------------------------------------------------- #
def run_ui(*, vio_endpoint: str = DEFAULT_VIO_ENDPOINT,
           slam_endpoint: str = DEFAULT_SLAM_ENDPOINT,
           capture_endpoint: str = DEFAULT_CAPTURE_ENDPOINT,
           calib_timeout_s: float = 60.0,
           default_view: str = "TOP",
           ba_window: bool = False) -> int:
    """Open the single-view Qt UI (5 trajectories) and block on the Qt event loop."""
    # Import Qt lazily so a headless smoke / CI run that doesn't need the GUI
    # can import this module without pulling PyQt6.
    from PyQt6.QtCore import Qt, QSocketNotifier
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import (
        QApplication, QHBoxLayout, QMainWindow, QPushButton,
        QToolBar, QVBoxLayout, QWidget,
    )

    from ui.qt import theme
    from ui.qt.viewer3d import Viewer3D, VIEW_PRESETS
    from ui.qt.panels import TelemetryPanel
    from ui.qt.synced_window import SyncedViewWindow
    from ui.qt.keypoints_window import KeypointTrackWindow
    from ui.qt.gyrofuse_window import GyroFuseWindow
    from ui.qt.loop_window import LoopClosureWindow
    from ui.qt.ba_window import BaWindow
    from ui.qt.map_window import MapWindow
    from ui.qt.calib_dialogs import GyroCalibDialog, AccelCalibDialog
    from ui.qt.camera_calib_dialog import CameraCalibWizard
    from ui.qt.calib_status_dialog import CalibrationStatusDialog
    from imu_camera.mathlib.device.calib_status import calibration_status
    from ui.modules import (
        IpcImuRawSource, IpcStereoRawSource, IpcGyroFuseSource,
        ipc_triplet_factory, ipc_keypoint_factory, ipc_slam_map_factory,
        ipc_loop_factory, ipc_ba_window_factory,
    )

    # 1. Wait for VIO + SLAM to be ready (and learn the capture resolution).
    LOG.info("ui: waiting for calib.bundle on %s ...", vio_endpoint)
    vio_bundle = _await_calib_bundle(vio_endpoint, calib_timeout_s)
    LOG.info("ui: vio ready (%dx%d)", vio_bundle.width, vio_bundle.height)
    LOG.info("ui: waiting for calib.bundle on %s ...", slam_endpoint)
    slam_bundle = _await_calib_bundle(slam_endpoint, calib_timeout_s)
    LOG.info("ui: slam ready (%dx%d)", slam_bundle.width, slam_bundle.height)

    # 2. Qt app. Share GL contexts across windows: the main Viewer3D and the
    # SLAM-Map room view are each a pyqtgraph GLViewWidget = a separate GL
    # context. Without sharing, the 2nd window's shader-program ids are not valid
    # in the 1st context, so glUseProgram throws GLError 1281/1282 ('invalid
    # value/operation') on every paint. AA_ShareOpenGLContexts must be set BEFORE
    # the QApplication is constructed.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv if sys.argv else ["ui"])
    app.setStyleSheet(theme.QSS)

    # 3. Build the SINGLE view: one Viewer3D + one TelemetryPanel + one live
    #    PoseHistory. The live marker = pose.odom (VIO); all 5 trajectory lines
    #    render in this one viewer, each fed by a snapshot getter on the single
    #    SlamMapTracker (which subscribes odom/vo/refined on the vio endpoint and
    #    slam.map on the slam endpoint).
    history = PoseHistory(capacity=200_000)
    viewer = Viewer3D(history, default_view=default_view)

    # The VIO marker source feeds the live PoseHistory (green trail + marker).
    vio_source = IpcPoseSource(vio_endpoint, label="vio",
                               connect_timeout_s=calib_timeout_s)
    # The single tracker owns the other 4 lines' data (VO / VIO-BA / corrected /
    # SLAM) across its two IPC clients.
    tracker = SlamMapTracker(slam_endpoint, vio_endpoint=vio_endpoint,
                             connect_timeout_s=calib_timeout_s)
    tracker.start()

    # Wire each viewer line to its tracker snapshot getter (green VIO line + the
    # live marker come from `history`, fed by vio_source.start below).
    viewer.set_vo_path_source(tracker.vo_snapshot)               # 1. VO (grey)
    viewer.set_ba_path_source(tracker.ba_snapshot)               # 3. VIO-BA (blue)
    viewer.set_corrected_path_source(tracker.corrected_vio_snapshot)  # 4. orange
    viewer.set_refined_path_source(tracker.refined_path_snapshot)     # 5. cyan
    viewer.set_overlay_source(tracker.slam_overlay_snapshot)          # 5. kf dots

    # 4. Push poses from the marker source -> PoseHistory. The viewer redraws on
    #    a QTimer (~60 Hz) inside Viewer3D and reads from PoseHistory under a
    #    lock, so it's safe to append from the IPC recv thread directly.
    vio_source.start(history.push)

    # 5. Compose the central widget: a top Controls row (toggles + Clear/Restart)
    #    handled by the toolbar below, then the viewer beside its telemetry panel.
    central = QWidget()
    root = QVBoxLayout(central)
    root.setContentsMargins(6, 6, 6, 6)
    root.setSpacing(6)
    body = QWidget()
    bh = QHBoxLayout(body)
    bh.setContentsMargins(0, 0, 0, 0)
    bh.setSpacing(6)
    bh.addWidget(viewer, 1)
    panel = TelemetryPanel(history, source_fps_getter=lambda: vio_source.fps)
    bh.addWidget(panel, 0)
    root.addWidget(body, 1)

    win = QMainWindow()
    win.setWindowTitle("OAK-D · 4-proc · 5-trajectory view")
    win.resize(1400, 860)
    win.setStyleSheet(theme.QSS)
    win.setCentralWidget(central)

    # 6. Menu bar (View / Visualize / Calibration). Mirrors the single-process
    #    main window menus, but device-agnostic: every action drives the
    #    UNCHANGED windows + dialogs over IPC (capture owns the device, the UI
    #    only consumes topics). No _release_device dance -- there is nothing to
    #    release here. There is now ONE viewer/history, so the View + Clear Trail
    #    actions target it directly.
    W, H = int(vio_bundle.width), int(vio_bundle.height)
    dev_id = vio_bundle.device_id or "default"

    # 6a. Always-visible Controls toolbar (top of the window). It holds the 5
    #     per-line toggle buttons, then "Clear Trail" (per-run trajectory reset)
    #     and "Restart". Restart is needed because the IPC bus is one-way
    #     (server->client): the UI cannot reset vio/slam in place, so it asks the
    #     launcher to respawn the whole pipeline by quitting with
    #     RESTART_EXIT_CODE (the launcher loops on it).
    restart_requested = [False]

    def _do_restart() -> None:
        restart_requested[0] = True
        app.quit()                              # launcher sees RESTART_EXIT_CODE

    tb = QToolBar("Controls")
    tb.setMovable(False)
    tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    win.addToolBar(tb)                          # docks at the top by default

    # 5 per-line toggles (checkable). DEFAULT: only VIO is on -- the other lines
    # (VO / VIO-BA / SLAM-corrected / SLAM) start OFF so the view isn't cluttered;
    # the user enables them as needed. Each drives its viewer visibility setter
    # via the toggled(bool) signal. Order = the drift/back-to-front order.
    # (label, viewer visibility setter, default-on)
    for _label, _setter, _on in (("VO", viewer.set_vo_visible, False),
                                 ("VIO", viewer.set_vio_visible, True),
                                 ("VIO-BA", viewer.set_ba_visible, False),
                                 ("SLAM-corrected VIO",
                                  viewer.set_corrected_visible, False),
                                 ("SLAM", viewer.set_slam_visible, False)):
        _btn = QPushButton(_label)
        _btn.setCheckable(True)
        _btn.setChecked(_on)
        _btn.toggled.connect(_setter)
        # setChecked(False) on a fresh button fires no toggled, so call the
        # setter explicitly to force each line's initial visibility to match.
        _setter(_on)
        tb.addWidget(_btn)
    tb.addSeparator()

    # "Clear Trail" wipes the DISPLAY of ALL 5 lines: the live PoseHistory (the
    # green VIO trail + marker) AND every tracker buffer (VO / VIO-BA /
    # SLAM-corrected / SLAM line + kf dots). This is a display wipe only -- the
    # vio/slam processes keep their state, so the lines rebuild from the next
    # incoming data (see SlamMapTracker.clear / PoseHistory.clear).
    def _clear_trail() -> None:
        history.clear()
        tracker.clear()

    clear_trail_act = QAction("Clear Trail", win)
    clear_trail_act.triggered.connect(lambda _c=False: _clear_trail())
    tb.addAction(clear_trail_act)
    restart_act = QAction("Restart", win)
    restart_act.triggered.connect(lambda _c=False: _do_restart())
    tb.addAction(restart_act)

    mbar = win.menuBar()
    # Render the menu bar INSIDE the window on every platform (Linux/Windows do
    # this anyway; macOS would otherwise hoist it into the global top-of-screen
    # bar). Forcing in-window keeps the UI identical across platforms -- the same
    # cross-platform goal as the rest of the proc4 stack. The menu itself is
    # plain Qt (QMenuBar/QAction), so it already runs on Linux; this only fixes
    # WHERE it draws.
    mbar.setNativeMenuBar(False)

    # View: preset cameras + follow, on the single viewer. ("Clear Trail" moved
    # to the always-visible Controls toolbar above.)
    view_menu = mbar.addMenu("View")
    for name in VIEW_PRESETS:
        act = QAction(name.title(), win)
        # Capture `name` (the loop var) per-iteration, else every action would
        # bind the last value; `_c` swallows QAction.triggered's bool arg.
        act.triggered.connect(lambda _c=False, n=name: viewer.set_view(n))
        view_menu.addAction(act)
    view_menu.addSeparator()
    follow_act = QAction("Follow Camera", win)
    follow_act.setCheckable(True)
    follow_act.toggled.connect(lambda on: viewer.set_follow(on))
    view_menu.addAction(follow_act)
    # (No "Clear Keyframes": proc4's SLAM source has no clear_slam_map and there
    #  is no UI->SLAM channel, so it would be a dead action.)

    # Visualize: the EXISTING windows, fed by the IPC adapters. Cache one
    # instance per kind on `win` so repeated opens reuse it (and its single IPC
    # worker) instead of stacking subscribers. `fps` is just the UI redraw
    # cadence -- the data itself is event-driven over IPC.
    vis_menu = mbar.addMenu("Visualize")

    def _open_triplet() -> None:
        if getattr(win, "_triplet_win", None) is None:
            win._triplet_win = SyncedViewWindow(
                ipc_triplet_factory(capture_endpoint, W, H),
                fps=20, parent=win)
        win._triplet_win.show()
        win._triplet_win.raise_()
        win._triplet_win.activateWindow()
        win._triplet_win.ensure_started()   # retry on every open
        win.statusBar().showMessage("Camera + Depth + IMU triplet opened.", 2500)

    triplet_act = QAction("Camera + Depth + IMU (triplet)…", win)
    triplet_act.triggered.connect(_open_triplet)
    vis_menu.addAction(triplet_act)

    def _open_keypoints() -> None:
        if getattr(win, "_keypoints_win", None) is None:
            win._keypoints_win = KeypointTrackWindow(
                ipc_keypoint_factory(capture_endpoint, vio_endpoint, W, H),
                fps=20, parent=win)
        win._keypoints_win.show()
        win._keypoints_win.raise_()
        win._keypoints_win.activateWindow()
        win._keypoints_win.ensure_started()
        win.statusBar().showMessage("Keypoint depth tracker opened.", 2500)

    keypoints_act = QAction("Keypoint Depth Tracker…", win)
    keypoints_act.triggered.connect(_open_keypoints)
    vis_menu.addAction(keypoints_act)

    def _open_gyrofuse() -> None:
        if getattr(win, "_gyrofuse_win", None) is None:
            # Source factory binds to VIO's endpoint (frame.gyrofuse publisher);
            # device-agnostic, like the other Visualize windows.
            win._gyrofuse_win = GyroFuseWindow(
                lambda: IpcGyroFuseSource(
                    vio_endpoint, connect_timeout_s=calib_timeout_s),
                parent=win)
        win._gyrofuse_win.show()
        win._gyrofuse_win.raise_()
        win._gyrofuse_win.activateWindow()
        win._gyrofuse_win.ensure_started()
        win.statusBar().showMessage("Gyro fusion strip chart opened.", 2500)

    gyrofuse_act = QAction("Gyro Fusion (strip chart)…", win)
    gyrofuse_act.triggered.connect(_open_gyrofuse)
    vis_menu.addAction(gyrofuse_act)

    def _open_loop() -> None:
        if getattr(win, "_loop_win", None) is None:
            # Source binds SLAM's endpoint (slam.loop match funnel) + VIO's
            # endpoint (keyframe gray + its kf rings); device-agnostic, like the
            # other Visualize windows.
            win._loop_win = LoopClosureWindow(
                ipc_loop_factory(slam_endpoint, vio_endpoint, W, H),
                parent=win)
        win._loop_win.show()
        win._loop_win.raise_()
        win._loop_win.activateWindow()
        win._loop_win.ensure_started()
        win.statusBar().showMessage("Loop Closure window opened.", 2500)

    loop_act = QAction("Loop Closure…", win)
    loop_act.triggered.connect(_open_loop)
    vis_menu.addAction(loop_act)

    def _open_ba_window() -> None:
        if getattr(win, "_ba_window_win", None) is None:
            # Source binds VIO's endpoint (the ba.window solve-snapshot publisher,
            # present only when VIO ran with --ba-window). live=True -> the window
            # defaults to "Follow latest" (rolling head); the user unchecks it to
            # scrub the buffered timeline. Device-agnostic, like the other windows.
            win._ba_window_win = BaWindow(
                ipc_ba_window_factory(vio_endpoint,
                                      connect_timeout_s=calib_timeout_s),
                live=True, parent=win)
        win._ba_window_win.show()
        win._ba_window_win.raise_()
        win._ba_window_win.activateWindow()
        win._ba_window_win.ensure_started()
        win.statusBar().showMessage("BA Window opened.", 2500)

    ba_window_act = QAction("BA Window…", win)
    ba_window_act.triggered.connect(_open_ba_window)
    if not ba_window:
        # The pipeline was not launched with --ba-window, so VIO never publishes
        # ba.window -- disable the action (with a hint) rather than open a window
        # that would sit forever on its "waiting" frame.
        ba_window_act.setEnabled(False)
        ba_window_act.setToolTip("Launch with --ba-window to enable the BA Window "
                                 "(VIO publishes ba.window only then).")
    vis_menu.addAction(ba_window_act)

    # SLAM Map (3D room): a ModalAI/VOXL-style VOXEL OCCUPANCY map of the room
    # (clean green voxel cubes -- floor grid + walls + furniture), in the same ENU
    # frame as the main Viewer3D. The IpcSlamMapSource consumes VIO's ``keyframe``
    # (denoised depth via VIO's kf rings) + SLAM's ``slam.map`` (corrected poses)
    # and runs a LOG-ODDS OCCUPANCY GRID with FREE-SPACE RAY CARVING (OctoMap/Voxblox
    # style): each ray adds occupied evidence at its hit and free evidence to the
    # voxels it passes through, so a wrongly-added (stereo-noise) voxel the camera
    # later sees through is CARVED back below threshold and removed -- the map
    # self-cleans as the camera moves. Its callback hands each fresh voxel set to the
    # window via the thread-safe `submit` (a queued signal onto the GUI thread); the
    # window renders light square world-unit points. Cached on `win`; the source is
    # stopped in run_ui's teardown.
    def _open_slam_map() -> None:
        if getattr(win, "_slam_map_win", None) is None:
            win._slam_map_win = MapWindow(title="SLAM Map (3D room)")
        wmap = win._slam_map_win
        # (Re)start the source on every open (the source is one-shot per run, like
        # ensure_started elsewhere): stop a prior one before spawning a fresh one.
        old = getattr(win, "_slam_map_src", None)
        if old is not None:
            try:
                old.stop()
            except Exception:                                      # noqa: BLE001
                pass
        src = ipc_slam_map_factory(vio_endpoint, slam_endpoint,
                                   vio_bundle.K, W, H)()
        win._slam_map_src = src
        src.start_cloud(wmap.submit)
        if src.error:
            win.statusBar().showMessage(f"SLAM map: {src.error}", 4000)
        wmap.show()
        wmap.raise_()
        wmap.activateWindow()
        win.statusBar().showMessage("SLAM Map (3D room) opened.", 2500)

    slam_map_act = QAction("SLAM Map (3D room)…", win)
    slam_map_act.triggered.connect(_open_slam_map)
    vis_menu.addAction(slam_map_act)

    # Calibration: each wizard gets a FRESH IPC IMU source (capture's raw imu.raw)
    # and a modal dialog. We inject `stream=src`, which sets the dialog's
    # `_owns_stream=False` -- so the dialog will NOT stop the stream and WE must,
    # in `finally`. The SAME resolved `dev_id` goes to both source and dialog so
    # the saved calib keys match what capture loads on its next start.
    cal_menu = mbar.addMenu("Calibration")

    def _open_gyro_calib() -> None:
        src = IpcImuRawSource(capture_endpoint, device_id=dev_id,
                              connect_timeout_s=calib_timeout_s)
        try:
            GyroCalibDialog(win, device_id=dev_id, stream=src).exec()
        finally:
            src.stop()

    gyro_act = QAction("Gyroscope Bias…", win)
    gyro_act.triggered.connect(_open_gyro_calib)
    cal_menu.addAction(gyro_act)

    def _open_accel_calib() -> None:
        src = IpcImuRawSource(capture_endpoint, device_id=dev_id,
                              connect_timeout_s=calib_timeout_s)
        try:
            AccelCalibDialog(win, device_id=dev_id, stream=src).exec()
        finally:
            src.stop()

    accel_act = QAction("Accelerometer (6-position)…", win)
    accel_act.triggered.connect(_open_accel_calib)
    cal_menu.addAction(accel_act)

    # Stereo camera calibration: like the IMU wizards, inject a FRESH RAW stereo
    # source (capture's imucam.sample left+right) at the bundle's W/H + dev_id, and
    # WE stop it in `finally` (the wizard does not own the injected stream).
    def _open_camera_calib() -> None:
        src = IpcStereoRawSource(capture_endpoint, W, H, device_id=dev_id,
                                 connect_timeout_s=calib_timeout_s)
        try:
            CameraCalibWizard(win, device_id=dev_id, width=W, height=H,
                              stream=src).exec()
        finally:
            src.stop()

    camera_act = QAction("Camera (stereo) calibration…", win)
    camera_act.triggered.connect(_open_camera_calib)
    cal_menu.addAction(camera_act)

    # 6b. Unified "Calibration status…" -- the ONE place showing all three calib
    #     states + a per-item "Open wizard". The status dialog is device-agnostic
    #     (keyed by `dev_id`) and cv2/depthai-free; it re-queries `calibration_status`
    #     on every show, so a wizard finished from inside it is reflected on the next
    #     open. We INSERT it ABOVE the three wizards (it references their handlers, so
    #     it's defined here but placed first via insertAction).
    def _open_calib_status() -> None:
        dlg = getattr(win, "_calib_status_dlg", None)
        if dlg is None:
            dlg = CalibrationStatusDialog(
                win,
                status_provider=lambda: calibration_status(dev_id),
                openers={"gyro": _open_gyro_calib,
                         "accel": _open_accel_calib,
                         "camera": _open_camera_calib})
            win._calib_status_dlg = dlg
        dlg.show()                              # showEvent re-queries the status
        dlg.raise_()
        dlg.activateWindow()

    status_act = QAction("Calibration status…", win)
    status_act.triggered.connect(_open_calib_status)
    cal_menu.insertAction(gyro_act, status_act)   # first item, above the wizards
    cal_menu.insertSeparator(gyro_act)            # divider before the three wizards

    # 6c. Startup calibration nag -- NON-BLOCKING. Query the unified status and, if
    #     anything is missing, surface a prominent (but dismissible) status-bar
    #     message + a PERSISTENT clickable toolbar indicator. No modal on launch.
    #     Factored into `install_calib_nag` so it's unit-testable offscreen.
    install_calib_nag(win, tb, calibration_status(dev_id), _open_calib_status)

    win.show()

    # SIGTERM from the launcher must exit the Qt loop within the launcher's
    # 10 s deadline (else it SIGKILLs us). A bare `signal.signal` handler is
    # NOT enough on macOS/Linux Qt: the handler runs on the main thread, but
    # only when the Qt event loop wakes -- and the loop is usually blocked in
    # a native syscall (`select` / `CFRunLoop`) with no reason to wake. Result:
    # the Python handler runs after a SIGKILL has already arrived.
    #
    # The Qt-recommended fix is `signal.set_wakeup_fd` + `QSocketNotifier`:
    # the kernel writes the signal number to a non-blocking socket, the
    # notifier turns the resulting "fd readable" event into a Qt slot
    # invocation on the GUI thread, and that slot calls `app.quit()` -- which
    # is now guaranteed to be observed because the event loop has been woken
    # by the socket I/O event itself. See
    # https://doc.qt.io/qt-6/qsocketnotifier.html and
    # https://docs.python.org/3/library/signal.html#signal.set_wakeup_fd
    #
    # `socketpair()` is UNIX-only; the project only targets macOS/Linux (same
    # constraint already imposed by AF_UNIX in `ui.comms`).
    wakeup_rd, wakeup_wr = socket.socketpair()
    wakeup_rd.setblocking(False)
    wakeup_wr.setblocking(False)
    # `set_wakeup_fd` returns the previous wakeup fd so we can restore it on
    # exit (otherwise a re-entrant Qt run in the same interpreter -- e.g. the
    # selftest re-using a `QApplication` singleton -- would leak a stale fd).
    prev_wakeup_fd = signal.set_wakeup_fd(wakeup_wr.fileno())

    stop = [False]

    def _on_sigterm(_signo, _frame):
        # Runs on the main thread. Just flip the flag; the QSocketNotifier
        # slot below (woken by the wakeup-fd byte the kernel just wrote) does
        # the actual `app.quit()`.
        stop[0] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    notifier = QSocketNotifier(wakeup_rd.fileno(), QSocketNotifier.Type.Read)

    def _on_signal_wake(_fd) -> None:
        # Drain whatever the kernel wrote (one byte per pending signal); the
        # exact contents don't matter -- the wake is the signal.
        try:
            wakeup_rd.recv(4096)
        except (BlockingIOError, OSError):
            pass
        if stop[0]:
            LOG.info("ui: SIGTERM -> quitting")
            app.quit()

    notifier.activated.connect(_on_signal_wake)

    LOG.info("ui: entering Qt event loop")
    try:
        rc = app.exec()
    finally:
        # Stop the IPC sources FIRST -- their recv threads otherwise keep the
        # process alive after `app.exec()` returns.
        vio_source.stop()
        tracker.stop()
        # The SLAM-map source is a standalone thread (its window is a plain
        # QMainWindow with no closeEvent worker-stop), so stop it explicitly here.
        _src = getattr(win, "_slam_map_src", None)
        if _src is not None:
            try:
                _src.stop()
            except Exception:                                      # noqa: BLE001
                pass
        # Close any Visualize child windows so their IPC workers stop cleanly
        # (closeEvent stops the worker). The calib dialogs are modal + scoped to
        # their handler's `finally`, so there's nothing to clean up for them here.
        for _attr in ("_triplet_win", "_keypoints_win", "_gyrofuse_win",
                      "_loop_win", "_ba_window_win", "_slam_map_win"):
            _w = getattr(win, _attr, None)
            if _w is not None:
                try:
                    _w.close()
                except Exception:                                  # noqa: BLE001
                    pass
        # Tear down the signal plumbing in reverse order so no signal racing
        # in during teardown writes to a closed socket. `set_wakeup_fd(-1)`
        # (or the previous fd) disables wakeups before we close ours.
        try:
            signal.set_wakeup_fd(prev_wakeup_fd)
        except (ValueError, OSError):
            pass
        notifier.setEnabled(False)
        wakeup_rd.close()
        wakeup_wr.close()
    # The Restart toolbar button flips `restart_requested` then calls app.quit();
    # surface that to the launcher as RESTART_EXIT_CODE so it respawns the whole
    # pipeline. Any normal close (window X, SIGTERM, Ctrl-C) returns the Qt rc.
    if restart_requested[0]:
        LOG.info("ui: restart requested -> exit %d", RESTART_EXIT_CODE)
        return RESTART_EXIT_CODE
    LOG.info("ui: bye (rc=%d)", int(rc))
    return int(rc)


# --------------------------------------------------------------------------- #
def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vio-endpoint", default=DEFAULT_VIO_ENDPOINT)
    ap.add_argument("--slam-endpoint", default=DEFAULT_SLAM_ENDPOINT)
    ap.add_argument("--capture-endpoint", default=DEFAULT_CAPTURE_ENDPOINT,
                    help="capture endpoint for the Visualize/Calibration menus")
    ap.add_argument("--calib-timeout", type=float, default=60.0,
                    help="seconds to wait for each retained calib.bundle")
    ap.add_argument("--default-view", default="TOP",
                    choices=("ISO", "TOP", "FRONT", "SIDE"),
                    help="initial 3D viewer preset (default: top-down)")
    ap.add_argument("--ba-window", action="store_true",
                    help="enable the Visualize ▸ BA Window action (VIO must run "
                         "with --ba-window so it publishes ba.window). Off => the "
                         "menu item is shown but disabled.")
    args = ap.parse_args()
    return run_ui(
        vio_endpoint=args.vio_endpoint,
        slam_endpoint=args.slam_endpoint,
        capture_endpoint=args.capture_endpoint,
        calib_timeout_s=args.calib_timeout,
        default_view=args.default_view,
        ba_window=args.ba_window,
    )


if __name__ == "__main__":
    import os as _os
    _rc = main()
    LOG.info("ui: main returned, calling os._exit(%d)", int(_rc))
    logging.shutdown()
    _os.sys.stdout.flush()
    _os.sys.stderr.flush()
    _os._exit(int(_rc))
