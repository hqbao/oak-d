"""IPC adapters so the EXISTING ``ui.qt`` windows + calib dialogs run in proc4.

The single-process windows drive their views off worker threads that build an
in-process acquisition / odometry graph on a private
:class:`~ui.comms.LocalPubSub` and tap it with a UI sink (see
``ui.qt.synced_window`` / ``ui.qt.keypoints_window``). In the 4-process
``./run.sh --proc`` topology there is no in-process graph: the data already lives
on the capture / VIO IPC servers. This module provides three drop-in adapters
that subscribe those IPC topics and republish them onto the very same local bus
the UNCHANGED UI sinks read -- so the windows + dialogs work identically without
any edit to ``ui.qt`` or ``ui.main``.

Device-agnostic by contract
---------------------------
This module is part of the proc4 UI plumbing, which must stay generic for a
future multi-chip port: it consumes only the abstract IPC topics + Wire POD
types and NEVER imports depthai (no device/chip library) -- that device-agnostic
guarantee is the one the multi-chip port depends on. It does pull PyQt6
transitively (the ``TripletWorker`` / ``KeypointWorker`` base classes live in
the Qt window modules), which is expected -- the UI is a Qt app; "generic" here
means independent of the camera/SoC, not of the GUI toolkit. It does NOT import
``ui.main`` (no import cycle), so ``ui.main`` can import it lazily inside
``run_ui`` to keep its own module import Qt-free.

What each adapter feeds
-----------------------
* :class:`IpcImuRawSource` -- duck-types the calib dialogs' default IMU stream
  for the gyro / accel calib dialogs. Subscribes capture's RAW IMU (``imu.raw``)
  and re-emits one ``(3,)`` sample at a time (the shape the dialog's stillness
  gate / six-face collector expect).
* :class:`IpcTripletWorker` -- a :class:`~ui.qt.synced_window.TripletWorker`
  whose ``_drive`` republishes capture's ``imucam.sample`` + ``frame.depth`` so
  the UNCHANGED :class:`~ui.qt.synced_window.SyncedViewWindow` sink renders the
  triplet.
* :class:`IpcKeypointWorker` -- a
  :class:`~ui.qt.keypoints_window.KeypointWorker` whose ``_drive`` republishes
  capture's ``frame.depth`` plus VIO's ``frame.tracks`` + ``frame.inliers`` (two
  endpoints) so the UNCHANGED
  :class:`~ui.qt.keypoints_window.KeypointTrackWindow` sink renders the
  keypoint overlay.

IPC client error model (how connect failures are surfaced)
----------------------------------------------------------
:meth:`ui.comms.IPCPubSub.start` (role="client") RAISES (``TimeoutError`` /
``ConnectionError``) when the socket never appears within ``connect_timeout_s``;
once connected, a runtime receive error instead sets its ``.error`` attribute.
:class:`~ui.comms.IPCSubscriber` swallows the ``start`` exception inside its own
``run`` (it logs and returns), so to surface a connect failure to the polling
window we (a) check each client's ``.error`` every loop tick and (b) detect a
subscriber that died at start (the client never connected) and report a connect
failure. The base worker ``run`` catches any exception we raise in ``_drive``
into ``self.error`` for the window to display.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ui.comms import topics
from ui.comms.messages import END
from ui.comms import IPCPubSub, IPCSubscriber, RingRegistry
from ui.comms.converters import to_local
from ui.comms.ring_registry import default_capture_specs, default_vio_specs
from ui.comms.lib.misc import geometry
from ui.qt.keypoints_window import KeypointWorker
from ui.qt.synced_window import TripletWorker


def _attach_capture_rings(endpoint: str, width: int, height: int) -> RingRegistry:
    """Attach capture's consumer-side shared-memory rings.

    The rings only exist while the capture process is running, so a failure here
    almost always means capture is down. Re-raise it as a clear, device-agnostic
    reason (the base worker ``run`` lifts it onto ``self.error`` for the window)
    instead of leaking a raw ``/<endpoint>.gray_left`` shared-memory path.
    """
    try:
        return RingRegistry().attach_all(default_capture_specs(
            endpoint=endpoint, width=int(width), height=int(height)))
    except FileNotFoundError as e:
        raise RuntimeError(
            f"capture stream not available on {endpoint!r} "
            f"(is capture running?)") from e


# --------------------------------------------------------------------------- #
# (1) IMU source for the calibration dialogs
# --------------------------------------------------------------------------- #
class IpcImuRawSource:
    """Duck-typed IMU stream over capture's RAW IMU IPC topic.

    The gyro / accel calibration dialogs (:mod:`ui.qt.calib_dialogs`) drive a
    stream object with exactly four touch-points -- ``start(callback)``,
    ``stop()``, ``.error`` and ``.device_id`` -- and feed each ``(3,)`` sample to
    a stillness gate / six-face collector. This adapter offers the same surface
    but sources the samples from capture's retained ``imu.raw`` topic instead of
    opening a device, so the SAME dialogs work unchanged in the 4-process UI.

    NOT an ``ImuStream`` subclass: it shares no implementation, only the duck
    type the dialogs rely on.

    ``imu.raw`` is the RAW, uncalibrated IMU (capture's ``_DATA_TOPICS`` publishes
    it) -- exactly what a calibration must consume (calibrating off an
    already-calibrated stream would be circular).
    """

    def __init__(self, capture_endpoint: str, *,
                 device_id: str = "default",
                 connect_timeout_s: float = 30.0) -> None:
        self._endpoint = capture_endpoint
        self._connect_timeout_s = float(connect_timeout_s)
        # Public attrs the dialog reads (mirror the IMU stream's contract).
        self.device_id: str = device_id
        self.error: str | None = None

        self._client: IPCPubSub | None = None
        # ``imu.raw`` is pure POD (no shared-memory ring), so a bare registry is
        # enough for the converter -- the ``rings`` arg is unused for this topic.
        self._rings = RingRegistry()
        # cb(gyro:(3,), accel:(3,), t_s_seconds) -> None
        self._cb = None

    # ------------------------------------------------------------------ #
    def start(self, callback) -> None:
        """Connect to capture and stream per-sample IMU rows to ``callback``.

        On connect failure set :attr:`error` and return (do NOT raise): the
        dialog polls :attr:`error` on its UI timer and surfaces it itself.
        """
        self._cb = callback
        client = IPCPubSub(self._endpoint, role="client",
                           connect_timeout_s=self._connect_timeout_s)
        client.subscribe(topics.IMU_RAW, self._on_imu)
        try:
            client.start()
        except Exception as e:                                     # noqa: BLE001
            # start() raises on connect timeout / refusal -- surface it for the
            # dialog's poll loop rather than crashing the UI thread.
            self.error = f"capture IMU stream connect failed: {e}"
            return
        self._client = client

    def _on_imu(self, wm) -> None:
        """Receive thread: split a wire IMU batch into per-sample callbacks."""
        if wm is END:
            return
        # IMU_RAW is pure POD; the rings arg is unused for this topic.
        imu = to_local(topics.IMU_RAW, wm, self._rings)
        if imu is END:                                # WireEnd -> local END
            return
        gyro = np.asarray(imu.gyro, dtype=np.float64).reshape(-1, 3)
        accel = np.asarray(imu.accel, dtype=np.float64).reshape(-1, 3)
        imu_ts = np.asarray(imu.imu_ts, dtype=np.int64).reshape(-1)
        m = int(min(gyro.shape[0], accel.shape[0], imu_ts.shape[0]))
        if m == 0:                                    # no samples this interval
            return
        cb = self._cb
        if cb is None:
            return
        # The dialog's collector takes ONE (3,) sample at a time with a float
        # SECONDS timestamp (it computes window_s = t_last - t_start and needs
        # >=80 gyro samples over >=1.0 s). The wire batch is (M, 3) in ns, so
        # emit row-by-row in seconds.
        for i in range(m):
            cb(gyro[i], accel[i], float(imu_ts[i]) * 1e-9)

    def stop(self) -> None:
        """Close the IPC client (idempotent; swallow teardown errors)."""
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.stop()
            except Exception:                                      # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# (1b) Gyro-fusion source for the strip-chart window
# --------------------------------------------------------------------------- #
class IpcGyroFuseSource:
    """Duck-typed gyro-fusion stream over VIO's ``frame.gyrofuse`` IPC topic.

    The "Gyro fusion" strip-chart window (:mod:`ui.qt.gyrofuse_window`) drives a
    stream object with exactly three touch-points -- ``start(callback)``,
    ``stop()`` and ``.error`` -- and feeds each per-frame
    :class:`~ui.comms.messages.FrameGyroFuse` to its chart. This adapter offers
    the same surface but sources the records from VIO's ``frame.gyrofuse`` topic
    (pure POD, no shared-memory ring), so the window needs no device handle.

    ``frame.gyrofuse`` is published ONLY on gyro-fused frames (the VIO publisher
    self-skips when gyro is off / PnP failed), so every record the callback sees
    is a genuine fusion observation -- the chart never gets a garbage frame.
    Mirrors :class:`IpcImuRawSource`'s connect-error model: ``start`` swallows a
    connect timeout onto :attr:`error` (the window polls it) rather than raising.
    """

    def __init__(self, vio_endpoint: str, *,
                 connect_timeout_s: float = 30.0) -> None:
        self._endpoint = vio_endpoint
        self._connect_timeout_s = float(connect_timeout_s)
        self.error: str | None = None
        self._client: IPCPubSub | None = None
        # frame.gyrofuse is pure POD (no ring), so a bare registry suffices for
        # the converter -- the ``rings`` arg is unused for this topic.
        self._rings = RingRegistry()
        self._cb = None

    def start(self, callback) -> None:
        """Connect to VIO and stream each FrameGyroFuse record to ``callback``."""
        self._cb = callback
        client = IPCPubSub(self._endpoint, role="client",
                           connect_timeout_s=self._connect_timeout_s)
        client.subscribe(topics.FRAME_GYROFUSE, self._on_msg)
        try:
            client.start()
        except Exception as e:                                     # noqa: BLE001
            self.error = f"VIO gyro-fusion stream connect failed: {e}"
            return
        self._client = client

    def _on_msg(self, wm) -> None:
        if wm is END:
            return
        msg = to_local(topics.FRAME_GYROFUSE, wm, self._rings)
        if msg is END:                                # WireEnd -> local END
            return
        cb = self._cb
        if cb is not None:
            cb(msg)

    def stop(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.stop()
            except Exception:                                      # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# (2) Triplet worker (image | depth | IMU) for SyncedViewWindow
# --------------------------------------------------------------------------- #
class IpcTripletWorker(TripletWorker):
    """Drive :class:`~ui.qt.synced_window.SyncedViewWindow` over IPC.

    Republishes capture's ``imucam.sample`` + ``frame.depth`` onto the local bus
    that the window's :class:`~ui.modules.triplet.UiTripletModule` sink joins by
    ``seq`` -- so the window renders the exact same triplet it would from the
    in-process front-end, without any edit to the window.
    """

    mode = "IPC"

    def __init__(self, capture_endpoint: str, width: int, height: int, *,
                 connect_timeout_s: float = 10.0) -> None:
        super().__init__()
        self._cap_ep = capture_endpoint
        self._w = int(width)
        self._h = int(height)
        self._connect_timeout_s = float(connect_timeout_s)

    def _drive(self, bus, sink) -> None:
        # Attach capture's shared-memory rings (consumer side) so the subscriber
        # bridge can ``read_copy`` the frame + depth arrays out of them. The
        # rings only exist while capture is up, so a missing ring == capture not
        # running -- surface that as a clear reason, not a raw shm-path error.
        cap_rings = _attach_capture_rings(self._cap_ep, self._w, self._h)
        client = IPCPubSub(self._cap_ep, role="client",
                           connect_timeout_s=self._connect_timeout_s)
        sub = IPCSubscriber(bus, client, cap_rings,
                            [topics.IMUCAM_SAMPLE, topics.FRAME_DEPTH])
        # Mirror the Replay/Live worker lifecycle: start the sink first, then the
        # source bridge; loop until stopped while surfacing the first error.
        sink.start()
        sub.start()
        try:
            while not self._stop.is_set():
                self._stop.wait(0.05)
                err = self._connect_or_runtime_error(client, sub)
                if err is not None:
                    self.error = err
                    break
        finally:
            sub.stop()
            sink.stop()
            cap_rings.close()

    @staticmethod
    def _connect_or_runtime_error(client: IPCPubSub,
                                  sub: IPCSubscriber) -> str | None:
        """First fatal reason from a client, or None.

        ``IPCPubSub.start`` raises on a failed connect; ``IPCSubscriber``
        catches that inside its ``run`` and returns, so a dead subscriber thread
        means the client never connected. A runtime receive error instead lands
        on ``client.error``.
        """
        if client.error:
            return client.error
        if not sub.is_alive():
            return f"capture stream connect failed ({client.endpoint})"
        return None


# --------------------------------------------------------------------------- #
# (3) Keypoint worker (frame + KLT tracks) for KeypointTrackWindow
# --------------------------------------------------------------------------- #
class IpcKeypointWorker(KeypointWorker):
    """Drive :class:`~ui.qt.keypoints_window.KeypointTrackWindow` over IPC.

    The overlay needs three streams from TWO endpoints: ``frame.depth`` (the
    rectified-left image + metric depth) comes from CAPTURE, while
    ``frame.tracks`` + ``frame.inliers`` (the KLT ids/pixels + PnP inliers) come
    from VIO. We republish all three onto the local bus the window's
    :class:`~ui.modules.tracks.UiTracksModule` sink reads.
    """

    mode = "IPC"
    #: Realtime live view -- keep latency bounded (latest-only sink).
    latest_only = True

    def __init__(self, capture_endpoint: str, vio_endpoint: str,
                 width: int, height: int, *,
                 connect_timeout_s: float = 10.0) -> None:
        super().__init__()
        self._cap_ep = capture_endpoint
        self._vio_ep = vio_endpoint
        self._w = int(width)
        self._h = int(height)
        self._connect_timeout_s = float(connect_timeout_s)

    def _drive(self, bus, sink) -> None:
        # Capture's depth ring must be attached so its frame.depth converts; VIO's
        # tracks/inliers are pure POD (no ring) so a bare registry suffices there.
        # A missing ring == capture not running -> surface a clear reason.
        cap_rings = _attach_capture_rings(self._cap_ep, self._w, self._h)
        cap_client = IPCPubSub(self._cap_ep, role="client",
                               connect_timeout_s=self._connect_timeout_s)
        vio_client = IPCPubSub(self._vio_ep, role="client",
                               connect_timeout_s=self._connect_timeout_s)
        # Depth first (per-seq image+depth); tracks/inliers (POD) from VIO.
        cap_sub = IPCSubscriber(bus, cap_client, cap_rings,
                                [topics.FRAME_DEPTH])
        vio_sub = IPCSubscriber(bus, vio_client, RingRegistry(),
                                [topics.FRAME_TRACKS, topics.FRAME_INLIERS])
        sink.start()
        cap_sub.start()
        vio_sub.start()
        try:
            while not self._stop.is_set():
                self._stop.wait(0.05)
                err = self._first_error(((cap_client, cap_sub),
                                         (vio_client, vio_sub)))
                if err is not None:
                    self.error = err
                    break
        finally:
            vio_sub.stop()
            cap_sub.stop()
            sink.stop()
            cap_rings.close()

    @staticmethod
    def _first_error(pairs) -> str | None:
        """First fatal reason across ``(client, sub)`` pairs, or None."""
        for client, sub in pairs:
            if client.error:
                return client.error
            if not sub.is_alive():
                return f"stream connect failed ({client.endpoint})"
        return None


# --------------------------------------------------------------------------- #
# (3b) SLAM 3D-map source (fused keyframe cloud) for MapWindow
# --------------------------------------------------------------------------- #
class IpcSlamMapSource(threading.Thread):
    """Fuse the per-keyframe DENSE depth maps into ONE clean room cloud for MapWindow.

    Subscribes BOTH endpoints the 3D room needs (mirrors :class:`IpcKeypointWorker`'s
    two-endpoint pattern, but the keyframe gray/depth ride VIO's dedicated
    ``kf_gray`` / ``kf_depth`` rings, NOT capture's frame rings):

    * ``keyframe`` (VIO) -> per keyframe ``seq`` we keep its gray image + metric
      depth (``read_copy``-ed out of VIO's kf rings by the converter) and the
      ROTATION of its ``T_world_cam``. The keyframe rate is low (kf_every), so a
      bounded dict of the last :data:`MAX_KEYFRAMES` keyframes caps memory.
    * ``slam.map`` (SLAM) -> the CONTINUOUS per-keyframe CORRECTED positions,
      keyed by source frame seq (``kf_ids`` aligned with ``kf_positions``). We use
      these corrected translations -- not the raw VIO keyframe pose -- so the room
      re-snaps after every loop closure.

    The fused pose per keyframe is therefore ``[R_keyframe | t_corrected]``:
    SLAM's overlay carries only the corrected camera POSITION (a ``(3,)`` per
    keyframe), so we keep each keyframe's own rotation and swap in the corrected
    translation. That is enough to re-anchor every back-projected point to the
    loop-corrected room (orientation drift between corrections is negligible at
    keyframe spacing), and it keeps this a pure CONSUMER of existing topics -- no
    new field, no data-path change.

    Cloud build (the "clean DENSE room" approach -- the middle ground between the
    laggy raw-dense fuse and the too-sparse PnP-inlier landmark set):

    1. :func:`geometry.keyframe_pointcloud` back-projects the DENSE depth map of
       every keyframe (subsampled by :data:`STRIDE`) and stacks them into one world
       cloud -- so the room SURFACE is reconstructed from every viewpoint, giving a
       RECOGNISABLE room (not the ~400 sparse landmarks). It is cleaned at source:
       :data:`EDGE_MAX` drops "flying pixels" at depth discontinuities and the
       ``[min_depth, max_depth]`` gate drops noisy far stereo depth.
    2. :func:`geometry.voxel_downsample` voxel-grid fuses the result: the SAME
       physical surface seen across many keyframes lands in one :data:`VOXEL_M`
       cell and collapses to a single centroid (DEDUP -- this is what keeps the
       overlapping dense maps from exploding the count), thin stereo noise (cells
       with ``< :data:`VOXEL_MIN_COUNT``` points) is dropped, and the total point
       count is BOUNDED by the number of occupied cells. The result is a dense but
       deduped/filtered room at a smooth point count (~tens of thousands), not the
       hundreds-of-thousands raw-dense fuse that lagged.

    ``K`` comes from VIO's retained ``calib.bundle`` (the rectified-left
    intrinsic for the full-res depth grid, the same one
    :func:`geometry.keyframe_pointcloud` expects).

    Rebuild policy: a full-cloud re-fuse over every kept keyframe is heavy, so we
    COALESCE -- a ``slam.map`` (or a new keyframe) just marks the model dirty, and
    a background loop rebuilds at most :data:`REBUILD_HZ` times a second, always
    from the freshest poses. The finished ``(points, colors, cams)`` go to the
    injected ``on_cloud`` callback (the window marshals it onto the GUI thread).

    Connect-error model mirrors :class:`IpcGyroFuseSource` /
    :class:`IpcKeypointWorker`: :meth:`start` swallows a connect timeout onto
    :attr:`error` (the window polls it) rather than raising.
    """

    #: Cap on kept keyframes (bounds memory like the other UI buffers). At
    #: kf_every=5 @ 20 fps a keyframe lands ~every 0.25 s, so 600 kf ~= 2.5 min of
    #: distinct keyframes -- far more than any room reconstruction needs.
    MAX_KEYFRAMES = 600
    #: Max full-cloud rebuilds per second. The DENSE rebuild is heavier than the
    #: old sparse landmark one (it back-projects + voxel-fuses every keyframe's
    #: depth), so we run at 4 Hz: bumped from 3 -> 4 to re-snap the room promptly
    #: on new keyframes + slam.map pose updates, but kept at 4 (not 5) so the
    #: ~0.25 s period stays safely ABOVE the measured rebuild cost as the keyframe
    #: count grows. The build is OFF the GUI thread (this source is a worker thread
    #: that submit()s the finished cloud), so a heavy rebuild never stutters the UI.
    REBUILD_HZ = 4.0
    # --- DENSE-room build tuning (all tunable; see class docstring step 1+2) --- #
    # Tuned on replay corridor_60s (640x400, ~30 placed keyframes): lands ~45k
    # output points in ~0.22 s/rebuild -- a recognisable room at a SMOOTH count,
    # between the laggy 329k raw-dense fuse and the too-sparse ~420 PnP landmarks.
    #: Depth subsample stride per keyframe. STRIDE=3 -> 1/9 the pixels: it both
    #: keeps the output in-band AND keeps the np.unique voxel fuse (the build's
    #: cost driver, ~O(input pixels)) under the rebuild period. The voxel fuse
    #: dedups the rest, so the room shape survives the subsample.
    STRIDE = 3
    #: Drop "flying pixels" at depth discontinuities: reject a pixel whose depth
    #: jumps more than this (m) from a 4-neighbour (foreground/background edges
    #: interpolate to points floating between the two real surfaces).
    EDGE_MAX = 0.10
    #: Voxel-grid edge (m) for the dedup/fuse pass: dense points within one 6 cm
    #: cell (same surface across overlapping keyframes) collapse to one centroid.
    VOXEL_M = 0.06
    #: Min point hits a voxel cell needs to survive: cells hit by >= 2 rays are
    #: real surface; thinner cells are stereo noise, dropped.
    VOXEL_MIN_COUNT = 2

    def __init__(self, vio_endpoint: str, slam_endpoint: str, K: np.ndarray, *,
                 connect_timeout_s: float = 10.0,
                 width: int | None = None, height: int | None = None) -> None:
        super().__init__(name=f"slam-map-src-{slam_endpoint}", daemon=True)
        self._vio_ep = vio_endpoint
        self._slam_ep = slam_endpoint
        self._K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self._connect_timeout_s = float(connect_timeout_s)
        # The kf rings are FIXED shape; if the caller knows the capture
        # resolution use it, else fall back to the canonical default specs.
        self._w = width
        self._h = height

        # Public attrs the window polls (mirror the other sources' contract).
        self.error: str | None = None

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._dirty = threading.Event()
        self._cb = None                       # on_cloud(points, colors, cams)

        self._vio_client: IPCPubSub | None = None
        self._slam_client: IPCPubSub | None = None
        self._vio_rings: RingRegistry | None = None

        # Per-keyframe accumulators, keyed by source frame seq. Insertion order is
        # the keyframe arrival order (Python dicts preserve it), which is the
        # order we fuse in -- the cloud is order-independent anyway.
        self._kf_gray: dict[int, np.ndarray] = {}
        self._kf_depth: dict[int, np.ndarray] = {}
        self._kf_R: dict[int, np.ndarray] = {}        # (3,3) keyframe rotation
        # Latest corrected camera positions from slam.map, keyed by seq.
        self._kf_corr_pos: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def start_cloud(self, on_cloud) -> None:
        """Connect both endpoints and stream fused clouds to ``on_cloud``.

        ``on_cloud(points (N,3) float32, colors (N,3) float32, cams (M,3)
        float32)`` is invoked from this source's REBUILD thread (the window
        marshals it onto the GUI thread). On connect failure :attr:`error` is set
        and the thread is NOT started (the window polls :attr:`error`).
        """
        self._cb = on_cloud
        # Attach VIO's keyframe rings (consumer side) so the keyframe converter
        # can read_copy the gray/depth arrays out of them. Missing rings == VIO
        # not running -> surface a clear, device-agnostic reason.
        try:
            self._vio_rings = self._attach_vio_rings()
        except RuntimeError as e:
            self.error = str(e)
            return

        vio_client = IPCPubSub(self._vio_ep, role="client",
                               connect_timeout_s=self._connect_timeout_s)
        vio_client.subscribe(topics.KEYFRAME, self._on_keyframe)
        slam_client = IPCPubSub(self._slam_ep, role="client",
                                connect_timeout_s=self._connect_timeout_s)
        slam_client.subscribe(topics.SLAM_MAP, self._on_slammap)
        try:
            vio_client.start()
            slam_client.start()
        except Exception as e:                                     # noqa: BLE001
            self.error = f"SLAM-map stream connect failed: {e}"
            for c in (vio_client, slam_client):
                try:
                    c.stop()
                except Exception:                                  # noqa: BLE001
                    pass
            if self._vio_rings is not None:
                self._vio_rings.close()
                self._vio_rings = None
            return
        self._vio_client = vio_client
        self._slam_client = slam_client
        self.start()                                  # spin the rebuild thread

    def _attach_vio_rings(self) -> RingRegistry:
        """Attach VIO's keyframe rings; map a missing ring to a clear reason."""
        # Use the caller's resolution when known, else the canonical default.
        kwargs = {}
        if self._w is not None and self._h is not None:
            kwargs = {"width": int(self._w), "height": int(self._h)}
        try:
            return RingRegistry().attach_all(
                default_vio_specs(endpoint=self._vio_ep, **kwargs))
        except FileNotFoundError as e:
            raise RuntimeError(
                f"VIO keyframe stream not available on {self._vio_ep!r} "
                f"(is VIO running?)") from e

    # ------------------------------------------------------------------ #
    def _on_keyframe(self, wm) -> None:
        """VIO recv thread: stash one keyframe's gray + depth + rotation.

        The dense room is built from each keyframe's full depth map (gray for
        colour), so we keep only the gray/depth grid + the keyframe's rotation --
        the corrected translation comes from ``slam.map``.
        """
        if wm is END:
            return
        kf = to_local(topics.KEYFRAME, wm, self._vio_rings)
        if kf is END:                                 # WireEnd -> local END
            return
        seq = int(kf.seq)
        R = np.asarray(kf.T_world_cam, dtype=np.float64)[:3, :3].copy()
        with self._lock:
            self._kf_gray[seq] = kf.gray_left          # already a private copy
            self._kf_depth[seq] = kf.depth_m
            self._kf_R[seq] = R
            self._evict_locked()
        self._dirty.set()

    def _on_slammap(self, wm) -> None:
        """SLAM recv thread: refresh the corrected per-keyframe positions."""
        if wm is END:
            return
        smap = to_local(topics.SLAM_MAP, wm, RingRegistry())   # POD, no rings
        if smap is END:
            return
        pos = np.asarray(smap.kf_positions, dtype=np.float64).reshape(-1, 3)
        seqs = np.asarray(smap.kf_seqs, dtype=np.int64).reshape(-1)
        if len(seqs) != len(pos):              # malformed -> ignore (keep last)
            return
        with self._lock:
            # slam.map carries the FULL current corrected map every keyframe, so
            # rebuild the dict wholesale (drops keyframes SLAM no longer keeps).
            self._kf_corr_pos = {int(s): pos[i] for i, s in enumerate(seqs)}
        self._dirty.set()

    def _evict_locked(self) -> None:
        """Drop the oldest keyframes once over the cap (call under the lock)."""
        n = len(self._kf_gray)
        if n <= self.MAX_KEYFRAMES:
            return
        # dict preserves insertion order -> the first keys are the oldest.
        for seq in list(self._kf_gray.keys())[: n - self.MAX_KEYFRAMES]:
            self._kf_gray.pop(seq, None)
            self._kf_depth.pop(seq, None)
            self._kf_R.pop(seq, None)

    # ------------------------------------------------------------------ #
    def _build_cloud(self):
        """Fuse the kept keyframes into one DENSE room cloud (CORRECTED poses).

        Returns ``(points, colors, cams)`` (all empty when nothing is usable).
        Only keyframes that have BOTH a stashed gray/depth AND a corrected
        position from slam.map are fused -- so a keyframe SLAM hasn't placed yet
        is skipped (not drawn at a stale raw pose).

        Two-stage clean DENSE build (see class docstring):

        1. :func:`geometry.keyframe_pointcloud` back-projects each keyframe's DENSE
           depth map (subsampled by :data:`STRIDE`) into the world, cleaned at
           source: :data:`EDGE_MAX` drops flying pixels at depth discontinuities
           and ``[min_depth, max_depth]`` gates noisy far stereo depth -- so the
           room SURFACE is recognisable from every viewpoint.
        2. :func:`geometry.voxel_downsample` voxel-fuses the overlapping dense maps:
           the same surface across keyframes collapses to one centroid per
           :data:`VOXEL_M` cell (DEDUP -> bounds the count), and cells with
           ``< :data:`VOXEL_MIN_COUNT``` points are dropped as thin stereo noise.
        """
        with self._lock:
            corr = dict(self._kf_corr_pos)
            seqs = [s for s in self._kf_gray if s in corr]
            grays = [self._kf_gray[s] for s in seqs]
            depths = [self._kf_depth[s] for s in seqs]
            Rs = [self._kf_R[s] for s in seqs]
            ts = [corr[s] for s in seqs]
        if not seqs:
            empty = np.zeros((0, 3), np.float32)
            return empty, empty, empty
        # Compose [R_keyframe | t_corrected] per keyframe (see class docstring).
        poses = []
        cams = np.empty((len(seqs), 3), np.float32)
        for i in range(len(seqs)):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rs[i]
            T[:3, 3] = ts[i]
            poses.append(T)
            cams[i] = ts[i].astype(np.float32)
        # (1) DENSE per-keyframe depth -> world surface, edge-rejected + depth-gated.
        points, colors = geometry.keyframe_pointcloud(
            poses=poses, depths=depths, grays=grays, K=self._K,
            stride=self.STRIDE, edge_max=self.EDGE_MAX)
        # (2) Voxel-fuse the overlapping dense maps: one centroid per occupied cell
        #     (dedup + count cap) and drop thin stereo noise (min_count).
        points, colors = geometry.voxel_downsample(
            points, colors, voxel=self.VOXEL_M, min_count=self.VOXEL_MIN_COUNT)
        return points, colors, cams

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Rebuild the cloud (coalesced) whenever the model goes dirty."""
        period = 1.0 / self.REBUILD_HZ
        while not self._stop.is_set():
            # Wait for a dirty mark (or shutdown); then throttle so a burst of
            # slam.map updates collapses into ONE rebuild per period.
            if not self._dirty.wait(timeout=0.25):
                continue
            if self._stop.is_set():
                break
            self._dirty.clear()
            try:
                points, colors, cams = self._build_cloud()
            except Exception:                          # noqa: BLE001
                # A malformed keyframe must not kill the rebuild thread; skip it.
                time.sleep(period)
                continue
            cb = self._cb
            if cb is not None:
                cb(points, colors, cams)
            # Coalesce: sleep out the rest of the period before honouring the
            # next dirty mark, so we never exceed REBUILD_HZ full rebuilds/s.
            self._stop.wait(period)

    def stop(self) -> None:
        """Close both clients + the kf rings (idempotent)."""
        self._stop.set()
        self._dirty.set()                              # wake the rebuild thread
        for c in (self._vio_client, self._slam_client):
            if c is not None:
                try:
                    c.stop()
                except Exception:                      # noqa: BLE001
                    pass
        self._vio_client = None
        self._slam_client = None
        rings = self._vio_rings
        self._vio_rings = None
        if rings is not None:
            try:
                rings.close()
            except Exception:                          # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# (4) Factory helpers -- the windows want a zero-arg ``worker_factory``
# --------------------------------------------------------------------------- #
def ipc_triplet_factory(capture_endpoint: str, width: int, height: int):
    """Return a zero-arg factory building an :class:`IpcTripletWorker`."""
    return lambda: IpcTripletWorker(capture_endpoint, width, height)


def ipc_keypoint_factory(capture_endpoint: str, vio_endpoint: str,
                         width: int, height: int):
    """Return a zero-arg factory building an :class:`IpcKeypointWorker`."""
    return lambda: IpcKeypointWorker(capture_endpoint, vio_endpoint,
                                     width, height)


def ipc_slam_map_factory(vio_endpoint: str, slam_endpoint: str, K: np.ndarray,
                         width: int, height: int):
    """Return a zero-arg factory building an :class:`IpcSlamMapSource`.

    Binds the VIO endpoint (the ``keyframe`` publisher + its kf rings), the SLAM
    endpoint (the ``slam.map`` corrected poses) and the rectified-left ``K`` from
    the retained calib bundle -- so the caller (``ui.main``) just opens MapWindow
    and starts the returned source.
    """
    return lambda: IpcSlamMapSource(vio_endpoint, slam_endpoint, K,
                                    width=width, height=height)
