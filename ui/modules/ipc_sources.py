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
# (3b) Shared keyframe accumulator (the SHM-ring + recv plumbing both 3D-map
#      sources build on -- so neither copy-pastes the attach / _on_keyframe /
#      evict code).
# --------------------------------------------------------------------------- #
class _KeyframeAccumulator(threading.Thread):
    """Attach VIO's keyframe rings and stash every keyframe for a 3D-map build.

    The occupancy voxel map (:class:`IpcSlamMapSource`) needs, per keyframe, the
    metric ``depth_m`` grid (``read_copy``-ed out of VIO's ``kf_depth`` ring by the
    converter) and the keyframe's FULL pose ``[R_keyframe | t_keyframe]`` (split
    from ``T_world_cam``) to back-project depth to the world. Rather than inline the
    ring attach + ``_on_keyframe`` stash + evict + the coalesced rebuild wiring in
    the source, that machinery lives here as a reusable base; a concrete source
    subclasses this and supplies only its own build (``_build`` -> a payload) + its
    rebuild rate. (Kept as a separate base -- not folded into the one current
    source -- because it is the natural seam for a second keyframe-fed map view and
    keeps the SHM/recv plumbing isolated from the map maths.)

    A bounded dict of the last :data:`MAX_KEYFRAMES` keyframes (keyed by source
    frame seq) caps memory; the subclass's build re-orders by seq, so insertion
    order is not relied on for any index assignment.

    Lifecycle (subclasses inherit it, only set ``_cb`` + spin via :meth:`start`):

    * :meth:`_attach_or_fail` attaches the kf rings (a missing ring == VIO not
      running -> a clear, device-agnostic reason on :attr:`error`).
    * :meth:`_make_keyframe_client` builds the VIO IPC client subscribed to
      ``keyframe`` (the concrete source adds any extra clients it needs, e.g.
      the occupancy map's ``slam.map``).
    * a coalesced rebuild loop (:meth:`run`) re-builds at most :attr:`REBUILD_HZ`
      times a second whenever the model goes dirty, OFF the GUI thread, and hands
      each finished payload to the injected callback (the window marshals it onto
      the GUI thread via its thread-safe ``submit``).

    Connect-error model mirrors the other sources: a connect timeout is swallowed
    onto :attr:`error` (the window polls it) rather than raising.
    """

    #: Cap on kept keyframes (bounds memory like the other UI buffers). At
    #: kf_every=5 @ 20 fps a keyframe lands ~every 0.25 s, so 600 kf ~= 2.5 min of
    #: distinct keyframes -- far more than any 3D map needs. The occupancy grid is
    #: PERSISTENT (it accumulates hit counts as keyframes arrive), so an evicted
    #: keyframe's hits stay folded into the grid -- this cap only bounds the raw
    #: depth-grid buffer, not the fused map.
    MAX_KEYFRAMES = 600
    #: Valid-depth band (m): outside this range stereo depth is too noisy to
    #: back-project. The occupancy build gates the whole depth grid to this band.
    MIN_DEPTH_M = 0.3
    MAX_DEPTH_M = 6.0
    #: Max full rebuilds per second. Subclasses override.
    REBUILD_HZ = 4.0

    def __init__(self, vio_endpoint: str, *, name: str,
                 connect_timeout_s: float = 10.0,
                 width: int | None = None, height: int | None = None) -> None:
        super().__init__(name=name, daemon=True)
        self._vio_ep = vio_endpoint
        self._connect_timeout_s = float(connect_timeout_s)
        # kf rings are FIXED shape; use the caller's resolution when known, else
        # the canonical default specs.
        self._w = width
        self._h = height

        # Public attr the window polls (mirrors the other sources' contract).
        self.error: str | None = None

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._dirty = threading.Event()
        self._cb = None                       # build payload sink (set by subclass)

        self._vio_client: IPCPubSub | None = None
        self._vio_rings: RingRegistry | None = None

        # Per-keyframe accumulators, keyed by source frame seq. The occupancy map
        # needs only the metric depth grid + each keyframe's own pose
        # (rotation AND translation) to back-project depth to the world.
        # ``MAX_KEYFRAMES`` bounds the raw depth-grid buffer.
        self._kf_depth: dict[int, np.ndarray] = {}
        self._kf_R: dict[int, np.ndarray] = {}        # (3,3) keyframe rotation
        self._kf_t: dict[int, np.ndarray] = {}        # (3,) VIO keyframe translation

    # ------------------------------------------------------------------ #
    def _attach_or_fail(self) -> bool:
        """Attach VIO's kf rings; on failure set :attr:`error` and return False."""
        try:
            self._vio_rings = self._attach_vio_rings()
            return True
        except RuntimeError as e:
            self.error = str(e)
            return False

    def _attach_vio_rings(self) -> RingRegistry:
        """Attach VIO's keyframe rings; map a missing ring to a clear reason."""
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

    def _make_keyframe_client(self) -> IPCPubSub:
        """Build the VIO IPC client subscribed to the ``keyframe`` topic."""
        client = IPCPubSub(self._vio_ep, role="client",
                           connect_timeout_s=self._connect_timeout_s)
        client.subscribe(topics.KEYFRAME, self._on_keyframe)
        self._vio_client = client
        return client

    def _on_keyframe(self, wm) -> None:
        """VIO recv thread: stash one keyframe's metric depth + FULL pose.

        Splits the (4,4) VIO world<-cam pose into rotation + translation; the
        build back-projects this keyframe's depth with ``Xw = R Xc + t`` so the
        hits land in a single consistent odom-frame grid (one pose source per
        keyframe -> no seams). The depth grid is the keyframe's DENOISED metric
        depth (``read_copy``-ed out of VIO's ``kf_depth`` ring by the converter).
        Marks the model dirty so the subclass NEXT rebuild folds it in (the
        occupancy build INCREMENTS the persistent hit-count grid from the
        not-yet-fused keyframes -- see :meth:`IpcSlamMapSource._build`).
        """
        if wm is END:
            return
        kf = to_local(topics.KEYFRAME, wm, self._vio_rings)
        if kf is END:                                 # WireEnd -> local END
            return
        seq = int(kf.seq)
        T = np.asarray(kf.T_world_cam, dtype=np.float64)
        R = T[:3, :3].copy()
        t = T[:3, 3].copy()
        with self._lock:
            self._kf_depth[seq] = kf.depth_m
            self._kf_R[seq] = R
            self._kf_t[seq] = t
            self._evict_locked()
        self._dirty.set()

    def _evict_locked(self) -> None:
        """Drop the oldest keyframes once over the cap (call under the lock)."""
        n = len(self._kf_depth)
        if n <= self.MAX_KEYFRAMES:
            return
        # dict preserves insertion order -> the first keys are the oldest.
        for seq in list(self._kf_depth.keys())[: n - self.MAX_KEYFRAMES]:
            self._kf_depth.pop(seq, None)
            self._kf_R.pop(seq, None)
            self._kf_t.pop(seq, None)
            self._on_evict_locked(seq)

    def _on_evict_locked(self, seq: int) -> None:
        """Hook for a subclass to drop its OWN per-seq state (call under lock)."""

    # ------------------------------------------------------------------ #
    def _build(self):
        """Build the source's payload from the accumulated keyframes.

        Implemented by each concrete source; the return value is passed straight
        to the injected callback. Runs OFF the GUI thread on the rebuild loop.
        """
        raise NotImplementedError

    def run(self) -> None:
        """Coalesced rebuild loop: re-build when dirty, capped at REBUILD_HZ."""
        period = 1.0 / self.REBUILD_HZ
        while not self._stop.is_set():
            # Wait for a dirty mark (or shutdown); then throttle so a burst of
            # updates collapses into ONE rebuild per period.
            if not self._dirty.wait(timeout=0.25):
                continue
            if self._stop.is_set():
                break
            self._dirty.clear()
            try:
                payload = self._build()
            except Exception:                          # noqa: BLE001
                # A malformed keyframe must not kill the rebuild thread; skip it.
                time.sleep(period)
                continue
            cb = self._cb
            if cb is not None:
                cb(*payload)
            # Coalesce: sleep out the rest of the period before honouring the next
            # dirty mark, so we never exceed REBUILD_HZ full rebuilds/s.
            self._stop.wait(period)

    def _stop_extra_clients(self) -> None:
        """Hook for a subclass to stop any EXTRA clients it opened (e.g. slam)."""

    def stop(self) -> None:
        """Close the VIO client + kf rings (+ any subclass clients); idempotent."""
        self._stop.set()
        self._dirty.set()                              # wake the rebuild thread
        client = self._vio_client
        self._vio_client = None
        if client is not None:
            try:
                client.stop()
            except Exception:                          # noqa: BLE001
                pass
        self._stop_extra_clients()
        rings = self._vio_rings
        self._vio_rings = None
        if rings is not None:
            try:
                rings.close()
            except Exception:                          # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# (3c) SLAM 3D-map source (ModalAI-style VOXEL OCCUPANCY map) for MapWindow
# --------------------------------------------------------------------------- #
class IpcSlamMapSource(_KeyframeAccumulator):
    """Build a ModalAI/VOXL-style VOXEL OCCUPANCY map of the room for MapWindow.

    Instead of a noisy per-keyframe point cloud, this builds the clean blocky
    occupancy map the user wants (floor grid + walls + furniture as green voxel
    cubes) the way an OctoMap does it: by TEMPORAL OCCUPANCY FUSION. The earlier
    naive per-keyframe voxel binning was noisy + laggy precisely because it lacked
    this fusion -- it re-binned every keyframe from scratch and kept every cell a
    ray ever touched (so transient stereo noise survived). Here a PERSISTENT
    per-voxel HIT-COUNT grid accumulates across keyframes and a cell only counts as
    OCCUPIED once it has been hit by ``>= OCC_HITS`` keyframes -- a real surface is
    re-observed many times and survives; random stereo noise hits a cell once or
    twice and is rejected. That fusion both CLEANS the map and keeps the voxel
    count (and therefore the render load) low.

    This is a pure CONSUMER of existing topics: no new field, no data-path change.

    Subscribes TWO endpoints (mirrors :class:`IpcKeypointWorker`'s two-endpoint
    pattern, but the keyframe depth rides VIO's dedicated ``kf_depth`` ring, NOT
    capture's frame rings):

    * ``keyframe`` (VIO) -> per keyframe ``seq`` we keep its DENOISED metric depth
      (``read_copy``-ed out of VIO's kf ring by the converter) and the FULL POSE
      (rotation AND translation) of its ``T_world_cam`` -- everything the
      back-projection needs. The keyframe rate is low (kf_every), so a bounded dict
      of the last :data:`MAX_KEYFRAMES` raw depth grids caps memory (the fused
      occupancy grid is persistent and unaffected by that cap).
    * ``slam.map`` (SLAM) -> the CONTINUOUS per-keyframe loop-closure-corrected
      positions, keyed by source frame seq. Retained (and kept fresh) as the source
      of truth for a future corrected-map variant + diagnostics, but it does NOT
      position the voxels: they are quantised from each keyframe's own VIO pose for
      a consistent, seam-free odom-frame grid. (Re-anchoring the grid to the
      loop-corrected translations is a deliberate later step.)

    Occupancy fusion (PERSISTENT + INCREMENTAL -- see :meth:`_fuse_keyframe_locked`
    and :meth:`_build`):

    1. :attr:`_hits` is a persistent ``{(ix,iy,iz) -> hit_count}`` dict. As each
       NEW keyframe arrives, its (denoised) depth is back-projected to the world by
       the keyframe's OWN VIO pose (strided + depth-gated + edge-rejected), each hit
       point is quantised to a :data:`VOXEL_M` cell, and that cell's hit_count is
       INCREMENTED (capped at :data:`HIT_CAP` so a long dwell can't overflow). The
       grid ACCUMULATES across keyframes; we never rebuild it from scratch. A set
       :attr:`_fused_seqs` records which keyframe seqs are already folded in, so a
       rebuild only folds the not-yet-fused keyframes (cheap, incremental).
    2. A cell is OCCUPIED when ``hit_count >= OCC_HITS`` -- the temporal noise
       filter. Render emits one voxel per occupied cell.
    3. Output: ALL occupied voxel CENTRES (the cell index * VOXEL_M + half a cell)
       + a GREEN gradient colour by HEIGHT (derived from the cell's world ``y``,
       the optical DOWN axis). We render the WHOLE occupied set so the map GROWS as
       the camera explores; a bounded room's occupied count PLATEAUS, and OCC_HITS
       keeps it light. :data:`MAX_VOXELS` is only a HIGH runaway safety cap (a fair
       random subsample if ever tripped -- NEVER a "keep top-N by hit_count" drop,
       which would erase the newest areas and froze the map at the start).

    Render choice (light, no-lag): the window draws the voxels as a
    ``GLScatterPlotItem`` of large SQUARE world-unit points (size == VOXEL_M,
    ``pxMode=False``), NOT an N-cube ``GLMeshItem`` -- a scatter of N points is far
    cheaper to upload + paint than 12*N triangles, and at this voxel size the
    squares read as the blocky cubes. The rebuild is coalesced + capped at
    :attr:`REBUILD_HZ` and only re-emits when the occupied set CHANGED -- grew or
    shifted (see :meth:`run` + :meth:`_occ_signature`), so the GUI never re-uploads
    an identical cloud yet always reflects newly-explored areas.

    FUTURE (v2, do NOT implement now unless cheap): free-space RAY-CARVING -- decrement
    the hit_count of voxels along each depth ray BEFORE the hit (the ones the ray
    passed through unobstructed) so a flying-voxel that a later ray sees through is
    actively cleared, exactly like OctoMap's miss updates. v1 only does hit updates
    (the >= OCC_HITS gate already rejects most one-off noise without the extra
    per-ray traversal cost).

    ``K`` comes from VIO's retained ``calib.bundle`` (the rectified-left intrinsic
    for the full-res depth grid the keyframe ``depth_m`` lives on).

    The VIO keyframe feed (ring attach + ``_on_keyframe`` stash + evict + the
    coalesced rebuild loop) is inherited from :class:`_KeyframeAccumulator`; this
    class adds the ``slam.map`` client + the occupancy fusion + the voxel build.

    Connect-error model mirrors :class:`IpcGyroFuseSource` /
    :class:`IpcKeypointWorker`: :meth:`start` swallows a connect timeout onto
    :attr:`error` (the window polls it) rather than raising.
    """

    # ------------------------------------------------------------------ #
    # Tunables (each commented with which way to turn it).
    # ------------------------------------------------------------------ #
    #: Max full-build re-emits per second. The build is OFF the GUI thread (this
    #: source is a worker thread that submit()s the finished voxels), so a rebuild
    #: never stutters the UI; 4 Hz re-snaps the map promptly as keyframes arrive.
    REBUILD_HZ = 4.0
    #: Voxel edge length (m). The cell every world hit is quantised to AND the
    #: rendered square-point size. Coarse on purpose (~0.10 m) -> the blocky VOXL
    #: look, FEWER voxels (lighter render) and stronger temporal fusion (more rays
    #: agree per cell). LOWER for finer structure (more voxels, heavier); RAISE for
    #: a coarser, lighter map.
    VOXEL_M = 0.10
    #: Occupancy threshold: a cell is OCCUPIED (rendered) only once it has been hit
    #: by >= OCC_HITS keyframes. This IS the temporal noise filter -- a real surface
    #: is re-observed across many keyframes (survives); a random stereo-noise hit
    #: touches a cell once (rejected). Because we now render ALL occupied cells (no
    #: top-N cap that favoured the start), OCC_HITS is also the RENDER-LOAD knob:
    #: 5 yields fewer, cleaner voxels per area so a whole noisy-OAK-D room stays a
    #: light scatter that plateaus around a few tens of thousands of points (vs the
    #: old 3, which on noisy stereo overran the 30k cap in the start area alone and
    #: starved every later area). RAISE for a cleaner/sparser/lighter map (slower to
    #: fill, needs more re-observation); LOWER toward 1 to fill faster (noisier,
    #: heavier).
    OCC_HITS = 5
    #: Depth-map subsample stride: back-project every STRIDE-th pixel in u and v.
    #: The occupancy grid only needs the room SHAPE, not every pixel, so a stride of
    #: 4 (1/16 the rays) keeps the per-keyframe fuse cheap while surfaces still get
    #: dense multi-keyframe support. LOWER (toward 1) for denser support (heavier
    #: fuse); RAISE for a lighter, sparser fuse.
    STRIDE = 4
    #: Edge-reject threshold (m): drop "flying pixels" on a depth discontinuity (a
    #: fg/bg edge back-projects to points floating BETWEEN the two surfaces). A
    #: pixel is kept only if BOTH its vertical and horizontal depth gradient are
    #: <= this. SAME idea as the shared geometry edge-reject; 0 disables it.
    EDGE_MAX_M = 0.1
    #: Per-cell hit-count ceiling. Without it a stationary dwell could pump one
    #: cell's count arbitrarily high (no harm to occupancy, but it would dominate
    #: any future confidence/decay logic + the colour normalisation). Capping keeps
    #: the counts in a sane band; well above OCC_HITS so it never affects the gate.
    HIT_CAP = 255
    #: Hard SAFETY cap on rendered voxels (runaway guard ONLY). We render ALL
    #: occupied cells (hit_count >= OCC_HITS) -- a bounded room's occupied count
    #: PLATEAUS (each surface re-observed but no new surface once explored), so a
    #: green height-coloured GLScatterPlotItem of a few tens of thousands of points
    #: stays light. This cap exists solely so a pathological unbounded sweep can't
    #: explode the render; it is set HIGH (well above a room's plateau) so it never
    #: trips in normal use. When it DOES trip we subsample FAIRLY (a uniform random
    #: draw -- see :meth:`_build`), NOT by lowest hit-count: dropping the lowest
    #: counts would erase exactly the newest, least-re-observed areas (the bug that
    #: froze the map at the start). LOWER only if a room genuinely overruns the
    #: render budget; RAISE for an even larger guard band.
    MAX_VOXELS = 150_000

    def __init__(self, vio_endpoint: str, slam_endpoint: str, K: np.ndarray, *,
                 connect_timeout_s: float = 10.0,
                 width: int | None = None, height: int | None = None) -> None:
        super().__init__(vio_endpoint, name=f"slam-map-src-{slam_endpoint}",
                         connect_timeout_s=connect_timeout_s,
                         width=width, height=height)
        self._slam_ep = slam_endpoint
        self._K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self._slam_client: IPCPubSub | None = None
        # PERSISTENT occupancy hit-count grid: {(ix,iy,iz) int -> hit_count int}.
        # Accumulated across keyframes (NEVER rebuilt from scratch); guarded by the
        # base's _lock together with the depth/pose dicts.
        self._hits: dict[tuple[int, int, int], int] = {}
        # The keyframe seqs already folded into _hits, so a rebuild only fuses the
        # NEW keyframes (incremental). Cleared keyframes stay folded in (the grid is
        # persistent), so an evicted seq is simply never re-fused.
        self._fused_seqs: set[int] = set()
        # Signature of the occupied set at the last emit -> skip re-emitting an
        # UNCHANGED cloud (avoids re-uploading the whole scatter to GL every rebuild
        # tick) while still re-emitting whenever the set GROWS or SHIFTS as new
        # keyframes fuse. A bare count is NOT enough: as the camera explores, the
        # count grows (caught) but a count that momentarily plateaus while voxels
        # appear in a new area + drop in an old one would be missed -- so we pair
        # the count with a cheap content hash of the occupied cell keys. -1 = never
        # emitted (force the first emit). See :meth:`run`.
        self._last_emit_sig: tuple[int, int] = (-1, 0)
        # Latest loop-closure-corrected camera positions from slam.map, keyed by
        # seq. NOT used to position the voxels (they build from the VIO keyframe's
        # own pose for a seam-free odom-frame grid); retained as the source of truth
        # for the future corrected-map variant + diagnostics.
        self._kf_corr_pos: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def start_cloud(self, on_cloud) -> None:
        """Connect both endpoints and stream voxel maps to ``on_cloud``.

        ``on_cloud(points (N,3) float32, colors (N,3) float32, cams (M,3)
        float32)`` is invoked from this source's REBUILD thread (the window
        marshals it onto the GUI thread). ``points`` are the occupied VOXEL CENTRES,
        ``colors`` the green-by-height gradient, ``cams`` the keyframe camera
        positions. On connect failure :attr:`error` is set and the thread is NOT
        started (the window polls :attr:`error`). Method name kept (``start_cloud``)
        so ``ui.main`` wires it unchanged.
        """
        self._cb = on_cloud
        # Attach VIO's keyframe rings (consumer side) so the keyframe converter can
        # read_copy the depth array out of them. Missing rings == VIO not running
        # -> surface a clear, device-agnostic reason.
        if not self._attach_or_fail():
            return

        vio_client = self._make_keyframe_client()
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
            self._vio_client = None
            if self._vio_rings is not None:
                self._vio_rings.close()
                self._vio_rings = None
            return
        self._slam_client = slam_client
        self.start()                                  # spin the rebuild thread

    # ------------------------------------------------------------------ #
    def _on_evict_locked(self, seq: int) -> None:
        """Drop this source's per-seq corrected-position state too (under lock).

        The keyframe's depth/pose are dropped by the base; its hits STAY in the
        persistent grid (already fused) -- we just never re-fuse an evicted seq.
        """
        self._kf_corr_pos.pop(seq, None)

    def _on_slammap(self, wm) -> None:
        """SLAM recv thread: refresh the loop-closure-corrected positions.

        These are kept fresh for the future corrected-map variant + diagnostics
        only; the voxels are quantised from each keyframe's own VIO pose (see
        :meth:`_fuse_keyframe_locked`), so this does NOT move the rendered grid.
        Marks dirty so the camera trail re-snaps (slam.map carries kf positions).
        """
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

    # ------------------------------------------------------------------ #
    def _fuse_keyframe_locked(self, depth: np.ndarray, R: np.ndarray,
                              t: np.ndarray) -> None:
        """Fold ONE keyframe's depth into the persistent hit-count grid (under lock).

        Back-projects the (denoised) depth to the world by the keyframe's OWN VIO
        pose ``Xw = R Xc + t`` (strided + depth-gated + edge-rejected), quantises
        each world hit to a :data:`VOXEL_M` cell, and INCREMENTS that cell's
        hit_count (capped at :data:`HIT_CAP`). Vectorised: one back-projection per
        keyframe, one ``np.unique`` over the hit cells, then a scatter add into the
        persistent dict -- so a cell hit by many rays in ONE keyframe still counts as
        a SINGLE keyframe observation (the +1 is per (keyframe, cell), the temporal
        fusion the OCC_HITS gate relies on).
        """
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim != 2:
            return
        h, w = d.shape
        s = max(1, int(self.STRIDE))
        # Per-pixel validity on the FULL grid first (so the edge gradient sees a
        # native-resolution discontinuity), THEN subsample by stride -- matching the
        # shared geometry helper's edge reject.
        m = (np.isfinite(d) & (d >= self.MIN_DEPTH_M) & (d <= self.MAX_DEPTH_M))
        if self.EDGE_MAX_M > 0.0:
            dv = np.abs(np.diff(d, axis=0, append=d[-1:]))
            dh = np.abs(np.diff(d, axis=1, append=d[:, -1:]))
            m &= (dv <= self.EDGE_MAX_M) & (dh <= self.EDGE_MAX_M)
        keep = m[::s, ::s]
        if not np.any(keep):
            return
        fx, fy = float(self._K[0, 0]), float(self._K[1, 1])
        cx, cy = float(self._K[0, 2]), float(self._K[1, 2])
        us = np.arange(0, w, s, dtype=np.float64)
        vs = np.arange(0, h, s, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs)
        z = d[::s, ::s].astype(np.float64)
        uu, vv, z = uu[keep], vv[keep], z[keep]                # flat (M,)
        # Pinhole back-projection to the camera frame, then to the world by the
        # keyframe's OWN pose, then quantise to integer voxel coords.
        cam = np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, z], axis=1)
        world = cam @ np.asarray(R, np.float64).reshape(3, 3).T \
            + np.asarray(t, np.float64).reshape(3)             # (M,3)
        keys = np.floor(world / float(self.VOXEL_M)).astype(np.int64)
        # Collapse multi-ray hits within THIS keyframe to one +1 per cell, then add
        # into the persistent grid (capped). One Python loop over the UNIQUE cells
        # this keyframe touched (a few thousand at most), not the raw rays.
        uniq = np.unique(keys, axis=0)
        cap = int(self.HIT_CAP)
        for cell in uniq:
            k = (int(cell[0]), int(cell[1]), int(cell[2]))
            c = self._hits.get(k, 0) + 1
            self._hits[k] = c if c < cap else cap

    # ------------------------------------------------------------------ #
    def _build(self):
        """Fold any NEW keyframes into the grid, then emit the occupied voxels.

        Returns ``(points, colors, cams)``:

        * ``points`` ``(N,3)`` float32 -- the CENTRE of EVERY OCCUPIED voxel
          (``hit_count >= OCC_HITS``). We render the WHOLE occupied set so the map
          GROWS as the camera explores (a bounded room's occupied count plateaus);
          only if it exceeds the high :data:`MAX_VOXELS` safety cap do we drop
          voxels by a FAIR uniform-random subsample (NOT by lowest hit-count, which
          would erase the newest areas). Optical-world frame (the window rotates it
          to ENU, same as the trajectory).
        * ``colors`` ``(N,3)`` float32 -- a GREEN gradient by HEIGHT (optical
          ``+y`` is world-DOWN), so the floor/walls read like ModalAI's height-tinted
          occupancy.
        * ``cams`` ``(M,3)`` float32 -- ALL VIO keyframe camera positions (the path).

        The fuse is INCREMENTAL: only keyframes not in :attr:`_fused_seqs` are
        folded (the persistent grid already holds the rest), so this stays cheap as
        keyframes accumulate.
        """
        with self._lock:
            # (1) Fold the NEW keyframes into the persistent grid (incremental).
            new_seqs = [s for s in self._kf_depth if s not in self._fused_seqs]
            for seq in sorted(new_seqs):
                self._fuse_keyframe_locked(self._kf_depth[seq],
                                           self._kf_R[seq], self._kf_t[seq])
                self._fused_seqs.add(seq)
            # (2) Snapshot the occupied cell KEYS + the camera trail under the
            #     lock, then build the arrays lock-free. We render ALL occupied
            #     cells, so only the keys are needed (the hit_count no longer
            #     selects a subset).
            occ = [k for k, c in self._hits.items() if c >= self.OCC_HITS]
            cam_ts = [self._kf_t[s] for s in self._kf_depth]
        cams = (np.asarray(cam_ts, np.float32).reshape(-1, 3)
                if cam_ts else np.zeros((0, 3), np.float32))
        if not occ:
            empty = np.zeros((0, 3), np.float32)
            return empty, empty, cams

        keys = np.asarray(occ, dtype=np.int64)                     # (N,3)
        # Render the WHOLE occupied set (the map must GROW as new areas are
        # explored). The high MAX_VOXELS safety cap is a runaway guard only; if it
        # ever trips we subsample UNIFORMLY AT RANDOM -- a spatially-fair draw that
        # thins the cloud everywhere instead of erasing the newest (lowest-hit)
        # areas the way a "keep top-N by hit_count" rule did (that rule permanently
        # favoured the start area and froze the map). Seeded for a stable cloud
        # across rebuilds (so the safety thinning doesn't shimmer frame to frame).
        if keys.shape[0] > self.MAX_VOXELS:
            rng = np.random.default_rng(0)
            sel = rng.choice(keys.shape[0], size=self.MAX_VOXELS, replace=False)
            keys = keys[sel]
        # Voxel centres: cell index * edge + half a cell (the cell's centre).
        points = ((keys.astype(np.float32) + 0.5) * np.float32(self.VOXEL_M))
        colors = self._green_by_height(points[:, 1])
        return points.astype(np.float32), colors, cams

    @staticmethod
    def _green_by_height(y_opt: np.ndarray) -> np.ndarray:
        """Optical ``+y`` (world-DOWN) per-voxel -> a GREEN height gradient (N,3).

        ModalAI tints occupancy by height; we mimic that cheaply. Optical ``+y`` is
        world-DOWN, so a LARGER ``y`` is LOWER (floor) and a SMALLER ``y`` is HIGHER
        (ceiling). We normalise ``-y`` (so high = bright) across the current voxel
        span and ramp a dark-green -> bright-green gradient with a slight blue lift
        up high, all in [0,1]. Pure numpy, vectorised.
        """
        n = y_opt.shape[0]
        if n == 0:
            return np.zeros((0, 3), np.float32)
        up = -np.asarray(y_opt, np.float32)            # height (up positive)
        lo, hi = float(up.min()), float(up.max())
        span = hi - lo
        # Flat span (all one height) -> mid gradient; else normalise to [0,1].
        h = (np.full(n, 0.5, np.float32) if span < 1e-6
             else (up - lo) / span)
        r = 0.05 + 0.10 * h                            # stays low (greenish)
        g = 0.35 + 0.65 * h                            # dark -> bright green
        b = 0.10 + 0.35 * h                            # slight blue lift up high
        return np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Coalesced rebuild loop with a "materially changed" emit guard.

        Same throttle as the base, but skips the callback when the occupied set is
        UNCHANGED since the last emit -- so the GUI never re-uploads an identical
        scatter to GL (the no-lag requirement) -- while ALWAYS re-emitting when the
        set GROWS or SHIFTS as new keyframes fuse (the bug was a frozen view). A new
        keyframe / slam.map marks dirty; we rebuild (folds new keyframes + recounts
        occupied) and emit when the signature -- (occupied count, content hash) --
        changed. Pairing the count with a content hash catches a spatial shift that
        leaves the count unchanged (new area appears, old voxel drops), which a bare
        count comparison would miss.
        """
        period = 1.0 / self.REBUILD_HZ
        while not self._stop.is_set():
            if not self._dirty.wait(timeout=0.25):
                continue
            if self._stop.is_set():
                break
            self._dirty.clear()
            try:
                points, colors, cams = self._build()
            except Exception:                          # noqa: BLE001
                # A malformed keyframe must not kill the rebuild thread; skip it.
                time.sleep(period)
                continue
            cb = self._cb
            sig = self._occ_signature(points)
            # Re-emit whenever the occupied set CHANGED (count or content); an idle
            # dirty tick that produced the identical set re-pushes nothing to GL.
            if cb is not None and sig != self._last_emit_sig:
                self._last_emit_sig = sig
                cb(points, colors, cams)
            self._stop.wait(period)

    @staticmethod
    def _occ_signature(points: np.ndarray) -> tuple[int, int]:
        """Cheap (count, content-hash) signature of the rendered voxel cloud.

        ``points`` are the occupied voxel centres (deterministic for a given
        occupied set + cap). The count alone can't tell a SHIFTED set from a stable
        one (same N, different cells), so we add a hash of the raw point bytes. Both
        are O(N) over a few-tens-of-thousands array (negligible vs the build) and
        change the instant the set grows OR shifts -- so the map re-emits as the
        camera explores instead of freezing.
        """
        n = int(points.shape[0])
        if n == 0:
            return (0, 0)
        # hash() over the C-contiguous float32 bytes: order-sensitive but the
        # occupied keys come out of the dict in a stable order per build, so an
        # unchanged set hashes identically while any add/drop/shift changes it.
        return (n, hash(np.ascontiguousarray(points).tobytes()))

    # ------------------------------------------------------------------ #
    def _stop_extra_clients(self) -> None:
        """Close the EXTRA slam.map client (the base closes the VIO one)."""
        client = self._slam_client
        self._slam_client = None
        if client is not None:
            try:
                client.stop()
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
    and starts the returned occupancy-voxel source.
    """
    return lambda: IpcSlamMapSource(vio_endpoint, slam_endpoint, K,
                                    width=width, height=height)
