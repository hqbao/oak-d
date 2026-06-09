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
from ui.viz.map_cloud import longest_consecutive_run
from ui.viz import floor_plan
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

    Both the sparse landmark map (:class:`IpcSlamMapSource`) and the 2D top-down
    floor plan (:class:`IpcFloorPlanSource`) need the EXACT same VIO keyframe
    feed -- the gray/depth grids (read_copy-ed out of VIO's ``kf_gray`` /
    ``kf_depth`` rings by the converter) and each keyframe's FULL pose
    ``[R_keyframe | t_keyframe]`` (split from ``T_world_cam``), plus the KLT track
    snapshot (``track_ids`` / ``track_px`` / ``inlier_ids``) the landmark map
    reads. Rather than duplicate the ring attach + ``_on_keyframe`` stash + evict
    wiring in two sources, that machinery lives ONCE here; each concrete source
    subclasses this and only supplies its own build (``_build`` -> a payload) +
    its rebuild rate.

    A bounded dict of the last :data:`MAX_KEYFRAMES` keyframes (keyed by source
    frame seq) caps memory; the subclass's build re-orders by seq, so insertion
    order is not relied on for any index assignment.

    Lifecycle (subclasses inherit it, only set ``_cb`` + spin via :meth:`start`):

    * :meth:`_attach_or_fail` attaches the kf rings (a missing ring == VIO not
      running -> a clear, device-agnostic reason on :attr:`error`).
    * :meth:`_make_keyframe_client` builds the VIO IPC client subscribed to
      ``keyframe`` (the concrete source adds any extra clients it needs, e.g.
      the landmark map's ``slam.map``).
    * a coalesced rebuild loop (:meth:`run`) re-builds at most :attr:`REBUILD_HZ`
      times a second whenever the model goes dirty, OFF the GUI thread, and hands
      each finished payload to the injected callback (the window marshals it onto
      the GUI thread via its thread-safe ``submit``).

    Connect-error model mirrors the other sources: a connect timeout is swallowed
    onto :attr:`error` (the window polls it) rather than raising.
    """

    #: Cap on kept keyframes (bounds memory like the other UI buffers). At
    #: kf_every=5 @ 20 fps a keyframe lands ~every 0.25 s, so 600 kf ~= 2.5 min of
    #: distinct keyframes -- far more than any 3D map needs.
    MAX_KEYFRAMES = 600
    #: Valid-depth band (m): outside this range stereo depth is too noisy to
    #: back-project. Both builds gate to this band (the landmark map samples per
    #: track; the floor plan bins the whole grid through the SAME band).
    MIN_DEPTH_M = 0.3
    MAX_DEPTH_M = 6.0
    #: Max full rebuilds per second. Subclasses override (the floor-plan scatter
    #: rebuild is cheap -> a higher rate).
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

        # Per-keyframe accumulators, keyed by source frame seq. Each keyframe keeps
        # its gray/depth grid + rotation AND translation + its KLT track snapshot
        # (ids + pixels) plus the PnP-inlier id subset. ``MAX_KEYFRAMES`` bounds it.
        self._kf_gray: dict[int, np.ndarray] = {}
        self._kf_depth: dict[int, np.ndarray] = {}
        self._kf_R: dict[int, np.ndarray] = {}        # (3,3) keyframe rotation
        self._kf_t: dict[int, np.ndarray] = {}        # (3,) VIO keyframe translation
        self._kf_track_ids: dict[int, np.ndarray] = {}   # (N,) track ids
        self._kf_track_px: dict[int, np.ndarray] = {}    # (N,2) pixels, id-aligned
        self._kf_inlier_ids: dict[int, np.ndarray] = {}  # (M,) PnP inlier ids

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
        """VIO recv thread: stash one keyframe's gray/depth + FULL pose + tracks.

        Splits the (4,4) VIO world<-cam pose into rotation + translation; every
        observation a build derives uses ``Xw = R Xc + t`` so positions stay
        self-consistent (one pose source per keyframe -> no seams). The track
        snapshot is OPTIONAL on the wire; it is stored as-is and a build skips
        keyframes lacking it.
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
            self._kf_gray[seq] = kf.gray_left          # already a private copy
            self._kf_depth[seq] = kf.depth_m
            self._kf_R[seq] = R
            self._kf_t[seq] = t
            self._kf_track_ids[seq] = kf.track_ids
            self._kf_track_px[seq] = kf.track_px
            self._kf_inlier_ids[seq] = kf.inlier_ids
            self._evict_locked()
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
            self._kf_t.pop(seq, None)
            self._kf_track_ids.pop(seq, None)
            self._kf_track_px.pop(seq, None)
            self._kf_inlier_ids.pop(seq, None)
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
# (3c) SLAM 3D-map source (sparse landmark cloud) for MapWindow
# --------------------------------------------------------------------------- #
class IpcSlamMapSource(_KeyframeAccumulator):
    """Build the sparse, ID-based SLAM landmark cloud for MapWindow.

    Shows ONE point per LANDMARK (KLT track id) that was a PnP INLIER across a run
    of ``>= PERSIST_KF`` SUCCESSIVE keyframes -- i.e. only consistently-tracked,
    motion-validated features (like a real SLAM map), NOT a dense-depth
    reconstruction. This is a pure CONSUMER of existing topics: no new field, no
    data-path change.

    Subscribes BOTH endpoints the landmark map needs (mirrors
    :class:`IpcKeypointWorker`'s two-endpoint pattern, but the keyframe gray/depth
    ride VIO's dedicated ``kf_gray`` / ``kf_depth`` rings, NOT capture's frame
    rings):

    * ``keyframe`` (VIO) -> per keyframe ``seq`` we keep its gray image + metric
      depth (``read_copy``-ed out of VIO's kf rings by the converter), the ids +
      pixels of its KLT tracks (``track_ids`` / ``track_px``), the subset PnP kept
      as INLIERS this frame (``inlier_ids``) and the FULL POSE (rotation AND
      translation) of its ``T_world_cam``. The keyframe rate is low (kf_every), so a
      bounded dict of the last :data:`MAX_KEYFRAMES` keyframes caps memory.
    * ``slam.map`` (SLAM) -> the CONTINUOUS per-keyframe loop-closure-corrected
      positions, keyed by source frame seq. Retained (and kept fresh) as the source
      of truth for a future corrected-map variant + diagnostics, but it does NOT
      position the cloud: the SLAM backend only places a SPARSE subset of keyframes
      (~1 in ~18), so gating on it starved the persistence run (see below).

    The cloud positions every keyframe from its OWN VIO pose ``[R_keyframe |
    t_keyframe]`` (``_kf_R`` / ``_kf_t``), so a landmark's observations across
    keyframes never mix two pose sources -> a consistent, seam-free odom-frame map.
    (Re-anchoring to the loop-corrected ``slam.map`` translations is a deliberate
    later step; positioning from one source first is what makes the map populate.)

    Cloud build (TRACK-ID persistence -- see :meth:`_build`):

    1. Order ALL VIO keyframes (every ``--kf-every`` frames) by ``seq`` and give
       each a 0-based sequential index ``k`` -- the index the consecutive-run gate
       counts in. Counting over the DENSE VIO keyframes (not the sparse ``slam.map``
       subset) is what lets a KLT track reach :data:`PERSIST_KF` consecutive.
    2. For each keyframe ``k`` and each ``track_id`` in its ``inlier_ids``: look up
       the track's pixel (``track_ids`` -> ``track_px``), sample ``depth_m`` there,
       back-project with the pinhole and transform to the world by
       ``[R_keyframe | t_keyframe]``; record ``(k, world_xyz, gray)`` under that
       ``track_id``. So we accumulate, per landmark, every keyframe it was an
       inlier in (with a world observation + colour).
    3. For each landmark, take the SET of keyframe indices it was an inlier in and
       compute its longest run of CONSECUTIVE integers
       (:func:`ui.viz.map_cloud.longest_consecutive_run`). KEEP the landmark only
       if that run reaches :data:`PERSIST_KF` -- so a track seen only in scattered
       or too-few keyframes is dropped.
    4. Output ONE point per kept landmark: the MEDIAN of its world observations
       (robust to a stray) with mean colour, plus the keyframe camera markers.

    This is much lighter than a dense fuse (only ~tens-hundreds of tracks per
    keyframe), so the rebuild is fast even as keyframes accumulate.

    ``K`` comes from VIO's retained ``calib.bundle`` (the rectified-left intrinsic
    for the full-res depth grid the keyframe ``depth_m`` lives on).

    Rebuild policy: a full re-fuse over every kept keyframe is coalesced -- a
    ``slam.map`` (or a new keyframe) just marks the model dirty, and a background
    loop rebuilds at most :data:`REBUILD_HZ` times a second, always from the
    freshest poses. The finished ``(points, colors, cams)`` go to the injected
    ``on_cloud`` callback (the window marshals it onto the GUI thread).

    The VIO keyframe feed (ring attach + ``_on_keyframe`` stash + evict + the
    coalesced rebuild loop) is inherited UNCHANGED from
    :class:`_KeyframeAccumulator`; this class adds ONLY the landmark-specific
    ``slam.map`` client + the track-id-persistence build.

    Connect-error model mirrors :class:`IpcGyroFuseSource` /
    :class:`IpcKeypointWorker`: :meth:`start` swallows a connect timeout onto
    :attr:`error` (the window polls it) rather than raising.
    """

    #: Max full-cloud rebuilds per second. The sparse landmark rebuild is light
    #: (only the ~tens-hundreds of inlier tracks per keyframe, not every depth
    #: pixel), so we run at 4 Hz to re-snap the map promptly on new keyframes +
    #: slam.map pose updates. The build is OFF the GUI thread (this source is a
    #: worker thread that submit()s the finished cloud), so a rebuild never stutters
    #: the UI.
    REBUILD_HZ = 4.0
    #: Persistence gate: a landmark (track id) shows only if it was a PnP inlier
    #: across >= PERSIST_KF SUCCESSIVE keyframes -> only consistently-tracked,
    #: motion-validated points (transient stereo noise never qualifies). 6 is a
    #: balance: strict enough to reject one-off noise, low enough that the map
    #: actually populates at the live --kf-every cadence. RAISE for higher
    #: confidence (sparser), LOWER to fill the room faster (noisier).
    PERSIST_KF = 6

    def __init__(self, vio_endpoint: str, slam_endpoint: str, K: np.ndarray, *,
                 connect_timeout_s: float = 10.0,
                 width: int | None = None, height: int | None = None) -> None:
        super().__init__(vio_endpoint, name=f"slam-map-src-{slam_endpoint}",
                         connect_timeout_s=connect_timeout_s,
                         width=width, height=height)
        self._slam_ep = slam_endpoint
        self._K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self._slam_client: IPCPubSub | None = None
        # Latest loop-closure-corrected camera positions from slam.map, keyed by
        # seq. NOT used to position the cloud (the cloud builds from the VIO
        # keyframe's own pose for a seam-free odom-frame map); retained as the
        # source of truth for the future corrected-map variant + diagnostics.
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
        """Drop this source's per-seq corrected-position state too (under lock)."""
        self._kf_corr_pos.pop(seq, None)

    def _on_slammap(self, wm) -> None:
        """SLAM recv thread: refresh the loop-closure-corrected positions.

        These are kept fresh for the future corrected-map variant + diagnostics
        only; the cloud is positioned from each keyframe's own VIO pose (see
        :meth:`_build`), so this does NOT gate or move the rendered map.
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
    def _build(self):
        """Build the sparse landmark cloud by TRACK-ID persistence (VIO poses).

        Returns ``(points, colors, cams)`` (points/colors empty when no landmark
        qualifies). EVERY VIO keyframe with a stashed gray/depth/track snapshot is
        used -- positioned from its OWN VIO pose ``[R | t]`` (``_kf_R`` / ``_kf_t``),
        NOT from the sparse ``slam.map`` subset. VIO emits a keyframe every
        ``--kf-every`` frames (dense), whereas the SLAM backend only places a coarse
        subset (~1 in ~18); gating the persistence run on that sparse subset meant a
        KLT track (living tens of frames -> only ~2 SLAM keyframes) could NEVER reach
        :data:`PERSIST_KF` consecutive, so the map stayed empty. Counting over the
        DENSE VIO keyframes lets a track that persists ``PERSIST_KF`` successive
        keyframes qualify. Using one pose source (the keyframe's own) for every
        observation keeps a landmark's positions self-consistent -> no seams.

        Track-id persistence (see class docstring):

        1. Order the usable keyframes by ``seq`` and give each a 0-based sequential
           index ``k`` (the index the consecutive-run gate counts in).
        2. Per keyframe ``k``, per ``track_id`` in its ``inlier_ids``: find the
           track's pixel (``track_ids`` -> ``track_px``), sample ``depth_m`` there,
           back-project with the pinhole (``X=(u-cx)/fx*z``, ``Y=(v-cy)/fy*z``,
           ``Z=z``) and transform to the world by ``[R_keyframe | t_keyframe]``.
           Record ``(k, world_xyz, gray)`` under that ``track_id`` (skip ids with
           no/invalid depth or off-grid pixel).
        3. Per landmark, the SET of keyframe indices it was an inlier in -> its
           longest run of CONSECUTIVE integers; KEEP only when the run reaches
           :data:`PERSIST_KF`.
        4. One output point per kept landmark = the MEDIAN of its world
           observations (robust to a stray), with mean colour.
        """
        with self._lock:
            # Use EVERY VIO keyframe that carries a full track snapshot (gray/depth
            # always present once stashed; track ids may be None on the wire). NO
            # ``slam.map`` gate -- positions come from each keyframe's own VIO pose.
            # Sort by SOURCE FRAME SEQ so the 0-based index ``k`` reflects true
            # keyframe order regardless of dict insertion / eviction.
            seqs = sorted(s for s in self._kf_gray
                          if self._kf_inlier_ids.get(s) is not None
                          and self._kf_track_ids.get(s) is not None
                          and self._kf_track_px.get(s) is not None)
            grays = [self._kf_gray[s] for s in seqs]
            depths = [self._kf_depth[s] for s in seqs]
            Rs = [self._kf_R[s] for s in seqs]
            ts = [self._kf_t[s] for s in seqs]
            track_ids = [self._kf_track_ids[s] for s in seqs]
            track_px = [self._kf_track_px[s] for s in seqs]
            inlier_ids = [self._kf_inlier_ids[s] for s in seqs]
            # Cams = ALL VIO keyframe positions (even ones without a usable track
            # snapshot) so the camera trail stays complete.
            cam_ts = [self._kf_t[s] for s in self._kf_gray]
        cams = (np.asarray(cam_ts, np.float32).reshape(-1, 3)
                if cam_ts else np.zeros((0, 3), np.float32))
        if not seqs:
            empty = np.zeros((0, 3), np.float32)
            return empty, empty, cams

        fx, fy = float(self._K[0, 0]), float(self._K[1, 1])
        cx, cy = float(self._K[0, 2]), float(self._K[1, 2])

        # (2) Accumulate, per landmark id, every keyframe it was an inlier in:
        #     {track_id -> ([k, ...], [world_xyz, ...], [gray, ...])}. We process
        #     each keyframe VECTORISED over its inlier ids (numpy gather + a single
        #     back-projection), then scatter the rows into the per-landmark lists.
        kf_seen: dict[int, list[int]] = {}        # track_id -> [keyframe index k]
        kf_xyz: dict[int, list[np.ndarray]] = {}  # track_id -> [world (3,)]
        kf_gray_v: dict[int, list[float]] = {}    # track_id -> [gray in [0,1]]
        for k in range(len(seqs)):
            ids = np.asarray(track_ids[k]).ravel()
            inl = np.asarray(inlier_ids[k]).ravel()
            if ids.size == 0 or inl.size == 0:
                continue
            px = np.asarray(track_px[k], dtype=np.float64).reshape(-1, 2)
            depth = np.asarray(depths[k], dtype=np.float32)
            h, w = depth.shape
            # Select this keyframe's inlier tracks (id-aligned pixel lookup).
            sel = np.isin(ids, inl)
            sel_ids = ids[sel]
            sel_px = px[sel]
            if sel_ids.size == 0:
                continue
            # Round pixels to the depth grid and gate to on-grid + valid depth.
            u = np.round(sel_px[:, 0]).astype(np.int64)
            v = np.round(sel_px[:, 1]).astype(np.int64)
            on = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not np.any(on):
                continue
            u, v, sel_ids = u[on], v[on], sel_ids[on]
            z = depth[v, u]
            ok = (np.isfinite(z) & (z >= self.MIN_DEPTH_M)
                  & (z <= self.MAX_DEPTH_M))
            if not np.any(ok):
                continue
            u, v, z, sel_ids = u[ok], v[ok], z[ok], sel_ids[ok]
            # Pinhole back-projection to the camera frame, then to world by the
            # keyframe's own VIO pose [R_keyframe | t_keyframe] (Xw = R Xc + t).
            cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
            world = cam @ Rs[k].T + ts[k]                       # (m, 3)
            gray_k = grays[k]
            gvals = (np.asarray(gray_k, dtype=np.float32)[v, u] / 255.0
                     if gray_k is not None
                     else np.full(z.size, 0.85, np.float32))
            # Scatter each landmark's observation into its accumulator.
            for j in range(sel_ids.size):
                tid = int(sel_ids[j])
                kf_seen.setdefault(tid, []).append(k)
                kf_xyz.setdefault(tid, []).append(world[j])
                kf_gray_v.setdefault(tid, []).append(float(gvals[j]))

        # (3+4) Keep a landmark only if its longest run of SUCCESSIVE keyframe
        #       indices reaches PERSIST_KF; emit the MEDIAN world point + mean
        #       colour. ``longest_consecutive_run`` needs a sorted UNIQUE index
        #       sequence (a keyframe that saw a track twice must count once).
        pts_out: list[np.ndarray] = []
        col_out: list[float] = []
        for tid, ks in kf_seen.items():
            run = longest_consecutive_run(sorted(set(ks)))
            if run < self.PERSIST_KF:
                continue
            obs = np.asarray(kf_xyz[tid], dtype=np.float64)     # (n_obs, 3)
            pts_out.append(np.median(obs, axis=0))
            col_out.append(float(np.mean(kf_gray_v[tid])))
        if not pts_out:
            empty = np.zeros((0, 3), np.float32)
            return empty, empty, cams
        points = np.asarray(pts_out, dtype=np.float32)
        # Mean gray -> grayscale RGB (the viewer takes (N,3) colour in [0,1]).
        g = np.asarray(col_out, dtype=np.float32)[:, None]
        colors = np.repeat(g, 3, axis=1).astype(np.float32)
        return points, colors, cams

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
# (3e) Floor-plan source (2D top-down occupancy raster) for FloorPlanWindow.
# --------------------------------------------------------------------------- #
class IpcFloorPlanSource(_KeyframeAccumulator):
    """Build a 2D TOP-DOWN floor-plan raster of the room for FloorPlanWindow.

    A LIGHT complement to the 3D landmark map (:class:`IpcSlamMapSource`):
    instead of a 3D cloud (heavy GL, hard to read in perspective on this Mac),
    this back-projects the SAME VIO keyframe
    feed to world points and bins them onto the horizontal GROUND plane into a 2D
    OCCUPANCY raster -- so the walls read as a top-down outline + the camera path,
    rendered as a cheap 2D ``ImageItem`` (no ``GLViewWidget``). Pure CONSUMER of
    the ``keyframe`` feed (no SLAM endpoint, no ``slam.map``); no new field, no
    data-path change.

    Reuses the inherited :class:`_KeyframeAccumulator` machinery WHOLESALE (the kf
    ring attach + ``_on_keyframe`` stash + evict + the coalesced rebuild loop) --
    only ONE VIO client, NO copy-paste of the SHM/recv code. This class adds ONLY
    the 2D ground-plane occupancy build (delegated to :mod:`ui.viz.floor_plan`).

    Build (:meth:`_build`, runs OFF the GUI thread):

    1. Back-project EVERY stashed keyframe's (denoised) depth to world points by
       its OWN VIO pose ``[R | t]`` (strided + the SAME depth gate + edge reject as
       the 3D builders) -- :func:`~ui.viz.floor_plan.keyframes_to_ground_points`.
    2. Bin the points onto the optical ``(x, z)`` GROUND plane (drop the vertical
       optical ``+y``/DOWN axis) into a 2D occupancy raster, scoring each cell by
       point count BOOSTED by the vertical extent (so walls outscore floor) and
       colour-mapping it -- :func:`~ui.viz.floor_plan.build_floor_plan`.
    3. Project ALL keyframe camera positions onto the SAME plane for the path
       overlay.

    The finished ``(rgb, path_px, cams, extent)`` go to the injected ``on_plan``
    callback (the window marshals it onto the GUI thread via its thread-safe
    ``submit``).

    Connect-error model mirrors the other sources.
    """

    #: The 2D histogram build is CHEAP (a single ``np.add.at`` scatter, no triangle
    #: stacking), so it can rebuild often -- 5 Hz keeps the plan snappy as keyframes
    #: accumulate without burning CPU.
    REBUILD_HZ = 5.0

    def __init__(self, vio_endpoint: str, K: np.ndarray, *,
                 connect_timeout_s: float = 10.0,
                 width: int | None = None, height: int | None = None) -> None:
        super().__init__(vio_endpoint, name=f"floor-plan-src-{vio_endpoint}",
                         connect_timeout_s=connect_timeout_s,
                         width=width, height=height)
        self._K = np.asarray(K, dtype=np.float64).reshape(3, 3)

    # ------------------------------------------------------------------ #
    def start_plan(self, on_plan) -> None:
        """Connect VIO's ``keyframe`` and stream floor plans to ``on_plan``.

        ``on_plan(rgb (H,W,3) uint8, path_px (M,2) float32, cams (M,3) float32,
        extent FloorPlanExtent)`` is invoked from this source's REBUILD thread (the
        window marshals it onto the GUI thread). On connect failure :attr:`error`
        is set and the thread is NOT started (the window polls it).
        """
        self._cb = on_plan
        if not self._attach_or_fail():
            return
        vio_client = self._make_keyframe_client()
        try:
            vio_client.start()
        except Exception as e:                                     # noqa: BLE001
            self.error = f"Floor-plan stream connect failed: {e}"
            try:
                vio_client.stop()
            except Exception:                                      # noqa: BLE001
                pass
            self._vio_client = None
            if self._vio_rings is not None:
                self._vio_rings.close()
                self._vio_rings = None
            return
        self.start()                                  # spin the rebuild thread

    # ------------------------------------------------------------------ #
    def _build(self):
        """Back-project the keyframes -> bin onto the ground plane -> raster.

        Returns ``(rgb, path_px, cams, extent)``: the 2D occupancy raster, the
        camera path projected to raster pixels, ALL VIO keyframe camera positions
        (optical world) and the world<->pixel extent. Every stashed keyframe's
        depth is used, back-projected by its OWN VIO pose; the floor-plan math is
        the pure-numpy :mod:`ui.viz.floor_plan` (no GL, no Qt).
        """
        with self._lock:
            # gray is unused for the plan (occupancy, not colour); the build needs
            # only depth + each keyframe's own pose. Sort by source seq for a stable
            # build order (the order doesn't affect the histogram, but keeps the
            # camera path in capture order).
            seqs = sorted(self._kf_depth)
            depths = [self._kf_depth[s] for s in seqs]
            Rs = [self._kf_R[s] for s in seqs]
            ts = [self._kf_t[s] for s in seqs]
            cam_ts = [self._kf_t[s] for s in self._kf_gray]
        cams = (np.asarray(cam_ts, np.float32).reshape(-1, 3)
                if cam_ts else np.zeros((0, 3), np.float32))

        # Back-project every keyframe's gated, strided depth to world points, then
        # bin onto the ground plane into the occupancy raster + camera path.
        points = floor_plan.keyframes_to_ground_points(depths, Rs, ts, self._K)
        rgb, path_px, extent = floor_plan.floor_plan_with_path(points, cams)
        return rgb, path_px, cams, extent


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


def ipc_floor_plan_factory(vio_endpoint: str, K: np.ndarray,
                           width: int, height: int):
    """Return a zero-arg factory building an :class:`IpcFloorPlanSource`.

    Binds the VIO endpoint (the ``keyframe`` publisher + its kf rings) and the
    rectified-left ``K`` from the retained calib bundle -- so the caller
    (``ui.main``) just opens FloorPlanWindow and starts the returned source. No
    SLAM endpoint: the floor plan builds from the keyframe depth + each keyframe's
    own VIO pose only (the same ``keyframe`` feed the SLAM-map source uses).
    """
    return lambda: IpcFloorPlanSource(vio_endpoint, K,
                                      width=width, height=height)
