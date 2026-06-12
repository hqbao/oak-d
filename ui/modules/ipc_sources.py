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

import collections
import logging
import threading
import time

import numpy as np

from ui.comms import topics
from ui.comms.messages import END
from ui.comms import IPCPubSub, IPCSubscriber, LocalPubSub, RingRegistry
from ui.comms.converters import to_local
from ui.comms.ring_registry import default_capture_specs, default_vio_specs
from ui.qt.keypoints_window import KeypointWorker
from ui.qt.synced_window import TripletWorker

#: Lifecycle-level diagnostics for the calib sources. Kept at INFO for the
#: handful of one-shot lifecycle events (rings attached, client connected, first
#: packet, watchdog) so the terminal the operator is watching pinpoints WHERE a
#: live-path stall stalls; per-frame work stays silent (no spam).
LOG = logging.getLogger("ui.modules.ipc_sources")


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
# (1a2) Stereo RAW-pair source for the camera-calibration wizard
# --------------------------------------------------------------------------- #
class IpcStereoRawSource:
    """Duck-typed RAW stereo-pair stream over capture's ``imucam.sample`` topic.

    The stereo camera-calibration wizard needs a stream of UNRECTIFIED left+right
    pairs (lens distortion must still be present -- a calibration-from-scratch is
    exactly what recovers it). Capture's ``imucam.sample`` (:class:`ImuCamPacket`)
    carries those RAW frames: ``read_cam.py``'s live source polls the OAK-D's raw
    mono outputs (``dev.poll("left")`` / ``dev.poll("right")``), NOT a rectified
    ``StereoDepth`` output, and the pack/calibrate steps pass the grays through
    unchanged (only the IMU is calibrated). So subscribing this topic delivers the
    raw pairs the wizard requires.

    Mirrors :class:`IpcImuRawSource`'s "stream contract" -- the four touch-points
    the calib UI drives: ``start(callback)``, ``stop()``, :attr:`error` and
    :attr:`device_id` -- but, because the grays ride capture's shared-memory rings
    (they are not pure POD like ``imu.raw``), it ALSO attaches the consumer-side
    capture rings so :func:`~ui.comms.converters.to_local` can ``read_copy`` each
    frame out of shared memory into a private array. Each delivered frame is a
    small record ``(seq, ts_ns, gray_left, gray_right)``.

    Mono guard
    ----------
    A mono recording publishes ``gray_right is None``. Stereo calibration needs
    BOTH frames, so on the first such packet we latch a clear :attr:`error` and
    STOP emitting -- the wizard polls :attr:`error` and never receives a half-pair
    (which it could not calibrate from anyway).

    Why the :class:`~ui.comms.IPCSubscriber` transport (not a bare client)
    ---------------------------------------------------------------------
    This source delivers ``imucam.sample`` the SAME proven way the live
    "Camera+Depth+IMU triplet" window does (:class:`IpcTripletWorker`): an
    :class:`~ui.comms.IPCSubscriber` re-hydrates each wire packet (``read_copy``-ing
    the grays out of capture's rings) and republishes the in-proc
    :class:`~ui.comms.messages.ImuCamPacket` onto a PRIVATE
    :class:`~ui.comms.LocalPubSub`, which we tap to emit the wizard callback. The
    triplet window demonstrably renders capture's live ``imucam.sample`` through
    exactly this bridge, so reusing it (rather than a bespoke
    ``IPCPubSub(role="client")`` + hand-rolled handler) removes any chance of a
    late-joining subscriber drifting from the path that is known to deliver this
    topic. ``IPCSubscriber`` also OWNS its client's lifecycle (it starts the client
    on its own thread inside ``run`` and stops it on ``stop``), so the connect /
    receive / teardown all match the triplet's behaviour byte-for-byte.

    No-frame watchdog
    -----------------
    If :meth:`start` succeeds (rings attached + the subscriber thread is alive) but
    ZERO frames arrive within :attr:`frame_timeout_s`, a watchdog latches a clear,
    actionable :attr:`error` ("connected to capture but no stereo frames arrived
    ...") so the wizard surfaces it instead of hanging on the placeholder preview
    forever. The wizard already polls :attr:`error` on every drain tick.

    Device-agnostic: consumes only the abstract ``imucam.sample`` topic + the Wire
    POD type; no depthai / OAK-D specifics cross into the UI.
    """

    #: Default no-frame watchdog window (s). If the subscriber connects but no
    #: stereo packet arrives within this long, :meth:`start`'s watchdog latches a
    #: clear error. ~5 s comfortably covers a live OAK-D's first-frame latency at
    #: any sane fps while still failing fast enough that the operator isn't left
    #: staring at a frozen placeholder.
    FRAME_TIMEOUT_S = 5.0

    def __init__(self, capture_endpoint: str, width: int, height: int, *,
                 device_id: str = "default",
                 connect_timeout_s: float = 30.0,
                 frame_timeout_s: float | None = None) -> None:
        self._endpoint = capture_endpoint
        self._w = int(width)
        self._h = int(height)
        self._connect_timeout_s = float(connect_timeout_s)
        self.frame_timeout_s = (self.FRAME_TIMEOUT_S if frame_timeout_s is None
                                else float(frame_timeout_s))
        # Public attrs the wizard reads (mirror the IMU stream's contract).
        self.device_id: str = device_id
        self.error: str | None = None

        # Proven triplet transport: IPCSubscriber re-hydrates capture's
        # imucam.sample onto this PRIVATE local bus; we tap it for the callback.
        self._rings: RingRegistry | None = None
        self._client: IPCPubSub | None = None
        self._sub: IPCSubscriber | None = None
        self._local = LocalPubSub()
        # Latched once a mono packet is seen, so we emit no further half-pairs.
        self._mono = False
        # Count of delivered packets -- the watchdog reads this to decide whether
        # any frame ever arrived. Written only on the subscriber's recv thread.
        self._n_packets = 0
        # cb(seq:int, ts_ns:int, gray_left:(H,W) uint8, gray_right:(H,W) uint8)
        self._cb = None
        # No-frame watchdog: fires once `frame_timeout_s` after a clean start if
        # `_n_packets` is still 0.
        self._watchdog: threading.Timer | None = None

    # ------------------------------------------------------------------ #
    def start(self, callback) -> None:
        """Attach capture's rings, connect, and stream RAW pairs to ``callback``.

        On a missing ring (capture down) or a connect failure set :attr:`error`
        and return (do NOT raise): the wizard polls :attr:`error` on its UI timer
        and surfaces it itself, exactly like :class:`IpcImuRawSource`. Mirrors the
        :class:`IpcTripletWorker` bring-up: attach rings -> build the client ->
        wire an :class:`~ui.comms.IPCSubscriber` -> ``start`` it (which starts the
        client on its own thread). A no-frame watchdog is armed last.
        """
        self._cb = callback
        try:
            # The grays live in capture's shared-memory rings; the converter
            # read_copies them out, so the consumer side must be attached first.
            # A missing ring almost always means capture is not running -> map it
            # to a clear, device-agnostic reason rather than a raw shm-path error.
            self._rings = _attach_capture_rings(self._endpoint, self._w, self._h)
        except RuntimeError as e:
            self.error = str(e)
            LOG.info("stereo-src[%s]: ring attach failed: %s", self._endpoint, e)
            return
        LOG.info("stereo-src[%s]: capture rings attached (%dx%d)",
                 self._endpoint, self._w, self._h)

        # Tap the private bus where the subscriber will republish each re-hydrated
        # ImuCamPacket (this is the exact stream the triplet window renders live).
        self._local.subscribe(topics.IMUCAM_SAMPLE, self._on_packet)
        client = IPCPubSub(self._endpoint, role="client",
                           connect_timeout_s=self._connect_timeout_s)
        # IPCSubscriber.__init__ calls client.subscribe for the topic; run() then
        # starts the client. It swallows a connect failure onto a log line + a
        # dead thread, so we detect that below (mirrors IpcTripletWorker).
        sub = IPCSubscriber(self._local, client, self._rings,
                            [topics.IMUCAM_SAMPLE])
        sub.start()
        # A connect failure leaves the subscriber thread dead before it can block
        # on its stop event; surface it as the wizard's error rather than hanging.
        if not sub.is_alive():
            self.error = (f"capture stereo stream connect failed "
                          f"({self._endpoint})")
            LOG.info("stereo-src[%s]: subscriber thread died at start -- "
                     "client never connected", self._endpoint)
            sub.stop()
            self._rings.close()
            self._rings = None
            return
        self._client = client
        self._sub = sub
        LOG.info("stereo-src[%s]: subscriber connected, awaiting first packet",
                 self._endpoint)
        # Arm the no-frame watchdog LAST, once the transport is genuinely up.
        self._watchdog = threading.Timer(self.frame_timeout_s,
                                         self._on_frame_timeout)
        self._watchdog.daemon = True
        self._watchdog.start()

    def _on_packet(self, pkt) -> None:
        """Local-bus tap (subscriber recv thread): emit one RAW stereo pair.

        Receives the ALREADY re-hydrated :class:`~ui.comms.messages.ImuCamPacket`
        (the :class:`~ui.comms.IPCSubscriber` ran ``to_local``, so the grays are
        already ``read_copy``-ed out of shared memory and the record owns them).
        """
        if pkt is END:                                # WireEnd -> local END
            return
        if self._mono:                                # already gave up on this stream
            return
        if self._n_packets == 0:
            LOG.info("stereo-src[%s]: first imucam.sample packet received "
                     "(seq=%s)", self._endpoint, getattr(pkt, "seq", "?"))
        self._n_packets += 1
        if pkt.gray_right is None:
            # Mono recording: latch the reason ONCE and stop emitting -- the
            # wizard needs both frames; a lone left frame is not calibratable.
            self._mono = True
            self.error = ("stereo calibration needs a stereo capture; "
                          "this stream has no right frame")
            LOG.info("stereo-src[%s]: mono packet (gray_right is None) -- "
                     "stereo calib needs both frames", self._endpoint)
            return
        cb = self._cb
        if cb is not None:
            cb(int(pkt.seq), int(pkt.ts_ns), pkt.gray_left, pkt.gray_right)

    def _on_frame_timeout(self) -> None:
        """Watchdog: latch a clear error if no frame arrived after the timeout.

        Runs on the timer thread. Only fires when the transport came up cleanly
        (no prior error) yet ZERO packets were delivered -- the exact silent-hang
        condition the operator hit. Does nothing once any packet has arrived or an
        error is already latched.
        """
        if self.error is not None or self._n_packets > 0:
            return
        self.error = (
            f"connected to capture on {self._endpoint!r} but no stereo frames "
            f"arrived within {self.frame_timeout_s:.0f}s -- is capture running "
            f"and producing a stereo pair?")
        LOG.info("stereo-src[%s]: WATCHDOG fired -- 0 frames in %.0fs",
                 self._endpoint, self.frame_timeout_s)

    def stop(self) -> None:
        """Stop the subscriber + detach the rings (idempotent; swallow errors)."""
        watchdog = self._watchdog
        self._watchdog = None
        if watchdog is not None:
            watchdog.cancel()
        # Stopping the IPCSubscriber stops its underlying client + joins the recv
        # thread; the client is owned by the subscriber, so we do NOT stop it
        # separately (that would double-close).
        sub = self._sub
        self._sub = None
        self._client = None
        if sub is not None:
            try:
                sub.stop()
            except Exception:                                      # noqa: BLE001
                pass
        rings = self._rings
        self._rings = None
        if rings is not None:
            try:
                rings.close()
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
# (1b') BA-window source for the "BA Window" visualiser
# --------------------------------------------------------------------------- #
class IpcBaWindowSource:
    """Duck-typed windowed-BA snapshot stream over VIO's ``ba.window`` IPC topic.

    The "BA Window" view (:mod:`ui.qt.ba_window`) drives a stream object with the
    same three touch-points as the other sources -- ``start(callback)`` /
    ``stop()`` / ``.error`` -- and feeds each :class:`~ui.comms.messages.BaWindow`
    to its renderer. This adapter sources the records from VIO's ``ba.window``
    topic (pure POD, no shared-memory ring) so the window needs no device handle,
    EXACTLY like :class:`IpcGyroFuseSource`.

    Unlike the per-frame gyro-fusion stream, the window ALSO buffers the last
    ``buffer`` snapshots in a bounded :class:`collections.deque` under a lock, so
    the window's timeline slider can scrub back to a previous solve in REPLAY mode
    (and rolling last-N in LIVE). The deque is the single source of truth for both
    modes; :meth:`snapshot_count` / :meth:`snapshot_at` expose it to the slider.

    ``ba.window`` is published ONLY when VIO ran with ``--ba-window`` (the opt-in
    capture engine); a healthy VIO without that flag simply never emits here, so
    the window stays on its "waiting" frame (NOT an error). Mirrors
    :class:`IpcGyroFuseSource`'s connect-error model: a connect timeout is
    swallowed onto :attr:`error` (the window polls it) rather than raising.
    """

    #: Default bounded buffer of recent snapshots (the slider's range). Generous
    #: enough to scrub a short replay segment; LIVE keeps the rolling last-N.
    DEFAULT_BUFFER = 240

    def __init__(self, vio_endpoint: str, *, buffer: int = DEFAULT_BUFFER,
                 connect_timeout_s: float = 30.0) -> None:
        self._endpoint = vio_endpoint
        self._connect_timeout_s = float(connect_timeout_s)
        self.error: str | None = None
        self._client: IPCPubSub | None = None
        # ba.window is pure POD (no ring), so a bare registry suffices for the
        # converter -- the ``rings`` arg is unused for this topic.
        self._rings = RingRegistry()
        self._cb = None
        self._lock = threading.Lock()
        self._buf: "collections.deque" = collections.deque(maxlen=int(buffer))

    def start(self, callback) -> None:
        """Connect to VIO and stream each BaWindow record to ``callback``.

        Each arrival is appended to the bounded buffer (oldest evicted) UNDER the
        lock BEFORE the callback runs, so the window's ``snapshot_at`` always sees
        a snapshot the callback already knows about.
        """
        self._cb = callback
        client = IPCPubSub(self._endpoint, role="client",
                           connect_timeout_s=self._connect_timeout_s)
        client.subscribe(topics.BA_WINDOW, self._on_msg)
        try:
            client.start()
        except Exception as e:                                     # noqa: BLE001
            self.error = f"VIO BA-window stream connect failed: {e}"
            return
        self._client = client

    def _on_msg(self, wm) -> None:
        if wm is END:
            return
        msg = to_local(topics.BA_WINDOW, wm, self._rings)
        if msg is END:                                # WireEnd -> local END
            return
        with self._lock:
            self._buf.append(msg)
        cb = self._cb
        if cb is not None:
            cb(msg)

    # -- slider / buffer access (thread-safe) ----------------------------- #
    def snapshot_count(self) -> int:
        """Number of buffered snapshots currently available (the slider range)."""
        with self._lock:
            return len(self._buf)

    def snapshot_at(self, i: int):
        """The buffered snapshot at index ``i`` (0 = oldest, -1 = newest), or None.

        Indices are into the rolling deque; an out-of-range index returns ``None``
        so the window's slider can never raise on a race with an eviction.
        """
        with self._lock:
            n = len(self._buf)
            if n == 0:
                return None
            if i < 0:
                i += n
            if 0 <= i < n:
                return self._buf[i]
            return None

    def stop(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.stop()
            except Exception:                                      # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# (1b2) Frontend-internals snapshot source for the "Frontend Internals" window
# --------------------------------------------------------------------------- #
class IpcFrontendVizSource:
    """Duck-typed frontend-internals stream over VIO's ``frame.frontend`` topic.

    The "Frontend Internals" view (:mod:`ui.qt.frontend_window`) drives a stream
    object with the same touch-points as the other sources -- ``start(callback)``
    / ``stop()`` / ``.error`` -- plus the buffered ``snapshot_count`` /
    ``snapshot_at`` the timeline slider scrubs. It sources each
    :class:`~ui.comms.messages.FrameFrontend` from VIO's ``frame.frontend`` topic
    (pure POD -- the quantised heatmap + flow arrays ride inline, no shared-memory
    ring), so the window needs no device handle. A direct structural twin of
    :class:`IpcBaWindowSource`.

    ``frame.frontend`` is published ONLY when VIO ran with ``--frontend-viz``; a
    healthy VIO without that flag simply never emits here, so the window stays on
    its "waiting" frame (NOT an error). A connect timeout is swallowed onto
    :attr:`error` (the window polls it) rather than raising.
    """

    #: Default bounded buffer of recent snapshots (the slider's range). At 20 Hz a
    #: 240-deep buffer scrubs the last ~12 s of frames; LIVE keeps the rolling.
    DEFAULT_BUFFER = 240

    def __init__(self, vio_endpoint: str, *, buffer: int = DEFAULT_BUFFER,
                 connect_timeout_s: float = 30.0) -> None:
        self._endpoint = vio_endpoint
        self._connect_timeout_s = float(connect_timeout_s)
        self.error: str | None = None
        self._client: IPCPubSub | None = None
        # frame.frontend is pure POD (no ring), so a bare registry suffices for
        # the converter -- the ``rings`` arg is unused for this topic.
        self._rings = RingRegistry()
        self._cb = None
        self._lock = threading.Lock()
        self._buf: "collections.deque" = collections.deque(maxlen=int(buffer))

    def start(self, callback) -> None:
        """Connect to VIO and stream each FrameFrontend record to ``callback``.

        Each arrival is appended to the bounded buffer (oldest evicted) UNDER the
        lock BEFORE the callback runs, so the window's ``snapshot_at`` always sees
        a snapshot the callback already knows about.
        """
        self._cb = callback
        client = IPCPubSub(self._endpoint, role="client",
                           connect_timeout_s=self._connect_timeout_s)
        client.subscribe(topics.FRAME_FRONTEND, self._on_msg)
        try:
            client.start()
        except Exception as e:                                     # noqa: BLE001
            self.error = f"VIO frontend-viz stream connect failed: {e}"
            return
        self._client = client

    def _on_msg(self, wm) -> None:
        if wm is END:
            return
        msg = to_local(topics.FRAME_FRONTEND, wm, self._rings)
        if msg is END:                                # WireEnd -> local END
            return
        with self._lock:
            self._buf.append(msg)
        cb = self._cb
        if cb is not None:
            cb(msg)

    # -- slider / buffer access (thread-safe) ----------------------------- #
    def snapshot_count(self) -> int:
        """Number of buffered snapshots currently available (the slider range)."""
        with self._lock:
            return len(self._buf)

    def snapshot_at(self, i: int):
        """The buffered snapshot at index ``i`` (0 = oldest, -1 = newest), or None.

        Indices are into the rolling deque; an out-of-range index returns ``None``
        so the window's slider can never raise on a race with an eviction (mirrors
        :meth:`IpcBaWindowSource.snapshot_at`).
        """
        with self._lock:
            n = len(self._buf)
            if n == 0:
                return None
            if i < 0:
                i += n
            if 0 <= i < n:
                return self._buf[i]
            return None

    def stop(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.stop()
            except Exception:                                      # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# (1c) Loop-closure match source for the "Loop Closure" window
# --------------------------------------------------------------------------- #
class IpcLoopMatchSource:
    """Duck-typed loop-closure stream: SLAM ``slam.loop`` + VIO ``keyframe`` grays.

    The "Loop Closure" window (:mod:`ui.qt.loop_window`) drives a stream object
    with three touch-points -- ``start(callback)`` / ``stop()`` / ``.error`` --
    and renders the LAST loop event (loops are sporadic). This adapter subscribes
    TWO endpoints (mirrors :class:`IpcKeypointWorker`'s two-endpoint pattern):

    * ``slam.loop`` (SLAM, pure POD) -> the per-loop-CANDIDATE match funnel (the
      matched ORB pixel pairs + per-match verification stage + funnel counts +
      rotation-gate verdict). There are NO keyframe images on this topic (SLAM
      keeps only descriptors), so we join by seq to the grays below.
    * ``keyframe`` (VIO, kf rings) -> every keyframe's GRAY image, buffered by
      source frame seq (like :class:`_KeyframeAccumulator`, bounded) so a loop
      event can look up its current + matched-old keyframe gray.

    On each ``slam.loop`` message the callback gets a finished
    :class:`~ui.viz.loop_render.LoopEvent` (the two looked-up grays -- ``None``
    when evicted -- plus the funnel). The window keeps the last one shown.

    Connect-error model mirrors :class:`IpcGyroFuseSource`: a connect timeout is
    swallowed onto :attr:`error` (the window polls it) rather than raising.
    """

    #: Cap on buffered keyframe grays (bounds memory). A loop can reach far back,
    #: so this is generous; an evicted keyframe simply renders as a placeholder
    #: pane (the funnel counts + verdict still show).
    MAX_GRAYS = 800

    def __init__(self, slam_endpoint: str, vio_endpoint: str, *,
                 width: int | None = None, height: int | None = None,
                 connect_timeout_s: float = 10.0) -> None:
        self._slam_ep = slam_endpoint
        self._vio_ep = vio_endpoint
        self._w = width
        self._h = height
        self._connect_timeout_s = float(connect_timeout_s)
        self.error: str | None = None

        self._slam_client: IPCPubSub | None = None
        self._vio_client: IPCPubSub | None = None
        self._vio_rings: RingRegistry | None = None
        self._lock = threading.Lock()
        self._grays: dict[int, np.ndarray] = {}       # seq -> (H, W) gray
        self._cb = None

    # ------------------------------------------------------------------ #
    def start(self, callback) -> None:
        """Connect both endpoints; stream a LoopEvent per slam.loop message."""
        self._cb = callback
        # Attach VIO's keyframe rings (consumer side) so the keyframe converter
        # can read_copy the gray array out of them. Missing rings == VIO not
        # running -> a clear, device-agnostic reason.
        try:
            kwargs = {}
            if self._w is not None and self._h is not None:
                kwargs = {"width": int(self._w), "height": int(self._h)}
            self._vio_rings = RingRegistry().attach_all(
                default_vio_specs(endpoint=self._vio_ep, **kwargs))
        except FileNotFoundError as e:
            self.error = (f"VIO keyframe stream not available on "
                          f"{self._vio_ep!r} (is VIO running?): {e}")
            return

        vio_client = IPCPubSub(self._vio_ep, role="client",
                               connect_timeout_s=self._connect_timeout_s)
        vio_client.subscribe(topics.KEYFRAME, self._on_keyframe)
        slam_client = IPCPubSub(self._slam_ep, role="client",
                                connect_timeout_s=self._connect_timeout_s)
        slam_client.subscribe(topics.SLAM_LOOP, self._on_loop)
        try:
            vio_client.start()
            slam_client.start()
        except Exception as e:                                     # noqa: BLE001
            self.error = f"loop-closure stream connect failed: {e}"
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

    # ------------------------------------------------------------------ #
    def _on_keyframe(self, wm) -> None:
        """VIO recv thread: buffer one keyframe's GRAY image, keyed by seq."""
        if wm is END:
            return
        kf = to_local(topics.KEYFRAME, wm, self._vio_rings)
        if kf is END:
            return
        seq = int(kf.seq)
        gray = np.asarray(kf.gray_left).copy()        # own the slot copy
        with self._lock:
            self._grays[seq] = gray
            # dict preserves insertion order -> drop the oldest over the cap.
            n = len(self._grays)
            if n > self.MAX_GRAYS:
                for old in list(self._grays.keys())[: n - self.MAX_GRAYS]:
                    self._grays.pop(old, None)

    def _on_loop(self, wm) -> None:
        """SLAM recv thread: build a LoopEvent (join the two grays by seq)."""
        if wm is END:
            return
        msg = to_local(topics.SLAM_LOOP, wm, RingRegistry())   # POD, no rings
        if msg is END:
            return
        from ui.viz.loop_render import LoopEvent

        cur_seq, old_seq = int(msg.cur_seq), int(msg.old_seq)
        with self._lock:
            cur_gray = self._grays.get(cur_seq)
            old_gray = self._grays.get(old_seq)
        ev = LoopEvent(
            cur_seq=cur_seq, old_seq=old_seq,
            cur_gray=cur_gray, old_gray=old_gray,
            cur_px=np.asarray(msg.cur_px, np.float32).reshape(-1, 2),
            old_px=np.asarray(msg.old_px, np.float32).reshape(-1, 2),
            stage=np.asarray(msg.stage, np.uint8).reshape(-1),
            n_appearance=int(msg.n_appearance), n_fmat=int(msg.n_fmat),
            n_pnp=int(msg.n_pnp), rot_deg=float(msg.rot_deg),
            rot_gate_deg=float(msg.rot_gate_deg), accepted=bool(msg.accepted))
        cb = self._cb
        if cb is not None:
            cb(ev)

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        """Close both IPC clients + the kf rings (idempotent)."""
        for attr in ("_slam_client", "_vio_client"):
            client = getattr(self, attr)
            setattr(self, attr, None)
            if client is not None:
                try:
                    client.stop()
                except Exception:                                  # noqa: BLE001
                    pass
        rings = self._vio_rings
        self._vio_rings = None
        if rings is not None:
            try:
                rings.close()
            except Exception:                                      # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# (1d) Pose-graph before/after source for the "Pose Graph" visualiser
# --------------------------------------------------------------------------- #
class IpcPoseGraphSource:
    """Duck-typed pose-graph before/after stream (ALGORITHMS.md §4.3).

    The "Pose Graph" window (:mod:`ui.qt.posegraph_window`) drives a stream object
    with the BA-window source's surface -- ``start(callback)`` / ``stop()`` /
    ``.error`` + ``snapshot_count()`` / ``snapshot_at(i)`` (the timeline slider
    scrubs the buffer) -- and feeds each :class:`~ui.viz.posegraph_render.PoseGraphSnapshot`
    to its renderer.

    This is a PURE CONSUMER of EXISTING topics -- there is NO new IPC field or
    data-path change (gap=0 trivially). It subscribes THREE endpoints (mirrors
    :class:`IpcSlamMapSource`'s two-endpoint pattern + the loop funnel):

    * ``pose.odom`` (VIO) -> the RAW, drifted dense VIO trajectory. Buffered as a
      bounded seq->position ring; this is the BEFORE state (pre-pose-graph) and the
      per-keyframe pre-PGO node anchor (raw VIO position at the keyframe's seq).
    * ``slam.map`` (SLAM) -> the CONTINUOUS pose-graph-CORRECTED keyframe positions
      (``kf_positions`` + ``kf_ids`` source seqs) + ``n_loops``. This is the AFTER
      state. A new ``n_loops`` value is the trigger to assemble + buffer a snapshot.
    * ``slam.loop`` (SLAM) -> the per-candidate match funnel; the CONFIRMED
      (``accepted``) ones give the loop EDGE ``(cur_seq, old_seq)`` chords.

    A :class:`~ui.viz.posegraph_render.PoseGraphSnapshot` is assembled whenever the
    correctedmap's ``n_loops`` increases (a loop just closed -> the graph was
    re-optimised), capturing the matched raw/corrected node pairs + the dense
    before/after trails + the loop edges KNOWN AT THAT MOMENT. The bounded deque
    keeps the recent snapshots so the timeline slider can scrub to each closure.

    All positions are world X-Z in the camera-optical frame (the frame ``slam.map``
    / ``pose.odom`` both publish); the renderer is top-down so no NED conversion is
    needed (unlike the main Viewer3D, which is intentionally a different lens).

    Connect-error model mirrors :class:`IpcSlamMapSource`'s: a connect timeout is
    swallowed onto :attr:`error` (the window polls it) rather than raising.
    """

    #: Cap on buffered raw VIO poses (the dense BEFORE trail + node anchors). A
    #: loop session is ~30-45 s; this is generous enough to keep the whole path.
    MAX_VIO = 6000
    #: Cap on confirmed loop edges retained (loops are sporadic, so tiny).
    MAX_LOOPS = 64
    #: Default bounded buffer of assembled snapshots (the slider's timeline range).
    DEFAULT_BUFFER = 64

    def __init__(self, vio_endpoint: str, slam_endpoint: str, *,
                 buffer: int = DEFAULT_BUFFER,
                 connect_timeout_s: float = 30.0) -> None:
        self._vio_ep = vio_endpoint
        self._slam_ep = slam_endpoint
        self._connect_timeout_s = float(connect_timeout_s)
        self.error: str | None = None

        self._vio_client: IPCPubSub | None = None
        self._slam_client: IPCPubSub | None = None
        # pose.odom / slam.map / slam.loop are all pure POD (no rings), so a bare
        # registry suffices for the converters (the ``rings`` arg is unused).
        self._rings = RingRegistry()
        self._cb = None

        self._lock = threading.Lock()
        # Raw VIO dense trail: parallel seq + optical (x, z) arrays (bounded).
        self._vio_seqs: np.ndarray = np.zeros(0, np.int64)
        self._vio_xz: np.ndarray = np.zeros((0, 2), np.float64)
        # Latest corrected keyframe map (the AFTER nodes), from slam.map.
        self._kf_seqs: np.ndarray = np.zeros(0, np.int64)
        self._kf_after_xz: np.ndarray = np.zeros((0, 2), np.float64)
        self._n_loops = 0
        # Confirmed loop edges seen so far: list of (cur_seq, old_seq).
        self._loop_edges: list[tuple[int, int]] = []
        # Assembled snapshots for the slider (one per loop closure event).
        self._buf: "collections.deque" = collections.deque(maxlen=int(buffer))

    # ------------------------------------------------------------------ #
    def start(self, callback) -> None:
        """Connect all three endpoints; assemble a snapshot per loop closure."""
        self._cb = callback
        vio_client = IPCPubSub(self._vio_ep, role="client",
                               connect_timeout_s=self._connect_timeout_s)
        vio_client.subscribe(topics.POSE_ODOM, self._on_pose)
        slam_client = IPCPubSub(self._slam_ep, role="client",
                                connect_timeout_s=self._connect_timeout_s)
        slam_client.subscribe(topics.SLAM_MAP, self._on_slammap)
        slam_client.subscribe(topics.SLAM_LOOP, self._on_loop)
        try:
            vio_client.start()
            slam_client.start()
        except Exception as e:                                     # noqa: BLE001
            self.error = f"pose-graph stream connect failed: {e}"
            for c in (vio_client, slam_client):
                try:
                    c.stop()
                except Exception:                                  # noqa: BLE001
                    pass
            return
        self._vio_client = vio_client
        self._slam_client = slam_client

    # ------------------------------------------------------------------ #
    def _on_pose(self, wm) -> None:
        """VIO recv thread: append one raw (drifted) VIO pose to the trail."""
        if wm is END:
            return
        msg = to_local(topics.POSE_ODOM, wm, self._rings)
        if msg is END:                                # WireEnd -> local END
            return
        # Optical-world translation -> top-down (x, z); the renderer is top-down.
        t = np.asarray(msg.T_world_cam, np.float64)[:3, 3]
        seq = int(msg.seq)
        with self._lock:
            self._vio_seqs = np.append(self._vio_seqs, seq)
            self._vio_xz = np.vstack((self._vio_xz, t[[0, 2]][None, :]))
            if len(self._vio_seqs) > self.MAX_VIO:
                self._vio_seqs = self._vio_seqs[-self.MAX_VIO:]
                self._vio_xz = self._vio_xz[-self.MAX_VIO:]

    def _on_slammap(self, wm) -> None:
        """SLAM recv thread: latch the corrected nodes; snapshot on a new loop."""
        if wm is END:
            return
        msg = to_local(topics.SLAM_MAP, wm, self._rings)
        if msg is END:
            return
        kf_pos = np.asarray(msg.kf_positions, np.float64).reshape(-1, 3)
        kf_seqs = np.asarray(msg.kf_seqs, np.int64).reshape(-1)
        n_loops = int(msg.n_loops)
        # Drop a malformed map (seqs must align with positions to anchor nodes).
        if len(kf_seqs) != len(kf_pos):
            return
        snap = None
        with self._lock:
            self._kf_seqs = kf_seqs
            self._kf_after_xz = (kf_pos[:, [0, 2]] if len(kf_pos)
                                 else np.zeros((0, 2), np.float64))
            if n_loops > self._n_loops:
                # A loop just closed -> the graph was re-optimised. Assemble a
                # before/after snapshot from the state KNOWN AT THIS MOMENT and
                # buffer it for the timeline slider.
                self._n_loops = n_loops
                snap = self._assemble_locked()
                if snap is not None:
                    self._buf.append(snap)
        if snap is not None and self._cb is not None:
            self._cb(snap)

    def _on_loop(self, wm) -> None:
        """SLAM recv thread: record a CONFIRMED loop edge (cur_seq, old_seq)."""
        if wm is END:
            return
        msg = to_local(topics.SLAM_LOOP, wm, self._rings)
        if msg is END:
            return
        if not bool(msg.accepted):                    # only confirmed -> an edge
            return
        edge = (int(msg.cur_seq), int(msg.old_seq))
        with self._lock:
            if edge not in self._loop_edges:
                self._loop_edges.append(edge)
                if len(self._loop_edges) > self.MAX_LOOPS:
                    self._loop_edges = self._loop_edges[-self.MAX_LOOPS:]

    # ------------------------------------------------------------------ #
    def _assemble_locked(self):
        """Build a :class:`PoseGraphSnapshot` from the latched state (lock held).

        The BEFORE node anchor for keyframe ``s`` is the raw VIO position at seq
        ``s`` (the pre-PGO estimate). Only keyframes whose seq still lives in the
        bounded VIO ring get a node (others rolled off and have no anchor). The
        AFTER trail is the BEFORE trail rubber-sheeted by the per-node correction
        delta, piecewise-linearly interpolated by seq (the SAME deform ``ui.main``
        applies to the corrected-VIO line) -- so the dense path also visibly
        spreads the correction, not just the nodes.
        """
        from ui.viz.posegraph_render import PoseGraphSnapshot

        vio_seqs = self._vio_seqs
        vio_xz = self._vio_xz
        kf_seqs = self._kf_seqs
        kf_after = self._kf_after_xz
        if len(vio_seqs) == 0 or len(kf_seqs) < 2:
            return None

        # seq -> raw VIO position (the pre-PGO node anchor). VIO seqs arrive
        # monotonic but sort defensively for searchsorted robustness.
        order = np.argsort(vio_seqs, kind="stable")
        vs = vio_seqs[order]
        vp = vio_xz[order]
        pos = np.searchsorted(vs, kf_seqs)
        in_range = pos < len(vs)
        hit = np.zeros(len(kf_seqs), bool)
        hit[in_range] = vs[pos[in_range]] == kf_seqs[in_range]
        if hit.sum() < 2:                             # need >=2 anchored nodes
            return None

        kf_seq_hit = kf_seqs[hit]
        kf_before = vp[pos[hit]]                       # (Kh, 2) raw node anchors
        kf_after_hit = kf_after[hit]                   # (Kh, 2) corrected nodes
        # Sort nodes by seq so odometry edges chain in graph order.
        ks_order = np.argsort(kf_seq_hit, kind="stable")
        kf_seq_hit = kf_seq_hit[ks_order]
        kf_before = kf_before[ks_order]
        kf_after_hit = kf_after_hit[ks_order]

        # Rubber-sheet the dense raw trail by the per-node correction delta (the
        # same np.interp-per-axis deform main.corrected_vio_snapshot uses).
        delta = kf_after_hit - kf_before              # (Kh, 2)
        ks = kf_seq_hit.astype(np.float64)
        dense_seq = vio_seqs.astype(np.float64)
        corr = np.empty_like(vio_xz)
        for axis in range(2):
            corr[:, axis] = np.interp(dense_seq, ks, delta[:, axis])
        after_traj = vio_xz + corr

        # Map confirmed loop edges (cur_seq, old_seq) to node indices.
        seq_to_idx = {int(s): i for i, s in enumerate(kf_seq_hit)}
        loop_edges = []
        for cur_s, old_s in self._loop_edges:
            ci, oi = seq_to_idx.get(cur_s), seq_to_idx.get(old_s)
            if ci is not None and oi is not None:
                loop_edges.append((ci, oi))

        return PoseGraphSnapshot(
            kf_seqs=kf_seq_hit.copy(),
            kf_before_xz=kf_before.copy(),
            kf_after_xz=kf_after_hit.copy(),
            before_traj_xz=vio_xz.copy(),
            after_traj_xz=after_traj,
            loop_edges=tuple(loop_edges),
            n_loops=self._n_loops)

    # -- slider / buffer access (thread-safe) ----------------------------- #
    def snapshot_count(self) -> int:
        """Number of buffered snapshots (one per loop closure) for the slider."""
        with self._lock:
            return len(self._buf)

    def snapshot_at(self, i: int):
        """The buffered snapshot at index ``i`` (0 = oldest, -1 = newest), or None.

        Out-of-range returns ``None`` so the slider can never raise on a race with
        an eviction (mirrors :meth:`IpcBaWindowSource.snapshot_at`).
        """
        with self._lock:
            n = len(self._buf)
            if n == 0:
                return None
            if i < 0:
                i += n
            if 0 <= i < n:
                return self._buf[i]
            return None

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        """Close both IPC clients (idempotent)."""
        for attr in ("_vio_client", "_slam_client"):
            client = getattr(self, attr)
            setattr(self, attr, None)
            if client is not None:
                try:
                    client.stop()
                except Exception:                                  # noqa: BLE001
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
    #: PERSISTENT (it accumulates log-odds evidence as keyframes arrive), so an
    #: evicted keyframe's evidence stays folded into the grid -- this cap only bounds
    #: the raw depth-grid buffer, not the fused map.
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
        occupancy build adds occupied + free log-odds evidence to the persistent
        grid from the not-yet-fused keyframes -- see :meth:`IpcSlamMapSource._build`).
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
    cubes) the way an OctoMap / Voxblox does it: a PROBABILISTIC LOG-ODDS OCCUPANCY
    GRID with FREE-SPACE RAY CARVING. This is how ModalAI VOXL gets a clean map out
    of NOISY STEREO (not a ToF sensor): every depth ray does TWO things -- it adds
    OCCUPIED evidence at its hit point AND adds FREE evidence to every voxel it
    passes THROUGH on the way there. A voxel that stereo noise wrongly populated
    from one viewpoint is later CROSSED by rays from new viewpoints (the camera can
    now see THROUGH it), and that free evidence drives its log-odds back below the
    occupied threshold so the voxel DISAPPEARS. The map self-cleans as the camera
    moves -- exactly the "remove already-added points when they're detected invalid"
    behaviour the user asked for.

    This supersedes the earlier hit-count-only grid (a cell occupied once it was hit
    by ``>= OCC_HITS`` keyframes). That gate only ever ADDED; it never removed, so
    persistent stereo artefacts (e.g. the garbage cone a textureless ceiling throws)
    that were re-hit a few times crossed the threshold and stuck forever, blobbing
    the map. Log-odds + carving is strictly better: it keeps the temporal-fusion
    cleanliness (a real surface is re-observed and its log-odds climbs and saturates)
    AND actively erases noise the camera later sees past.

    This is a pure CONSUMER of existing topics: no new field, no data-path change.

    Subscribes TWO endpoints (mirrors :class:`IpcKeypointWorker`'s two-endpoint
    pattern, but the keyframe depth rides VIO's dedicated ``kf_depth`` ring, NOT
    capture's frame rings):

    * ``keyframe`` (VIO) -> per keyframe ``seq`` we keep its DENOISED metric depth
      (``read_copy``-ed out of VIO's kf ring by the converter) and the FULL POSE
      (rotation AND translation) of its ``T_world_cam`` -- everything the
      back-projection needs. The translation is also the camera ORIGIN ``C`` each
      ray is carved FROM. The keyframe rate is low (kf_every), so a bounded dict of
      the last :data:`MAX_KEYFRAMES` raw depth grids caps memory (the fused log-odds
      grid is persistent and unaffected by that cap).
    * ``slam.map`` (SLAM) -> the CONTINUOUS per-keyframe loop-closure-corrected
      positions, keyed by source frame seq. Retained (and kept fresh) as the source
      of truth for a future corrected-map variant + diagnostics, but it does NOT
      position the voxels: they are quantised from each keyframe's own VIO pose for
      a consistent, seam-free odom-frame grid. (Re-anchoring the grid to the
      loop-corrected translations is a deliberate later step.)

    Log-odds occupancy fusion (PERSISTENT + INCREMENTAL -- see
    :meth:`_fuse_keyframe_locked` and :meth:`_build`):

    1. :attr:`_log` is a persistent ``{(ix,iy,iz) -> log_odds (float)}`` dict.
       log_odds is ``log(p_occ / (1 - p_occ))``; >0 leans occupied, <0 leans free.
       As each NEW keyframe arrives, its (denoised) depth is back-projected to the
       world by the keyframe's OWN VIO pose ``P = R Xc + t`` (strided + depth-gated
       + edge-rejected), with the camera origin ``C = t``. Then, per keyframe:

       * HIT update: the voxel containing each world hit ``P`` gets ``+= L_OCC``.
       * FREE carving: every voxel the ray ``C -> P`` passes THROUGH (from ``C`` up
         to JUST BEFORE the hit voxel) gets ``+= L_FREE`` -- a Voxblox/OctoMap miss
         update. Vectorised 3D voxel traversal (amanatides-woo DDA stepped in
         lockstep across all rays) so it is numpy-fast over every ray at once; the
         carve range is capped at :attr:`MAX_DEPTH_M` so a ray never traverses an
         unbounded voxel line. Each cell is updated ONCE per (keyframe, kind) -- a
         cell hit/crossed by many rays in ONE keyframe still moves by a single
         ``L_OCC``/``L_FREE`` (the +1 is per (keyframe, cell), the temporal fusion).
         A cell both crossed AND hit in the same keyframe is treated as a HIT (the
         endpoint wins -- it is the measured surface).

       The accumulated log-odds is clamped to ``[L_MIN, L_MAX]`` so a long dwell
       can't pin a cell so high that later free evidence can never carve it (the
       OctoMap "clamping update policy"). The grid ACCUMULATES across keyframes; we
       never rebuild it from scratch. A set :attr:`_fused_seqs` records which seqs
       are already folded in, so a rebuild only folds the not-yet-fused keyframes
       (cheap, incremental).
    2. A cell is OCCUPIED (the internal fusion notion) when ``log_odds >=
       L_OCC_THRESH``. RENDER, however, is gated SEPARATELY and HIGHER at
       ``log_odds >= L_DISPLAY``: the UPDATE math (carving) is unchanged -- the grid
       keeps every cell's low/near-zero evidence so a later crossing ray can still
       carve it -- but the VIEW shows only HIGH-confidence surfaces. A real wall is
       re-hit by many consistent rays so its log_odds climbs and SATURATES well above
       L_DISPLAY and renders; behind-the-wall stereo spray (which carving CANNOT reach,
       because rays stop at the wall surface and nothing crosses the space behind it)
       is hit only once or twice, stays below L_DISPLAY, and is filtered out of the
       view. A noise voxel in REACHABLE free space is additionally carved by later
       crossing rays back below even L_OCC_THRESH and disappears outright.
    3. Output: ALL DISPLAYABLE voxel CENTRES (the cell index * VOXEL_M + half a cell)
       + a GREEN gradient colour by HEIGHT (derived from the cell's world ``y``,
       the optical DOWN axis). We render the WHOLE occupied set so the map GROWS as
       the camera explores; carving keeps it CLEAN (noise removed) and a bounded
       room's occupied count PLATEAUS. :data:`MAX_VOXELS` is only a HIGH runaway
       safety cap (a fair random subsample if ever tripped -- NEVER a "keep top-N"
       drop, which would erase the newest areas and froze the map at the start).

    Render choice (light, no-lag): the window draws the voxels as a
    ``GLScatterPlotItem`` of large SQUARE world-unit points (size == VOXEL_M,
    ``pxMode=False``), NOT an N-cube ``GLMeshItem`` -- a scatter of N points is far
    cheaper to upload + paint than 12*N triangles, and at this voxel size the
    squares read as the blocky cubes. The rebuild is coalesced + capped at
    :attr:`REBUILD_HZ` and only re-emits when the occupied set CHANGED -- grew or
    shifted (see :meth:`run` + :meth:`_occ_signature`), so the GUI never re-uploads
    an identical cloud yet always reflects newly-explored / newly-carved areas.

    Perf (the user is sensitive to lag): carving is the cost -- each ray touches
    ~range/VOXEL_M voxels. It is mitigated three ways: (a) the DDA is fully
    vectorised over all rays (no per-ray Python loop), (b) :attr:`STRIDE` limits the
    ray count, (c) the carve range is capped at :attr:`MAX_DEPTH_M`. The whole build
    runs OFF the GUI thread (this source is a worker that ``submit()``s the finished
    voxels) and is throttled at :attr:`REBUILD_HZ`, so the UI never stalls; and the
    rendered result is LIGHTER than the old hit-only map because carving removes
    voxels. The functional probe reports the per-keyframe fuse time.

    ``K`` comes from VIO's retained ``calib.bundle`` (the rectified-left intrinsic
    for the full-res depth grid the keyframe ``depth_m`` lives on).

    The VIO keyframe feed (ring attach + ``_on_keyframe`` stash + evict + the
    coalesced rebuild loop) is inherited from :class:`_KeyframeAccumulator`; this
    class adds the ``slam.map`` client + the log-odds fusion + the voxel build.

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
    #: Depth-map subsample stride: back-project every STRIDE-th pixel in u and v.
    #: The occupancy grid only needs the room SHAPE, not every pixel, so a stride of
    #: 4 (1/16 the rays) keeps the per-keyframe fuse + carve cheap while surfaces
    #: still get dense multi-keyframe support. Carving cost scales with the ray
    #: count, so STRIDE is the primary perf knob. LOWER (toward 1) for denser support
    #: (heavier fuse + carve); RAISE for a lighter, sparser fuse.
    STRIDE = 4
    #: Edge-reject threshold (m): drop "flying pixels" on a depth discontinuity (a
    #: fg/bg edge back-projects to points floating BETWEEN the two surfaces). A
    #: pixel is kept only if BOTH its vertical and horizontal depth gradient are
    #: <= this. SAME idea as the shared geometry edge-reject; 0 disables it.
    EDGE_MAX_M = 0.1

    # --- Log-odds occupancy constants (OctoMap/Voxblox-style; see class doc). ---
    #: OCCUPIED evidence added to the hit voxel per keyframe (a "hit" sensor model
    #: update, ~log(0.7/0.3) ~= +0.85). RAISE so surfaces lock in faster / resist
    #: carving more; LOWER so noise carves away more easily.
    L_OCC = 0.85
    #: FREE evidence added to every voxel a ray passes THROUGH per keyframe (a
    #: "miss" update, ~log(0.38/0.62) ~= -0.50). This is what REMOVES wrongly-added
    #: voxels: a noise cell crossed by later rays accumulates this until it drops
    #: below L_OCC_THRESH and disappears. |L_FREE| < L_OCC so a single grazing
    #: free-ray can't erase a well-supported surface, but a couple of crossings can
    #: carve a once-seen noise voxel. Strengthened slightly (-0.40 -> -0.50) so a
    #: crossing ray drives REACHABLE noise down faster; still well under L_OCC so a
    #: real thin surface is not over-carved by one grazing miss. RAISE the magnitude
    #: (toward L_OCC) to carve more aggressively; LOWER to carve more conservatively.
    L_FREE = -0.50
    #: Clamp band on the accumulated log-odds (OctoMap's "clamping update policy").
    #: L_MAX bounds how confident a cell can get so a long dwell can't pin it so high
    #: that later free evidence can NEVER carve it (defeating the whole point);
    #: L_MIN bounds the free side. L_MAX raised (3.5 -> 5.0) so a consistently-observed
    #: surface can climb WELL above the render gate L_DISPLAY (a wall re-hit ~6x reaches
    #: 5.0, vs L_DISPLAY=2.0) while sporadic behind-wall noise (1-2 hits) stays under it
    #: -- widening the confidence gap the display gate separates on. L_MAX/L_OCC ~= 6
    #: hits to saturate; even saturated, sustained free evidence still carves it back
    #: (10 crossings at L_FREE=-0.50 span the full [L_MIN, L_MAX] band).
    L_MIN = -2.5
    L_MAX = 5.0
    #: Occupancy threshold: a cell is OCCUPIED (the INTERNAL occupied set used by the
    #: fusion bookkeeping) when its log_odds is >= this (p_occ ~= 0.62). One un-carved
    #: hit (L_OCC=0.85) already crosses it, so a surface enters the set immediately;
    #: one later free crossing (L_FREE=-0.50) pulls a once-hit noise voxel to +0.35 <
    #: thresh and it leaves the set. This gate is kept LOW on purpose: the UPDATE math
    #: (carving) needs the grid to retain low/near-zero evidence so a later crossing ray
    #: can still drive a noise cell back down -- raising it would not change what carving
    #: can reach. RAISE only to change the internal occupied-set membership; the RENDER
    #: gate is the separate, higher :data:`L_DISPLAY` below.
    L_OCC_THRESH = 0.5
    #: RENDER confidence gate (SEPARATE from L_OCC_THRESH; the principled fix for the
    #: behind-the-wall noise). The UPDATE math is unchanged -- carving still drives every
    #: voxel's log_odds correctly -- but the VIEW shows only voxels with
    #: ``log_odds >= L_DISPLAY``, set HIGHER than L_OCC_THRESH. Rationale: a real wall is
    #: a consistently-observed surface re-hit by many rays from many viewpoints, so its
    #: log_odds climbs and SATURATES near L_MAX -> well above L_DISPLAY -> it renders.
    #: The spray BEHIND the wall is sporadic stereo noise that carving cannot reach (rays
    #: stop at the wall surface, nothing crosses the space behind it) but which is only
    #: ever hit ONCE or a few times -> its log_odds stays low (~L_OCC..a couple of L_OCC)
    #: -> it falls below L_DISPLAY and is filtered out of the view. So we DISPLAY
    #: high-confidence surfaces only, without disturbing the carving that cleans the
    #: reachable space. RAISE toward L_MAX for a stricter view (only the most-observed
    #: surfaces; risks thinning a real but lightly-seen thin surface); LOWER toward
    #: L_OCC_THRESH to show more (noisier). Chosen from the PNG sweep (see
    #: ui/tests/_map_display_sweep.py): +2.0 keeps the wall crisp while the behind-wall
    #: tail drops out.
    L_DISPLAY = 2.0

    # --- Spatial outlier removal (point-cloud SOR on the DISPLAYED voxels). ---
    #: After the L_DISPLAY gate, the wall renders -- but a sparse spray of ISOLATED
    #: noise voxels can still pass the gate OUTSIDE the wall (stereo specks that were
    #: each hit a couple of times yet carving could not reach). The principled,
    #: standard fix is the SAME radius-outlier filter PCL/Open3D apply to a point
    #: cloud: a REAL wall is a DENSE surface -- every occupied voxel has MANY occupied
    #: neighbours (a flat surface gives ~10-26 in a 3x3x3 box); isolated noise has FEW
    #: (0-a-handful). So we KEEP a displayed voxel only if it has at least
    #: :data:`MIN_NEIGHBORS` OTHER displayed voxels within the (2r+1)^3 neighbourhood
    #: -- the dense walls survive (they are dense), the lone specks are dropped. This
    #: is a pure RENDER-time filter on the L_DISPLAY-gated set; it does NOT touch the
    #: log-odds grid / carving / fusion (see :meth:`_spatial_outlier_filter`).
    #:
    #: Neighbourhood radius in voxels: the box is (2*NEIGHBOR_RADIUS+1)^3. r=1 -> the
    #: 26-neighbourhood (the immediately-adjacent cells), which is what a "dense
    #: surface" means at this VOXEL_M. RAISE to count over a wider box (tolerates
    #: gappier surfaces but also lets bigger specks survive); LOWER is not meaningful
    #: below 1.
    NEIGHBOR_RADIUS = 1
    #: Minimum OTHER displayed neighbours a voxel must have inside the box to be KEPT.
    #: 0 disables the filter (render every L_DISPLAY voxel). HIGHER = more aggressive
    #: isolated-noise removal: a dense wall voxel has ~10-26 neighbours so it easily
    #: clears any value up to ~20; an isolated speck has 0-few and is dropped. Chosen
    #: from the PNG sweep (see ui/tests/_map_sor_sweep.py): the value that cleanly
    #: removes the outside-wall spray while keeping the walls solid + connected. RAISE
    #: to prune harder (risks thinning a genuinely thin/lightly-seen real surface);
    #: LOWER to keep more (noisier).
    MIN_NEIGHBORS = 6

    #: Hard SAFETY cap on rendered voxels (runaway guard ONLY). We render ALL
    #: occupied cells (log_odds >= L_OCC_THRESH) -- carving keeps the set CLEAN and a
    #: bounded room's occupied count PLATEAUS, so a green height-coloured
    #: GLScatterPlotItem of a few tens of thousands of points stays light. This cap
    #: exists solely so a pathological unbounded sweep can't explode the render; it is
    #: set HIGH (well above a room's plateau) so it never trips in normal use. When it
    #: DOES trip we subsample FAIRLY (a uniform random draw -- see :meth:`_build`),
    #: NOT by lowest log-odds: dropping the lowest would erase exactly the newest,
    #: least-re-observed areas (the bug that froze the map at the start). LOWER only if
    #: a room genuinely overruns the render budget; RAISE for an even larger guard band.
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
        # PERSISTENT log-odds occupancy grid: {(ix,iy,iz) int -> log_odds float}.
        # log_odds = log(p_occ / (1 - p_occ)); >0 leans occupied, <0 leans free.
        # Each keyframe's rays push hit voxels up by L_OCC and crossed voxels down by
        # L_FREE (clamped to [L_MIN, L_MAX]), so noise the camera later sees through
        # gets carved back below L_OCC_THRESH. Accumulated across keyframes (NEVER
        # rebuilt from scratch); guarded by the base's _lock with the depth/pose dicts.
        self._log: dict[tuple[int, int, int], float] = {}
        # The keyframe seqs already folded into _log, so a rebuild only fuses the
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

        The keyframe's depth/pose are dropped by the base; its log-odds evidence
        STAYS in the persistent grid (already fused) -- we just never re-fuse an
        evicted seq.
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
        """Fold ONE keyframe's depth into the persistent log-odds grid (under lock).

        Back-projects the (denoised) depth to the world by the keyframe's OWN VIO
        pose ``P = R Xc + t`` (strided + depth-gated + edge-rejected). The camera
        origin is the keyframe translation ``C = t``. Then, for THIS keyframe:

        * HIT update -- the voxel containing each world hit ``P`` gets ``+= L_OCC``.
        * FREE carving -- every voxel each ray ``C -> P`` passes THROUGH (from ``C``
          up to JUST BEFORE the hit voxel) gets ``+= L_FREE`` (a Voxblox/OctoMap miss
          update). This is what REMOVES wrongly-added voxels: a noise cell a later
          ray sees through accumulates free evidence and drops below L_OCC_THRESH.

        Both kinds collapse to ONE update per (keyframe, cell) via ``np.unique`` -- a
        cell hit/crossed by many rays in ONE keyframe still moves by a single
        L_OCC/L_FREE (the temporal-fusion principle). A cell both crossed AND hit in
        the same keyframe is treated as a HIT: the hit cells are SUBTRACTED from the
        free cells so the endpoint (the measured surface) wins. All accumulated
        log-odds are clamped to ``[L_MIN, L_MAX]``. Fully vectorised: one
        back-projection + one vectorised DDA over all rays, then a scatter add into
        the persistent dict over only the UNIQUE cells touched.
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
        # keyframe's OWN pose -> the hit points P (M,3) in voxel UNITS (so the DDA
        # and the quantise share one coordinate system). The camera origin C is the
        # keyframe translation, likewise in voxel units.
        vm = float(self.VOXEL_M)
        cam = np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, z], axis=1)
        world = cam @ np.asarray(R, np.float64).reshape(3, 3).T \
            + np.asarray(t, np.float64).reshape(3)             # (M,3)
        P = world / vm                                         # hit points (vox)
        C = np.asarray(t, np.float64).reshape(3) / vm          # origin (vox)

        # HIT cells: the voxel each ray ENDS in (floor of the world hit).
        hit_keys = np.floor(P).astype(np.int64)                # (M,3)
        # FREE cells: every voxel each ray crosses from C up to (not incl.) the hit
        # voxel, via the vectorised DDA. May be empty (origin == hit voxel).
        free_keys = self._carve_free_cells(C, P)               # (F,3)

        # Collapse to one update per (keyframe, cell) on a PACKED 1-D int64 key (the
        # 3 voxel coords bit-packed) -- ``np.unique`` over a 1-D int array is an order
        # of magnitude faster than the lexsort ``np.unique(..., axis=0)`` does over
        # 700k 3-col rows (which dominated the fuse time). A cell both hit AND crossed
        # in this keyframe is a HIT (the endpoint, the measured surface, wins), so
        # drop any free cell that is also a hit cell (``np.isin`` on the 1-D keys).
        hit_u = np.unique(self._pack_keys(hit_keys)) if hit_keys.size \
            else np.empty(0, np.int64)
        if free_keys.size:
            free_u = np.unique(self._pack_keys(free_keys))
            free_u = free_u[~np.isin(free_u, hit_u)]
        else:
            free_u = np.empty(0, np.int64)

        lmin, lmax = float(self.L_MIN), float(self.L_MAX)
        lfree, locc = float(self.L_FREE), float(self.L_OCC)
        # Scatter the evidence into the persistent grid (clamped). One Python pass
        # over the UNIQUE cells this keyframe touched (a few thousand), not the raw
        # rays * voxels-per-ray. The packed keys decode back to (ix,iy,iz) tuples.
        for k in self._unpack_to_tuples(free_u):
            v = self._log.get(k, 0.0) + lfree
            self._log[k] = lmin if v < lmin else (lmax if v > lmax else v)
        for k in self._unpack_to_tuples(hit_u):
            v = self._log.get(k, 0.0) + locc
            self._log[k] = lmin if v < lmin else (lmax if v > lmax else v)

    def _carve_free_cells(self, C: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Vectorised amanatides-woo DDA: voxels every ray ``C -> P`` crosses.

        Returns an ``(F,3)`` int array of the integer voxel coords every ray passes
        THROUGH, from the origin voxel up to (but NOT including) the hit voxel
        ``floor(P)`` -- i.e. the FREE space along each ray. ``C`` is the shared
        camera origin and ``P`` the (M,3) hit points, BOTH in voxel units (world /
        VOXEL_M), so a unit step is one voxel.

        Algorithm (the standard amanatides-woo grid traversal, but stepped across all
        rays at once so it stays numpy-vectorised -- no per-ray Python loop):

        * Each ray starts in voxel ``floor(C)`` and walks ONE voxel per iteration
          along whichever axis has the nearest grid-plane crossing (``tMax`` per
          axis), which guarantees a CONTIGUOUS, gap-free voxel line.
        * The ACTIVE set is COMPACTED every iteration: a ray drops out the moment it
          reaches its hit voxel or its t passes ``max_t``, and the per-ray state
          arrays are sliced down to the survivors -- so per-step work shrinks with
          the active count instead of staying O(M). This matters a lot in a corridor
          where near-wall rays finish in a few steps while far rays run the full
          range; without compaction every step re-touches the long-dead near rays.
        * The carve range is capped at :attr:`MAX_DEPTH_M` (so a ray never traverses
          an unbounded line), bounding the iteration count to ~MAX_DEPTH_M/VOXEL_M.

        Vectorisation note: this loops at most ``ceil(range/VOXEL_M)`` times (a small
        constant, e.g. 60 for 6 m / 0.10 m), and EACH iteration is a handful of numpy
        ops over the (shrinking) active ray batch -- the cost is in C, not Python. The
        collected per-step cells are concatenated once at the end.
        """
        M = P.shape[0]
        if M == 0:
            return np.empty((0, 3), np.int64)
        Cv = np.asarray(C, np.float64).reshape(1, 3)
        dirv = P - Cv                                          # ray vectors (M,3)
        seg_len = np.linalg.norm(dirv, axis=1)                 # |C->P| in voxels
        # Cap the per-ray traversal length: never carve past the hit voxel (t up to
        # seg_len) NOR beyond MAX_DEPTH_M (in voxels), whichever is closer.
        max_t = np.minimum(seg_len, float(self.MAX_DEPTH_M) / float(self.VOXEL_M))
        # Keep only non-degenerate rays (hit voxel != origin voxel); guards /0 and
        # drops zero-length rays up front so the active arrays start tight.
        keep = seg_len > 1e-9
        if not np.any(keep):
            return np.empty((0, 3), np.int64)

        # Per-ray state, sliced to the active set; updated in-place each step.
        cur = np.floor(Cv).astype(np.int64)                    # origin voxel (1,3)
        cur = np.broadcast_to(cur, (M, 3))[keep].copy()        # (A,3) current voxel
        target = np.floor(P[keep]).astype(np.int64)            # (A,3) hit voxel
        unit = dirv[keep] / seg_len[keep, None]                # (A,3) unit direction
        max_t = max_t[keep]                                    # (A,)
        # amanatides-woo per-axis setup. step = sign of the direction; tDelta = t to
        # cross one voxel along each axis; tMax = t at the FIRST grid-plane crossing.
        step = np.sign(unit).astype(np.int64)
        absu = np.abs(unit)
        safe = absu > 1e-12                                    # avoid /0 on flat axes
        t_delta = np.full(unit.shape, np.inf)
        np.divide(1.0, absu, out=t_delta, where=safe)
        # Distance from C to the next grid plane along each axis (in voxels). For
        # +step it is (1 - frac); for -step it is frac; on a plane it's a full voxel.
        frac = (Cv - np.floor(Cv))                             # (1,3) shared origin
        first = np.where(step > 0, 1.0 - frac, frac)
        first = np.where(first <= 1e-12, 1.0, first)
        t_max = np.full(unit.shape, np.inf)
        np.divide(first, absu, out=t_max, where=safe)

        max_iters = int(np.ceil(float(max_t.max()))) + 3

        collected: list[np.ndarray] = []
        # The ORIGIN voxel is free space too (the camera sits in clear air), unless it
        # already is the hit voxel -- emit it for rays whose origin != hit.
        origin_free = np.any(cur != target, axis=1)
        if np.any(origin_free):
            collected.append(cur[origin_free].copy())

        rows = np.arange(cur.shape[0])
        for _ in range(max_iters):
            if cur.shape[0] == 0:
                break
            # Step each ray along its axis of MINIMUM tMax (the next plane it crosses)
            # -- the amanatides-woo advance. argmin over the 3 tMax picks the axis.
            axis = np.argmin(t_max, axis=1)                    # (A,) chosen axis
            t_cross = t_max[rows, axis]                        # t at this crossing
            cur[rows, axis] += step[rows, axis]                # advance the voxel
            t_max[rows, axis] += t_delta[rows, axis]           # advance its tMax
            # A ray stays active only if, AFTER stepping, it has NOT yet reached the
            # hit voxel AND has not run past max_t. The cell just stepped INTO is free
            # space (strictly before the hit voxel for still-active rays).
            still = ~np.all(cur == target, axis=1) & (t_cross < max_t)
            if not np.all(still):                              # COMPACT to survivors
                cur, target, step, t_delta, t_max, max_t = (
                    cur[still], target[still], step[still], t_delta[still],
                    t_max[still], max_t[still])
                rows = np.arange(cur.shape[0])
            if cur.shape[0]:
                collected.append(cur.copy())

        if not collected:
            return np.empty((0, 3), np.int64)
        return np.concatenate(collected, axis=0)

    #: Bits per voxel axis when packing an (ix,iy,iz) cell into one int64 key.
    #: 21 bits -> a signed range of +-2^20 (~1.05M) voxels per axis = +-105 km at
    #: VOXEL_M=0.10 m, vastly beyond any room; 3*21 = 63 bits fits a signed int64.
    #: A bias of 2^20 shifts the signed coord into [0, 2^21) before packing.
    _PACK_BITS = 21
    _PACK_BIAS = 1 << 20

    @classmethod
    def _pack_keys(cls, cells: np.ndarray) -> np.ndarray:
        """Pack an int ``(N,3)`` voxel-coord array into a 1-D int64 key array.

        Bit-packs (ix,iy,iz) into ONE int64 so ``np.unique`` / ``np.isin`` run on a
        flat int array (an order of magnitude faster than ``np.unique(..., axis=0)``,
        which lexsorts 3-col rows and dominated the carve cost). Each coord is biased
        into a non-negative range first so negative voxel indices pack cleanly.
        """
        b, bias = cls._PACK_BITS, cls._PACK_BIAS
        c = cells.astype(np.int64)
        return (((c[:, 0] + bias) << (2 * b))
                | ((c[:, 1] + bias) << b)
                | (c[:, 2] + bias))

    @classmethod
    def _unpack_to_tuples(cls, keys: np.ndarray):
        """Decode packed int64 keys back to ``(ix,iy,iz)`` int tuples (a generator).

        Yields the dict keys for the scatter step (the inverse of :meth:`_pack_keys`).
        Done in numpy then ``.tolist()`` so the per-cell Python work is just tuple
        construction, not arithmetic.
        """
        b, bias = cls._PACK_BITS, cls._PACK_BIAS
        mask = (1 << b) - 1
        k = np.asarray(keys, np.int64)
        ix = (k >> (2 * b)) - bias
        iy = ((k >> b) & mask) - bias
        iz = (k & mask) - bias
        return zip(ix.tolist(), iy.tolist(), iz.tolist())

    # ------------------------------------------------------------------ #
    def _spatial_outlier_filter(self, cells: np.ndarray) -> np.ndarray:
        """Radius-outlier (SOR) filter on the displayed voxel coords (vectorised).

        Standard point-cloud spatial outlier removal, applied to the integer voxel
        grid so it needs no kd-tree: KEEP only cells that have at least
        :data:`MIN_NEIGHBORS` OTHER displayed cells inside the
        ``(2*NEIGHBOR_RADIUS+1)^3`` box around them. A real wall is a DENSE surface
        (each cell has ~10-26 occupied neighbours) so it survives; an isolated stereo
        speck has 0-few and is dropped -- without eroding the walls (they are dense).

        ``cells`` is the ``(N,3)`` int array of the L_DISPLAY-gated voxel coords.
        Returns the kept subset (an ``(K,3)`` int array, K <= N).

        Vectorisation (NO scipy/skimage -- not installed): pack the cells into the
        same 1-D int64 keys :meth:`_pack_keys` uses and SORT that key table ONCE; then
        for EACH of the up to ``(2r+1)^3 - 1`` neighbour OFFSETS, pack ``cells +
        offset`` and test membership against the sorted table with one
        ``np.searchsorted`` (a C-level binary search), summing the hits per cell. The
        offset is added to the UNPACKED integer coords and re-packed (a packed key is
        bit-FIELDS, so adding a constant to it would carry across axes -- wrong).
        Sorting once and reusing it (vs ``np.isin``, which re-sorts the table on every
        call) cuts the work to a single O(N log N) sort + a small constant (26 at r=1)
        number of O(N log N) searches -- ~70 ms over a 50k-voxel set, comfortably light
        within the off-thread 4 Hz (250 ms) rebuild budget.
        """
        n = cells.shape[0]
        r = int(self.NEIGHBOR_RADIUS)
        min_n = int(self.MIN_NEIGHBORS)
        # min_n <= 0 disables the filter; r <= 0 or an empty set -> nothing to do.
        if min_n <= 0 or r <= 0 or n == 0:
            return cells
        c = cells.astype(np.int64)
        # The membership table: all displayed cells as packed int64 keys, SORTED once
        # so every offset's lookup is a binary search against the SAME table (instead
        # of np.isin re-sorting it 26 times).
        ks = np.sort(self._pack_keys(c))
        # Tally OTHER displayed neighbours per cell over every box offset except the
        # centre (0,0,0) -- the cell itself is not its own neighbour.
        counts = np.zeros(n, dtype=np.int32)
        rng = range(-r, r + 1)
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    # Re-pack the SHIFTED coords (add the offset to the unpacked ints,
                    # then pack) and binary-search the sorted table. searchsorted gives
                    # the INSERTION index; clip it in-range and confirm the table entry
                    # there EQUALS the shifted key -> a present neighbour. (Keys are
                    # >= 0 by the pack bias, so a clip to [0, size-1] is safe.)
                    shifted = self._pack_keys(c + np.array((dx, dy, dz), np.int64))
                    idx = np.clip(np.searchsorted(ks, shifted), 0, ks.size - 1)
                    counts += (ks[idx] == shifted)
        return c[counts >= min_n]

    # ------------------------------------------------------------------ #
    def _build(self):
        """Fold any NEW keyframes into the grid, then emit the occupied voxels.

        Returns ``(points, colors, cams)``:

        * ``points`` ``(N,3)`` float32 -- the CENTRE of every DISPLAYABLE voxel that
          ALSO survives the spatial outlier filter. A voxel is displayable when
          ``log_odds >= L_DISPLAY`` (the higher RENDER gate -- so only HIGH-confidence
          surfaces show and the low-confidence behind-wall spray is filtered out); on
          top of that a SPATIAL OUTLIER REMOVAL (:meth:`_spatial_outlier_filter`, the
          standard point-cloud radius-outlier filter) drops ISOLATED voxels -- the
          sparse spray OUTSIDE the wall -- by neighbour count, so only DENSE surfaces
          (the walls) remain. We render the WHOLE surviving set so the map GROWS as the
          camera explores (a bounded room's count plateaus) and SELF-CLEANS (a carved
          noise voxel drops in log-odds and falls out of the view); only if it exceeds
          the high :data:`MAX_VOXELS` safety cap do we drop voxels by a FAIR
          uniform-random subsample (NOT by lowest log-odds, which would erase the
          newest areas). Optical-world frame (the window rotates it to ENU, same as the
          trajectory).
        * ``colors`` ``(N,3)`` float32 -- a GREEN gradient by HEIGHT (optical
          ``+y`` is world-DOWN), so the floor/walls read like ModalAI's height-tinted
          occupancy.
        * ``cams`` ``(M,3)`` float32 -- ALL VIO keyframe camera positions (the path).

        The fuse is INCREMENTAL: only keyframes not in :attr:`_fused_seqs` are
        folded (the persistent grid already holds the rest), so this stays cheap as
        keyframes accumulate.
        """
        with self._lock:
            # (1) Fold the NEW keyframes into the persistent grid (incremental):
            #     each adds occupied evidence at its hits and carves free space along
            #     its rays, so a later keyframe can REMOVE an earlier noise voxel.
            new_seqs = [s for s in self._kf_depth if s not in self._fused_seqs]
            for seq in sorted(new_seqs):
                self._fuse_keyframe_locked(self._kf_depth[seq],
                                           self._kf_R[seq], self._kf_t[seq])
                self._fused_seqs.add(seq)
            # (2) Snapshot the DISPLAYABLE cell KEYS under the lock, then build the
            #     arrays lock-free. The RENDER gate is L_DISPLAY (the separate, higher
            #     confidence threshold), NOT L_OCC_THRESH: the grid keeps ALL its
            #     log-odds evidence (the UPDATE math / carving is untouched -- it still
            #     needs low/near-zero cells so a later crossing ray can carve them), but
            #     the VIEW shows only HIGH-confidence surfaces. A real wall is re-hit by
            #     many rays so its log_odds saturates well above L_DISPLAY and renders;
            #     the sporadic behind-the-wall spray (carving can't reach it -- rays stop
            #     at the wall) is hit only once or twice, stays below L_DISPLAY, and is
            #     filtered out of the view. We render ALL displayable cells, so only the
            #     keys are needed.
            thresh = float(self.L_DISPLAY)
            occ = [k for k, lo in self._log.items() if lo >= thresh]
            cam_ts = [self._kf_t[s] for s in self._kf_depth]
        cams = (np.asarray(cam_ts, np.float32).reshape(-1, 3)
                if cam_ts else np.zeros((0, 3), np.float32))
        if not occ:
            empty = np.zeros((0, 3), np.float32)
            return empty, empty, cams

        keys = np.asarray(occ, dtype=np.int64)                     # (N,3)
        # (2b) SPATIAL OUTLIER REMOVAL on the displayed set: drop ISOLATED voxels
        #      (the sparse spray OUTSIDE the wall) by neighbour count -- a dense wall
        #      voxel has many occupied neighbours and survives; a lone speck has few
        #      and is removed. Pure render-time filter (the log-odds grid is
        #      untouched). MIN_NEIGHBORS=0 makes this a no-op.
        keys = self._spatial_outlier_filter(keys)
        if keys.shape[0] == 0:                                     # all pruned
            empty = np.zeros((0, 3), np.float32)
            return empty, empty, cams
        # Render the WHOLE occupied set (the map must GROW as new areas are
        # explored). The high MAX_VOXELS safety cap is a runaway guard only; if it
        # ever trips we subsample UNIFORMLY AT RANDOM -- a spatially-fair draw that
        # thins the cloud everywhere instead of erasing the newest (lowest log-odds)
        # areas the way a "keep top-N" rule did (that rule permanently favoured the
        # start area and froze the map). Seeded for a stable cloud across rebuilds
        # (so the safety thinning doesn't shimmer frame to frame).
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


def ipc_loop_factory(slam_endpoint: str, vio_endpoint: str,
                     width: int, height: int):
    """Return a zero-arg factory building an :class:`IpcLoopMatchSource`.

    Binds the SLAM endpoint (the ``slam.loop`` match-funnel publisher) and the
    VIO endpoint (the ``keyframe`` gray publisher + its kf rings), so the caller
    (``ui.main``) just opens the Loop-Closure window with the returned source.
    """
    return lambda: IpcLoopMatchSource(slam_endpoint, vio_endpoint,
                                      width=width, height=height)


def ipc_ba_window_factory(vio_endpoint: str, *, buffer: int = 240,
                          connect_timeout_s: float = 30.0):
    """Return a zero-arg factory building an :class:`IpcBaWindowSource`.

    Binds VIO's endpoint (the ``ba.window`` solve-snapshot publisher, present only
    when VIO ran with ``--ba-window``), so the caller (``ui.main``) just opens the
    BA Window with the returned source. Pure POD topic -- no rings, no width/height
    needed (unlike the loop / keypoint sources that read kf grays).
    """
    return lambda: IpcBaWindowSource(vio_endpoint, buffer=buffer,
                                     connect_timeout_s=connect_timeout_s)


def ipc_frontend_viz_factory(vio_endpoint: str, *, buffer: int = 240,
                             connect_timeout_s: float = 30.0):
    """Return a zero-arg factory building an :class:`IpcFrontendVizSource`.

    Binds VIO's endpoint (the ``frame.frontend`` snapshot publisher, present only
    when VIO ran with ``--frontend-viz``), so the caller (``ui.main``) just opens
    the Frontend Internals window with the returned source. Pure POD topic -- no
    rings, no width/height needed (the heatmap rides inline, quantised).
    """
    return lambda: IpcFrontendVizSource(vio_endpoint, buffer=buffer,
                                        connect_timeout_s=connect_timeout_s)


def ipc_pose_graph_factory(vio_endpoint: str, slam_endpoint: str, *,
                           connect_timeout_s: float = 30.0):
    """Return a zero-arg factory building an :class:`IpcPoseGraphSource`.

    Binds VIO's endpoint (the raw ``pose.odom`` drifted trail = BEFORE) and SLAM's
    endpoint (``slam.map`` corrected nodes = AFTER + ``slam.loop`` edges), so the
    caller (``ui.main``) just opens the Pose Graph window with the returned source.
    All pure POD topics -- no rings, no width/height needed. A pure CONSUMER of
    existing topics (no new IPC field), so the byte-parity oracle is unaffected.
    """
    return lambda: IpcPoseGraphSource(vio_endpoint, slam_endpoint,
                                      connect_timeout_s=connect_timeout_s)
