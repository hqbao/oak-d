"""imu_cam acquisition front-end, as PROCEDURAL Python.

This replaces the old reactive ``ImuCamModule(Module)`` + ``Step`` chain. The
per-trigger work is now a plain function (:func:`process_cam_sync`) that calls the
step functions in the EXACT same order the reactive chain ran -- so every output
stream (``imu.raw`` / ``imucam.sample`` / ``frame.depth``) stays byte-identical
and the gap=0 oracle (which replays through THIS front-end) holds. The framework
indirection (``Module.on`` / ``_routes`` / ``_run_chain`` short-circuit-on-None /
``ctx.state["matcher"]``) is gone; the data flow reads as straight-line code.

What did NOT vanish: the CAM_SYNC inbox + the IMU buffer + the optional
``latest_only`` coalescing + the END-forward to multiple downstream topics. These
are LOAD-BEARING and are replicated EXPLICITLY in :class:`ImuCamWorker` -- a plain
thread that owns the inbox, the coalescing, and the END forwarding, instead of
inheriting the comms reactive substrate.

The IMU buffer is filled by the injected ``source`` on its OWN I/O thread (a
hardware producer, not a worker) -- the worker only drains it per camera trigger.

``ImuCamModule`` is kept as a public alias for the procedural :class:`ImuCamWorker`
(the in-process selftest imports ``ImuCamModule``).

The :func:`build_replay_frontend` / :func:`build_live_frontend` helpers (was
``ours.app.build_*``) wire ONLY the acquisition front-end (``read_cam`` +
``imu_cam`` with depth) -- everything downstream of ``imucam.sample`` /
``frame.depth`` lives in the other projects (vio / slam / ui).
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

from sky.sensors.imu_calib import ImuCalibration

from imu_camera.comms import LocalPubSub, topics
from imu_camera.comms.messages import END
from imu_camera.io.reader import SessionReader
from sky.imu.timed_buffer import TimedImuBuffer
from sky.depth.stereo import (
    SGMConfig, SGMStereoMatcher)

from .apply_calibration import CalibrationResolver, apply_calibration
from .compute_depth import compute_depth
from .pack_synced import pack_synced
from .publish_depth import publish_depth
from .publish_imu_raw import publish_imu_raw
from .publish_imucam import publish_imucam
from .read_cam import ReadCamModule, ReplayCamSource
from .read_imu import ImuSource, ReplayImuSource
from .tof_downsample import tof_downsample

#: VL53L9CX-class ToF sensor simulation output grid. The OAK-D is the stand-in:
#: depth is computed on it at the SOURCE resolution (``--width``/``--height``,
#: where stereo actually works) and then DOWNSAMPLED to this fixed ToF grid --
#: a clean, dense, accurate low-res depth (median-of-valid fills holes + averages
#: noise), exactly what a real ToF returns. NOT a direct low-res stereo solve.
TOF_W = 54
TOF_H = 42

#: Inbox payload marker for the coalescing path: "the real message is the current
#: self._latest". Mirrors the old ``Module._LATEST`` token.
_LATEST = object()
#: Inbox sentinel to unblock ``queue.get`` on ``stop()``. Mirrors ``Module._SENTINEL``.
_SENTINEL = object()


# --------------------------------------------------------------------------- #
def process_cam_sync(worker: "ImuCamWorker", msg) -> None:
    """Run the full per-trigger chain for one ``cam.sync`` message.

    Ordering is byte-identical to the old reactive chain
    (``ImuCamModule.__init__`` built ``[pack, publish_imu_raw,
    apply_calibration]`` then appended a depth-mode-specific tail):

    1. ``pack_synced`` -- drain the IMU buffer to the frame time, build the packet.
    2. ``publish_imu_raw`` -- publish the RAW IMU on ``imu.raw`` (returns the
       packet unchanged).
    3. ``apply_calibration`` -- correct the packet's IMU (or pass through).
    4. depth-mode tail:

       * ``matcher is None`` (UI triplet view): ``publish_imucam`` only.
       * ToF sim: ``tof_downsample`` -- SGM at source res, downsample gray + depth
         to 54x42, publish BOTH ``imucam.sample`` + ``frame.depth`` (terminal).
       * Normal: ``publish_imucam`` then ``compute_depth`` then ``publish_depth``.

    The old chain short-circuited on a step returning ``None``; here the equivalent
    control flow is explicit (the ToF / no-depth branches simply stop).
    """
    bus = worker.bus
    packet = pack_synced(worker.buffer, worker.wait_timeout, msg)
    packet = publish_imu_raw(bus, packet)
    packet = apply_calibration(worker.calib_resolver.get(), packet)

    matcher = worker.matcher
    if matcher is None:
        # No matcher (UI triplet view): just publish the synced packet.
        publish_imucam(bus, packet)
        return
    if worker.tof_sim:
        # VL53L9CX simulation: this single call REPLACES the normal
        # publish_imucam / compute_depth / publish_depth trio -- it computes depth
        # at source res, downsamples gray + depth to 54x42, and publishes both
        # imucam.sample (54x42 gray + calibrated IMU) AND frame.depth (54x42).
        tof_downsample(matcher, bus, TOF_W, TOF_H, packet)
        return
    # Normal path (byte-identical to before): publish the synced packet at source
    # res, then compute + publish dense SGM depth.
    publish_imucam(bus, packet)
    publish_depth(bus, compute_depth(matcher, packet))


# --------------------------------------------------------------------------- #
class ImuCamWorker(threading.Thread):
    """One thread that buffers IMU, packs per camera trigger, computes depth.

    A plain procedural replacement for the old reactive ``ImuCamModule(Module)``.
    It owns the CAM_SYNC inbox, the IMU buffer, the optional ``latest_only``
    coalescing, and the END-forward to its downstream topics, all as explicit code
    rather than framework hooks.

    ``source`` supplies the raw IMU (``ReplayImuSource`` offline, ``LiveImuSource``
    on the bench) on its OWN I/O thread -- the worker only drains the buffer.
    ``wait_timeout`` bounds how long packing a frame waits for the IMU stream to
    cover its timestamp before draining what is available (so the run never hangs
    on the final frame).

    For every camera trigger the worker publishes the uncalibrated samples on
    ``topics.IMU_RAW`` (honest: exactly what the sensor reported) and, on
    ``topics.IMUCAM_SAMPLE``, the frames bundled with the CALIBRATED IMU.
    ``calibration`` (or the lazy ``calibration_provider``, used on the live path
    where the device id is known only after the device opens) supplies the
    per-device correction; with none, the calibrated packet equals the raw one.

    ``matcher`` makes depth a step IN this worker: when supplied, the chain also
    runs SGM on the same stereo pair and publishes ``topics.FRAME_DEPTH``. The
    camera/IMU visualiser passes ``matcher=None`` (it only wants the synced
    packet, no depth).

    END handling: ``read_cam`` emits END on ``cam.sync`` when its source exhausts;
    this worker forwards END ONCE to ALL downstream topics it publishes
    (``[IMU_RAW, IMUCAM_SAMPLE]`` plus ``FRAME_DEPTH`` when a matcher is wired),
    then sets :attr:`done`. This is a single-input worker (one inbox topic), so the
    first END is terminal -- the multi-END here is FORWARD-fan-out, not a join.

    WARNING: ``latest_only=True`` makes the CAM_SYNC inbox coalesce, which drops
    camera triggers when packing/SGM falls behind -- the downstream
    ``imucam.sample`` + ``frame.depth`` (both in ``topics.VIO_PATH_TOPICS``) then
    skip those frames, breaking VIO's gyro continuity (PreintegratePrior) and KLT
    continuity (TrackFeatures). ONLY pass ``latest_only=True`` when this worker
    feeds a UI-only graph with no odometry downstream (e.g. the triplet view).
    For VIO / replay / the capture process, keep the default FIFO inbox and put
    backpressure at the IPC boundary (``IPCPubSub(blocking=False)``).
    """

    def __init__(self, bus: LocalPubSub, source: ImuSource, *,
                 matcher: SGMStereoMatcher | None = None,
                 buffer_capacity: int = 8192, wait_timeout: float = 0.5,
                 calibration: ImuCalibration | None = None,
                 calibration_provider:
                     Callable[[], ImuCalibration | None] | None = None,
                 latest_only: bool = False,
                 tof_sim: bool = False,
                 ) -> None:
        super().__init__(name="imu-cam", daemon=True)
        self.bus = bus
        self.source = source
        self.buffer = TimedImuBuffer(capacity=buffer_capacity)
        self.matcher = matcher
        self.wait_timeout = float(wait_timeout)
        self.tof_sim = bool(tof_sim)
        self.calib_resolver = CalibrationResolver(
            calibration, provider=calibration_provider)

        # Downstream topics END is forwarded to (was Module.forwards_to). IMU_RAW +
        # IMUCAM_SAMPLE always; FRAME_DEPTH when depth runs (matcher present).
        self._downstream = [topics.IMU_RAW, topics.IMUCAM_SAMPLE]
        if matcher is not None:
            self._downstream.append(topics.FRAME_DEPTH)

        self._latest_only = bool(latest_only)
        self._inbox: "queue.Queue" = queue.Queue()
        self._latest: Any = _SENTINEL          # single-slot newest unprocessed msg
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self.done = threading.Event()          #: set after END is handled
        self._emitted_end = False

        # Subscribe the inbox feeder to cam.sync in __init__ (matches the old
        # Module.on timing) so a trigger published between construction and start()
        # is never lost.
        self.bus.subscribe(topics.CAM_SYNC, self._on_cam_sync)

    # -- inbox feeder (runs on the PUBLISHER's thread, kept cheap) ----------- #
    def _on_cam_sync(self, msg: Any) -> None:
        """Bus handler for ``cam.sync``: enqueue (coalescing or strict FIFO)."""
        if not self._latest_only:
            # Strict FIFO: every trigger (and END) processed in order. Required by
            # the VIO + deterministic replay paths -- dropping one breaks gyro/KLT
            # continuity (the gap=0 oracle replays through here).
            self._inbox.put(msg)
            return
        # Coalescing (LIVE UI-only): keep only the newest unprocessed trigger in
        # the single slot; enqueue a wake-up token only when nothing was pending
        # -- EXCEPT END, which always enqueues a token so it is delivered even if
        # it overwrites a pending data frame (losing the last frame is fine;
        # dropping END is not). Byte-for-byte the old Module._coalesce, specialised
        # to this worker's single cam.sync topic.
        with self._latest_lock:
            pending = self._latest is not _SENTINEL
            self._latest = msg
            enqueue = (not pending) or (msg is END)
        if enqueue:
            self._inbox.put(_LATEST)

    # -- thread body -------------------------------------------------------- #
    def stop(self) -> None:
        self._stop.set()
        self._inbox.put(_SENTINEL)             # unblock the queue.get

    def run(self) -> None:
        # Continuous IMU read on the source's own I/O thread; close the buffer
        # when a replay source exhausts so any pending wait_until returns at once.
        self.source.start(self.buffer.append, on_exhausted=self.buffer.close)
        try:
            self._loop()
        finally:
            self.source.stop()
            self.buffer.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            item = self._inbox.get()
            if item is _SENTINEL:
                break
            if item is _LATEST:
                # Coalescing inbox: the token says "drain the slot". Pull the
                # current newest trigger (already drained by an earlier token ->
                # _SENTINEL -> skip).
                with self._latest_lock:
                    msg, self._latest = self._latest, _SENTINEL
                if msg is _SENTINEL:
                    continue
            else:
                msg = item                      # strict-FIFO payload
            if msg is END:
                self._handle_end()
                continue
            process_cam_sync(self, msg)

    def _handle_end(self) -> None:
        # Forward END to every downstream topic exactly once, then signal done.
        # Single-input worker (one cam.sync inbox), so the first END is terminal
        # (the old Module.expected_ends defaulted to 1 for this case).
        if not self._emitted_end:
            self._emitted_end = True
            for topic in self._downstream:
                self.bus.publish(topic, END)
        self.done.set()


#: Public name kept for the in-process selftest (imports ``ImuCamModule``). It is
#: now the procedural worker, not a reactive Module.
ImuCamModule = ImuCamWorker


# --------------------------------------------------------------------------- #
# Front-end builders (was ours.app.build_replay_frontend / build_live_frontend).
# --------------------------------------------------------------------------- #
def _replay_imu_startup(reader: SessionReader, use_gyro: bool):
    """Startup IMU references for a recorded session.

    Returns ``(R_imu_cam, accel_align, gyro_bias)`` mirroring what the live
    front-end measures once at boot:

    * ``R_imu_cam`` -- IMU->camera rotation (gyro prior conjugation), or ``None``
      when the session has no IMU extrinsics (pure-vision fallback).
    * ``accel_align`` -- mean startup accelerometer (camera frame) over the first
      ~0.3 s, the gravity-leveling reference.
    * ``gyro_bias`` -- mean gyro over the first ~1 s (near-static start), removed
      from every sample by the imu_cam worker so the rotation prior is unbiased.
    """
    if not (use_gyro and reader.calib.has_imu_extrinsics):
        return None, None, None
    imu = reader.load_imu()
    ts = imu["ts_ns"]
    if ts.size <= 1:
        return None, None, None
    R_imu_cam = reader.calib.T_imu_left[:3, :3]
    t0 = int(ts[0])
    gyro_bias = None
    gwin = ts <= t0 + int(1.0 * 1e9)                 # first ~1 s
    if gwin.any():
        gyro_bias = imu["gyro"][gwin].mean(axis=0)
    accel_align = None
    awin = ts <= t0 + int(0.3 * 1e9)                 # first ~0.3 s
    if awin.any():
        accel_align = R_imu_cam @ imu["accel"][awin].mean(axis=0)
    return R_imu_cam, accel_align, gyro_bias


def build_replay_frontend(bus: LocalPubSub, reader: SessionReader, *,
                          depth_fast: bool = False, max_frames: int = 0,
                          calibration: ImuCalibration | None = None,
                          latest_only: bool = False,
                          tof_sim: bool = False):
    """Wire ONLY the acquisition front-end (``read_cam`` + ``imu_cam`` with depth)
    from a recorded session.

    For consumers that read ``frame.depth`` / ``imucam.sample`` directly (e.g. the
    capture process, or the image|depth|IMU triplet) and need no
    odometry/backend/slam. ``calibration`` is the IMU correction the imu_cam
    worker applies (``None`` -> the published IMU is raw); ``latest_only`` makes
    the front-end coalescing so a realtime visualiser stays fresh.
    ``tof_sim`` enables the VL53L9CX simulation: depth is computed at the session's
    SOURCE resolution then gray + depth are downsampled to ``TOF_W x TOF_H`` before
    publish. Returns ``(cam_module, imu_module)``.
    """
    sgm = SGMConfig.live() if depth_fast else SGMConfig()
    matcher = SGMStereoMatcher.from_calib(reader.calib, sgm)
    imu_module = ImuCamWorker(bus, ReplayImuSource(reader), matcher=matcher,
                              calibration=calibration, latest_only=latest_only,
                              tof_sim=tof_sim)
    cam_module = ReadCamModule(
        bus, ReplayCamSource(reader, max_frames=max_frames), fps=20)
    return cam_module, imu_module


def build_live_frontend(bus: LocalPubSub, *, width: int = 640, height: int = 400,
                        fps: int = 20, use_gyro: bool = True,
                        depth_fast: bool = True, recalibrate_bias: bool = False,
                        use_camera_calib: bool = False,
                        latest_only: bool = False,
                        tof_sim: bool = False):
    """Open the OAK-D and wire ONLY the acquisition front-end (``read_cam`` +
    ``imu_cam`` with depth) off ONE shared device.

    For consumers that read ``frame.depth`` / ``imucam.sample`` directly and need
    no odometry/backend/slam. The depth matcher rectifies BOTH cameras
    (``rectify_left=True``) since the raw left is unrectified.
    ``use_camera_calib`` opts into the operator's saved per-device stereo calib
    (default off -> the trusted factory calib is used). ``tof_sim`` enables
    the VL53L9CX simulation (depth at source res, then gray + depth downsampled to
    ``TOF_W x TOF_H`` before publish). Returns
    ``(device, cam_module, imu_module, cal)`` where ``cal`` is the
    live-calibration bundle (``cal.imu_calibration`` etc.); the caller starts the
    threads and releases ``device`` when the run ends. Hardware-only.
    """
    from imu_camera.device.oak_live import SharedLiveDevice
    from imu_camera.device.live_calib import read_live_calibration
    from .read_cam import LiveCamSource
    from .read_imu import LiveImuSource

    device = SharedLiveDevice(width=width, height=height, fps=fps)
    # Compile the KLT + SGM numba kernels on a background thread NOW, so the
    # ~1-3 s LLVM compile (cold cache) overlaps the device boot + the startup
    # IMU still-window below instead of stalling frame one. Uses the live configs
    # so the compiled signatures match. Never blocks: a slow warmup just means
    # frame one compiles as it always did.
    import threading as _threading
    from imu_camera.warmup import warmup_sgm
    from imu_camera.comms.lib.config.resolution import ResolutionProfile
    from imu_camera.resolution_build import sgm_config
    _res = ResolutionProfile.for_resolution(width, height)
    # imu_camera warms ONLY its own SGM kernel (it does not run KLT -- vio does
    # and warms KLT in its own process).
    _threading.Thread(
        target=warmup_sgm,
        kwargs={"sgm_cfg": sgm_config(_res, fast=depth_fast)},
        name="jit-warmup", daemon=True).start()

    cal = read_live_calibration(device, width=width, height=height,
                                use_gyro=use_gyro, depth_fast=depth_fast,
                                recalibrate_bias=recalibrate_bias,
                                use_camera_calib=use_camera_calib)
    matcher = SGMStereoMatcher.from_calib(cal.calib, cal.sgm_cfg,
                                          rectify_left=True)
    imu_module = ImuCamWorker(bus, LiveImuSource(device), matcher=matcher,
                              calibration=cal.imu_calibration,
                              latest_only=latest_only, tof_sim=tof_sim)
    cam_module = ReadCamModule(bus, LiveCamSource(device), fps=fps, realtime=True)
    return device, cam_module, imu_module, cal
