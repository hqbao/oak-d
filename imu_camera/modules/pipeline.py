"""The :class:`ImuCamModule` -- buffer IMU, pack per trigger, compute depth --
plus the live/replay acquisition front-end wiring.

The other half of the acquisition front-end (``read_cam`` is the other). It reads
the IMU continuously into a timestamped buffer and, for each
:class:`~imu_camera.comms.messages.CamSync` trigger the ``read_cam`` module
publishes on ``cam.sync``, drains the buffer up to that frame's device timestamp
and publishes an :class:`~imu_camera.comms.messages.ImuCamPacket` (the frames
bundled with exactly the inertial samples in that frame's interval) on
``imucam.sample``. When given a stereo matcher it also computes dense depth for
the same pair INLINE and publishes ``frame.depth`` -- depth is a step in this
module, not a separate one, since it is just a transform of the stereo pair this
module already produces (it still runs on this module's thread, in parallel with
the odometry thread that consumes the result).

The :func:`build_replay_frontend` / :func:`build_live_frontend` helpers (was
``ours.app.build_*``) wire ONLY the acquisition front-end (``read_cam`` +
``imu_cam`` with depth) -- everything downstream of ``imucam.sample`` /
``frame.depth`` lives in the other projects (vio / slam / ui).
"""
from __future__ import annotations

from collections.abc import Callable

from sky.sensors.imu_calib import ImuCalibration

from imu_camera.comms import LocalPubSub, Module, topics
from imu_camera.io.reader import SessionReader
from imu_camera.mathlib.imu.timed_buffer import TimedImuBuffer
from sky.depth.stereo import (
    SGMConfig, SGMStereoMatcher)

from .apply_calibration import ApplyCalibrationStep
from .compute_depth import ComputeDepthStep
from .pack_synced import PackSyncedStep
from .publish_depth import PublishDepthStep
from .publish_imu_raw import PublishImuRawStep
from .publish_imucam import PublishImuCamStep
from .read_cam import ReadCamModule, ReplayCamSource
from .read_imu import ImuSource, ReplayImuSource
from .tof_downsample import ToFDownsampleStep

#: VL53L9CX-class ToF sensor simulation output grid. The OAK-D is the stand-in:
#: depth is computed on it at the SOURCE resolution (``--width``/``--height``,
#: where stereo actually works) and then DOWNSAMPLED to this fixed ToF grid --
#: a clean, dense, accurate low-res depth (median-of-valid fills holes + averages
#: noise), exactly what a real ToF returns. NOT a direct low-res stereo solve.
TOF_W = 54
TOF_H = 42


class ImuCamModule(Module):
    """Reactive module: buffer IMU, pack per camera trigger, compute dense depth.

    ``source`` supplies the raw IMU (``ReplayImuSource`` offline,
    ``LiveImuSource`` on the bench). ``wait_timeout`` bounds how long packing a
    frame waits for the IMU stream to cover its timestamp before draining what is
    available (so the run never hangs on the final frame).

    For every camera trigger the module publishes the uncalibrated samples on
    ``topics.IMU_RAW`` (honest: exactly what the sensor reported) and, on
    ``topics.IMUCAM_SAMPLE``, the frames bundled with the CALIBRATED IMU.
    ``calibration`` (or the lazy ``calibration_provider``, used on the live path
    where the device id is known only after the device opens) supplies the
    per-device correction; with none, the calibrated packet equals the raw one.

    ``matcher`` makes depth a step IN this module: when supplied, the chain also
    runs SGM on the same stereo pair and publishes ``topics.FRAME_DEPTH``. The
    camera/IMU visualiser passes ``matcher=None`` (it only wants the synced
    packet, no depth).

    Note on threads: the *module* owns one thread (it drains the inbox and runs
    the pack/publish/depth chain). The injected ``source`` runs the continuous
    high-rate IMU read on its OWN I/O thread -- a hardware producer, not a module.
    No module logic runs on that thread; it only fills the thread-safe buffer.

    WARNING: ``latest_only=True`` makes THIS module's CAM_SYNC inbox coalesce,
    which drops camera triggers when packing/SGM falls behind -- the downstream
    ``imucam.sample`` + ``frame.depth`` (both in ``topics.VIO_PATH_TOPICS``) then
    skip those frames, breaking VIO's gyro continuity (PreintegratePrior) and KLT
    continuity (TrackFeatures). ONLY pass ``latest_only=True`` when this module
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
        super().__init__("imu-cam", bus, latest_only=latest_only)
        self.source = source
        self.buffer = TimedImuBuffer(capacity=buffer_capacity)

        chain = [
            PackSyncedStep(self.buffer, wait_timeout),
            PublishImuRawStep(),
            ApplyCalibrationStep(calibration, provider=calibration_provider),
        ]
        downstream = [topics.IMU_RAW, topics.IMUCAM_SAMPLE]
        if matcher is not None:
            self.ctx.state["matcher"] = matcher
            if tof_sim:
                # VL53L9CX simulation: depth is still computed at the SOURCE
                # resolution inside ToFDownsampleStep (where SGM works), then
                # gray + depth are downsampled to TOF_W x TOF_H. The single step
                # publishes the 54x42 imucam.sample (gray + calibrated IMU) AND
                # the 54x42 frame.depth, so it REPLACES the normal
                # PublishImuCam/ComputeDepth/PublishDepth trio. The published
                # frames now match the 54x42 capture rings + the anisotropically
                # scaled calib.bundle K (see imu_camera.main).
                chain += [ToFDownsampleStep(TOF_W, TOF_H)]
            else:
                # Normal path (byte-identical to before): publish the synced
                # packet at source res, then compute + publish dense SGM depth.
                chain += [PublishImuCamStep(),
                          ComputeDepthStep(), PublishDepthStep()]
            downstream.append(topics.FRAME_DEPTH)
        else:
            # No matcher (UI triplet view): just publish the synced packet.
            chain += [PublishImuCamStep()]

        self.forwards_to(*downstream)
        self.on(topics.CAM_SYNC, chain)

    def run(self) -> None:
        # Continuous IMU read on the source's own I/O thread; close the buffer
        # when a replay source exhausts so any pending wait_until returns at once.
        self.source.start(self.buffer.append, on_exhausted=self.buffer.close)
        try:
            super().run()
        finally:
            self.source.stop()
            self.buffer.close()


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
      from every sample by the imu_cam module so the rotation prior is unbiased.
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
    module applies (``None`` -> the published IMU is raw); ``latest_only`` makes
    the front-end coalescing so a realtime visualiser stays fresh.
    ``tof_sim`` enables the VL53L9CX simulation: depth is computed at the session's
    SOURCE resolution then gray + depth are downsampled to ``TOF_W x TOF_H`` before
    publish. Returns ``(cam_module, imu_module)``.
    """
    sgm = SGMConfig.live() if depth_fast else SGMConfig()
    matcher = SGMStereoMatcher.from_calib(reader.calib, sgm)
    imu_module = ImuCamModule(bus, ReplayImuSource(reader), matcher=matcher,
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
    from imu_camera.mathlib.device.oak_live import SharedLiveDevice
    from imu_camera.mathlib.device.live_calib import read_live_calibration
    from .read_cam import LiveCamSource
    from .read_imu import LiveImuSource

    device = SharedLiveDevice(width=width, height=height, fps=fps)
    # Compile the KLT + SGM numba kernels on a background thread NOW, so the
    # ~1-3 s LLVM compile (cold cache) overlaps the device boot + the startup
    # IMU still-window below instead of stalling frame one. Uses the live configs
    # so the compiled signatures match. Never blocks: a slow warmup just means
    # frame one compiles as it always did.
    import threading
    from imu_camera.mathlib.warmup import warmup_sgm
    from imu_camera.comms.lib.config.resolution import ResolutionProfile
    from imu_camera.mathlib.resolution_build import sgm_config
    _res = ResolutionProfile.for_resolution(width, height)
    # imu_camera warms ONLY its own SGM kernel (it does not run KLT -- vio does
    # and warms KLT in its own process).
    threading.Thread(
        target=warmup_sgm,
        kwargs={"sgm_cfg": sgm_config(_res, fast=depth_fast)},
        name="jit-warmup", daemon=True).start()

    cal = read_live_calibration(device, width=width, height=height,
                                use_gyro=use_gyro, depth_fast=depth_fast,
                                recalibrate_bias=recalibrate_bias,
                                use_camera_calib=use_camera_calib)
    matcher = SGMStereoMatcher.from_calib(cal.calib, cal.sgm_cfg,
                                          rectify_left=True)
    imu_module = ImuCamModule(bus, LiveImuSource(device), matcher=matcher,
                              calibration=cal.imu_calibration,
                              latest_only=latest_only, tof_sim=tof_sim)
    cam_module = ReadCamModule(bus, LiveCamSource(device), fps=fps, realtime=True)
    return device, cam_module, imu_module, cal
