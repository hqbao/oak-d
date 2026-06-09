"""Message types passed between modules over the in-process pub/sub bus.

These are plain immutable carriers -- one per topic in :mod:`comms.topics`.
Keeping them here documents exactly what each module consumes and produces, and
keeps the modules themselves free of ad-hoc dicts. They ride
:class:`comms.pubsub.LocalPubSub` directly (zero serialization); the cross-process
wire forms (one per topic) live in :mod:`comms.wire`.

The :data:`END` sentinel is published on a topic when its upstream module has no
more data (e.g. the recorded session ran out). Reactive modules forward it to
their own downstream topics so the whole graph drains cleanly; the UI module uses
it to know the run is finished.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Published on a topic to signal "no more messages will follow on this topic".
END = object()


@dataclass(frozen=True)
class ImuPrior:
    """Per-frame IMU prior, built inside the odometry module from each packet.

    The odometry module owns the IMU->prior fusion (``PreintegratePrior``),
    turning the synced :class:`ImuCamPacket` into:

    * ``R_prior`` -- inter-frame camera-frame rotation ``R_cam(prev->cur)`` from
      the gyro (the rotation prior handed to PnP), or ``None`` on the first frame.
    * ``accel_cam`` / ``at_rest`` -- the camera-frame accelerometer this frame and
      whether the camera was still; supplied so a keyframe can carry a gravity
      prior into the back-end. ``at_rest`` is ``False`` (accel ``None``) when no
      usable gravity measurement is available.
    * ``imu_moving`` -- loose stillness gate (gyro rate or accel deviation above a
      "definitely moving" threshold). Distinct from ``at_rest`` (strict): the
      middle band is neither, so a freeze gate that vetoes ONLY on confirmed
      motion does not fire on borderline samples. Consumed by the RGB-D PnP to
      keep the low-inlier translation freeze (textureless wall) from pinning the
      marker through a real motion-blurred shake.

    It is stashed in the module's ``priors[seq]`` (never put on the bus) so the
    matching depth frame can pick it up by ``seq``.
    """

    seq: int
    R_prior: np.ndarray | None
    accel_cam: np.ndarray | None = None
    at_rest: bool = False
    imu_moving: bool = False


@dataclass(frozen=True)
class DepthFrame:
    """A left image with a metric depth map aligned to it."""

    seq: int
    ts_ns: int
    gray_left: np.ndarray
    depth_m: np.ndarray


@dataclass(frozen=True)
class FrameTracks:
    """One frame's KLT tracks for the keypoint-depth visualiser.

    Published on ``topics.FRAME_TRACKS`` by the odometry module's
    ``PublishTracks`` step. The ``ids`` / ``points`` are the REAL frontend tracks
    the motion estimate consumes (the same ``{id: pixel}`` ``Tracked`` carries) --
    not a parallel detector.

    NOTE: this message carries ONLY the per-frame tracks (ids + pixels). The
    rectified-left image and depth map needed to render the overlay travel on
    ``topics.FRAME_DEPTH`` (already published by the ``imu_cam`` module); the UI
    sink joins them by ``seq``. This keeps the cross-process layout honest -- in
    the 4-proc topology the capture process is the SINGLE writer of the gray /
    depth shared-memory rings, and the VIO process (which publishes
    ``frame.tracks``) does not race the capture process for the same ring slots.

    * ``ids`` -- ``(N,)`` int64 persistent track ids.
    * ``points`` -- ``(N, 2)`` float32 pixel coordinates (same order as ``ids``).
    """

    seq: int
    ts_ns: int
    ids: np.ndarray
    points: np.ndarray


@dataclass(frozen=True)
class FrameInliers:
    """One frame's PnP reprojection diagnostic for the keypoint-depth visualiser.

    Published on ``topics.FRAME_INLIERS`` by the odometry module's
    ``PublishInliers`` step, AFTER ``EstimateMotion`` solves the RGB-D PnP.

    Carries, per PnP correspondence (all ``M`` points fed to the RANSAC, NOT just
    the inliers), the reprojection of its prev-frame 3D point through the SAME
    ``(R, t)`` the RANSAC produced -- the pose that DEFINED the inlier set. The UI
    draws a measured-pixel -> reprojected-pixel stub per point (a REAL odometry
    output read from ``last_info``, not a re-derivation), tiny + green for
    inliers, long + red for outliers, so "minimise reprojection error" is visible.

    * ``ids`` -- ``(M,)`` int64 PnP point track ids (all correspondences, in PnP
      order). The classic "inlier ids" any consumer wants = ``ids[inlier]``.
    * ``reproj`` -- ``(M, 2)`` float32 reprojected pixel per point (same order as
      ``ids``); ``pinhole(K, R @ obj_i + t)``.
    * ``inlier`` -- ``(M,)`` bool RANSAC inlier mask (same order as ``ids``).
    """

    seq: int
    ts_ns: int
    ids: np.ndarray
    reproj: np.ndarray
    inlier: np.ndarray


@dataclass(frozen=True)
class PoseMsg:
    """An estimated camera pose (4x4 ``T_world_cam``) for one frame."""

    seq: int
    ts_ns: int
    T_world_cam: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Keyframe:
    """A keyframe handed to the back-end / SLAM modules.

    Carries both the high-level snapshot the SLAM map needs (pose + image +
    depth) and the low-level track snapshot the sliding-window BA needs
    (``track_ids`` + ``track_px`` from the odometry front-end, plus an optional
    at-rest ``accel`` for the gravity prior).
    """

    seq: int
    T_world_cam: np.ndarray
    gray_left: np.ndarray
    depth_m: np.ndarray
    track_ids: np.ndarray | None = None
    track_px: np.ndarray | None = None
    accel: np.ndarray | None = None
    #: subset of ``track_ids`` the RGB-D PnP kept as INLIERS this frame (the clean,
    #: motion-consistent features). The 3D-map viewer back-projects only these, so
    #: the noisy dense-depth points the solve rejected are never drawn.
    inlier_ids: np.ndarray | None = None


@dataclass(frozen=True)
class LoopCorrection:
    """A pose-graph correction: rewritten keyframe poses after loop closure."""

    seq: int
    kf_poses: dict[int, np.ndarray]
    n_loops: int


@dataclass(frozen=True)
class SlamOverlay:
    """Continuous SLAM keyframe-map snapshot for the live overlay (slam.map).

    Positions are CAMERA-OPTICAL world frame; the UI applies the optical->NED
    display transform. Distinct from LoopCorrection (which is the loop-event
    pose-graph rewrite on loop.correction).
    """

    kf_positions: np.ndarray          # (N, 3) optical-world keyframe positions
    n_loops: int
    last_match: np.ndarray | None = None   # (M, 3) optical, the just-closed loop's kfs (flash)
    #: ``(N,)`` int64 source frame seq of each keyframe, SAME order as
    #: ``kf_positions``. Lets the UI match each corrected keyframe to its dense
    #: VIO pose for the rubber-sheet "corrected VIO" line. Empty when unset.
    kf_seqs: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))


@dataclass(frozen=True)
class CamSync:
    """A stereo pair the ``cam`` module publishes as a sync trigger.

    Published on ``topics.CAM_SYNC`` by the ``ReadCam`` module once per scheduled
    frame. It carries the frames *and* their device timestamp so the ``imu_cam``
    module can both (a) drain its buffer up to ``ts_ns`` and (b) pack the very
    same frames into the combined packet -- no second lookup, no shared state
    between the two modules beyond this message.

    ``ts_ns`` is the frame device timestamp (left camera), the cut used to select
    the inertial samples that belong to this frame's interval.
    """

    seq: int
    ts_ns: int
    gray_left: np.ndarray
    gray_right: np.ndarray | None


@dataclass(frozen=True)
class ImuCamPacket:
    """A camera frame bundled with all IMU samples up to its timestamp.

    Published on ``topics.IMUCAM_SAMPLE`` by the ``imu_cam`` module in response to
    each :class:`CamSync`. This is the synchronised unit downstream consumers
    (state estimation, visualiser) work on: a stereo pair plus exactly the
    inertial measurements that fall in this frame's interval
    ``(prev_frame_ts, ts_ns]``, selected by device timestamp (the only clock the
    IMU shares with the camera).

    * ``imu_ts`` -- ``(M,)`` device timestamps (ns) of the samples, time-ordered.
    * ``gyro`` / ``accel`` -- ``(M, 3)`` angular rate (rad/s) and specific force
      (m/s^2). ``M`` may be 0 (e.g. the first frame, or a dropped IMU interval).
    """

    seq: int
    ts_ns: int
    gray_left: np.ndarray
    gray_right: np.ndarray | None
    imu_ts: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray


@dataclass(frozen=True)
class ImuRaw:
    """The RAW IMU samples for one frame interval, before any calibration.

    Published on ``topics.IMU_RAW`` by the ``imu_cam`` module for every
    :class:`CamSync`, carrying exactly what the sensor reported (no bias/scale
    correction) so a consumer can see the uncalibrated signal. The matching
    :class:`ImuCamPacket` on ``topics.IMUCAM_SAMPLE`` carries the SAME interval's
    samples after calibration. ``M`` may be 0 for an empty interval.

    * ``imu_ts`` -- ``(M,)`` device timestamps (ns), time-ordered.
    * ``gyro`` / ``accel`` -- ``(M, 3)`` raw angular rate (rad/s) and specific
      force (m/s^2).
    """

    seq: int
    ts_ns: int
    imu_ts: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
