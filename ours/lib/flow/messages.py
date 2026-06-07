"""Message types passed between flows over the pub/sub bus.

These are plain immutable carriers -- one per topic in ``ours.lib.flow.topics``. Keeping
them here documents exactly what each flow consumes and produces, and keeps the
flows themselves free of ad-hoc dicts.

The :data:`END` sentinel is published on a topic when its upstream flow has no
more data (e.g. the recorded session ran out). Reactive flows forward it to their
own downstream topics so the whole graph drains cleanly; the UI flow uses it to
know the run is finished.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Published on a topic to signal "no more messages will follow on this topic".
END = object()


@dataclass(frozen=True)
class ImuPrior:
    """Per-frame IMU prior, built inside the odometry flow from each packet.

    The odometry flow owns the IMU->prior fusion (``PreintegratePrior``), turning
    the synced :class:`ImuCamPacket` into:

    * ``R_prior`` -- inter-frame camera-frame rotation ``R_cam(prev->cur)`` from
      the gyro (the rotation prior handed to PnP), or ``None`` on the first frame.
    * ``accel_cam`` / ``at_rest`` -- the camera-frame accelerometer this frame and
      whether the camera was still; supplied so a keyframe can carry a gravity
      prior into the back-end. ``at_rest`` is ``False`` (accel ``None``) when no
      usable gravity measurement is available.
    * ``imu_moving`` -- loose stillness gate (gyro rate or accel deviation above a
      "definitely moving" threshold). Distinct from ``at_rest`` (strict): the
      middle band is neither, so a freeze gate that vetoes ONLY on confirmed
      motion does not fire on borderline samples. Consumed by
      :meth:`~ours.lib.odometry.odometry.RGBDVisualOdometry.estimate` to keep the
      low-inlier translation freeze (textureless wall) from pinning the marker
      through a real motion-blurred shake -- the same IMU signal that
      ``OdometryConfig.min_inliers_for_translation`` was designed around.

    It is stashed in the flow's ``priors[seq]`` (never put on the bus) so the
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

    Published on ``topics.FRAME_TRACKS`` by the odometry flow's
    :class:`~ours.flows.odometry.publish_tracks.PublishTracks` task. The ``ids`` /
    ``points`` are the REAL frontend tracks the motion estimate consumes (the same
    ``{id: pixel}`` :class:`~ours.flows.odometry.tracked.Tracked` carries) -- not a
    parallel detector.

    NOTE: this message carries ONLY the per-frame tracks (ids + pixels). The
    rectified-left image and depth map needed to render the overlay travel on
    ``topics.FRAME_DEPTH`` (already published by the ``imu_cam`` flow); the UI
    sink (:class:`~ours.flows.ui.tracks.UiTracksFlow`) joins them by ``seq``. This
    keeps the cross-process layout honest -- in the 4-proc topology the capture
    process is the SINGLE writer of the gray / depth shared-memory rings, and the
    VIO process (which publishes ``frame.tracks``) does not race the capture
    process for the same ring slots.

    * ``ids`` -- ``(N,)`` int64 persistent track ids.
    * ``points`` -- ``(N, 2)`` float32 pixel coordinates (same order as ``ids``).
    """

    seq: int
    ts_ns: int
    ids: np.ndarray
    points: np.ndarray


@dataclass(frozen=True)
class FrameInliers:
    """One frame's PnP inlier track ids for the keypoint-depth visualiser.

    Published on ``topics.FRAME_INLIERS`` by the odometry flow's
    :class:`~ours.flows.odometry.publish_inliers.PublishInliers` task, AFTER
    :class:`~ours.flows.odometry.estimate_motion.EstimateMotion` solves the
    RGB-D PnP. ``ids`` are the subset of that frame's tracks the PnP RANSAC kept
    as inliers -- the clean points the motion solve actually trusted (a REAL
    odometry output read from ``last_info``, not a re-derivation). The UI marks
    these so the operator can see which tracks survived outlier rejection.

    * ``ids`` -- ``(M,)`` int64 inlier track ids (subset of the frame's tracks).
    """

    seq: int
    ts_ns: int
    ids: np.ndarray


@dataclass(frozen=True)
class PoseMsg:
    """An estimated camera pose (4x4 ``T_world_cam``) for one frame."""

    seq: int
    ts_ns: int
    T_world_cam: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Keyframe:
    """A keyframe handed to the back-end / SLAM flows.

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
class CamSync:
    """A stereo pair the ``cam`` flow publishes as a sync trigger.

    Published on ``topics.CAM_SYNC`` by :class:`~ours.flows.cam.CamFlow`
    once per scheduled frame. It carries the frames *and* their device timestamp
    so the ``imu_cam`` flow can both (a) drain its buffer up to ``ts_ns`` and
    (b) pack the very same frames into the combined packet -- no second lookup,
    no shared state between the two flows beyond this message.

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

    Published on ``topics.IMUCAM_SAMPLE`` by
    :class:`~ours.flows.imu_cam.ImuCamFlow` in response to each
    :class:`CamSync`. This is the synchronised unit downstream consumers (state
    estimation, visualiser) work on: a stereo pair plus exactly the inertial
    measurements that fall in this frame's interval ``(prev_frame_ts, ts_ns]``,
    selected by device timestamp (the only clock the IMU shares with the camera).

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

    Published on ``topics.IMU_RAW`` by
    :class:`~ours.flows.imu_cam.ImuCamFlow` for every :class:`CamSync`,
    carrying exactly what the sensor reported (no bias/scale correction) so a
    consumer can see the uncalibrated signal. The matching
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
