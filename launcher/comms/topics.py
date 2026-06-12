"""Canonical pub/sub topic names -- the cross-project data-flow contract.

Each constant is the string key used on :class:`comms.pubsub.LocalPubSub` (and,
across processes, on :class:`comms.ipc.IPCPubSub`). The STRINGS here are frozen:
they are the contract every one of the split projects (imu_camera, depth, vio,
slam, ui) agrees on, so they MUST NOT be renamed.

There is ONE acquisition front-end (``cam`` + ``imu_cam``); the odometry module
consumes its synced ``imucam.sample`` and the ``frame.depth`` its depth step
emits::

    cam      --cam.sync------> imu_cam        (trigger: stereo pair + ts)
    imu_cam  --imu.raw-------> (visualiser)    (raw IMU for the interval)
    imu_cam  --imucam.sample-> odometry        (frames + CALIBRATED IMU)
    imu_cam  --frame.depth---> odometry        (depth step output)
    odometry --pose.odom-----> ui-collector, ui-render
    odometry --pose.vo-------> ui             (pure-vision VO line, LIVE-only)
    odometry --frame.tracks--> ui-tracks       (KLT tracks + depth, visualiser)
    odometry --keyframe------> backend, slam
    backend  --pose.refined--> ui-collector
    slam     --loop.correction-> ui-collector
    slam     --slam.map-------> ui-collector    (LIVE-only continuous overlay)

``cam.sync`` carries the frames so the ``imu_cam`` module packs them with the
inertial samples it drains up to the frame timestamp. For each trigger the
``imu_cam`` module emits the uncalibrated samples on ``imu.raw`` (honest, what
the sensor reported), the frames bundled with the *calibrated* IMU on
``imucam.sample``, and -- in the VIO path -- the ``frame.depth`` computed by its
own depth step. The odometry module owns the IMU->prior fusion itself.

Note: the back-end and SLAM modules both trigger off ``keyframe`` (NOT
``pose.odom``); the SLAM ``loop.correction`` is currently consumed only by the
UI collector -- it is not yet fed back into odometry (no closed loop on the live
pose path).

``slam.map`` is the LIVE-ONLY continuous keyframe overlay published EVERY
keyframe by the SLAM module (``publish_map=True``); the UI's SLAM tab draws its
keyframe dots from it instead of waiting on ``loop.correction`` (which only fires
ON a loop closure). The offline path keeps ``publish_map=False`` so it is never
published there and the deterministic ``loop.correction`` scoring stays
byte-identical.
"""
from __future__ import annotations

FRAME_DEPTH = "frame.depth"
POSE_ODOM = "pose.odom"
# Pure-vision frame-to-frame trajectory (no IMU / no BA), live-only -- the UI's
# "VO" line to compare against the VIO ``pose.odom``. Carries a PoseMsg.
POSE_VO = "pose.vo"
KEYFRAME = "keyframe"
POSE_REFINED = "pose.refined"
LOOP_CORRECTION = "loop.correction"
# Continuous SLAM keyframe-map overlay (live-only; loop.correction stays the loop-event correction).
SLAM_MAP = "slam.map"

# Per-loop-CANDIDATE match funnel (live-only) for the UI's "Loop Closure" window.
# Published by the SLAM engine for every verified candidate -- CONFIRMED or
# REJECTED -- so the UI can show WHY a loop fired or was rejected: the matched ORB
# keypoint pixel pairs in the two keyframes, a per-match verification-stage label
# (appearance / epipolar / PnP), the funnel counts, and the rotation-gate verdict.
# Additive + LIVE-only (the offline/oracle path never publishes it), so pose math
# + the byte-parity oracle are UNAFFECTED. Carries NO keyframe images (SLAM does
# not retain the gray); the UI joins these by seq to grays it buffers off
# ``keyframe``. Consumed only by the UI.
SLAM_LOOP = "slam.loop"

# Per-frame KLT tracks the odometry frontend produced (ids + pixels) bundled with
# the frame + its depth, for the keypoint-depth visualiser. The SAME tracks the
# motion estimate consumes -- no parallel detector. Consumed only by the UI.
FRAME_TRACKS = "frame.tracks"

# Per-frame PnP inlier track ids -- the clean subset the RGB-D PnP RANSAC actually
# kept for the motion solve (a REAL odometry output, not a re-derivation). Lets the
# keypoint-depth visualiser mark which tracks survived outlier rejection. Consumed
# only by the UI; published after EstimateMotion runs.
FRAME_INLIERS = "frame.inliers"

# Per-frame gyro-fusion diagnostic -- the vision-vs-gyro inter-frame rotation, their
# disagreement, the resulting correction gain + translation-trust, and the config
# gate thresholds (ALGORITHMS.md #5). A REAL odometry output read from last_info,
# published only on gyro-fused frames. Drives the UI "Gyro fusion" strip chart that
# explains why the fused VIO stays straight where pure-vision drifts. UI-only.
FRAME_GYROFUSE = "frame.gyrofuse"

# Acquisition front-end (``cam`` <-> ``imu_cam``).
CAM_SYNC = "cam.sync"
IMUCAM_SAMPLE = "imucam.sample"
# Raw IMU for each frame interval, published BEFORE calibration (honest, what
# the sensor reported). ``imucam.sample`` carries the CALIBRATED IMU.
IMU_RAW = "imu.raw"

# One-shot retained calibration broadcast (intrinsics + extrinsics) the capture
# process publishes on boot; read DIRECTLY by VIO / SLAM / UI (no flow message).
CALIB_BUNDLE = "calib.bundle"

# Periodic snapshot of the VIO process's windowed-BA refined trajectory, read
# directly by the UI (no flow message, like calib.bundle).
VIO_MAP = "vio.map"

# Per-keyframe windowed-BA solve snapshot for the UI's "BA Window" visualiser
# (window keyframe poses + shared 3D landmarks + observation rays + reprojection
# error + the PRE-solve state for a before/after toggle). Published by the VIO
# process ONLY when the opt-in ``--ba-window`` flag is on -- the default-OFF path
# never captures it, so the byte-parity oracle is UNAFFECTED. Carries NO images
# (mirrors slam.loop). Consumed only by the UI.
BA_WINDOW = "ba.window"

# Per-frame frontend-internals snapshot for the UI's "Frontend Internals" view:
# the Shi-Tomasi (lambda_min) response heatmap (quantised producer-side), the
# accepted corners + detection geometry (min_distance / quality / grid), and the
# KLT flow field (prev->next per-track pixels coloured by forward-backward error,
# with the culled mask). Shows HOW the frontend finds + tracks features.
# Published by the VIO process ONLY when the opt-in ``--frontend-viz`` flag is on
# -- the default-OFF path builds the plain KLTFrontend (no capture), so the
# returned tracks AND the byte-parity oracle are UNAFFECTED. The heatmap rides
# inline as a quantised uint8 ndarray (no shared-memory ring, like ba.window).
# Consumed only by the UI.
FRAME_FRONTEND = "frame.frontend"

# Topics that MUST stay FIFO end-to-end -- VIO + deterministic replay break if
# any frame is coalesced away. A module built with ``latest_only=True`` whose
# inbox or downstream chain feeds the odometry compute path (PreintegratePrior /
# TrackFeatures / EstimateMotion) corrupts the gyro continuity and the KLT track
# continuity. The set documents the contract; UI-ONLY sinks may subscribe these
# topics on a latest-only inbox (they consume frames for display, not VIO state).
VIO_PATH_TOPICS = frozenset({
    CAM_SYNC,
    IMUCAM_SAMPLE,
    FRAME_DEPTH,
})
