# oak-d

Companion-computer project to turn an **OAK-D W** stereo camera into a 6-DoF
position source for the flight-controller. Runs on a Mac mini today, will move
to a Raspberry Pi 5 later.

```
oak-d/
  ours/                  OUR from-scratch pipeline (library-free; self-contained)
    app.py               wire + run the 6 flows over a pub/sub bus (live + replay)
    depthai_ours_vio.py  live OAK-D source driving ours.lib (ours/-ba/-slam/-vio)
    lib/                 the algorithm library + runtime building blocks
      pose.py            own Pose dataclass + ring buffer (copy, kept independent)
      frames.py          own camera/body/world transforms (copy)
      pngio.py           own pure-Python PNG codec (copy)
      geometry.py        SO(3)/SE(3) Lie-algebra helpers
      pubsub.py flow.py task.py  thread-per-flow actor model + topics/messages
      frontend/          KLT + Shi-Tomasi feature frontend (cv2-free; Numba KLT)
      stereo/            own SGM dense depth + rectification
      imu/               Forster IMU preintegration + translation filter
      odometry/          frame-to-frame RGB-D PnP (+ gyro fusion) + own RANSAC PnP
      backend/           sliding-window bundle adjustment (analytic Schur)
      loop/              own ORB + F-matrix loop closure + SE(3) pose graph + SLAM
      io/                recorded-session reader + time-synced bundles
      config/            resolution-aware tuning profiles
    flows/               live-pipeline orchestration (one thread + tasks per flow)
      capture/ depth/ odometry/ backend/ slam/ ui/   (capture = replay + live)
      live_source.py     bridge: run the live flow graph into the Qt viewer
    ui/                  own Qt 3D viewer + PoseSource base + fake source (copy)
    tools/               offline scoring + self-tests (call ours.lib directly)
      view_pose3d.py     live 3D viewer (run.sh entry; ours backends)
      vio_run.py         offline scoring of ours f2f/ba/slam/vio vs Basalt
      live_replay.py     replay a recorded session through the live ours pipeline
      synced_view.py     inspect the synced (image, depth, IMU) triplet
      imucam_view.py     cv2 view of the split cam/IMU front-end (left|right|gyro|accel)
      *_selftest.py      regression guards (klt, ba, posegraph, imu_preint, vio_ba, imucam_*)
  baseline/              DepthAI library pipeline (BasaltVIO + RTABMapSLAM)
    oakd/                baseline-only core (its Pose/frames/pngio/sources/ui)
      frames.py          camera <-> body (FRD) <-> world (NED) transforms
      pose.py            Pose dataclass + ring buffer
      recorder.py        live-run logger (C0..C9 streams to sessions/<name>/)
      pngio.py           pure-Python 8-bit PNG codec (replaces cv2.imread/imwrite)
      sources/           PoseSource base + the device-free fake source
      ui/                Qt 3D viewer (theme/viewer3d/panels/mainwindow)
    depthai_vio.py       real stereo-inertial VIO (dai.node.BasaltVIO)
    depthai_slam.py      VIO + SLAM with loop closure (BasaltVIO + RTABMapSLAM)
    tools/
      view_pose3d.py     live 3D viewer from OAK over USB (Basalt backends)
      record_session.py  dump C0..C9 + PCL from a live run to sessions/<name>/
      viz_session.py     offline multi-tab replay of a recorded session
      compare_sessions.py  ATE/RPE between two pose streams (VIO vs SLAM, etc.)
      baseline_report.py scan sessions/gold/, emit Markdown baseline report
      extract_kf_from_db.py  pull keyframes/loops out of rtabmap.db
  run.sh
  requirements.txt
```

See `docs/PIPELINE_CHECKPOINTS.md` for the recording schema + migration
plan, and `docs/GOLD_SESSIONS.md` for the regression suite scenarios.

## Camera mount

- Body of camera mounts on the FRONT of the drone, looking FORWARD.
- USB connector of the camera block points UP.
- The "WIDE" label on the front of the lens block reads correctly when viewed
  from above.

This gives camera-frame axes (right-handed, OpenCV convention):
  `Xc = right`, `Yc = down`, `Zc = forward` — already aligned with the drone's
  body FRD frame, so `R_body_cam = I`.

## Coordinate conventions

- World: **NED** (North-East-Down), origin = start pose.
- Body:  **FRD** (Forward-Right-Down).
- Viewer renders ENU (East, North, Up) for natural pilot perspective; the
  underlying state stays NED.

## Quick start

`run.sh` launches the viewer for **our** pipeline (the two roots share no code):

```bash
./run.sh                       # our frame-to-frame RGB-D PnP VO (default)
./run.sh --source fake         # device-free procedural trajectory
./run.sh --source ours         # our frame-to-frame RGB-D PnP VO
./run.sh --source ours-ba      # + sliding-window bundle adjustment (depth-anchored)
./run.sh --source ours-slam    # + ORB loop closure + SE(3) pose graph (full SLAM)
```

The **baseline** (DepthAI/Basalt) viewer is a separate entry point:

```bash
.venv/bin/python baseline/tools/view_pose3d.py --source oak    # BasaltVIO
.venv/bin/python baseline/tools/view_pose3d.py --source slam   # BasaltVIO + RTABMapSLAM
```

`--source ours` is the flow pipeline (capture → depth → odometry → backend →
slam → ui flows over a pub/sub bus); the older single-file source is still
available as `--source ours-legacy`. For a device run with no GUI:

```bash
.venv/bin/python -m ours.app --live          # headless: stream the OAK-D, print pose/loops
```

At START the live path measures only the **gravity-align level** (which depends
on how the camera is held); the **gyro bias** is a sensor constant, so it is
calibrated once, saved per device under `.cache/imu_calib.json`, and reused on
later runs. Force a fresh bias measurement with `--recalibrate-bias`.

The 3D viewer groups its features in a **menu bar** (the toolbar keeps only the
primary START/STOP):

- **View** — camera presets (ISO/TOP/FRONT/BACK/LEFT/RIGHT), Follow Camera,
  Clear Trail, and Clear Keyframes (SLAM backend only).
- **Calibration** — guided wizards that own the device while they run:
  **Gyroscope Bias** (one still window) and **Accelerometer (6-position)** (hold
  each face up/down; solves bias + scale + misalignment as the affine
  `a_cal = T·(a_raw − b)`). Both persist per device into `.cache/imu_calib.json`
  and the live path applies the accel calibration automatically on the next run.
- **Visualize** — inspect the raw sensor streams. These need exclusive device
  access, so the live VIO pipeline is released first.
  - **Camera + IMU (synced, live)** — opens an *in-app* window (no subprocess)
    that runs the split `cam_reader` + `imu_reader` flows live and draws every
    synchronised `ImuCamPacket` in three honest panels (each is exactly what the
    packet carries, no parallel pipeline):
    - **cameras** — `left | right` stereo pair;
    - **gyro** — an **auto-scaling** scrolling line chart (deg/s); the Y axis
      tracks the signal with a minimum span + expand-fast/shrink-slow
      hysteresis, so a still IMU doesn't strobe;
    - **accel** — a **real interactive 3D** vector view (OpenGL): the specific
      force is drawn as a solid arrow you can orbit with the mouse and snap to
      **BACK / LEFT / TOP**. A checkerboard floor below + 1 G reference rings +
      an X/Y/Z body-axis triad make the magnitude and direction readable.

    This is the view for verifying the camera↔IMU time-sync.
  - **Camera + Depth + IMU** triplet opens an in-app Qt window
    (`ours/ui/synced_window.py`): cameras on top (`image | depth`) and the IMU
    panels below (the same gyro chart + interactive 3D accel view), with a
    fixed-range single-hue khaki depth ramp (+ scale bar, `valid %`). The IMU shown is
    **calibrated** (`gyro − bias`, accel affine) when a per-device calibration is
    cached — the panel title reads `IMU · CALIBRATED` vs `IMU · RAW`. Live off the
    OAK-D (host SGM) or a recorded session.
  - **Keypoint Depth Tracker** opens an in-app Qt window
    (`ours/ui/keypoints_window.py`): the rectified-left frame with every live
    **KLT-frontend track** drawn on it (the SAME frontend the odometry runs, not
    a parallel detector). Each dot's **colour = that keypoint's metric depth**
    (the same fixed khaki 0.3–8 m ramp + scale bar as the depth panel), so colour
    means the same distance everywhere; keypoints with no stereo return are
    **hollow grey rings** (never a faked colour), fresh tracks get an amber ring.
    A faint per-id **trail** shows where the *same* keypoint moved over the last
    20 frames. The footer prints honest stats (`trk`, `valid-z %`, `mean-age`,
    `new`). Live off the OAK-D (host SGM) or a recorded session.

The same synced split front-end can be inspected **without the GUI** — a cv2
window over a recorded session or the live device:

```bash
.venv/bin/python -m ours.tools.imucam_view --session sessions/gold/lab_loop_30s   # replay
.venv/bin/python -m ours.tools.imucam_view --live                                 # OAK-D
```

To self-verify the front-end with numbers instead of eyeballs (no device):

```bash
.venv/bin/python -m ours.tools.imucam_sync_selftest --session sessions/gold/lab_loop_30s
.venv/bin/python -m ours.tools.imu_calib_selftest    # raw IMU on imu.raw, calibrated IMU in imucam.sample
QT_QPA_PLATFORM=offscreen .venv/bin/python -m ours.tools.imucam_window_selftest
QT_QPA_PLATFORM=offscreen .venv/bin/python -m ours.tools.synced_window_selftest   # image|depth|IMU triplet window
QT_QPA_PLATFORM=offscreen .venv/bin/python -m ours.tools.keypoints_window_selftest # keypoints coloured by depth + per-id trails
```

The imu-reader publishes the **raw** IMU for every frame interval on `imu.raw`
(exactly what the sensor reported) and bundles the **calibrated** IMU
(`gyro − bias`, `a = T·(a_raw − b)`) into the synced `imucam.sample` packet when a
per-device calibration is cached; with none, the packet carries the raw samples.
The live path loads the calibration lazily by device id once the shared OAK-D
opens (`ours/lib/imu/imu_calib.py`).

The calibration math and capture state machines live in `ours/lib/imu/`
(`accel_calib.py`, `calib_collect.py`, `calib_store.py`, `imu_calib.py`) and are
covered by offline self-tests (`accel_calib_selftest`, `calib_collect_selftest`,
`calib_store_selftest`, `imu_calib_selftest`, `ui_calib_selftest`) that run
without an OAK-D.


Both `ours-ba` and `ours-slam` run their heavy optimisation on a background
thread, so the display stays smooth. The accelerometer levels roll/pitch to
gravity at rest, while the **gyroscope** drives the inter-frame rotation prior:
vision (PnP) corrects that rotation weighted by its inlier confidence, *and* by
how far it disagrees with the gyro — so when a fast yaw makes the KLT tracker
slip (PnP still reports inliers but under-rotates) the gyro takes over the
rotation. When vision fails outright during the hardest part of a turn (too few
tracks to even attempt PnP) the gyro still propagates the rotation, so the body
frame keeps turning instead of freezing. On a healthy frame (plenty of inliers,
small disagreement) the fusion collapses to pure vision, so there is no accuracy
cost on good data. Position is still vision-only — this is loosely-coupled VIO,
not Basalt's tight-coupled optimisation.

Because the position is vision-only, three **opt-in `OdometryConfig` guards**
(all off by default → offline gold scoring stays byte-identical; the live
`ours` source turns them on) stop the PnP solver from injecting *phantom*
translation when vision cannot be trusted. Each was tuned by measurement on the
gold sessions, not guessed:

- **`max_translation_speed`** (live 4.0 m/s) — under a hard shake or very fast
  yaw the surviving KLT tracks are low-parallax and PnP reads the rotational
  image flow as a per-frame translation *jump* far larger than any real hand
  motion; integrated, the path wobbles ("đi tàu lượn"). A hand cannot move the
  camera faster than a few m/s, so the per-frame translation is clamped to that
  physical bound (needs the per-frame `dt_s`) — caps only the non-physical
  spikes, real in-budget motion is untouched.
- **`min_inliers_for_translation`** (live 12) — pointing at a textureless
  surface (white wall / blank screen) KLT still fills its corner budget with
  *garbage* corners, so `n_tracks` stays high, but PnP keeps only a handful of
  inliers (measured: white-wall median 0 inliers, p95 11; a real fast push
  median ~140). solvePnP still "succeeds" on the garbage and walks the body off
  in a random direction. Below the gate the translation is **frozen** (rotation
  still tracked by the gyro, position held put — the honest behaviour when
  vision is untrustworthy, same as a covered camera). The gate sits well below
  any real motion (fast-push p25 = 33 inliers), so normal use is untouched; the
  few fast-push frames that dip below it genuinely lost tracking, where freezing
  one frame is correct anyway (measured white-wall path-jitter 4.3 → 2.0,
  fast-push ATE 2.14% → 1.82%).
- **`resolve_translation_on_disagree`** — kept available but **left off live**:
  measured on `push_shake_20s` its disagreement gate fires on only ~8% of frames
  and never zeroed the translation, so it was ineffective; the freeze under hard
  shake is the missing tight-accel term, not this gate.


Tuning knobs (all optional, shown with their defaults):

```bash
# SLAM update cadence — the main lever. Lower = more frequent loop closure AND a
# smaller/cheaper pose graph. More frequent is not strictly better (see below).
./run.sh --source ours-slam --slam-kf-every 5      # insert+loop-detect every N frames
./run.sh --source ours-slam --slam-radius 0        # spatial loop gate (m), 0 = check all

# Keyframe budget for long runs (default off = grows with run time). The motion
# gate bounds the map by TRAJECTORY length instead — a hovering/stationary drone
# stops adding keyframes (cuts ~36% KF on lab_loop, ATE unchanged). Prefer this.
./run.sh --source ours-slam --slam-kf-min-trans 0.10 --slam-kf-min-rot 8
# Absolute safety cap (drops oldest). WARNING: forgets old places, so set it well
# above your largest excursion or loops there can no longer close.
./run.sh --source ours-slam --slam-max-kf 500

# Bundle-adjustment tuning (ours-ba)
./run.sh --source ours-ba --ba-window 6 --ba-kf-every 5 --ba-iters 5
./run.sh --source ours --fps 20                    # camera frame rate (any ours-* source)

# Run lighter at a lower resolution (any ours-* source). Cost scales with the
# pixel count, so half the width = ~1/4 the work. The pipeline auto-scales its
# pixel-unit vision thresholds from the 640x400 baseline; seven per-resolution
# knobs can be overridden to co-tune. See docs/RESOLUTION_TUNING.md.
./run.sh --source ours --width 320 --height 200    # half res, auto-scaled
./run.sh --source ours --width 320 --height 200 --max-corners 240 --klt-win 13

# --width/--height also set the capture resolution of the Visualize windows
# (Camera + IMU, Camera + Depth + IMU), so what they show matches the pipeline.

# Optical flow tracking AND corner detection have our own pure-NumPy
# implementations (pyramidal Lucas-Kanade + Shi-Tomasi, no library). The KLT
# inner loop is JIT-compiled with Numba (optional dep) so our own frontend runs
# in real time live (~15 ms/frame, vs ~140 ms pure-NumPy); without numba it
# falls back to a lighter live preset. The live `ours`/`ours-ba` path uses this
# library-free frontend unconditionally (no cv2, no flag); offline scoring uses
# the full config.
```

Offline scoring of the same backends against the Basalt reference:

```bash
.venv/bin/python ours/tools/vio_run.py --all --backend f2f    # frame-to-frame VO
.venv/bin/python ours/tools/vio_run.py --all --backend ba     # + windowed BA
.venv/bin/python ours/tools/vio_run.py --all --backend slam   # + loop closure
.venv/bin/python ours/tools/vio_run.py --all --backend slam --slam-kf-every 8
.venv/bin/python ours/tools/vio_run.py --all --backend vio    # tight-coupled VIO (experimental)
```

`--backend vio` is the **experimental tight-coupled** path: it folds the IMU
preintegration factors (rotation + velocity + position increments) and the
visual reprojection + depth into ONE joint optimisation per window, solving for
each keyframe's pose, velocity and gyro/accel bias together with the landmarks
(true Basalt style, vs the loosely-coupled gyro fusion above). The math is
validated by self-tests (`imu_preint_selftest.py`, `vio_ba_selftest.py`); on
real gold it currently **regresses vs `ba`** on healthy motion (the dense
finite-difference solver is rougher than the analytic Schur BA, and long
sessions show slow accel/gravity drift -- corridor scale ~1.15). It is opt-in
and touches no production path; `ours/tools/vio_diag.py` A/Bs the IMU factor on/off to
attribute the gap. Closing it needs online gravity-direction estimation, which
is the next step.

**How our pipeline differs from BasaltVIO** (and the ordered roadmap to match
it) is documented in [`docs/OURS_VS_BASALT.md`](docs/OURS_VS_BASALT.md) — read
that first before any tight-coupling work.

Self-tests (run before/after touching the from-scratch VIO):

```bash
.venv/bin/python ours/tools/klt_selftest.py        # our optical flow + corners vs OpenCV
.venv/bin/python ours/tools/ba_selftest.py         # sliding-window BA core
.venv/bin/python ours/tools/posegraph_selftest.py  # SE(3) pose-graph + loop closure
.venv/bin/python ours/tools/imu_preint_selftest.py # IMU preintegration vs closed form
.venv/bin/python ours/tools/vio_ba_selftest.py     # tight-coupled VIO joint solve
.venv/bin/python -m ours.tools.imucam_sync_selftest  # split cam/IMU sync contract (1 pkt/frame, samples in (prev,ts])
.venv/bin/python -m ours.tools.oak_live_selftest     # single-client shared OAK-D (cam+IMU open the device once)
QT_QPA_PLATFORM=offscreen .venv/bin/python -m ours.tools.imucam_window_selftest  # in-app synced view renders (offscreen Qt)
QT_QPA_PLATFORM=offscreen .venv/bin/python -m ours.tools.synced_window_selftest  # image|depth|IMU triplet window renders (offscreen Qt)
QT_QPA_PLATFORM=offscreen .venv/bin/python -m ours.tools.keypoints_window_selftest # keypoints coloured by depth + per-id trails (offscreen Qt)
```

`klt_selftest.py` is the regression guard for the library-free frontend: it
proves correctness against a synthetic known-shift ground truth (independent of
OpenCV), checks our corners + flow agree with OpenCV to sub-pixel, and prints
per-frame timing vs the 20 fps live budget so a performance regression shows up
in the numbers instead of as lag on the device.


## Status

- [x] Project scaffold + dark 3D viewer
- [x] Fake pose source (figure-8) for UI bring-up
- [x] Real visual-inertial odometry from OAK-D (BasaltVIO)
- [x] SLAM with loop closure (RTABMapSLAM)
- [x] From-scratch RGB-D VIO (`ours` f2f → `ours-ba` windowed BA → `ours-slam`
      ORB loop closure + SE(3) pose graph); gravity-leveled, scored vs Basalt in
      `ours/tools/vio_run.py` (corridor ATE 0.61%, see `docs/SKYSLAM_ROADMAP.md`)
- [x] Gyro complementary fusion (loosely-coupled): gyro rotation prior +
      vision correction gated on inliers AND vision/gyro disagreement; gyro
      propagates rotation when vision fails, so fast yaw no longer freezes the
      pose. No-op on well-tracked frames (gold ATE unchanged)
- [~] Tight-coupled VIO core (`ours/lib/backend/vio_window.py`): Forster on-manifold
      IMU preintegration + joint visual-inertial window solve (pose + velocity
      + gyro/accel bias + landmarks), self-test validated, wired offline as the
      opt-in `--backend vio`. Experimental: still regresses vs `ba` on healthy
      gold (rough dense FD solver + long-horizon accel/gravity drift); needs
      online gravity estimation before it replaces the loosely-coupled path
- [x] Own pure-NumPy optical flow (pyramidal Lucas-Kanade, `ours/lib/frontend/klt.py`)
      and corner detection (Shi-Tomasi, `ours/lib/frontend/corners.py`) replacing cv2;
      KLT inner loop JIT-accelerated with Numba (`ours/lib/frontend/klt_numba.py`,
      optional) so the library-free frontend runs live (~15 ms/frame)
- [x] Own library-free PnP (`ours/lib/odometry/pnp.py`: RANSAC DLT + robust-LM seed
      rescue + plain-LS Levenberg-Marquardt refine) replacing
      `cv2.solvePnPRansac` as the default. Measured vs cv2 on gold f2f: better
      on the genuine forward-motion sessions (corridor 0.79->0.77,
      lab_straight 1.11->1.09, push_straight_fast 1.65->1.20) and a wash
      through windowed BA. The live `ours`/`ours-ba` path (own KLT + own PnP)
      and the offline f2f/ba scoring are now fully cv2-free; cv2 is lazily
      imported only for ORB loop closure (`ours-slam`) and the dev-only PnP A/B
      oracle (`OAKD_OWN_PNP=0`)
- [x] Pure-Python PNG codec (`ours/lib/pngio.py`, 8-bit grayscale, all 5 PNG
      filters) for frame IO, replacing `cv2.imread`/`imwrite` (decode verified
      byte-for-byte vs cv2)
- [x] Logging + offline replay (`baseline/tools/record_session.py` + `baseline/tools/viz_session.py`)
- [x] Transparent time-synced (image, depth, IMU) input building block
      (`ours/lib/io/synced.py`) + inspector `ours/tools/synced_view.py` (replay + `--live`:
      image | depth | gyro angular-velocity chart + 3D accel vector)
- [x] Persistent SLAM database (auto save `rtabmap.db` + extract KF/loop via `baseline/tools/extract_kf_from_db.py`)
- [x] Gold regression suite (12 sessions, see `docs/GOLD_SESSIONS.md`)
- [ ] UDP / UART link to flight-controller
- [ ] Tracking-lost UI badge
- [ ] Calibration check tool
- [ ] Port to RPi5
- [ ] `skyslam` Python package (replace Basalt + RTABMap)

## Long-term

- **Software plan (current, research-backed)**: [docs/SKYSLAM_RESEARCH.md](docs/SKYSLAM_RESEARCH.md)
  — plan v3 with 9 phases, `numpy + opencv + gtsam + pyDBoW3`, acceptance gates,
  written after a thorough read of depthai-core / basalt / rtabmap / ORB-SLAM3 /
  OpenVINS source code.
- **Pipeline checkpoints (debug contract)**: [docs/PIPELINE_CHECKPOINTS.md](docs/PIPELINE_CHECKPOINTS.md)
  — schema C0–C9 used to compare skyslam against the baseline while building.
- **Hardware vision (long-term)**: [docs/SKYSLAM_ROADMAP.md](docs/SKYSLAM_ROADMAP.md)
  — read only the HW V1 / V2 / FC link parts (Section 3 software architecture
  has been superseded by SKYSLAM_RESEARCH.md).
