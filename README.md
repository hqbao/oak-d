# oak-d

Companion-computer project to turn an **OAK-D W** stereo camera into a 6-DoF
position source for the flight-controller. Runs on a Mac mini today, will move
to a Raspberry Pi 5 later.

```
oak-d/
  oakd/                package
    frames.py          camera <-> body (FRD) <-> world (NED) transforms
    pose.py            Pose dataclass + ring buffer
    sources/           pose providers
      base.py          PoseSource ABC
      fake.py          procedural figure-8 trajectory (UI bring-up)
      depthai_vio.py   real stereo-inertial VIO (dai.node.BasaltVIO)
      depthai_slam.py  VIO + SLAM with loop closure (BasaltVIO + RTABMapSLAM)
    ui/
      theme.py         military dark palette (mirrors flight-controller/_ui.py)
      viewer3d.py      pyqtgraph GLViewWidget — trajectory, drone triad, grid
      panels.py        telemetry side-panel
      mainwindow.py    top-level QMainWindow + toolbar (view presets)
  tools/
    view_pose3d.py       live 3D viewer from OAK over USB (run.sh entry)
    record_session.py    dump C0..C6 + PCL from a live run to sessions/<name>/
    viz_session.py       offline multi-tab replay of a recorded session
    compare_sessions.py  ATE/RPE between two pose streams (VIO vs SLAM, etc.)
    baseline_report.py   scan sessions/gold/, emit Markdown baseline report
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

```bash
./run.sh                       # launches viewer with fake pose source
./run.sh --source fake         # explicit
./run.sh --source oak          # real VIO (BasaltVIO, low latency, no loop closure)
./run.sh --source slam         # VIO + SLAM with loop closure (BasaltVIO + RTABMapSLAM)
```

### From-scratch VIO/SLAM sources (our own pipeline, replacing Basalt)

```bash
./run.sh --source ours         # our frame-to-frame RGB-D PnP VO
./run.sh --source ours-ba      # + sliding-window bundle adjustment (depth-anchored)
./run.sh --source ours-slam    # + ORB loop closure + SE(3) pose graph (full SLAM)
```

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

# Optical flow tracking AND corner detection have our own pure-NumPy
# implementations (pyramidal Lucas-Kanade + Shi-Tomasi, no library). The KLT
# inner loop is JIT-compiled with Numba (optional dep) so our own frontend runs
# in real time live (~15 ms/frame, vs ~140 ms pure-NumPy); without numba it
# falls back to a lighter live preset. Offline scoring uses the full config.
# The live viewer defaults to cv2; pass --own-klt to run our own frontend live:
./run.sh --source ours --own-klt                   # library-free frontend, live
```

Offline scoring of the same backends against the Basalt reference:

```bash
.venv/bin/python tools/vio_run.py --all --backend f2f    # frame-to-frame VO
.venv/bin/python tools/vio_run.py --all --backend ba     # + windowed BA
.venv/bin/python tools/vio_run.py --all --backend slam   # + loop closure
.venv/bin/python tools/vio_run.py --all --backend slam --slam-kf-every 8
.venv/bin/python tools/vio_run.py --all --backend vio    # tight-coupled VIO (experimental)
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
and touches no production path; `tools/vio_diag.py` A/Bs the IMU factor on/off to
attribute the gap. Closing it needs online gravity-direction estimation, which
is the next step.

Self-tests (run before/after touching the from-scratch VIO):

```bash
.venv/bin/python tools/klt_selftest.py        # our optical flow + corners vs OpenCV
.venv/bin/python tools/ba_selftest.py         # sliding-window BA core
.venv/bin/python tools/posegraph_selftest.py  # SE(3) pose-graph + loop closure
.venv/bin/python tools/imu_preint_selftest.py # IMU preintegration vs closed form
.venv/bin/python tools/vio_ba_selftest.py     # tight-coupled VIO joint solve
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
      `tools/vio_run.py` (corridor ATE 0.61%, see `docs/SKYSLAM_ROADMAP.md`)
- [x] Gyro complementary fusion (loosely-coupled): gyro rotation prior +
      vision correction gated on inliers AND vision/gyro disagreement; gyro
      propagates rotation when vision fails, so fast yaw no longer freezes the
      pose. No-op on well-tracked frames (gold ATE unchanged)
- [~] Tight-coupled VIO core (`oakd/vio/vio_window.py`): Forster on-manifold
      IMU preintegration + joint visual-inertial window solve (pose + velocity
      + gyro/accel bias + landmarks), self-test validated, wired offline as the
      opt-in `--backend vio`. Experimental: still regresses vs `ba` on healthy
      gold (rough dense FD solver + long-horizon accel/gravity drift); needs
      online gravity estimation before it replaces the loosely-coupled path
- [x] Own pure-NumPy optical flow (pyramidal Lucas-Kanade, `oakd/vio/klt.py`)
      and corner detection (Shi-Tomasi, `oakd/vio/corners.py`) replacing cv2;
      KLT inner loop JIT-accelerated with Numba (`oakd/vio/klt_numba.py`,
      optional) so the library-free frontend runs live (~15 ms/frame)
- [x] Logging + offline replay (`tools/record_session.py` + `tools/viz_session.py`)
- [x] Persistent SLAM database (auto save `rtabmap.db` + extract KF/loop via `tools/extract_kf_from_db.py`)
- [x] Gold regression suite (6 sessions, see `docs/GOLD_SESSIONS.md`)
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
