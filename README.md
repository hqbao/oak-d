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
thread, so the display stays smooth; gravity (accel) levels roll/pitch as the
final step while vision owns yaw + position. Tuning knobs (all optional, shown
with their defaults):

```bash
# SLAM update cadence — the main lever. Lower = more frequent loop closure AND a
# smaller/cheaper pose graph. More frequent is not strictly better (see below).
./run.sh --source ours-slam --slam-kf-every 5      # insert+loop-detect every N frames
./run.sh --source ours-slam --slam-radius 0        # spatial loop gate (m), 0 = check all

# Bundle-adjustment tuning (ours-ba)
./run.sh --source ours-ba --ba-window 6 --ba-kf-every 5 --ba-iters 5
./run.sh --source ours --fps 20                    # camera frame rate (any ours-* source)

# Optical flow tracking AND corner detection are our own pure-NumPy
# implementations by default (pyramidal Lucas-Kanade + Shi-Tomasi, no library).
# They are ~25x slower than cv2, so for a smoother live display you can opt back
# into cv2 (offline ATE is unaffected either way):
./run.sh --source ours --cv2-klt                   # use cv2 flow + corners (faster live)
```

Offline scoring of the same backends against the Basalt reference:

```bash
.venv/bin/python tools/vio_run.py --all --backend f2f    # frame-to-frame VO
.venv/bin/python tools/vio_run.py --all --backend ba     # + windowed BA
.venv/bin/python tools/vio_run.py --all --backend slam   # + loop closure
.venv/bin/python tools/vio_run.py --all --backend slam --slam-kf-every 8
```


## Status

- [x] Project scaffold + dark 3D viewer
- [x] Fake pose source (figure-8) for UI bring-up
- [x] Real visual-inertial odometry from OAK-D (BasaltVIO)
- [x] SLAM with loop closure (RTABMapSLAM)
- [x] From-scratch RGB-D VIO (`ours` f2f → `ours-ba` windowed BA → `ours-slam`
      ORB loop closure + SE(3) pose graph); gravity-leveled, scored vs Basalt in
      `tools/vio_run.py` (corridor ATE 0.61%, see `docs/SKYSLAM_ROADMAP.md`)
- [x] Own pure-NumPy optical flow (pyramidal Lucas-Kanade, `oakd/vio/klt.py`)
      and corner detection (Shi-Tomasi, `oakd/vio/corners.py`) replacing cv2;
      default on, `--cv2-klt` falls back for faster live display
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
