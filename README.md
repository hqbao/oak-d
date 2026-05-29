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
      depthai_vo.py    real stereo-inertial VIO (dai.node.BasaltVIO)
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

## Status

- [x] Project scaffold + dark 3D viewer
- [x] Fake pose source (figure-8) for UI bring-up
- [x] Real visual-inertial odometry from OAK-D (BasaltVIO)
- [x] SLAM with loop closure (RTABMapSLAM)
- [ ] UDP / UART link to flight-controller
- [ ] Persistent SLAM database (save / reload map across sessions)
- [ ] Tracking-lost UI badge
- [ ] Logging + offline replay source
- [ ] Calibration check tool
- [ ] Port to RPi5

## Long-term

See [docs/SKYSLAM_ROADMAP.md](docs/SKYSLAM_ROADMAP.md) for the plan to rewrite
the SLAM stack from scratch in C99 targeting custom hardware (no Basalt /
RTAB-Map / depthai at runtime).
