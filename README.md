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
      depthai_vo.py    TODO: real visual-inertial odometry from OAK-D
    ui/
      theme.py         military dark palette (mirrors flight-controller/_ui.py)
      viewer3d.py      pyqtgraph GLViewWidget — trajectory, drone triad, grid
      panels.py        telemetry side-panel
      mainwindow.py    top-level QMainWindow + toolbar (view presets)
  legacy/              reference DepthAI scripts from rtr-research (Apr 2025)
  tools/
    view_pose3d.py     entry — launches the 3D viewer
  run.sh
  requirements.txt
```

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
./run.sh --source oak          # not implemented yet (Phase 2)
```

## Status

- [x] Project scaffold + dark 3D viewer
- [x] Fake pose source (figure-8) for UI bring-up
- [ ] Real visual-inertial odometry from OAK-D
- [ ] UDP / UART link to flight-controller
- [ ] Port to RPi5
