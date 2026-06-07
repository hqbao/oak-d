# 4-Process Live Architecture

> **Status:** design (decided 2026-06-07 by user). Replaces the single-process
> live path under `ours.app.build_live` with four cooperating processes that
> communicate over a stdlib-only IPC layer. The OFFLINE / replay path
> (`ours.app.run_replay` + `flow_replay_selftest`) keeps the single-process
> codepath unchanged — determinism + byte-identical output depend on it.

## 1. Motivation

The single-process live graph already worked (out-of-process *engine* for the
heavy BA/SLAM solve, in-process Bus for everything else), but it has three
limits we want gone:

1. **One blow-up kills everything.** A Qt UI crash or a SLAM solver wedge
   takes the whole pipeline down (incl. the device pipeline, which then trips
   the OAK-D firmware watchdog and forces a USB replug).
2. **VIO and SLAM share a Python interpreter** → still pay for incidental GIL
   sharing whenever a Python step in one of them runs. `SubprocessEngine` only
   moves the inner solve out — the flow tasks (`RunBA`, `SlamStep`, the bus
   forwarders) still run in the main interpreter.
3. **Calibration / visualisation tools steal the device.** Today every wizard
   first stops the VIO source (the OAK-D is single-client) and reopens its own
   depthai pipeline. With a dedicated **capture** process that owns the device
   forever, the wizards subscribe to its stream and never touch the link.

## 2. Process layout (the decisions)

Four long-lived processes, plus transient tool processes that come and go:

| Process    | Owns                                  | Subscribes (IPC)               | Publishes (IPC) |
|---         |---                                    |---                             |---|
| `capture`  | OAK-D device + `CamFlow` + `ImuCamFlow` | —                            | `cam.sync`, `imu.raw`, `imucam.sample`, `frame.depth`, `calib.bundle` |
| `vio`      | `OdometryFlow` + `BackendFlow`         | `imucam.sample`, `frame.depth`, `calib.bundle` | `pose.odom`, `keyframe`, `frame.tracks`, `frame.inliers`, `pose.refined`, `vio.map` |
| `slam`     | `SlamFlow` (loop closure + pose graph) | `keyframe`, `calib.bundle`     | `loop.correction`, `slam.map` |
| `ui`       | Qt `MainWindow`, tabs, viewer3D        | every read-only topic above    | — (sink) |

Transient (started on-demand from the UI menu):

| Process / runner         | Subscribes                                     | Notes |
|---                       |---                                             |---|
| `tool.imu_stream`        | `imu.raw`                                      | gyro / accel calibration wizards |
| `tool.synced_window`     | `frame.depth`, `imucam.sample`                 | image+depth+IMU triplet |
| `tool.keypoints_window`  | `frame.depth`, `frame.tracks`, `frame.inliers` | keypoint-depth viewer |
| `tool.imucam_window`     | `imucam.sample`                                | camera + IMU live view |

The crucial design rule: **the tools no longer open the OAK-D**. They open a
read-only subscription to the capture process's IpcBus, exactly the way the UI
process does. Nobody fights `capture` for the device.

### 2.1 Decisions captured (asked + answered 2026-06-07)

| Question | Decision |
|---|---|
| Who owns the device? | Dedicated `capture` process (4 procs total). |
| What is "VIO's own map"? | VIO = frame-to-frame PnP + windowed BA (`BackendFlow`). |
| IPC mechanism? | `multiprocessing.Queue` for metadata + `shared_memory` ring for images. |
| UI display modes? | Two separate tabs (VIO tab, SLAM tab), each with its own viewer. |
| Calib / visualise tools? | Subscribe to capture's stream via IPC; don't open the device. |
| Offline replay? | **Stays single-process** — determinism + byte-identical output. |

## 3. IPC layer — `ours/lib/ipc/`

A thin substrate that exposes the same `publish(topic, msg)` / `subscribe(topic, handler)`
API as the in-process `Bus`, but works across processes. Everything is **stdlib only**
(`multiprocessing`, `multiprocessing.connection`, `multiprocessing.shared_memory`,
`pickle`, `struct`). The existing flows do not change; they still use the
in-process `Bus` inside their own process — a tiny **bridge flow** wires the
in-proc Bus to the IpcBus at the process boundary.

### 3.1 `ours/lib/ipc/shared_array.py` — `SharedArrayRing`

A fixed-shape, fixed-dtype ring of `N` shared-memory slots for one stream
(e.g. one ring for `gray_left`, one for `depth_m`). The producer rotates
`slot = seq % N`; consumers read by slot index out of metadata. Subscribers
who need to hold the array beyond one frame must copy it (cheap: a single
numpy copy of one frame is ~0.1 ms for 640×400).

- `N` is sized so a moderate consumer backlog can never wrap around (default
  `N=8` at 20 fps = 0.4 s of slack — well above the 50 ms / 60 ms latest-only
  inbox cadence used downstream).
- No locks: rotation is single-producer single-cursor. The consumer's
  responsibility is to read fast or copy out. Worst case: a stuck consumer
  reads stale frames, but the live latest-only sinks already drop those.

### 3.2 `ours/lib/ipc/bus.py` — `IpcBus`

Pub/sub over `multiprocessing.connection.Listener` (Unix-domain socket on
macOS/Linux). One central socket per *publisher process*. Every subscriber
connects with a `SUBSCRIBE([topic, ...])` handshake; the publisher then
forwards each `publish` to all matching connections via `Connection.send`
(pickle under the hood). Big numpy arrays are **never** pickled — they ride
the `SharedArrayRing` and the wire-message carries only `(slot, shape, dtype, ts)`.

The API mirrors `ours.lib.flow.pubsub.Bus`:

```python
bus = IpcServerBus(endpoint="ipc.capture")          # publisher side
bus.publish("imucam.sample", IpcImuCamRef(seq, ts, slot_gray, slot_right, imu_ts, gyro, accel))
```

```python
bus = IpcClientBus(endpoint="ipc.capture")          # subscriber side
bus.subscribe("imucam.sample", lambda ref: ...)
bus.start()
```

### 3.3 `ours/lib/ipc/messages.py` — wire messages

For every existing message type that crosses a process boundary, a sibling
wire-message exists carrying only:

- POD fields (`seq`, `ts_ns`, ids, etc.)
- `SharedArrayRef(ring_name, slot, shape, dtype)` for every large array.

The receiving bridge flow re-hydrates by copying from shared memory back into
a regular `np.ndarray` and constructing the in-proc dataclass (`ImuCamPacket`,
`DepthFrame`, `Keyframe`, ...). The wire messages live next to the in-proc
ones so the contract is visible in one place.

## 4. Bridge flows — `ours/flows/bridge/`

The bridge keeps the existing flows unchanged. Each side has one tiny flow:

- **`IpcPublisherFlow`** — subscribes to N in-proc topics, copies the payload
  into a shared-memory slot (if it has arrays), wraps it in the matching wire
  message, and `IpcServerBus.publish`es it. One per process boundary.
- **`IpcSubscriberFlow`** — subscribes (via `IpcClientBus`) to topics on a
  remote publisher, re-hydrates wire messages into in-proc dataclasses, and
  publishes them on the local in-proc Bus. Other flows in this process
  consume from the in-proc Bus exactly as before.

The whole IPC layer is therefore invisible to `OdometryFlow`, `BackendFlow`,
`SlamFlow`, the UI sinks, and every existing self-test.

## 5. Process entry points — `ours/proc/`

One module per process, each exposes a `main()` so it can be spawned as a
standalone Python process.

### 5.1 `ours/proc/capture.py`

```
LocalBus
  ├── CamFlow (LiveCamSource or ReplayCamSource)
  ├── ImuCamFlow (LiveImuSource or ReplayImuSource)
  └── IpcPublisherFlow → IpcServerBus(endpoint="oak.capture")
      └── publishes: cam.sync, imu.raw, imucam.sample, frame.depth, calib.bundle
```

`calib.bundle` is a one-shot retained message: when a new subscriber connects
it gets the latest cached bundle immediately (so VIO / SLAM can boot without
guessing). Re-published on device re-open.

### 5.2 `ours/proc/vio.py`

```
IpcClientBus(endpoint="oak.capture")
  └── IpcSubscriberFlow → LocalBus
        ├── OdometryFlow
        ├── BackendFlow            (worker=False — solve in-process here is fine;
        │                           this whole process is already "out-of-main")
        └── IpcPublisherFlow → IpcServerBus(endpoint="oak.vio")
              └── publishes: pose.odom, keyframe, frame.tracks, frame.inliers,
                             pose.refined, vio.map
```

`vio.map` is the windowed-BA refined-keyframe trajectory (the same payload
`Engine.poll_overlay()` produces today), published as a periodic snapshot.

### 5.3 `ours/proc/slam.py`

```
IpcClientBus(endpoint="oak.vio")
  └── IpcSubscriberFlow → LocalBus
        ├── SlamFlow              (worker=False; this process owns the heavy solve)
        └── IpcPublisherFlow → IpcServerBus(endpoint="oak.slam")
              └── publishes: loop.correction, slam.map
```

SLAM subscribes to `keyframe` **from VIO** (not capture), so SLAM never sees a
keyframe VIO hasn't already accepted. The pose-graph is SLAM's own map.

### 5.4 `ours/proc/ui.py`

```
IpcClientBus(endpoint="oak.capture")   # imu.raw, imucam.sample, frame.depth (for tools)
IpcClientBus(endpoint="oak.vio")       # pose.odom, keyframe, tracks, inliers, refined, vio.map
IpcClientBus(endpoint="oak.slam")      # loop.correction, slam.map
  └── IpcSubscriberFlows → LocalBus → Qt MainWindow
        ├── Viewer tab "VIO"  : marker = pose.odom, map = vio.map
        └── Viewer tab "SLAM" : marker = pose.odom + loop.correction, map = slam.map
```

The Qt main thread sees only the in-proc Bus, so the existing UI sinks
(`UiRenderFlow`, `UiTracksFlow`, `UiTripletFlow`) are reused unchanged.

## 6. UI changes — `ours/ui/`

Minimal surface area:

- **Tabs.** `mainwindow.py` adds a `QTabWidget` with two viewers: one bound to
  a `VioPoseSource`, one to a `SlamPoseSource`. Each source consumes the same
  in-proc Bus but listens on different topics + applies the correct map overlay.
- **Live source split.** `live_source.py` `FlowPoseSource` is replaced by
  `VioPoseSource` (reads `pose.odom` + `vio.map`) and `SlamPoseSource` (reads
  `pose.odom` with `loop.correction` applied + `slam.map`). Both are thin —
  they consume from the local Bus and push `Pose` into the viewer.
- **Calibration dialogs.** `ImuStream` becomes a thin wrapper around an
  `IpcClientBus` subscription to `imu.raw` (no depthai import in the UI proc).
  Falls back to its old in-proc behaviour if no capture endpoint exists, so the
  offline UI bring-up `fake` source keeps working.
- **Visualisation windows.** `synced_window` / `keypoints_window` / `imucam_window`
  workers replace their `build_live_frontend` call with an `IpcClientBus`
  subscription. No second device, no second SGM (depth already published by
  capture). This is the cleanup the original "honest pipeline visualisation"
  memory called for.

## 7. Launcher — `ours/proc/launcher.py` + `run.sh`

`launcher.py main()` spawns the three background processes (capture → vio →
slam, in that order so each subscriber boots after its publisher's endpoint
exists), waits a few hundred ms between each, then `exec`s the UI process in
the foreground. On UI exit it sends `SIGTERM` to the three background processes
and joins them; on any of them dying it shuts the others down with a clear
diagnostic.

`run.sh` gains:
- `./run.sh ...` — unchanged single-process live (default; the existing
  `ours.tools.view_pose3d` path stays for one release as a fallback).
- `./run.sh --proc ...` — new 4-process launcher.

The offline `ours.app --session ...` path is untouched.

## 8. Testing

| Test | Scope | Status |
|---|---|---|
| `ours.tools.ipc_bus_selftest` | `SharedArrayRing` roundtrip + wrap, `IpcServerBus`/`IpcClientBus` 2-proc echo, `IpcPublisherFlow → IpcSubscriberFlow` byte-for-byte `ImuCamPacket` roundtrip. | PASSING |
| `ours.tools.proc4_replay_selftest` | All four processes (capture + vio + slam + headless UI sink) spawned against a recorded session. Asserts identical pose.odom count / density to the single-proc `flow_replay_selftest`, plus expected keyframe + refined counts and clean END propagation. | PASSING |
| `ours.tools.proc4_ui_selftest` | Same 4-proc spawn but drives `IpcPoseSource` + `SlamMapTracker` + a Qt `MainWindow` construction. Catches GUI-side regressions without needing a display in the event loop. | PASSING |
| Existing `flow_replay_selftest` + every `_selftest` | **Unchanged.** Single-process path is the reference and must stay green. | PASSING |

Verified end-to-end (`proc4_replay_selftest` against `sessions/gold/lab_loop_30s`):
- 30 frames → 30 odom, 6 keyframes, 4 refined poses
- 60 frames → 60 odom, 12 keyframes, 10 refined poses
- Identical to single-proc `flow_replay_selftest` (60 odom, 10 refined).

## 9. Invariants

1. The IPC layer is stdlib-only. No new pip deps.
2. Existing flows (`OdometryFlow`, `BackendFlow`, `SlamFlow`, every UI sink)
   are reused unchanged. The bridge flow is the only new flow type.
3. The offline replay path (`ours.app`, `flow_replay_selftest`) stays
   byte-identical and single-process.
4. Tools never open the OAK-D. Capture is the only owner of the device.
5. No process holds another process's data (every numpy array crossing the
   bridge is copied out of shared memory on the receiving side before any
   downstream task runs).
6. **Ring slots > IPC outbox capacity.** A wire message in an outbox
   references a ring slot the producer must NOT have overwritten by the time
   the consumer reads it. Default slots=64 strictly exceeds outbox cap=32
   so a publisher that's outbox-full is at most 32 frames ahead of a
   consumer, and the 32 outbox-queued items reference slots `[N-32, N-1] mod
   64` — all distinct from the producer's next-write slot `N mod 64`. See
   `default_capture_specs` / `default_vio_specs` and
   `SharedArrayRing.create` for the assertion.
7. **Drain before stop.** `Flow.stop()` checks `_stop` at the TOP of every
   loop iteration and discards any items still queued. So a process that
   wants to publish END must wait for the flow's `done` event (set inside
   `_handle_end` after `expected_ends` ENDs have been processed) BEFORE
   calling `stop()`. Capture waits on `imu_flow.done`; VIO on `odom.done` +
   `backend.done`; SLAM on `slam.done`. Discovered the hard way -- without
   it, capture lost 4 of 5 frames.
8. **IpcServerBus.close drains outbox.** `close()` sets `_stopped` to gate
   new publishes, then puts BYE on each subscriber's outbox and joins the
   fanout thread (which sends every pending wire message in order, then BYE,
   then exits). `state.alive` is ONLY flipped by send-errors -- close does
   NOT flip it. Older code that flipped alive in close caused the fanout to
   discard everything queued at close-time.
9. **`SharedMemory(track=False)` on attach.** The attaching process must
   not register the shm with its own resource_tracker (the creator does).
   `SharedArrayRing.attach` passes `track=False` (Python ≥ 3.13). Without
   it, the attacher prints spurious "leaked shared_memory" warnings on exit
   even though only the creator should unlink. See
   https://bugs.python.org/issue38119.
10. **Readiness barrier = retained calib.bundle re-publish.** VIO subscribes
    to capture's retained `calib.bundle`, then re-publishes the SAME bundle
    on its own retained endpoint AFTER allocating its kf_* rings. SLAM (and
    the UI) wait on VIO's calib bundle as a "VIO is ready, rings exist"
    signal. Without this barrier, downstream procs race the ring creation
    and fail with `FileNotFoundError`.

## 10. Migration order

The phases match the todo list. Each is shippable on its own. **Done items
in bold.**

1. **IPC primitives** + selftest. No process changes yet.
2. **Bridge flows** + selftest. Still single-process; bridge is exercised by
   a unit test that wires both ends in-process.
3. **Capture process** + smoke test (replay backend → IPC publish).
4. **VIO process** + smoke test (replay capture → VIO → collect pose.odom).
5. **SLAM process** + smoke test (replay capture → VIO → SLAM → collect).
6. **UI process** + tabs + `IpcPoseSource` + `SlamMapTracker`.
7. Calib / visualize tools rewired to the IPC capture stream (Phase 10).
8. **`run.sh --proc`** launcher (`ours.proc.launcher`). Old single-process
   path stays as the default `./run.sh` for one release, removed only after
   live validation.
