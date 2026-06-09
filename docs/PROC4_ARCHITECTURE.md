# Live Architecture — the 5-project split (4 live processes)

> **Status:** shipped. The single-process `ours/` monolith was split into FIVE
> independent projects (`imu_camera`, `depth`, `vio`, `slam`, `ui`) + a `launcher`
> + a `verification` harness. The DepthAI/Basalt reference (`baseline/`) is kept
> for ATE comparison. End-to-end byte-parity vs the pre-split baseline is
> `gap = 0`, verified live on a real OAK-D.
>
> **Runtime = 4 processes** (`imu_camera`, `vio`, `slam`, `ui`). Depth runs INLINE
> on the capture process's `imu_cam` thread, so the launcher never spawns a depth
> process; `depth/` is an independent SOURCE TREE, promotable to a 5th process via
> its own `depth.main` harness.
>
> The OFFLINE / replay byte-parity oracle is **in-process** (single `LocalPubSub`,
> no `IPCPubSub`) and lives in `verification/` — determinism + byte-identical
> output depend on it staying single-process. (The filename keeps the historical
> "PROC4" name; the architecture is the 5-project split.)

## 1. Motivation

The pre-split single-process live graph already worked, but it had three limits:

1. **One blow-up kills everything.** A Qt UI crash or a SLAM solver wedge took the
   whole pipeline down (incl. the device pipeline, which then tripped the OAK-D
   firmware watchdog and forced a USB replug).
2. **VIO and SLAM shared a Python interpreter** → still paid for incidental GIL
   sharing whenever a Python step in one of them ran. Moving the inner solve
   out-of-process only moved the inner solve; the reactive modules still ran in the
   main interpreter.
3. **Calibration / visualisation tools stole the device.** Every wizard first
   stopped the VIO source (the OAK-D is single-client) and reopened its own depthai
   pipeline. With a dedicated **capture** process (`imu_camera`) that owns the
   device forever, the wizards subscribe to its stream and never touch the link.

## 2. Process layout (the decisions)

Four long-lived processes, plus transient tool processes that come and go:

| Process      | Owns                                          | Subscribes (IPC)               | Publishes (IPC) |
|---           |---                                            |---                             |---|
| `imu_camera` | OAK-D device + cam/IMU sync + IMU calib + **inline SGM depth** | —              | `cam.sync`, `imu.raw`, `imucam.sample`, `frame.depth`, `calib.bundle` |
| `vio`        | RGB-D PnP odometry + windowed BA              | `imucam.sample`, `frame.depth`, `calib.bundle` | `pose.odom`, `pose.vo` (pure-vision, LIVE-only), `keyframe`, `frame.tracks`, `frame.inliers`, `pose.refined` |
| `slam`       | ORB loop closure + SE(3) pose graph          | `keyframe`, `calib.bundle` (from VIO) | `loop.correction` (loop-event rewrite) **and** `slam.map` (continuous keyframe overlay, LIVE-only) |
| `ui`         | Qt `MainWindow`, single 5-trajectory Viewer3D + View/Visualize/Calibration menus | `pose.odom`, `pose.vo`, `pose.refined`, `calib.bundle` (vio); `slam.map`, `calib.bundle` (slam); on-demand: `imucam.sample`, `frame.depth`, `imu.raw` (capture) + `frame.tracks`, `frame.inliers` (vio) | — (sink) |

> The capture process is named `imu_camera`; its endpoint is `oak.capture` and its
> entrypoint is `imu_camera.main`. Throughout this doc "capture" = `imu_camera`.

```mermaid
flowchart LR
    subgraph cap["imu_camera (capture) · oak.capture"]
        CAM[read_cam] --> SYNC[imu_cam: sync + IMU calib]
        SYNC --> DEPTH["depth steps (SGM, INLINE)"]
    end
    subgraph vio["vio · oak.vio"]
        ODOM[OdometryModule: KLT + RGB-D PnP + gyro] --> BA[BackendModule: windowed BA]
    end
    subgraph slam["slam · oak.slam"]
        SLM[SlamModule: ORB loop closure + SE3 PGO]
    end
    subgraph ui["ui (foreground)"]
        VIEW["Viewer3D — VO / VIO / VIO-BA / SLAM-corr / SLAM"]
    end

    cap -- "imucam.sample · frame.depth · calib.bundle" --> vio
    vio -- "keyframe · calib.bundle" --> slam
    vio -- "pose.odom · pose.vo · pose.refined · calib.bundle" --> ui
    slam -- "slam.map · calib.bundle" --> ui
    cap -. "imu.raw · imucam.sample · frame.depth (on-demand)" .-> ui
    vio -. "frame.tracks · frame.inliers (on-demand)" .-> ui
```

The UI's Visualize / Calibration windows are **not** separate transient processes:
they run **in the UI process** as plain child `QMainWindow` / modal dialogs, fed
over IPC by the adapters in `ui/modules/ipc_sources.py` (see §6). Each adapter opens
its own read-only `IPCPubSub(role="client")` subscription on demand and tears it
down when the window/dialog closes — the same read-only subscription pattern the
long-lived trajectory sources use:

| In-UI window / dialog         | Adapter (`ui/modules/ipc_sources.py`) | Subscribes (endpoint · topics) |
|---                            |---                            |---|
| Gyro / Accel calibration      | `IpcImuRawSource`             | capture · `imu.raw` (RAW IMU) |
| Camera + Depth + IMU triplet  | `IpcTripletWorker`           | capture · `imucam.sample`, `frame.depth` |
| Keypoint Depth Tracker        | `IpcKeypointWorker`          | capture · `frame.depth`  +  vio · `frame.tracks`, `frame.inliers` |
| Gyro Fusion (strip chart)     | `IpcGyroFuseSource`          | vio · `frame.gyrofuse` |
| SLAM Map (landmarks)          | `IpcSlamMapSource`           | vio · `keyframe` (gray/depth/`track_ids`/`track_px`/`inlier_ids` via VIO's kf rings) + slam · `slam.map` (corrected poses) |
| Floor Plan (top-down)         | `IpcFloorPlanSource`         | vio · `keyframe` (depth via VIO's kf rings) |

The crucial design rule: **nothing but `imu_camera` opens the OAK-D**. The UI never
fights capture for the device, and the UI process imports no depthai — it is
device-agnostic by contract.

### 2.1 Decisions captured

| Question | Decision |
|---|---|
| Who owns the device? | Dedicated `imu_camera` (capture) process. |
| What is "VIO's own map"? | VIO = frame-to-frame PnP + windowed BA (`BackendModule`). |
| IPC mechanism? | `IPCPubSub` over a Unix-domain socket for metadata + `SharedArrayRing` shared memory for images. The wire is the class-path-independent `comms.codec`, NOT pickle. |
| UI display modes? | A SINGLE `Viewer3D` (no tabs) drawing 5 toggleable trajectory lines: VO / VIO / VIO-BA / SLAM-corrected VIO / SLAM. |
| Calib / visualise tools? | Subscribe to capture's stream via IPC; don't open the device. |
| Offline replay? | **Stays single-process, in-process** — determinism + byte-identical output (the `verification/` oracle). |

## 3. The comms contract — `<project>/comms/`

Every project vendors a **byte-identical** `comms/` package (single source of truth
= `imu_camera/comms/`; a `diff -r` CI gate keeps `depth`/`vio`/`slam`/`ui`/`launcher`
in lock-step). It is the merge + rename of the pre-split runtime layer; **the word
"flow" is gone**, the topic strings are unchanged. It imports no depthai / no PyQt6
/ no cv2 (headless-safe), and all internal imports are RELATIVE, so it drops into
any project unchanged. Full byte layout + rename map:
[`imu_camera/comms/README.md`](../imu_camera/comms/README.md).

| New name | Was | Role |
|---|---|---|
| `LocalPubSub` (`pubsub.py`) | `Bus` | in-process pub/sub — passes Python objects **directly** (zero serialization). |
| `IPCPubSub(endpoint, role=…)` (`ipc.py`) | `IpcServerBus` + `IpcClientBus` | cross-process pub/sub over a Unix-domain socket. |
| `Module` / `SourceModule` / `ModuleContext` (`module.py`) | `Flow` / `SourceFlow` / `FlowContext` | threaded reactive substrate. |
| `Step` (`step.py`) | `Task` | smallest input→output stage. |
| `IPCPublisher` / `IPCSubscriber` (`bridge.py`) | `IpcPublisherFlow` / `IpcSubscriberFlow` | bridge a `LocalPubSub` ↔ an `IPCPubSub` at a process boundary. |
| `SharedArrayRing` / `SharedArrayRef` (`shared_array.py`) | (unchanged binary layout) | single-segment shared-memory ring for large arrays. |

```python
# publisher side
bus = IPCPubSub(endpoint="oak.capture", role="server")
bus.publish("imucam.sample", wire_msg)        # wire_msg encoded via comms.codec
```
```python
# subscriber side
bus = IPCPubSub(endpoint="oak.capture", role="client")
bus.subscribe("imucam.sample", lambda ref: ...)
bus.start()
```

### 3.1 `comms/shared_array.py` — `SharedArrayRing`

A fixed-shape, fixed-dtype ring of `N` slots for one stream (e.g. one ring for
`gray_left`, one for `depth_m`). The ring is backed by **ONE `SharedMemory`
segment** named exactly `{name}` (no per-slot `.{i}` suffix), sized `N * nbytes`;
slot `i` is the byte-offset window `[i*nbytes : (i+1)*nbytes]` (a numpy offset-view).
The producer rotates `slot = seq % N`; consumers read by slot index out of the wire
metadata. Subscribers who need to hold the array beyond one frame must copy it
(cheap: ~0.1 ms for 640×400).

- The single-segment layout keeps the open-file-descriptor cost a **small constant
  per ring, independent of `N`** (CPython's `SharedMemory(create=True)` holds ~2 fds
  per segment on macOS). The earlier design used one segment per slot, so fd cost
  scaled linearly with slots — a capture process attaching 3 rings × 64 slots tripped
  macOS's 256-fd default (`shm_open` → EMFILE) at boot. Total RAM is identical.
- `N` is sized so a moderate consumer backlog can never wrap around.
- No locks: rotation is single-producer single-cursor. The live latest-only sinks
  already drop stale frames.

### 3.2 The wire codec — `comms/codec.py` (replaces pickle)

`pickle` bakes the publisher's **module path** into the bytes, so a decoder in a
*different* vendored copy (`imu_camera.comms.wire.WirePoseMsg` vs
`vio.comms.wire.WirePoseMsg`) could fail to resolve or mismatch identity. The codec
is keyed by **`(topic → Wire* class, dataclass-field-ORDER)`** from
`wire.TOPIC_WIRE` — never the module path — so any copy decodes any other copy's
bytes bit-identically, into the *decoder's own* `Wire*` type. The only wire change
vs the pre-split layer:

```
# OLD (implicit pickle):  conn.send(("M", topic, msg)) / conn.recv()
# NEW (raw codec bytes):  conn.send_bytes(codec.encode(topic, msg))
#                         topic, msg = codec.decode(conn.recv_bytes())
```

Big numpy arrays are **never** encoded — they ride the `SharedArrayRing` and the
wire message carries only `SharedArrayRef(ring_name, slot, shape, dtype)`. Public
API: `encode(topic, msg) -> bytes`, `decode(data) -> (topic, msg)`. The
cross-copy byte-parity oracle is `imu_camera/tests/codec_roundtrip_selftest.py`
(frozen sha256 per `Wire*`).

### 3.3 `comms/wire.py` + `comms/messages.py` — wire vs local messages

For every existing message type that crosses a process boundary, a sibling
`Wire*` dataclass in `wire.py` carries only POD fields + `SharedArrayRef` for each
large array. The receiving bridge re-hydrates by copying from shared memory back
into a regular `np.ndarray` and constructing the local dataclass (`ImuCamPacket`,
`DepthFrame`, `Keyframe`, …) defined in `messages.py`. `wire.TOPIC_WIRE` (the codec
key) also includes the **retained / read-directly** topics that have no converter
(`calib.bundle → WireCalibBundle`) so consumers reading them straight off the wire
can still decode.

## 4. The bridge — `comms/bridge.py`

The bridge keeps the existing reactive modules unchanged. Each side has one tiny
class:

- **`IPCPublisher`** — subscribes to N in-proc topics on the local `LocalPubSub`,
  copies the payload into a shared-memory slot (if it has arrays), wraps it in the
  matching `Wire*` message, and `IPCPubSub(role="server").publish`es it. One per
  process boundary.
- **`IPCSubscriber`** — subscribes (via `IPCPubSub(role="client")`) to topics on a
  remote publisher, re-hydrates wire messages into local dataclasses, and publishes
  them on the local in-proc `LocalPubSub`. Other modules in this process consume
  from the local bus exactly as before.

The whole IPC layer is therefore invisible to `OdometryModule`, `BackendModule`,
`SlamModule`, the UI sinks, and every existing self-test.

## 5. Process entry points — `<project>/main.py`

One module per process, each exposes a `main()` so it can be spawned standalone.

### 5.1 `imu_camera/main.py` (capture · `oak.capture`)

```
LocalPubSub
  ├── read_cam (live OAK-D or session replay)
  ├── imu_cam (sync + IMU calibration; INLINE depth steps: compute_depth → publish_depth)
  └── IPCPublisher → IPCPubSub(endpoint="oak.capture", role="server")
      └── publishes: cam.sync, imu.raw, imucam.sample, frame.depth, calib.bundle
```

`imu_camera.main` **defaults to replay** and takes an explicit `--live` for hardware
(the launcher passes `--live` on the live branch, `--session PATH [--max-frames N]`
on the replay branch). Depth is computed INLINE on the `imu_cam` thread by the
`compute_depth` → `publish_depth` steps — the same SGM math `depth/` owns (a
byte-identical vendored copy, `diff -r` gated).

`calib.bundle` is a one-shot **retained** message: when a new subscriber connects it
gets the latest cached bundle immediately (so VIO / SLAM can boot without guessing).
Re-published on device re-open.

### 5.2 `vio/main.py` (VIO · `oak.vio`)

```
IPCPubSub(endpoint="oak.capture", role="client")
  └── IPCSubscriber → LocalPubSub
        ├── OdometryModule(publish_vo=True, level_tilt=True, OdometryConfig(gyro_fuse=…))
        ├── BackendModule (worker=False — solve in-process here; this process is already off-main)
        └── IPCPublisher → IPCPubSub(endpoint="oak.vio", role="server")
              └── publishes: pose.odom, pose.vo, keyframe, frame.tracks,
                             frame.inliers, pose.refined
```

`OdometryModule` joins `imucam.sample` (IMU prior) + `frame.depth` (KLT track →
RGB-D PnP → gyro fusion → pose). Two-client startup: a **calib client** blocks on
the retained `calib.bundle`, then a **data client** for `imucam.sample` +
`frame.depth`. VIO re-broadcasts the retained `calib.bundle` on its own endpoint
AFTER allocating its `kf_*` rings (readiness barrier, §9 invariant 10). The
worker-engine subprocess boundary (`--worker`) stays on stdlib pickle
(`multiprocessing.Queue`, same-project classes) — it is **not** routed through the
cross-process codec.

`pose.vo` (`topics.POSE_VO`) is the PURE-VISION frame-to-frame trajectory — raw PnP
R/t only, **no gyro fusion, no tilt leveling, no BA**. It is accumulated by
`RGBDVisualOdometry.pose_vo` (a separate accumulator from the gyro-fused `pose`,
`vio/mathlib/odometry/odometry.py`) and emitted by the `publish_vo` step
(`vio/modules/publish_vo.py`), wired into the frame chain **only** when
`OdometryModule` is built with `publish_vo=True` (`vio/main.py` sets it). It is
**LIVE-only**: the offline / deterministic oracle leaves `publish_vo=False`, so it
never runs and `pose.odom` byte-parity is unaffected (§9 invariant 15).

### 5.3 `slam/main.py` (SLAM · `oak.slam`)

```
IPCPubSub(endpoint="oak.vio", role="client")
  └── IPCSubscriber → LocalPubSub
        ├── SlamModule(latest_only=True, publish_map=True, worker=False,
        │              SlamConfig(loop_max_odom_rot_deg=30.0, kf_min_trans_m=0.1, kf_min_rot_deg=5.0))
        └── IPCPublisher → IPCPubSub(endpoint="oak.slam", role="server")
              └── publishes: loop.correction, slam.map
```

SLAM subscribes to `keyframe` **from VIO** (not capture), so SLAM never sees a
keyframe VIO hasn't already accepted. The pose graph is SLAM's own map. The new
`slam.main` is a **pure VIO consumer** — it deliberately does not subscribe to
capture at all (the pre-split `--capture-endpoint` flag is gone). A single calib
client blocks on VIO's retained `calib.bundle` (its arrival proves VIO is up,
intrinsics are known, and the keyframe rings exist), then SLAM attaches to VIO's
`kf_*` rings.

**Keyframe motion-gating (proc-LIVE).** `SlamConfig(kf_min_trans_m=0.1,
kf_min_rot_deg=5.0)`: a keyframe joins the pose graph **only if the camera moved
≥10 cm OR rotated ≥5°** since the last inserted keyframe, so a hovering /
near-stationary drone stops adding redundant near-identical keyframes (bounds the
graph by trajectory length, not run time). The offline `SlamModule` keeps the
`SlamConfig` default `kf_min_trans_m=0.0` / `kf_min_rot_deg=0.0` (gate off), so
offline scoring is unchanged (§9 invariant 16).

**Latest-only (coalescing) inbox (proc-LIVE).** `latest_only=True`: the ORB +
pose-graph solve cost grows with the map, so a strict FIFO inbox backed up without
bound and the `slam.map` overlay lagged further behind real time. A coalescing inbox
**drops the backlog and always solves the FRESHEST keyframe**, so the live map stays
current; `END` is never coalesced so clean shutdown still propagates. The offline /
replay oracle keeps the `SlamModule` default `latest_only=False` (strict FIFO) for
determinism (§9 invariant 14).

**`worker=False` is the default:** the heavy BA/SLAM solves run **in-process** (this
process is already off the main interpreter), so there is no worker subprocess and
no `resource_tracker` semaphore noise at shutdown / Restart. `--worker` is an opt-in
that runs those solves GIL-free in child subprocesses.

`publish_map=True` (the LIVE-only flag) adds the `publish_slam_map` step so SLAM
emits **two** topics, distinct in cadence:

- `loop.correction` — the loop-event pose-graph rewrite, emitted ONLY on a confirmed
  loop closure. Byte-identical to the offline path.
- `slam.map` — a CONTINUOUS overlay published EVERY keyframe (`SlamOverlay`),
  carrying the current corrected camera-optical keyframe positions + `n_loops` +
  `last_match`. LIVE-only: the offline path keeps `publish_map=False`, so neither the
  step nor the topic exists there (§9 invariant 12).

### 5.4 `ui/main.py` (UI · foreground)

```
IPCPubSub(endpoint="oak.vio",     role="client")  # pose.odom, pose.vo, pose.refined, calib.bundle (always)
IPCPubSub(endpoint="oak.slam",    role="client")  # slam.map, calib.bundle (always)
IPCPubSub(endpoint="oak.capture", role="client")  # imu.raw, imucam.sample, frame.depth (on-demand, menus)
  └── IPCSubscribers → LocalPubSub → Qt MainWindow
        ├── ONE Viewer3D (no tabs): live marker = pose.odom (vio), drawing 5 lines
        │     VO                 : pose.vo     (vio) — grey,  pure vision
        │     VIO                : pose.odom   (vio) — green, f2f PnP + gyro
        │     VIO-BA             : pose.refined(vio) — blue,  windowed BA
        │     SLAM-corrected VIO : pose.odom deformed by slam.map corrections — orange (teleport red)
        │     SLAM               : slam.map    (slam) — cyan kf line + amber dots
        ├── Controls toolbar (always-visible, top of window):
        │     [VO][VIO][VIO-BA][SLAM-corrected VIO][SLAM]  : per-line show/hide
        │     Clear Trail  : clear the live trajectory trail
        │     Restart      : quit with RESTART_EXIT_CODE=42 → launcher respawns all
        └── Menu bar (renders in-window on every platform; setNativeMenuBar(False)):
              View         : VIEW_PRESETS / Follow Camera (on the single viewer)
              Visualize    : triplet window  ← capture imucam.sample/frame.depth
                             keypoint tracker ← capture frame.depth + vio tracks/inliers
              Calibration  : gyro / accel dialogs ← capture imu.raw (RAW)
```

A single `SlamMapTracker` subscribes `slam.map` (slam endpoint) plus `pose.odom` /
`pose.vo` / `pose.refined` (vio endpoint) for the lifetime of the process and
exposes one snapshot getter per line; `IpcPoseSource` feeds the live green marker +
trail off `pose.odom`. The **menu** subscriptions are opened lazily by the
`ui/modules/ipc_sources.py` adapters only when a Visualize/Calibration action fires,
and closed when the window/dialog closes.

The Qt main thread sees only the local `LocalPubSub`, so the existing UI sinks and
the `ui/qt` calib dialogs are reused unchanged — the adapters republish the IPC
topics onto the very same local bus those sinks already read.

### 5.5 Two different optimisers: VIO = windowed BA, SLAM = PGO

VIO and SLAM run **two distinct optimisers** — this is the key fact behind the five
UI lines:

- **VIO runs windowed Bundle Adjustment (BA).** `BackendModule` (`run_ba` step)
  solves a sliding window jointly over **keyframe poses AND landmarks** (3D points),
  minimising reprojection error — analytic Schur in `vio/mathlib/backend/`. Output:
  `pose.refined`, the blue **VIO-BA** line. BA refines the *local* geometry of the
  recent window.
- **SLAM runs Pose-Graph Optimization (PGO).** `SlamModule` (`slam_step`) runs ORB
  loop detection, then on a confirmed loop optimises a graph of **poses only — no
  landmarks** (`slam/mathlib/loop/`). The graph has odometry edges (relative motion
  between consecutive keyframes) + loop-closure edges (the relative motion implied by
  a revisited place); PGO **distributes the accumulated drift over the whole
  trajectory** so the loop closes consistently. Output: `loop.correction` (the
  loop-event rewrite) + `slam.map` (the continuous corrected keyframe map, the cyan
  **SLAM** line).

So BA ≠ PGO: BA is a local windowed landmark+pose solve (metric refinement); PGO is
a global pose-only solve fired by loop closure (drift redistribution).

## 6. UI — `ui/main.py` + `ui/modules/ipc_sources.py`

The UI is a single `QMainWindow` with **one** `Viewer3D` (no tabs) **and a menu
bar**. It imports **no depthai**: everything it shows is fed over IPC. The existing
`ui/qt` windows and calib dialogs are reused **unchanged** — `ui/main.py` only builds
the viewer + toolbar + menus and wires them, and `ui/modules/ipc_sources.py` supplies
three injectable adapters that bridge the IPC topics onto the same local
`LocalPubSub` those windows already read.

### 6.1 The single 5-trajectory view

| # | Line | Colour (`ui/qt/theme`) | Source topic | Meaning |
|---|---|---|---|---|
| 1 | **VO**                 | grey   (`VO_PATH`)        | `pose.vo` (vio)         | PURE-VISION frame-to-frame path — raw PnP R/t, **no IMU, no BA**. Drifts most. |
| 2 | **VIO**                | green  (`TRACE_PATH`)     | `pose.odom` (vio)       | Frame-to-frame RGB-D PnP **+ gyro fusion**, no BA. The responsive live marker + trail (never lags — never waits on a back-end). |
| 3 | **VIO-BA**             | blue/violet (`BA_PATH`)   | `pose.refined` (vio)    | Windowed **Bundle Adjustment** keyframe trajectory (landmarks + poses). Sparse. |
| 4 | **SLAM-corrected VIO** | orange (`CORRECTED_PATH`) | `pose.odom` deformed by `slam.map` | The dense VIO trail rubber-sheeted by SLAM's per-keyframe pose-graph correction (`np.interp` of the per-keyframe correction delta, matched by keyframe seq). Segments where the correction magnitude exceeds ~0.15 m (`TELEPORT_M`) are flagged "teleport" and drawn in **red** (`TELEPORT`). |
| 5 | **SLAM**               | cyan   (`REFINED_PATH`)   | `slam.map` (slam)       | The loop-corrected keyframe path + **amber keyframe dots**, with the just-revisited keyframes flashed on each loop closure (`last_match` + `n_loops`). |

The live green marker + VIO trail come from `IpcPoseSource` (`pose.odom`) feeding the
viewer's `PoseHistory`. The other four lines are fed by snapshot getters on a single
`SlamMapTracker` (`vo_snapshot`, `ba_snapshot`, `corrected_vio_snapshot`,
`refined_path_snapshot` + `slam_overlay_snapshot`), which subscribes — across two IPC
clients — to `slam.map` on the slam endpoint and `pose.odom` / `pose.vo` /
`pose.refined` on the vio endpoint. The SLAM-corrected VIO line needs BOTH the dense
`pose.odom` trail (with frame seqs) and the per-keyframe corrected positions SLAM
publishes (with their source seqs in `kf_ids`): the tracker matches each keyframe to
its dense VIO anchor, computes the correction delta, and interpolates it
piecewise-linearly by seq across the dense trail before adding it back.

`slam.map` **supersedes** the old `loop.correction`-driven overlay, which only fired
ON a loop closure — so there were no keyframe dots along the path until the first loop
closed (the bug this design fixes). `loop.correction` is still published (the
loop-event pose-graph rewrite), but the live keyframe-dots overlay no longer waits on
it.

### 6.2 Controls toolbar + menu bar

A small always-visible **Controls** `QToolBar` (docked top, `setMovable(False)`)
carries the **five per-line toggle buttons**, then **Clear Trail** and **Restart**.

- **Line toggles** — five checkable buttons, **VO** / **VIO** / **VIO-BA** /
  **SLAM-corrected VIO** / **SLAM** (in back-to-front / drift order). All start
  CHECKED; each `toggled(bool)` drives its viewer visibility setter, so the operator
  can isolate any one trajectory.
- **Clear Trail** — clears the live trajectory trail (`history.clear()`). With one
  viewer there is no "active tab" — it targets the single `PoseHistory` directly.
- **Restart** — respawn the whole pipeline fresh. Because the IPC bus is one-way
  (server→client) the UI **cannot** reset vio/slam in place, so it sets a flag and
  calls `app.quit()`; `run_ui` then returns **`RESTART_EXIT_CODE = 42`**. The
  launcher's restart loop sees code 42, `_cleanup_orphans()`es the prior generation,
  and respawns capture + vio + slam + ui from scratch (§7, §9 invariant 13).

The menu is plain Qt (`QMenuBar` / `QAction`); `ui.main` calls
`mbar.setNativeMenuBar(False)` so the bar renders **in-window on every platform**.

- **View** — `VIEW_PRESETS` camera presets and **Follow Camera**. There is no "Clear
  Keyframes" — there is no UI→SLAM channel, so it would be a dead action.
- **Visualize** — **"Camera + Depth + IMU (triplet)…"** (`SyncedViewWindow`, driven
  by `IpcTripletWorker`), **"Keypoint Depth Tracker…"** (`KeypointTrackWindow`, driven
  by `IpcKeypointWorker`), **"Gyro Fusion (strip chart)…"** (`GyroFuseWindow`, driven
  by `IpcGyroFuseSource`), and **"SLAM Map (landmarks)…"** (`MapWindow`, driven by
  `IpcSlamMapSource` — the **sparse, ID-based landmark map**: ONE point per KLT track id
  that was a PnP INLIER across ≥ `PERSIST_KF` (=20) SUCCESSIVE keyframes, re-snapped to
  SLAM's loop-corrected poses, in the same ENU frame as `Viewer3D`; the gate is a
  **longest-consecutive-keyframe-run** filter — `ui/viz/map_cloud.py::longest_consecutive_run`,
  UI-only — so only consistently-tracked, motion-validated points show, NOT a dense
  reconstruction), and **"Floor Plan (top-down)…"**
  (`FloorPlanWindow`, driven by `IpcFloorPlanSource` — a **LIGHT 2D top-down
  wall-outline raster**, NOT OpenGL). The floor plan is a cheap, readable alternative
  to the 3D maps (heavy GL on a Mac, noisy in perspective): it back-projects each
  keyframe's depth by its own VIO pose, drops the world-vertical optical-`+y` axis
  to bin the points onto the optical `(x,z)` GROUND plane, then builds the room's
  **occupied region** (a cell needs enough rays AND a real vertical column —
  `extent_m >= FLOOR_EXTENT_M`, the explicit *wall = vertical extent* gate that drops
  the flat floor) and reduces it to a **crisp wall line** with cheap 2D `cv2` ops:
  `MORPH_OPEN` scrubs the radial *star-burst* streaks, `MORPH_CLOSE` bridges
  depth-dropout gaps, `connectedComponentsWithStats` drops the isolated noise islands,
  and `MORPH_GRADIENT` takes the region **boundary** (top-down, a wall *is* the
  boundary between occupied and free space). The bright outline is drawn over a faint
  raw-occupancy context wash, with the **camera path** + latest-pose marker on top. It
  uses a 2D pyqtgraph `PlotWidget` (`ImageItem` + `PlotDataItem`) — **no
  `GLViewWidget`** — so it never stutters the UI, and the raster can be written to a
  PNG with pure numpy/cv2 for offscreen visual verification. The builder math is the
  pure-numpy+cv2 `ui/viz/floor_plan.py` (no Qt/GL). Each window is cached so repeated
  opens reuse the one IPC source.
- **Calibration** — **"Gyroscope Bias…"** (`GyroCalibDialog`) and **"Accelerometer
  (6-position)…"** (`AccelCalibDialog`). Each opens with a fresh `IpcImuRawSource`
  injected as its `stream`; the menu handler owns the stream and closes it in its
  `finally`.

### 6.3 IPC adapters — `ui/modules/ipc_sources.py`

Three drop-in adapters let the unchanged `ui/qt` windows/dialogs run with no
in-process acquisition graph. The module is **device-agnostic by contract**: it
consumes only the abstract IPC topics + wire POD types and never imports depthai.

| Adapter             | Duck-types / extends                  | Consumes (endpoint · topics)                                   | Notes |
|---                  |---                                    |---                                                            |---|
| `IpcImuRawSource`   | `ui.qt` IMU stream contract           | capture · `imu.raw`                                            | Subscribes capture's **RAW** IMU and re-emits one `(3,)` gyro+accel sample at a time with a **seconds** timestamp. RAW — not calibrated — is correct: calibrating off an already-calibrated stream would be circular. |
| `IpcTripletWorker`  | `ui.qt.synced_window` triplet worker  | capture · `imucam.sample`, `frame.depth`                       | Republishes both topics onto a local `LocalPubSub`; the unchanged triplet sink joins them by `seq` and renders. |
| `IpcKeypointWorker` | `ui.qt.keypoints_window` worker       | capture · `frame.depth`  +  **vio** · `frame.tracks`, `frame.inliers` | Two endpoints: depth imagery from capture, KLT tracks + PnP inliers from VIO. The unchanged tracks sink joins them by `seq`. Keeps `FrameTracks` pure POD so VIO never writes capture's rings (§9 invariant 6). |

Each adapter opens its own read-only `IPCPubSub(role="client")` on demand, attaches
only the capture rings it needs, and surfaces a connect failure (capture down) as a
clear reason rather than a raw shared-memory path error.

Beside these three duck-typed adapters, the same module hosts the **keyframe-map
builder** sources — `IpcSlamMapSource` (sparse landmark cloud) and `IpcFloorPlanSource`
(2D top-down wall-outline raster). Both subclass a shared `_KeyframeAccumulator` base
(VIO `keyframe` ring attach + stash + evict + a coalesced off-GUI rebuild loop), so
each adds **only** its own build (`_build`) with **no copy-paste** of the SHM/recv
wiring; the floor-plan build
delegates to the pure-numpy+cv2 `ui/viz/floor_plan.py` so its projection + wall-mask
cleanup are testable headless.

### 6.4 Calibration semantic — "saves for the NEXT capture start"

The UI does **not** own the device; `imu_camera` does. So a calibration the UI saves
is **not** applied live mid-run. The dialog keys the saved value (gyro bias / accel
calib) by `device_id` (from the calib bundle, §9 invariant 11) and writes it to the
per-device store; `imu_camera` **loads** it by the same key on its **next start**
(`load_gyro_bias` / `load_accel_calib` in `imu_camera/mathlib/device/live_calib.py`).
The dialog shows "Saved for device `<id>`" to make the deferred effect explicit.

## 7. Launcher — `launcher/main.py` + `run.sh`

`launcher.main` spawns the three background processes (capture → vio → slam, in that
order so each subscriber boots after its publisher's endpoint exists), waits a few
hundred ms between each, then runs the UI process in the foreground. On UI exit it
sends `SIGTERM` to the three background processes and joins them; on any of them
dying it shuts the others down with a clear diagnostic.

`launcher.main` stays **Qt-free**: it imports only `RESTART_EXIT_CODE` from
`ui.main` (which lazy-imports PyQt6 inside `run_ui`). It vendors `launcher/comms/`
(byte-identical copy, `diff -r` gated) for the `SharedArrayRing.cleanup_stale` +
`ring_registry` it needs for orphan reclaim.

**Restart loop.** The spawn → run-UI → teardown sequence is a **loop**. Each
iteration `_spawn_pipeline()`s a fresh capture + vio + slam generation, blocks on
`ui_proc.wait()`, then `_terminate()`s that generation on the main thread (no
waitpid race — the UI is already reaped by `wait()`). If the UI returned
`RESTART_EXIT_CODE` (42) the loop `_cleanup_orphans()`es and respawns the whole
pipeline; any other exit code breaks the loop and the launcher exits normally. The
endpoint names are computed ONCE (`--auto-suffix` derives them from the launcher
PID), so each restart re-creates the same-named endpoints + rings;
`_cleanup_orphans()` reclaims the prior generation's stale SHM/sockets each
iteration. `_RING_NAMES_BY_ROLE` is cap=`gray_left`/`gray_right`/`depth_m`,
vio=`kf_gray`/`kf_depth`, slm=none.

The **`--no-ui`** path runs the pipeline exactly **once** (no Restart button without
a UI): it spawns capture + vio + slam, waits for capture to exit, lets vio + slam
drain, then tears them down.

The launcher's **SIGTERM handler** (registered once) forwards SIGTERM to the current
generation's children and `os._exit(143)` immediately. It deliberately does **not**
call `_terminate()` — `_terminate` polls `os.waitpid(pid, WNOHANG)` on the same pid
the main thread is blocked in `ui_proc.wait()` on, and the two waitpid callers would
race for the single reap event.

**`--worker` is an opt-in (default off).** With it off, vio + slam run their heavy
BA/SLAM solves **in-process** and SLAM stays responsive via its latest-only inbox
(§5.3) — no worker subprocess, no `resource_tracker` noise. Passing `--worker`
propagates `--worker` to both the vio and slam children.

`run.sh` forwards to `python -m launcher.main --auto-suffix "$@"`:
- `./run.sh ...` — the live 4-process pipeline (default).
- `./run.sh --proc ...` — explicit alias for the same pipeline (the flag is stripped).
- `./run.sh --session PATH ...` — replay a recorded session through the pipeline.
- `./run.sh --no-ui ...` — headless capture + vio + slam (runs once).

Two intentional differences from the pre-split launcher, forced by the new projects'
argparse:

- **capture argv inversion** — `imu_camera.main` DEFAULTS to replay and takes an
  explicit `--live` for hardware, so the launcher's live branch passes `--live` and
  the replay branch passes `--session PATH [--max-frames N]`.
- **slam dropped `--capture-endpoint`** — the new `slam.main` is a pure consumer of
  VIO's output, so the launcher wires slam with only `--vio-endpoint` / `--endpoint`
  (passing `--capture-endpoint` would make slam's argparse abort).

## 8. Verification & testing

The byte-parity oracle lives in `verification/` and is **in-process** (single
`LocalPubSub`, no `IPCPubSub`) because the live pipeline is separate OS processes
over IPC, and process scheduling is nondeterministic. Two independent things are
proven (see [`verification/README.md`](../verification/README.md)):

1. **End-to-end math parity** — the in-process oracle drives the split projects'
   verbatim-ported math (`imu_camera` / `vio` / `slam`) through the SAME ATE/Sim3
   scoring loop the pre-split `vio_run` used, and reproduces the frozen
   `baseline_metrics.json`. Observed gap: `0.000e+00`.
2. **IPC contract parity** — the vendored `comms/` package is byte-identical across
   all copies, every copy's codec produces identical bytes for a fixed test-vector
   set, and the bridge + shared-memory rings round-trip a message intact.

| Test | Scope | Status |
|---|---|---|
| `verification/oracle_replay_selftest.py` | Byte-parity gate: split-project math == frozen baseline within `TOL_MM=1e-6` mm. Has a verified negative control. | PASSING (gap = 0) |
| `verification/ipc_comms_selftest.py` | 5-copy `comms` dir-diff + codec sha256 + cross-decode + `SharedArrayRing` round-trip + full bridge round-trip over a real Unix socket. | PASSING |
| `imu_camera/tests/codec_roundtrip_selftest.py` | Codec round-trip + frozen sha256 per `Wire*` (vendored into each copy → identical `codec_vectors.json`). | PASSING |
| `slam/tests/proc3_smoke_selftest.py` | 3-proc spawn (imu_camera replay + vio + slam) over a gold loop; asserts rc=0, `slam.map` advances, `loop.correction` n_loops matches the oracle. | PASSING |
| Per-project math selftests | `vio.tests.vio_ba_selftest`, `vio.tests.odometry_selftest`, `slam.tests.loop_closure_selftest`, `depth.tests.stereo_sgm_selftest` — each == the pre-split numbers line-for-line. | PASSING |

## 9. Invariants

1. The IPC layer is stdlib-only (sockets + shared memory). No new pip deps.
2. The reactive modules (`OdometryModule`, `BackendModule`, `SlamModule`, every UI
   sink) are reused unchanged. The bridge (`IPCPublisher` / `IPCSubscriber`) is the
   only IPC-aware glue.
3. The offline replay path (the `verification/` oracle) stays byte-identical and
   in-process (single `LocalPubSub`).
4. Tools never open the OAK-D. `imu_camera` is the only owner of the device.
5. No process holds another process's data (every numpy array crossing the bridge is
   copied out of shared memory on the receiving side before any downstream step runs).
6. **Ring slots > IPC outbox capacity.** A wire message in an outbox references a ring
   slot the producer must NOT have overwritten by the time the consumer reads it.
   Default slots=64 strictly exceeds outbox cap=32, so the 32 outbox-queued items
   reference slots `[N-32, N-1] mod 64` — all distinct from the producer's next-write
   slot `N mod 64`. See `default_capture_specs` / `default_vio_specs` in
   `comms/ring_registry.py` and `SharedArrayRing.create` for the assertion.
7. **Drain before stop.** `Module.stop()` checks `_stop` at the TOP of every loop
   iteration and discards any items still queued. So a process that wants to publish
   END must wait for the module's `done` event BEFORE calling `stop()`. Capture waits
   on the imu_cam module's `done`; VIO on odometry + backend; SLAM on slam.
8. **`IPCPubSub.close` drains the outbox.** `close()` sets `_stopped` to gate new
   publishes, then puts BYE on each subscriber's outbox and joins the fanout thread
   (which sends every pending wire message in order, then BYE, then exits).
   `state.alive` is ONLY flipped by send-errors — close does NOT flip it.
9. **`SharedMemory(track=False)` on attach.** The attaching process must not register
   the shm with its own resource_tracker (the creator does). `SharedArrayRing.attach`
   passes `track=False` (Python ≥ 3.13). See https://bugs.python.org/issue38119.
10. **Readiness barrier = retained `calib.bundle` re-publish.** VIO subscribes to
    capture's retained `calib.bundle`, then re-publishes the SAME bundle on its own
    retained endpoint AFTER allocating its `kf_*` rings. SLAM (and the UI) wait on
    VIO's calib bundle as a "VIO is ready, rings exist" signal. Without this barrier
    downstream procs race the ring creation and fail with `FileNotFoundError`.
11. **`WireCalibBundle.device_id` is the calibration key, carried on the bundle.**
    `WireCalibBundle` (`comms/wire.py`) carries an OPTIONAL `device_id: str | None`.
    **Producer:** capture fills it from the real device id
    (`imu_camera/mathlib/device/live_calib.py` → the calib bundle builder in
    `imu_camera/main.py`); replay sets it to `None`. VIO **re-broadcasts the same
    bundle**, so the UI reads `device_id` off VIO's bundle. **Consumer:** the UI keys
    any calibration it saves by this id, which is IDENTICALLY the key capture LOADS
    with on its NEXT start — that match is what makes a UI-saved calibration actually
    take effect (it is NOT applied live; §6.4). When `device_id` is `None` (replay)
    the UI falls back to `"default"`. *This is a cross-language wire contract:*
    `device_id` is a deliberate **additive, backward-compatible OPTIONAL** field
    (default + placed AFTER the existing optional fields), so the codec stays safe and
    any older subscriber simply ignores it.
12. **`slam.map` is a LIVE-ONLY overlay; it never touches the offline scoring path.**
    SLAM publishes a continuous keyframe-map overlay on `slam.map` (`topics.SLAM_MAP`),
    carrying the local POD `SlamOverlay` over the wire as `WireSlamMap`. **Producer:**
    the `publish_slam_map` step (`slam/modules/publish_slam_map.py`) emits it EVERY
    keyframe — but ONLY when `SlamModule` is built with `publish_map=True`. The `slam`
    process sets that flag (`slam/main.py`); the **offline / replay path keeps
    `publish_map=False`**, so neither the step nor the topic exists there.
    **Consumer:** the UI's SLAM line (`SlamMapTracker` in `ui/main.py`) draws the
    continuous keyframe dots from `slam.map` instead of waiting on `loop.correction`.
    The invariant: `slam.map` is **purely additive and live-only** —
    `loop.correction` and the deterministic offline scoring path stay
    **byte-identical** whether or not the overlay exists.
13. **Restart = full respawn via `RESTART_EXIT_CODE`; there is no reverse IPC
    channel.** The IPC bus is one-way (server→client), so the UI cannot reset
    vio/slam in place. The Restart toolbar button (`ui/main.py`) instead quits the Qt
    loop and `run_ui` returns `RESTART_EXIT_CODE = 42`; the launcher
    (`launcher/main.py`) loops on that code, `_cleanup_orphans()`es, and respawns
    capture + vio + slam + ui from scratch. Any other UI exit code ends the launcher
    normally. The `--no-ui` path bypasses the loop. The launcher's SIGTERM handler
    must NOT call `_terminate()` (waitpid race with `ui_proc.wait()` — see §7).
14. **proc-LIVE SLAM uses a latest-only (coalescing) inbox; the offline scoring path
    does not.** The `slam` process builds `SlamModule(latest_only=True, …)`
    (`slam/main.py`) so the LIVE viewer drops a keyframe backlog and always solves the
    freshest keyframe. `END` is never coalesced, so shutdown still propagates. The
    deterministic offline / replay path keeps the default `latest_only=False` (strict
    FIFO), so its scoring stays byte-identical. In-process solve is the default
    (`worker=False`); `--worker` moves the heavy solves to GIL-free child subprocesses.
15. **`pose.vo` is LIVE-only; `pose.odom` byte-parity is preserved.** The pure-vision
    frame-to-frame trajectory (`topics.POSE_VO`, the UI's VO line) is emitted by the
    `publish_vo` step (`vio/modules/publish_vo.py`), wired into the frame chain **only**
    when `OdometryModule` is built with `publish_vo=True`. The `vio` process sets that
    flag (`vio/main.py`); the offline / deterministic path leaves `publish_vo=False`,
    so it never runs. `pose_vo` is a SEPARATE accumulator on `RGBDVisualOdometry` (raw
    PnP R/t — no gyro, no tilt, no BA) that is read-only w.r.t. the gyro-fused `pose`,
    so adding it does not perturb the `pose.odom` solve: offline `pose.odom` output
    stays **byte-identical**.
16. **SLAM keyframe motion-gating is a proc-LIVE setting; the offline default is 0/0
    (gate off).** The `slam` process builds `SlamConfig(kf_min_trans_m=0.1,
    kf_min_rot_deg=5.0)` (`slam/main.py`) so a keyframe joins the pose graph only after
    ≥10 cm of translation OR ≥5° of rotation since the last inserted keyframe. The
    `SlamConfig` defaults are `kf_min_trans_m=0.0` / `kf_min_rot_deg=0.0`
    (`slam/mathlib/loop/slam.py`), so the offline path keeps the gate OFF and its
    deterministic scoring is unchanged.

## 10. The split (history)

The split was shipped in phases, each independently verifiable. The end state:

1. `imu_camera` — capture process + inline SGM depth (the template the others
   replicate).
2. `depth` — SGM source-of-truth + standalone depth-as-a-process harness.
3. `vio` — odometry + windowed BA process.
4. `slam` — loop closure + pose graph process (pure VIO consumer).
5. `ui` — single 5-trajectory `Viewer3D` + Visualize/Calibration menus over IPC.
6. `launcher` — process lifecycle (spawn / restart loop / orphan cleanup);
   `./run.sh --proc` → `launcher.main`.
7. `verification` — in-process byte-parity oracle (gap = 0) + cross-copy comms gate.

Each project vendors a byte-identical `comms/` (the codec replaces pickle), all
internal imports are RELATIVE, and the package pulls no depthai / PyQt6 / cv2 — so
every project is independently portable.
</content>
