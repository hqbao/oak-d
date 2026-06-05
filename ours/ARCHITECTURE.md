# `ours` — Architecture

From-scratch RGB-D Visual-Inertial Odometry / SLAM for the OAK-D, written to
replace a black-box baseline (DepthAI + Basalt) **one module at a time**. This
tree is **fully standalone**: it imports zero `oakd` / `baseline` code.

> Long-term goal: every block here can be swapped for our own improved
> implementation in isolation. The architecture exists to make that safe.

---

## 1. The two layers

```
ours/
├── lib/      LIBRARIES   — reusable code, no runtime/threads of its own
└── flows/    FLOWS       — the live threaded pipeline that USES the libraries
```

- **`ours.lib`** = pure libraries: algorithms (stereo, odometry, IMU, backend,
  loop) + shared helpers (`misc`) + the **flow-framework library** (`lib/flow`).
  A library has *no behaviour of its own* — it is called, it does not run.
- **`ours.flows`** = the concrete pipeline. Each flow is one thread that wires a
  short chain of tasks using the libraries. Flows hold **no maths**.

Offline tools (`ours.tools.*`) call the `ours.lib` algorithm libraries directly
and never touch `ours.flows`. The flows exist only for the live, threaded run.

---

## 2. HARD RULE — flows communicate ONLY via pub/sub topics

**Flows NEVER call each other directly.** The only way one flow influences
another is by publishing a message on a `Bus` topic the other flow subscribes
to. No flow imports, holds a reference to, or calls a method on another flow.

```python
# produce
ctx.bus.publish(topics.POSE_ODOM, PoseMsg(...))
# consume / forward
self.on(topics.FRAME_DEPTH, [ComputeDepth(), PublishDepth()])
self.forwards_to(topics.POSE_ODOM, topics.KEYFRAME)
```

The **inter-flow contract** is therefore exactly:

- the topic names in `lib/flow/topics.py`, and
- the message types in `lib/flow/messages.py`.

Nothing else couples two flows. The whole pipeline graph is fully described by
*which topics each flow reads and writes*. This is what makes each flow
independently testable and swappable.

> Anti-pattern reminder: do not visualize/record data a module does not actually
> emit. Every UI/recorded stream must trace back to a real output of the system
> being replaced (see project memory `honest-pipeline-visualization`).

---

## 3. The flow framework — `ours.lib.flow`

The threaded, message-passing substrate. It is a *library* (reusable machinery),
so it lives in `lib/` next to the other libraries; the concrete flows import it
just like they import `lib.stereo` or `lib.odometry`.

| Module | Role |
|---|---|
| `flow.py` | `Flow` / `SourceFlow` / `FlowContext` — one thread running a fixed task chain (FIFO or latest-only inbox) |
| `task.py` | `Task` — the smallest input→output step in a chain |
| `pubsub.py` | `Bus` — thread-safe publish/subscribe between flows |
| `messages.py` | one immutable message type per topic (the flow contract) |
| `topics.py` | topic-name constants |
| `runtime.py` | process-wide guards (e.g. `NUMBA_PARALLEL_LOCK`) |

Layering: depends only on the standard library + numpy. It does **not** import
the concrete flows or the algorithm libraries.

### Inbox modes: FIFO vs latest-only
A reactive `Flow` drains an inbox queue on its own thread. `Bus.publish` is
synchronous but only drops the message into the subscriber's inbox, so the heavy
work runs on the *subscribing* flow's thread (actor model) — the publish stays
cheap.

- **FIFO (default):** an unbounded queue, every message processed in order. The
  VIO + deterministic replay require this — dropping a frame would corrupt the
  pose. It is correct as long as the consumer keeps up with the producer.
- **latest-only (`Flow(latest_only=True)`):** a *coalescing* inbox that keeps only
  the newest unprocessed message per topic. A **realtime visualiser** needs this:
  a free-running producer (the live cam paces 20 fps) feeding a FIFO whose
  consumer is even slightly slower (SGM+KLT are serialized by the numba lock, so
  the chain runs < 20 fps) makes the inbox grow **without bound** → the view falls
  seconds behind and keeps drifting. Latest-only drops the backlog so each stage
  always works on the freshest frame; latency is bounded to ~one frame per stage.
  `END` is a control signal and is never coalesced away. The keypoint-depth view
  builds its whole graph latest-only (`build_*(realtime_latest=True)`); the VIO
  and replay never do.

### Numba concurrency guard
`numba parallel=True` is used only in `lib/stereo` (SGM) and
`lib/frontend/klt_numba` (KLT). The default `workqueue` numba layer is **not**
threadsafe across Python threads, so running the depth task (SGM, on the `imu_cam`
thread) and the odometry flow (KLT) concurrently crashes. `runtime.NUMBA_PARALLEL_LOCK` serializes the two
parallel regions; all other flows (pure numpy) run free. `tbb`/`omp` layers are
not installable on this host (macOS arm64, py3.13). Note this lock caps the
cam→imu_cam→odometry chain below the camera frame rate, which is exactly why the
realtime visualiser path needs the latest-only inbox above.

---

## 4. The flows — `ours.flows`

One thread per flow; **one Task per file**; a `*_flow.py` only wires the tasks.

### Topic-level data flow (which flow publishes/subscribes which topic)

```
cam ──cam.sync──► imu_cam ──imucam.sample──► odometry ──pose.odom──► ui-collector, ui-render
                          ──frame.depth─────►          ──frame.tracks─► ui-tracks (keypoints view)
                          ──imu.raw───────► (visualiser) ──keyframe──► backend, slam
                                                       backend ──pose.refined──► ui-collector
                                                       slam    ──loop.correction──► ui-collector
```

Edges above are exactly the `self.on(...)` subscriptions in each `*_flow.py`.
There is ONE acquisition front-end (`cam` + `imu_cam`) shared by the VIO
and the camera/IMU visualiser — no separate capture monolith. Things worth
noting because the obvious guess is wrong:

- **depth is a task INSIDE the `imu_cam` flow**, not a separate flow: it is just a
  transform of the stereo pair `imu_cam` already produces, so when a matcher is
  wired in (the VIO path) `imu_cam` runs SGM inline and publishes `frame.depth`.
  The visualiser builds `imu_cam` with `matcher=None`, so it skips depth.
- **odometry consumes both `imucam.sample` and `frame.depth`** (both published by
  `imu_cam`); it integrates the packet's gyro into the per-frame rotation prior
  (`PreintegratePrior`) and runs RGB-D PnP against the depth (`EstimateMotion`).
- **the keypoint-depth view subscribes `frame.tracks`** (the odometry frontend's
  real `{id: pixel}` tracks, published by `PublishTracks`); it does NOT run its
  own frontend — it is a UI sink like ui-render, honest about its data source.
- **odometry is a two-input join** (`imucam.sample` + `frame.depth`); it sees an
  END on each before it drains (`expected_ends = 2`).
- **backend and slam both trigger off `keyframe`**, not `pose.odom`. odometry
  emits a keyframe every `kf_every` frames; backend refines it, slam loop-closes.
- **`loop.correction` is consumed only by the UI collector** today — it is *not*
  fed back into odometry, so the live pose path has no closed loop yet. (When
  that feedback is added, give odometry a `self.on(LOOP_CORRECTION, ...)`.)

### Task-level wiring (who receives from whom)

```mermaid
flowchart TD
    subgraph CAM["cam flow (SourceFlow)"]
        PRODC["produce()"] --> PCS["PublishCamSync"]
    end
    subgraph IMU["imu_cam flow (pack + depth)"]
        PIC["PackImuCam"] --> PIR["PublishImuRaw"] --> AC["ApplyCalibration"] --> PICAM["PublishImuCam"] --> CD["ComputeDepth"] --> PD["PublishDepth"]
    end
    subgraph ODO["odometry flow"]
        PIP["PreintegratePrior"]
        TF["TrackFeatures"] --> PT["PublishTracks"] --> PV["EstimateMotion"] --> PP["PublishPose"] --> EK["EmitKeyframe"]
    end
    subgraph BCK["backend flow"]
        RB["RunBA"] --> PR["PublishRefined"]
    end
    subgraph SLM["slam flow"]
        SS["SlamStep"] --> PC2["PublishCorrection"]
    end
    subgraph UIC["ui-collector flow"]
        COd["CollectOdom"]
        CRf["CollectRefined"]
        CCo["CollectCorrection"]
    end
    subgraph UIR["ui-render flow"]
        RP["RenderPose"]
    end
    subgraph UITR["ui-tracks flow (keypoints view)"]
        RT["RenderTracks"]
    end

    PCS -- "cam.sync" --> PIC
    PICAM -- "imucam.sample" --> PIP
    PD -- "frame.depth" --> TF
    PP -- "pose.odom" --> COd
    PP -- "pose.odom" --> RP
    PT -- "frame.tracks" --> RT
    EK -- "keyframe" --> RB
    EK -- "keyframe" --> SS
    PR -- "pose.refined" --> CRf
    PC2 -- "loop.correction" --> CCo

    PIP -. "ctx.state['priors'][seq]\n(same thread, not via Bus)" .-> PV
```

The dotted edge is the one **intra-flow** hand-off: `PreintegratePrior` stashes
the gyro prior for sequence `seq` in the odometry flow's own `ctx.state`, and
`EstimateMotion` pops it when the matching depth frame arrives. This is shared
state inside a single thread/flow — it does **not** cross the Bus and does **not**
violate the §2 rule (which only forbids *cross-flow* calls).

| Flow | Tasks (in order) | Subscribes | Publishes |
|---|---|---|---|
| **cam** | `produce` → `PublishCamSync` | — (source) | `cam.sync` |
| **imu_cam** | `PackImuCam` → `PublishImuRaw` → `ApplyCalibration` → `PublishImuCam` → `ComputeDepth` → `PublishDepth` | `cam.sync` | `imu.raw`, `imucam.sample`, `frame.depth` |
| **odometry** | `PreintegratePrior` ⟂ `TrackFeatures` → `PublishTracks` → `EstimateMotion` → `PublishPose` → `EmitKeyframe` | `imucam.sample`, `frame.depth` | `pose.odom`, `keyframe`, `frame.tracks` |
| **backend** | `RunBA` → `PublishRefined` | `keyframe` | `pose.refined` |
| **slam** | `SlamStep` → `PublishCorrection` | `keyframe` | `loop.correction` |
| **ui-collector** | `CollectOdom` / `CollectRefined` / `CollectCorrection` | `pose.odom`, `pose.refined`, `loop.correction` | — (sink) |
| **ui-render** | `RenderPose` | `pose.odom` | — (sink) |
| **ui-tracks** | `RenderTracks` | `frame.tracks` | — (sink) |

`cam` + `imu_cam` are the only device-specific flows; their sources are
injected (`ReplayCamSource`/`ReplayImuSource` offline, `LiveCamSource`/
`LiveImuSource` off one shared OAK-D on the bench), so odometry→ui are unchanged on
hardware. The replay path subtracts a startup gyro bias (mean of the first ~1 s)
in `ApplyCalibration` and seeds the odometry gravity-align from the first ~0.3 s
of accel, mirroring what the live front-end measures once at boot.

**Live device safety (host-side).** The OAK-D is single-client: the `cam` and
`imu_cam` live sources share ONE `SharedLiveDevice` pipeline (reference-counted).
Every read of a depthai output queue goes through `SharedLiveDevice.poll`, which
holds the same lock the teardown (`release` → `handle.stop`) holds. So the two
reader threads never enter the depthai link concurrently, and a queue is never
read while another thread is destroying the pipeline — the lifetime race that
aborted the host with `mutex lock failed: Invalid argument` and (by starving the
XLink) tripped the device firmware watchdog. Verified offline by
`oak_live_selftest` (readers hammer `poll` while a concurrent `release` destroys a
queue that raises if read post-stop).

### Per-flow files
- `cam/`: `sources.py` (replay/live `CamSource`), `publish_cam_sync.py`,
  `cam_flow.py`.
- `imu_cam/`: `sources.py` (replay/live `ImuSource`), `pack_imucam.py`,
  `apply_calibration.py`, `publish_imu_raw.py`, `publish_imucam.py`,
  `compute_depth.py`, `publish_depth.py` (depth as a task in this flow),
  `imu_stream.py` (IMU-only reader for the calib wizards), `imu_cam_flow.py`.
- `odometry/`: `preintegrate_prior.py`, `track_features.py` (KLT, holds the numba
  parallel lock), `publish_tracks.py` (emits the KLT tracks on `frame.tracks` for
  the keypoints view), `estimate_motion.py` (pure-NumPy PnP + gyro fusion,
  lock-free), `tracked.py` (TrackFeatures→EstimateMotion carrier),
  `publish_pose.py`, `emit_keyframe.py`, `step.py` (carrier), `odometry_flow.py`.
- `backend/`: `run_ba.py`, `publish_refined.py`, `backend_flow.py`.
- `slam/`: `slam_step.py`, `publish_correction.py`, `slam_flow.py`.
- `ui/`: `collect_odom.py`, `collect_refined.py`, `collect_correction.py`,
  `collector.py`, `render_pose.py`, `render.py`, `render_tracks.py`, `tracks.py`
  (the keypoint-depth sink).

---

## 5. The libraries — `ours.lib`

All 11 subpackages are live (each referenced from ≥5 places — **no dead code**;
the many folders are domain decomposition, not leftovers).

| Package | Contents | What it does |
|---|---|---|
| `frontend/` | corners, klt, klt_numba, frontend | feature detection + KLT tracking |
| `stereo/` | stereo | rectification + SGM dense depth |
| `imu/` | imu, inertial_filter, accel_calib, calib_collect, calib_store, bias_store | gyro preintegration, inertial filter, IMU calibration |
| `odometry/` | odometry, pnp | RGB-D visual odometry |
| `backend/` | bundle, windowed, vio_window | windowed bundle adjustment / VIO |
| `loop/` | orb, loopclosure, posegraph, slam | ORB loop closure + pose-graph SLAM |
| `io/` | reader, synced | session readers, frame/IMU sync |
| `config/` | resolution | resolution profiles |
| `misc/` | frames, geometry, pose, pngio | shared `Pose` / frame / geometry / PNG helpers |
| `flow/` | flow, task, pubsub, messages, topics, runtime | the flow framework (see §3) |

The stable public API is the flat re-export from `ours/lib/__init__.py`
(`from ours.lib import RGBDVisualOdometry, ORB, SessionReader, ...`).

---

## 6. UI and entry points

- **`ours/app.py`** — live-pipeline assembler. Builds one `Bus`, the six flows,
  starts their threads. Replay harness (validates the flow graph against the same
  data as the offline oracle):
  ```
  python -m ours.app --session sessions/gold/lab_straight_20s --depth-fast
  ```
- **`ours/ui/`** — PyQt6 GUI. `mainwindow.py` (feature menu + 3D viewer),
  `viewer3d.py`, `panels.py`, `calib_dialogs.py` (gyro/accel calib wizards),
  `source.py` (`PoseSource` ABC + `FakePoseSource`), `live_source.py`
  (`FlowPoseSource` bridges the live flow graph → Qt viewer, optical→NED).
- **`ours/tools/`** — offline scripts that call `ours.lib` directly: per-module
  self-tests (`*_selftest.py`), the `vio_run` oracle, diagnostics, viewers.
- **`ours/legacy/depthai_ours_vio.py`** — LEGACY monolith, kept temporarily until the
  live flow graph is verified on-device; demoted from default.

---

## 7. Invariants to keep

1. `ours` imports **zero** `oakd` / `baseline`. Verify:
   ```
   python -c "import ours.app, ours.ui.mainwindow, sys; \
     assert not [m for m in sys.modules if m=='oakd' or m.startswith('oakd.')]"
   ```
2. `depthai` stays **lazy** — importing `ours.lib` / `ours.app` must not import it.
3. **Package-only**: no loose files in `lib/` or `flows/` roots; one Task per file.
4. Flows talk **only** via Bus topics (§2). Libraries never import flows.
5. Offline self-test sweep + replay parity must stay green before committing:
   ```
   for t in orb klt stereo vio_ba ba posegraph imu_preint motion_predict \
            inertial_filter accel_calib calib_store calib_collect \
            imucam_sync flow_replay; do
     python -m ours.tools.${t}_selftest; done
   QT_QPA_PLATFORM=offscreen python -m ours.tools.ui_calib_selftest
   python -m ours.app --session sessions/gold/lab_loop_30s --max-frames 60 --depth-fast
   ```
   Expected replay: 60 `pose.odom` + 10 `pose.refined`. (`flow_replay_selftest`
   now gates this app-graph contract automatically.)

---

## 8. Migration roadmap (replace the black box one block at a time)

The pub/sub contract (§2) is what makes this safe: swap the implementation behind
a flow, keep its topics/messages, and the rest of the graph is unchanged. Each
swap is validated by (a) the module's self-test and (b) replay parity vs the
`vio_run` oracle on the gold sessions.
