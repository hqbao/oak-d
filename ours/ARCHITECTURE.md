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
| `flow.py` | `Flow` / `SourceFlow` / `FlowContext` — one thread running a fixed task chain |
| `task.py` | `Task` — the smallest input→output step in a chain |
| `pubsub.py` | `Bus` — thread-safe publish/subscribe between flows |
| `messages.py` | one immutable message type per topic (the flow contract) |
| `topics.py` | topic-name constants |
| `runtime.py` | process-wide guards (e.g. `NUMBA_PARALLEL_LOCK`) |

Layering: depends only on the standard library + numpy. It does **not** import
the concrete flows or the algorithm libraries.

### Numba concurrency guard
`numba parallel=True` is used only in `lib/stereo` (SGM) and
`lib/frontend/klt_numba` (KLT). The default `workqueue` numba layer is **not**
threadsafe across Python threads, so running the depth flow (SGM) and odometry
flow (KLT) concurrently crashes. `runtime.NUMBA_PARALLEL_LOCK` serializes the two
parallel regions; all other flows (pure numpy) run free. `tbb`/`omp` layers are
not installable on this host (macOS arm64, py3.13).

---

## 4. The flows — `ours.flows`

One thread per flow; **one Task per file**; a `*_flow.py` only wires the tasks.

### Topic-level data flow (which flow publishes/subscribes which topic)

```
cam-reader ──cam.sync──► imu-reader ──imucam.sample──► depth ──frame.depth──► odometry ──pose.odom──► ui-collector, ui-render
                                    ──imucam.sample──────────────────────────►          ──keyframe──► backend, slam
                                    ──imu.raw───────► (visualiser)
                                                                       backend ──pose.refined──► ui-collector
                                                                       slam    ──loop.correction──► ui-collector
```

Edges above are exactly the `self.on(...)` subscriptions in each `*_flow.py`.
There is ONE acquisition front-end (cam-reader + imu-reader) shared by the VIO
and the camera/IMU visualiser — no separate capture monolith. Things worth
noting because the obvious guess is wrong:

- **depth and odometry both consume `imucam.sample`** (the synced frame+IMU
  packet); depth uses only the stereo pair, odometry also integrates the packet's
  gyro into the per-frame rotation prior (`PreintegratePrior`).
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
    subgraph CAM["cam-reader flow (SourceFlow)"]
        PRODC["produce()"] --> PCS["PublishCamSync"]
    end
    subgraph IMU["imu-reader flow"]
        PIC["PackImuCam"] --> PIR["PublishImuRaw"] --> AC["ApplyCalibration"] --> PICAM["PublishImuCam"]
    end
    subgraph DEP["depth flow"]
        CD["ComputeDepth"] --> PD["PublishDepth"]
    end
    subgraph ODO["odometry flow"]
        PIP["PreintegratePrior"]
        PV["ProcessVO"] --> PP["PublishPose"] --> EK["EmitKeyframe"]
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

    PCS -- "cam.sync" --> PIC
    PICAM -- "imucam.sample" --> CD
    PICAM -- "imucam.sample" --> PIP
    PD -- "frame.depth" --> PV
    PP -- "pose.odom" --> COd
    PP -- "pose.odom" --> RP
    EK -- "keyframe" --> RB
    EK -- "keyframe" --> SS
    PR -- "pose.refined" --> CRf
    PC2 -- "loop.correction" --> CCo

    PIP -. "ctx.state['priors'][seq]\n(same thread, not via Bus)" .-> PV
```

The dotted edge is the one **intra-flow** hand-off: `PreintegratePrior` stashes
the gyro prior for sequence `seq` in the odometry flow's own `ctx.state`, and
`ProcessVO` pops it when the matching depth frame arrives. This is shared state
inside a single thread/flow — it does **not** cross the Bus and does **not**
violate the §2 rule (which only forbids *cross-flow* calls).

| Flow | Tasks (in order) | Subscribes | Publishes |
|---|---|---|---|
| **cam-reader** | `produce` → `PublishCamSync` | — (source) | `cam.sync` |
| **imu-reader** | `AdmitFrame` → `PackImuCam` → `PublishImuRaw` → `ApplyCalibration` → `PublishImuCam` ; `CompleteAdmission` | `cam.sync`, `frame.done` | `imu.raw`, `imucam.sample` |
| **depth** | `ComputeDepth` → `PublishDepth` | `imucam.sample` | `frame.depth` |
| **odometry** | `PreintegratePrior` ⟂ `ProcessVO` → `PublishPose` → `EmitKeyframe` → `SignalDone` | `imucam.sample`, `frame.depth` | `pose.odom`, `keyframe`, `frame.done` |
| **backend** | `RunBA` → `PublishRefined` | `keyframe` | `pose.refined` |
| **slam** | `SlamStep` → `PublishCorrection` | `keyframe` | `loop.correction` |
| **ui-collector** | `CollectOdom` / `CollectRefined` / `CollectCorrection` | `pose.odom`, `pose.refined`, `loop.correction` | — (sink) |
| **ui-render** | `RenderPose` | `pose.odom` | — (sink) |

`cam-reader` + `imu-reader` are the only device-specific flows; their sources are
injected (`ReplayCamSource`/`ReplayImuSource` offline, `LiveCamSource`/
`LiveImuSource` off one shared OAK-D on the bench), so depth→ui are unchanged on
hardware. The replay path subtracts a startup gyro bias (mean of the first ~1 s)
in `ApplyCalibration` and seeds the odometry gravity-align from the first ~0.3 s
of accel, mirroring what the live front-end measures once at boot.

**Realtime backpressure (live only).** The camera streams at a fixed fps but the
VIO sustains less; with unbounded inboxes the surplus ~0.5 MB stereo packets pile
up on the host until memory pressure starves the depthai link and the device
firmware **watchdog crashes the camera**. The imu-reader therefore admits at most
`N` frames in flight (`BudgetAdmission`, default 2): `AdmitFrame` (first in the
chain, before the IMU is drained) skips a frame at the source when over budget —
so the skipped interval's IMU folds into the next admitted frame and the diamond
never desyncs — and `SignalDone` (odometry tail) publishes `frame.done` to free a
credit. The gate sits BEFORE the depth/odometry fan-out, so both branches always
see the identical surviving subset. Replay injects `AdmitAll` (admit everything,
never count) so it stays byte-for-byte deterministic (60-for-60).

### Per-flow files
- `cam_reader/`: `sources.py` (replay/live `CamSource`), `publish_cam_sync.py`,
  `cam_reader_flow.py`.
- `imu_reader/`: `sources.py` (replay/live `ImuSource`), `pack_imucam.py`,
  `apply_calibration.py`, `publish_imu_raw.py`, `publish_imucam.py`,
  `admission.py` (realtime credit gate) + `admit_frame.py` / `complete_admission.py`,
  `imu_stream.py` (IMU-only reader for the calib wizards), `imu_reader_flow.py`.
- `depth/`: `compute_depth.py`, `publish_depth.py`, `depth_flow.py`.
- `odometry/`: `preintegrate_prior.py`, `process_vo.py`, `publish_pose.py`,
  `emit_keyframe.py`, `signal_done.py` (backpressure credit), `step.py` (carrier),
  `odometry_flow.py`.
- `backend/`: `run_ba.py`, `publish_refined.py`, `backend_flow.py`.
- `slam/`: `slam_step.py`, `publish_correction.py`, `slam_flow.py`.
- `ui/`: `collect_odom.py`, `collect_refined.py`, `collect_correction.py`,
  `collector.py`, `render_pose.py`, `render.py`.

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
            imucam_sync flow_replay admission; do
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
