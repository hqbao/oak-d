# SkySLAM — Pipeline Checkpoints for Future Comparison

**Purpose**: Define the checkpoints in the current pipeline
(BasaltVIO + RTABMapSLAM) so that, when writing `skyslam` from scratch, we can
compare each stage's output against the baseline for debugging and accuracy
measurement.

**Created**: 2026-05-29 (rewrite, principle: "only show real data")
**Reference pipeline**: `oakd/sources/depthai_slam.py` (live) + `depthai_vo.py` (VIO-only)
**Comparison target**: `skyslam/` (not written yet — detailed plan in `docs/SKYSLAM_RESEARCH.md`)

---

## 0. Top principle: HONEST VISUALIZATION

> **The recorder + viewer may ONLY show real data coming from the running
> pipeline.** They must NEVER fabricate auxiliary data (a parallel
> FeatureTracker, a hand-rolled depth lookup, a guessed sparse map, …) and
> label it as if it were output of Basalt/RTABMap.

Why it matters: the long-term strategy is to **replace one module at a time**
in the depthai pipeline. If the viewer shows fake data, replacing a module
later makes it impossible to tell whether the new result is correct (you'd
be comparing against your own previous fake → meaningless).

**Hard rules:**
- Everything the UI shows must trace back to **one real depthai Output**
  that has been subscribed to.
- Do NOT use auxiliary nodes (`FeatureTracker`, `ImageManip`, …) to "pretend"
  to visualize the internals of another blob.
- Internals of a blob (Basalt features, RTABMap keyframes/BoW, …) can only
  be exposed once we write that module ourselves. Until then: **leave it
  blank**.

---

## 1. Black-box boundaries (depthai 3.6.1)

```
┌───────────────── OAK-D (RVC2 chip) ─────────────────┐
│                                                      │
│   Camera ─┐                                          │
│           ├─► StereoDepth ──┬─► rectifiedLeft  ◄── REAL  (C0)
│   Camera ─┘   (SGBM, blob)  ├─► rectifiedRight ◄── REAL  (C0)
│                             ├─► depth           ◄── REAL  (C0)
│                             ├─► disparity       ◄── REAL  (optional)
│                             └─► confidenceMap   ◄── REAL  (optional)
│                                                      │
│   IMU ──► IMU.out                              ◄── REAL  (C1)
│                                                      │
│   ┌──────────────────────────────────┐               │
│   │ BasaltVIO (blob — BLACK BOX)     │               │
│   │  in : left, right, imu           │               │
│   │  internals: KLT, multi-view tri, │               │
│   │             sliding-window BA,   │               │
│   │             IMU preintegration   │               │
│   │  out: transform                  │ ──► pose ◄── REAL  (C2)
│   └──────────────────────────────────┘               │
│                                                      │
│   ┌──────────────────────────────────┐               │
│   │ RTABMapSLAM (blob — BLACK BOX)   │               │
│   │  in : odom, rect, depth          │               │
│   │  internals: KF policy, BoW loop, │               │
│   │             pose-graph optim,    │               │
│   │             dense map gen        │               │
│   │  out: transform                  │ ──► pose ◄── REAL  (C3)
│   │       odomCorrection             │ ──► map<-odom delta ◄── REAL  (C5 source)
│   │       obstaclePCL, groundPCL     │ ──► world cloud ◄── REAL  (PCL)
│   │       passthroughRect/Odom/...   │ ──► debug, REAL when wired
│   └──────────────────────────────────┘               │
└──────────────────────────────────────────────────────┘
```

**The EMPTY SEATS (depthai does NOT expose them):**

| Hidden internal | Why we want it | When we'll get it |
|---|---|---|
| Basalt feature tracks (2D KLT) | overlay corners on image | when skyslam frontend written |
| Basalt sparse 3D landmarks | true "feature point cloud" | when skyslam frontend written |
| Basalt BA residuals / covariance | optimizer health | when skyslam backend written |
| Basalt keyframe poses | KF policy comparison | when skyslam backend written |
| RTABMap keyframe DB | KF reasons, descriptors | extract from `rtabmap.db` SQLite (offline) |
| RTABMap BoW similarity score | loop precision/recall | same — SQLite |
| RTABMap loop constraint edges | pose-graph topology | same — SQLite |

---

## 2. Organisation principles

1. **Black-box what we can't change**: Basalt and RTABMap are binary blobs.
   Do NOT touch internals. Only log input + output streams.
2. **JSONL for metadata, binary for dense data**: timestamps + pose go to
   JSONL (easy to diff/grep); images + depth + cloud go to raw binary.
3. **Timestamps are gold**: every record carries `ts_ns` (nanosecond,
   monotonic from pipeline start). This is the join key between streams and
   between two pipelines.
4. **Reproducible**: a fully recorded session can be replayed through both
   the old pipeline (sanity) and the new pipeline (skyslam). Same input →
   only then is output comparison meaningful.
5. **Observability levels** (only what depthai actually exposes):
   - L0: input streams (camera, IMU) — always recordable
   - L1: final pose output — always recordable
   - L2: pose-graph correction (odomCorrection) — RTABMap exposes
   - L2: dense reconstruction (obstacle/ground PCL) — RTABMap exposes
   - L3+: internal features/landmarks/keyframes — **NOT exposed, leave blank**

---

## 3. Checkpoint matrix (status as of 2026-05-29)

| # | Checkpoint | Status | Source | Notes |
|---|---|---|---|---|
| C0 | Input: stereo frame (L+R+D) | ✅ live | `StereoDepth.rectifiedLeft/syncedRight/depth` | PNG + raw16 |
| C1 | Input: IMU sample | ✅ live | `IMU.out` | 200 Hz |
| C2 | VIO pose (Basalt) | ✅ live | `BasaltVIO.transform` | FLU world |
| C3 | SLAM pose (RTABMap, loop-corrected) | ✅ live | `RTABMapSLAM.transform` | FLU world |
| C4 | Keyframe selected | ✅ offline | `rtabmap.db` SQLite (`Node` table) | `tools/extract_kf_from_db.py` — auto-runs after record |
| C4+ | Loop closure links (rich) | ✅ offline | `rtabmap.db` SQLite (`Link` types 1/2/3) | same tool → `kf_loops.jsonl` |
| C5 | Loop closure detected (derived) | ⚠️ derived | jump > thr in `odomCorrection` stream | precision OK, no live BoW score |
| C6 | Tracking lost/recovered | ⚠️ derived | gap > thr in SLAM pose stream | enough to draw a timeline |
| C7 | Frontend features per frame | ❌ blob hidden | **only skyslam will provide** | do NOT fake with FeatureTracker |
| C8 | IMU preintegration delta | ❌ blob hidden | only skyslam | |
| C9 | Optimizer state (Jacobians) | ❌ blob hidden | only skyslam | |
| — | Odom correction (raw) | ✅ live | `RTABMapSLAM.odomCorrection` | input for C5 derivation |
| — | Dense point cloud | ✅ live | `RTABMapSLAM.obstaclePCL + groundPCL` | dense reconstruction, NOT a sparse map |

Legend: ✅ = real stream from the pipeline, ⚠️ = derived from a real stream,
❌ = blob does not expose it → not done yet.

---

## 4. Migration strategy: replace one blob at a time

Goal: progressively replace `BasaltVIO` then `RTABMapSLAM` with our own
skyslam modules. Each replacement UNLOCKS more checkpoints (because the
module we wrote will expose its internals).

### Phase A (current) — record an honest baseline
Record sessions with BasaltVIO + RTABMapSLAM, capturing C0, C1, C2, C3, C5,
C6, PCL. C4/C7/C8/C9 are absent. This is the **ground-truth pipeline** to
compare against.

### Phase B — replace the RTABMap backend with `skyslam_backend` (pose graph + loop)
- Input: still use `BasaltVIO.transform` (C2) as odometry.
- Output: new pose (replacing C3), plus:
  - C4 (keyframe selection): when we pick the KFs ourselves
  - Full C5 (kf_query/kf_match/BoW score/inliers): when we detect loops
  - Sparse pose-graph topology
- Validation: compare `skyslam_backend.transform` vs `RTABMapSLAM.transform`
  on the same C2 input → ATE/RPE must be ≤ baseline.

### Phase C — replace the Basalt frontend with `skyslam_frontend` (VIO)
- Input: C0 + C1 (stereo + IMU).
- Output: pose (replacing C2), plus:
  - C7 (tracked 2D features)
  - C8 (IMU preintegration deltas)
  - Sparse 3D landmarks ← **this is the real "feature point cloud"**, a few
    hundred points, completely different from RTABMap's dense PCL.
- Validation: run `skyslam_frontend` and `BasaltVIO` in parallel, compare
  pose streams + check whether the feature overlay is correlated.

### Phase D — full skyslam stack
- `BasaltVIO` and `RTABMapSLAM` are kept only for regression tests.
- All of C0…C9 are skyslam data, the viewer can overlay everything.

---

## 3. Detailed schemas

Every record has two mandatory fields: `ts_ns` (uint64) + `seq` (uint32,
monotonic).

### C0 — image_pair (stereo frame)

```jsonl
{"ts_ns": 1716950123456789000, "seq": 42, "type": "stereo",
 "left_path":  "img/000042_L.png",
 "right_path": "img/000042_R.png",
 "depth_path": "img/000042_D.raw16",
 "width": 640, "height": 400,
 "intrinsics_left":  {"fx": 285.7, "fy": 285.7, "cx": 319.5, "cy": 199.5,
                       "dist": [0,0,0,0,0]},
 "intrinsics_right": {...},
 "T_left_right": [[...]]}
```

- Image: PNG 8-bit grayscale (rectified).
- Depth: raw16 little-endian, mm units, 0 = invalid.
- Store intrinsics once in `calib.json`; the schema can reference an ID to
  stay compact.

### C1 — imu

```jsonl
{"ts_ns": 1716950123450000000, "seq": 8421,
 "gyro": [0.001, -0.002, 0.0005],
 "accel": [0.12, -0.05, 9.81],
 "temp_c": 32.5}
```

200 Hz stream. Field names are fixed to make diffs easy.

### C2 — pose6dof (raw VIO, FLU world frame as Basalt outputs)

```jsonl
{"ts_ns": 1716950123456789000, "seq": 42,
 "frame_id": "flu_world",
 "pos": [0.123, -0.456, 0.789],
 "quat_wxyz": [0.9998, 0.001, -0.002, 0.005],
 "vel": [0.05, 0.01, 0.0],
 "tracking_ok": true,
 "source": "basalt_vio"}
```

- `pos` in metres, `quat` unit, `vel` in m/s.
- `source` enum: `basalt_vio` | `rtabmap_slam` | `sky_vio` | `sky_slam`.
- Store **FLU world** (Basalt's native frame), do NOT convert to NED — so
  Basalt and skyvio can be compared apples-to-apples. NED conversion is
  post-processing.

### C3 — pose6dof (SLAM loop-corrected)

Same schema as C2, with `source = rtabmap_slam` or `sky_slam`.

### C4 — kf_event ✅ DONE (offline)

```jsonl
{"ts_ns": 2402326697, "kf_id": 1, "weight": 0,
 "pos": [0.0007, 0.0022, 0.0707],
 "quat_wxyz": [0.6513, -0.0671, -0.7532, -0.0629]}
```

- Source: SQLite `rtabmap.db` table `Node`; the pose is a 48-byte BLOB
  (3x4 float32 row-major `[R|t]`), with timestamp in seconds from device
  boot.
- Extract tool: `tools/extract_kf_from_db.py` — auto-runs at the end of
  `record_session.py` (see the last log line: `kf: N keyframes, M loop
  links`).
- Auxiliary loop links (C4+) are written in parallel to `kf_loops.jsonl`
  from the `Link` table for types ∈ {1=global, 2=local_space,
  3=local_time}:

```jsonl
{"ts_ns": ..., "from_kf": 12, "to_kf": 3, "type": 1, "type_name": "global",
 "transform_pos": [...], "transform_quat_wxyz": [...]}
```

- `weight=-9` denotes a RTABMap intermediate node (graph reduction); `weight=0`
  is a normal keyframe.
- The viewer (`viz_session.py`) renders each KF as a yellow dot and each loop
  closure as a yellow line between two KFs in the Pose 3D tab.

### C5 — loop_event

```jsonl
{"ts_ns": ..., "event": "loop_closure",
 "pos_jump_m": 0.18, "rot_jump_deg": 3.4,
 "correction_pos": [...], "correction_quat_wxyz": [...]}
```

- Detected by tapping `RTABMapSLAM.odomCorrection` (the map←odom transform)
  and measuring the delta between consecutive samples. A large jump
  (default `pos > 10 cm` or `rot > 5°`) = a loop closure correction was
  applied.
- Skip the first `LOOP_WARMUP_S` (default 3 s) because RTABMap publishes a
  noisy correction while the map is still initialising (causes dozens of
  false positives).
- The BoW schema (`kf_query`, `score`, `inliers`) is only available via
  SQLite — `tools/extract_kf_from_db.py` already extracts the `Link` table
  (loop edges with transform + type) into `kf_loops.jsonl`. The score +
  inlier count are present in the `Link.information` column but are not
  parsed yet (TODO if needed).
- The raw stream is also dumped in parallel to
  `basalt/odom_correction.jsonl` for debugging.

### C6 — track_event

```jsonl
{"ts_ns": ..., "event": "tracking_lost", "last_pose_seq": 142, "gap_s": 0.72}
{"ts_ns": ..., "event": "tracking_recovered", "first_pose_seq": 158}
```

- Derived by scanning the SLAM pose stream for gaps > `TRACK_GAP_S` (default
  0.5 s). Computed in `SessionRecorder.close()`.

### C7 — features (tracked corners) — ❌ NOT IMPLEMENTED

The depthai `BasaltVIO` blob does NOT expose internal feature tracks. We
previously tried using `dai.node.FeatureTracker` in parallel to pretend —
**dropped because it violates Section 0** (FeatureTracker uses Harris /
Shi-Tomasi, which is completely different from Basalt's internal KLT, so the
overlay would misrepresent what the VIO is actually tracking).

To be implemented in Phase C (see Section 4) when writing
`skyslam_frontend`.

### Point cloud (RTABMap dense reconstruction)

Index `basalt/pointcloud.jsonl`:
```jsonl
{"ts_ns": ..., "seq": 0, "kind": "obstacle", "n_points": 5421,
 "path": "pointcloud/000000_obstacle.f32"}
```
- Binary file: raw `Nx3 float32` little-endian, world (FLU map) frame.
- RTABMap republishes the FULL cloud at every keyframe → the viewer only
  loads the most recent emission per kind (to avoid duplication).
- **This is DENSE reconstruction from the depth map**, NOT the sparse VIO
  map. Every valid depth pixel becomes one 3D point after a voxel filter
  (default 5 cm). That yields hundreds of thousands of points for a small
  room.
- Disable with `record_session --no-pcl`.

### C8-C9 — only implemented once skyslam is written

---

## 4. Session folder layout

```
sessions/
└── 2026-05-29_oak_lab_loop1/
    ├── calib.json              # intrinsics + extrinsics, written once
    ├── meta.json               # pipeline version, params, host info
    ├── input/
    │   ├── imu.jsonl           # C1
    │   ├── frames.jsonl        # C0 metadata
    │   └── img/                # PNG + raw16, ~50 MB / min
    ├── basalt/                 # output from the current pipeline (REAL streams)
    │   ├── vio_pose.jsonl         # C2 — BasaltVIO.transform
    │   ├── slam_pose.jsonl        # C3 — RTABMapSLAM.transform
    │   ├── odom_correction.jsonl  # raw RTABMapSLAM.odomCorrection
    │   ├── loop_events.jsonl      # C5 (derived from odom_correction jumps)
    │   ├── track_events.jsonl    # C6 (derived from slam pose gaps)
    │   ├── pointcloud.jsonl      # index of dense PCL emissions
    │   └── pointcloud/NNNN_*.f32  # RTABMap obstacle/ground PCL, Nx3 f32
    └── skyslam/                # output from future skyslam
        ├── vio_pose.jsonl         # C2 replacement (Phase C)
        ├── slam_pose.jsonl        # C3 replacement (Phase B)
        ├── kf_events.jsonl       # C4 (UNLOCKED with skyslam backend)
        ├── features.jsonl        # C7 (UNLOCKED with skyslam frontend)
        ├── landmarks.jsonl       # sparse 3D landmarks (UNLOCKED with frontend)
        └── ...
```

`meta.json`:
```json
{
  "session_id": "2026-05-29_oak_lab_loop1",
  "pipeline": "basalt_vio + rtabmap_slam",
  "depthai_version": "3.6.1",
  "host": "macmini-m1",
  "sensor": "OAK-D W",
  "duration_s": 124.5,
  "params": {
    "width": 640, "height": 400, "fps": 20,
    "imu_rate_hz": 200,
    "rtabmap_detection_rate": 1
  }
}
```

---

## 5. How to record from the current pipeline

Need an additional module `oakd/recorder.py`:

```python
class SessionRecorder:
    def __init__(self, out_dir: Path): ...
    def on_stereo_frame(self, left, right, depth, ts_ns, seq): ...
    def on_imu_sample(self, sample, ts_ns, seq): ...
    def on_vio_pose(self, transform, ts_ns, seq): ...
    def on_slam_pose(self, transform, ts_ns, seq): ...
    def on_kf_event(self, kf_data): ...
    def on_loop_event(self, loop_data): ...
    def close(self): ...
```

Wire it into `OakBasaltSlamSource`: add taps so every output queue fans out
into the recorder. New CLI flag: `--record sessions/<name>`.

Need to study the RTABMap nodes API to know which queue emits KF/loop
events (possibly `slam.passthroughRect` + statistics output, or query the
database file after the run).

---

## 6. Replay protocol for future skyslam

Once skyslam reaches Phase 1 (sensor abstraction):

1. `sky_replay --session sessions/2026-05-29_oak_lab_loop1` pushes input
   C0+C1 into skyvio/skyslam.
2. skyslam emits C2+C3+C4+C5+C6 to `sessions/.../skyslam/`.
3. The tool `compare_sessions.py basalt/ skyslam/` computes metrics.

→ Same input, two pipelines, two sets of output, objective difference
measurement.

---

## 7. Comparison metrics

### 7.1 Pose accuracy (C2, C3)

- **ATE** (Absolute Trajectory Error): after aligning the trajectory
  (Umeyama SE3), RMSE of position error. Unit: metres.
- **RPE** (Relative Pose Error): error in a 1 m / 1 s sliding window.
  Split translation and rotation components.
- Implementation reference: `evo` toolbox (Python, MIT), or write a custom
  `compare_pose.py` ~200 LOC.

### 7.2 Keyframe agreement (C4)

- **KF count ratio**: `n_kf_skyslam / n_kf_basalt`. Target 0.7–1.5.
- **KF temporal overlap**: % of Basalt KFs that have a skyslam KF within
  ±0.5 s.

### 7.3 Loop closure (C5)

- **Recall**: % of Basalt loops that skyslam also detects.
- **Precision**: % of skyslam loops that are correct (Basalt as ground
  truth).
- **Drift correction**: ATE before vs after loop, measured at points
  classified as "passing by a previously visited place".

### 7.4 Robustness (C6)

- **Tracking uptime**: % of time with `tracking_ok=true`.
- **Recovery time**: mean seconds from `lost` to `recovered`.

### 7.5 Performance (not a checkpoint, measured alongside)

- **FPS** mean + p99 latency.
- **CPU %** mean.
- **RAM** peak.

Stored in each run's `meta.json`.

---

## 8. Baseline gold standard

Before building skyslam, record 3–5 "gold" sessions as a fixed baseline:

| Session | Description | Purpose |
|---|---|---|
| `lab_loop1_120s` | Desk → one lap of the room → return | Basic loop closure |
| `lab_figure8_180s` | Figure-8 in the room, 2 loops | Multi-loop |
| `hallway_oneway_60s` | Walk straight down a corridor, no loop | Pure VIO, no SLAM correction |
| `static_30s` | Sit still on a desk for 30 s | Drift baseline, IMU bias |
| `aggressive_motion_60s` | Fast shaking, sudden rotations | Robustness stress |

Each session records full C0+C1+C2+C3+C4+C5+C6 from the Basalt+RTABMap
pipeline. Store under `sessions/` (gitignored, use an external drive or
Git LFS).

When skyslam reaches each phase, replay these 5 sessions → compute metrics
→ plot a "skyslam vs basalt" regression chart over time.

---

## 9. Implementation TODO

Pre-skyslam infra ✅ DONE:

- [x] `oakd/recorder.py` — fan-out tap on every queue
- [x] `tools/record_session.py` — CLI with `--duration`, `--no-pcl`, `-f`
- [x] `tools/compare_sessions.py` — ATE/RPE between two pose streams
- [x] Recorded 6 gold sessions (see `docs/GOLD_SESSIONS.md`)
- [x] `tools/extract_kf_from_db.py` — extract KF + loops from rtabmap.db
- [x] `tools/baseline_report.py` — emit Markdown baseline
- [x] Frozen baseline (`docs/GOLD_BASELINE.md`)

Skyslam work — see **`docs/SKYSLAM_RESEARCH.md`** Part 3 for plan v3
(9 phases with acceptance gates).

---

## 10. Cross-references

- **Skyslam plan v3 (research-backed)**: `docs/SKYSLAM_RESEARCH.md`
- Gold regression suite: `docs/GOLD_SESSIONS.md` + `docs/GOLD_BASELINE.md`
- Long-term hardware/FC vision: `docs/SKYSLAM_ROADMAP.md`
- Current SLAM source: `oakd/sources/depthai_slam.py`
- Current VIO source: `oakd/sources/depthai_vo.py`
- Pose data structure: `oakd/pose.py`

---

## 11. History

| Date | Change | Author |
|---|---|---|
| 2026-05-29 | First draft | Bao + Copilot |
| 2026-05-29 | Honest pipeline rewrite (drop fake FeatureTracker overlay) | Bao + Copilot |
| 2026-05-29 | C4 done: KF + loop links from rtabmap.db SQLite | Bao + Copilot |
| 2026-05-29 | Gold suite expanded → 6 sessions (added loop_closure_45s) | Bao + Copilot |
| 2026-05-29 | Translate document to English | Bao + Copilot |
