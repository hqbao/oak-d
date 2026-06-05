# SkySLAM — Research & Implementation Plan v3

**Status:** Research complete (2026-05-29). Implementation pending.
**Goal:** Replace depthai's `RTABMapSLAM` (and eventually `BasaltVIO`) blobs with our own Python+C/Cython package `skyslam/`, with near-equal accuracy to baseline, full visibility into internals.

---

## Part 0 — Decision summary (TL;DR)

If you read nothing else, read this.

1. **Skyslam will run on host CPU** (not on OAK-D device). Same as depthai's blobs — they're also host-side `ThreadedHostNode`s, NOT device-side firmware. So CPU budget is the same constraint either way.

2. **VIO backend choice: loosely-coupled EKF first, MSCKF as upgrade.** Research shows ATE gap is real:
   - Loosely-coupled EKF: ~10-20 cm ATE on EuRoC
   - MSCKF (OpenVINS): ~3-5 cm
   - SQRT-VIO (Basalt): ~2-6 cm
   - **Strategy**: ship loosely-coupled EKF MVP (~2 weeks), prove pipeline end-to-end via gold suite, then upgrade backend if accuracy insufficient. Risk-mitigation > performance-first.

3. **Frontend: patch-based KLT with grid detection** (Basalt-style, NOT pure OpenCV). Reasons in §2.3.

4. **Two feature types needed**: KLT patches for VIO frontend + ORB descriptors at keyframes for loop closure (DBoW2 needs descriptors). Standard ORB-SLAM3 architecture.

5. **Loop closure: ORB + DBoW2 + Consistency Groups + Sim(3)/SE(3) RANSAC + Essential Graph optim.** Use gtsam for backend. Do NOT skip consistency groups (this is what makes ORB-SLAM3 work in practice).

6. **Calibration**: pinhole + radtan (OAK-D rectified output already pinhole). Use depthai `calib.json` for camera intrinsics + IMU-cam extrinsic. **DO NOT** estimate online — overkill for our case.

7. **Quaternion convention: Hamilton (scalar-first, body→world)**. Already used in our codebase. Avoid JPL (OpenVINS uses it — that's why their math looks different).

8. **Library stack confirmed**: numpy + opencv (frontend) + gtsam (backend optim) + pyDBoW3 (vocab). cython/numba for hot loops if needed.

9. **Total effort estimate**: 6-9 weeks part-time, in 9 phases.

---

## Part 1 — Research synthesis

### 1.1 depthai BasaltVIO node

**Architecture:** Host CPU, inherits `ThreadedHostNode`. TBB multi-threading internally.

**Inputs (4):** left (Sync), right (Sync), IMU packet stream.
**Outputs (2):** `transform` (TransformData = pose + velocity + bias), passthrough left image.

**~60 VioConfig parameters exposed**, all are pass-through to upstream basalt config struct. Key ones we'll mirror in skyslam:
- `optical_flow_detection_grid_size` (default 50)
- `optical_flow_max_recovered_dist2` (0.04 px²)
- `optical_flow_epipolar_error` (0.005)
- `vio_max_states` (3), `vio_max_kfs` (7)
- `vio_obs_huber_thresh` (1.0)
- `vio_outlier_threshold` (3.0 σ)
- `vio_init_pose_weight` (1e8), `vio_init_ba_weight` (1e1), `vio_init_bg_weight` (1e2)

**Upstream fork:** Luxonis custom commit `ef61684` of `Basalt` on branch `depthai` (not upstream master).

**What's NOT exposed:**
- Keyframe poses, landmark map, IMU preintegrated factors.
- Internal state covariance, marginalization prior.
- Per-frame inlier counts (only final pose).
- Loop closure events (BasaltVIO doesn't do loop closure — that's RTABMap's job).

---

### 1.2 depthai RTABMapSLAM node

**Architecture:** Host CPU, ThreadedHostNode, SINGLE-THREADED (cannot parallelize SLAM backend). `setFreq()` defaults to 1 Hz — SLAM processes at 1 Hz while inputs may be higher rate.

**Inputs (4):** rectified image, depth, IMU (optional), odometry (typically from BasaltVIO).
**Outputs (9):** `transform`, `odomCorrection`, `obstaclePCL`, `groundPCL`, `occupancyGrid` (2D), `obstacleGridMap` (3D), passthroughs.

**Exposed setters (~10):**
- `setDatabasePath`, `setSaveDatabaseOnClose` (we use these — C4)
- `setLoadDatabaseOnStart`
- `setPublishTF`, `setPublishGroundCloud`, `setPublishObstacleCloud`
- `setLocalTransform` (sensor-to-base TF)
- `setFreq`, `setAlphaScaling` (depth disparity)
- `triggerNewMap()` (soft session reset)

**Upstream:** Official `introlab/rtabmap` commit `08f031e` (v0.21 + 8 Luxonis patches, mostly build-system).

**What's NOT exposed (this is huge — ~1500 RTABMap params are locked):**
- Feature detector choice (locked to ORB, 500 features/frame, 8 pyramid levels).
- BoW vocabulary path (uses bundled).
- Loop threshold (`Rtabmap/LoopThr` locked at default 0.9).
- Optimizer backend (`Optimizer/Strategy` locked at GTSAM if present).
- All memory tier thresholds (`Mem/STMSize=10` etc.).
- Bayesian filter prior (`Bayes/PredictionLC`).

**Best workaround we already use:** `rtabmap.db` SQLite persistence + offline extraction tool. Database contains EVERYTHING (poses, features, BoW words, link types, statistics) — see §1.4.

---

### 1.3 Basalt VIO algorithm (what runs inside the blob)

**Reference:** Usenko et al. 2019 "Visual-Inertial Mapping with Non-Linear Factor Recovery", Demmel et al. 2021 "Square Root Marginalization for Sliding-Window BA".

#### Frontend (multiscale frame-to-frame optical flow)

- **Detector:** FAST on spatial grid (50×50 cells, 1 feature per cell → ~300-800 features). Detection on ALL pyramid levels (3 levels default), enabling long-term tracking of small features.
- **Tracker:** Patch-based Lucas-Kanade with **SE(2) affine warp** (not just 2D translation). Iterative Gauss-Newton, 5 iterations max.
- **Stereo:** Tracks left and right independently, then 3 geometric filters:
  1. Forward-backward consistency: threshold 0.04 px² on recovered position.
  2. Epipolar constraint via essential matrix (pre-computed from calib): threshold 0.005.
  3. Spatial inhibition for re-detection (avoid clustering).
- **Patch size:** 8×8 or 16×16 (pattern code 51).

**Insight:** SE(2) affine warp matters for fish-eye/wide FOV. For our 90° OAK-D, pure translation KLT is fine. Skip affine, save 40% compute.

#### IMU preintegration (Forster-style on-manifold)

State per preintegrated interval: 9-DoF $(\Delta R, \Delta v, \Delta p)$ between keyframes.

**Bias correction Jacobian trick** (critical for performance):
$$\Delta x_{\text{corrected}} = \Delta x_{\text{nominal}} + J_{b_a}(b_a - \bar{b}_a) + J_{b_g}(b_g - \bar{b}_g)$$
Jacobians computed during integration, allow bias update WITHOUT re-integrating from scratch. Without this, every bias update = re-process all IMU samples in interval = 10× slowdown.

#### Backend (SQRT-keypoint-VIO sliding window)

**Two pose representations:**
1. **Frame poses** (6-DoF SE(3)) — for non-keyframes
2. **Frame states** (15-DoF) — pose + velocity + biases, only at keyframes, connected by IMU preintegrated factors

**Window:** `max_states=3` (full 15-DoF), `max_kfs=7` (keyframes total).

**Landmarks:** Anchored inverse depth (1-D per landmark + host KF ID + 2D bearing direction). Bounds scale drift at far depths.

**Cost function:**
$$\text{Cost} = \sum_{ij} w_{\text{vis}} \rho_H(r_{\text{vis}}^{ij}) + \sum_k w_{\text{imu}} \, r_{\text{imu}}^T \Sigma_{\text{imu}}^{-1} r_{\text{imu}}$$
Huber threshold 1.0, LM solver, QR-based square-root marginalization.

**Square-root marginalization (key insight):**
- Avoids explicit Schur complement → numerically stable in float32.
- $\text{cond}(R) = \sqrt{\text{cond}(R^T R)}$ → halves condition number.
- Uses Givens rotations for online updates.

**Initialization:**
1. Align first IMU sample's acceleration to gravity (yaw unknown).
2. Strong pose prior (weight $10^8$), weak bias priors ($10$–$100$).
3. Landmark triangulation only after `vio_min_frames_after_kf=5` frames AND < 70% tracking ratio.

**Reported performance:**
- EuRoC V1-02 Easy: 2.6 cm ATE
- EuRoC V1-03 Difficult: 5.8 cm ATE
- 30-50 FPS on Intel i7 single core (with TBB)

**Camera model used:** Double-sphere (Kannala-Brandt). **We don't need this** — OAK-D rectified output is pinhole.

---

### 1.4 RTABMap algorithm (what runs inside the blob)

**Reference:** Labbé & Michaud 2019 "RTAB-Map as an open-source lidar and visual SLAM library".

#### Three-tier memory

- **STM** (Short-Term Memory): 10 most recent signatures in RAM.
- **WM** (Working Memory): bounded cache, candidate set for loop queries.
- **LTM** (Long-Term Memory): SQLite DB on disk, loaded on-demand.

**Forgetting policy:** weak nodes with old timestamps move STM→WM→LTM. Recent neighbors of loop closures protected.

#### Signature / BoW

- **Features:** ORB (default), 500 per frame, 8 pyramid levels, scale factor 1.2, FAST threshold 20 (adaptive down to 7).
- **Descriptor:** rBRIEF 256-bit (32 bytes).
- **Vocabulary:** DBoW2 (NOT DBoW3), hierarchical tree.
- **TF-IDF:** $\text{weight} = \frac{n_{w,f}}{N_f} \cdot \log\frac{N}{n_w}$
- **Query score:** L1-normalized cosine similarity.

#### Loop closure pipeline

1. **BoW retrieval**: top-K candidates from WM+LTM.
2. **Bayesian filter** (this is the part we'll skip in simpler implementations):
   $$P(H_i \mid Z_t) = \frac{P(Z_t \mid H_i) \cdot P(H_i \mid Z_{t-1})}{\sum_j P(Z_t \mid H_j) \cdot P(H_j \mid Z_{t-1})}$$
   Graph-distance prior weights `[0.1, 0.36, 0.30, 0.16, 0.062, ...]` — neighbors more likely than far nodes. Virtual hypothesis ID=-1 absorbs "no match" probability.
3. **Acceptance**: posterior ≥ `LoopThr=0.9` AND top/second ratio ≥ `LoopRatio`.
4. **Verification**: PnP RANSAC on matched descriptors.
5. **Pose graph optim** (default GTSAM iSAM2): triggered after every accepted closure.

#### SQLite schema (full version, supplements what we already probed)

| Table | What's in it | Use for skyslam |
|---|---|---|
| `Node` | id, map_id, weight, pose (3×4 f32), stamp, label, ground_truth_pose, velocity, gps, env_sensors | Already extracting. |
| `Link` | from_id, to_id, type (0..8), transform (3×4 f32), information_matrix (6×6 f64), user_data | Already extracting. **NEW:** types 7=gravity, 8=IMU constraints exist. |
| `Data` | id, image (JPEG), depth (RVL/PNG), depth_confidence, scan, scan_info, user_data, occupancy_grid, calibration | **NEW**: contains compressed raw frame per node! Can extract for offline replay. |
| `Feature` | node_id, word_id, pos_x/y, size, dir, response, octave, depth_x/y/z, descriptor_size, descriptor (32 bytes) | **NEW**: full 2D+3D feature positions + ORB descriptors. Huge for analysis. |
| `Word` | id, descriptor_size, descriptor (cluster centroid) | DBoW2 vocabulary used. |
| `GlobalDescriptor` | node_id, type, descriptor, nui | Global place-descriptor per node. |
| `Statistics` | id, stamp, process_time, memory_used, db_memory_used, visual_inliers, hypothesis_ratio, ... (~50 fields) | **NEW**: per-frame metrics — can plot RTABMap performance over time. |
| `Info` | key, value | Build/version metadata. |
| `Admin` | id, version, key, value | Params used during recording. |

**Action item for future**: extend `baseline/tools/extract_kf_from_db.py` to dump `Feature` and `Statistics` tables too. Free data we've been ignoring.

#### Link types (full enum)

| Type | Name | When |
|---|---|---|
| 0 | kNeighbor | Sequential odom link (every KF gets one) |
| 1 | kGlobalClosure | Bayesian-accepted loop closure |
| 2 | kLocalSpaceClosure | Spatial proximity (< 5m, 2D grid lookup) |
| 3 | kLocalTimeClosure | Temporal proximity (last N frames) |
| 4 | kUserClosure | Manual (UI) |
| 5 | kVirtualClosure | Graph reduction parent-child |
| 6 | kLandmark | Fiducial markers |
| 7 | kGravity | Gravity-axis prior (IMU) |
| 8 | kIMU | IMU preintegration constraint |

Our extractor currently only filters types 1-3. Add 0 (neighbor) for full topology.

---

### 1.5 ORB-SLAM3 architecture (best reference for clean python port)

**Reference:** Campos et al. 2021 "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM".

#### Four threads

1. **Tracking** (main): per-frame ORB + motion model + local map tracking + KF decision.
2. **LocalMapping**: process new KFs → triangulate → local BA → KF culling.
3. **LoopClosing**: BoW query → Consistency Groups → Sim3/SE3 RANSAC → loop fusion + Essential Graph optim.
4. **Viewer** (optional).

Thread synchronization via mutex + condition variables. LoopClosing can pause LocalMapping during corrections.

#### Tracking state machine

```
NO_IMAGES_YET → NOT_INITIALIZED → OK ⇄ RECENTLY_LOST → LOST
                                   ↑
                                   └── after 5s of failure, spawn new map
```

**Stereo init is immediate** (no parallax needed — depth from stereo). Critical difference vs monocular which needs 2-frame triangulation.

**Per-frame in OK state:**
1. `TrackWithMotionModel()`: predict via constant-velocity or IMU integration → project last frame's MPs to current → match → PnP.
2. `TrackReferenceKeyFrame()` fallback if motion model fails.
3. `TrackLocalMap()`: collect covisible KFs and MPs → match → refine pose.

#### LocalMapping

Triangulates new MPs from current KF + covisible neighbors (NOT just stereo).
**Local BA window**: current KF + 1st-neighbor covisibles (~10-20 KFs), fixed KFs = 2nd-neighbor covisibles.
**KF culling**: drop KF if 50%+ MPs observed by ≥3 other KFs at similar scale (stereo: 50%, mono: 90%).

#### LoopClosing — Consistency Groups

**This is the secret sauce.** Single BoW match has ~10% false positive rate. ORB-SLAM3 requires:

1. DBoW2 returns top-K candidates.
2. For each candidate, compute Sim3 via 3-point RANSAC → reproject MPs → if ≥80 reprojection matches → mark as consistent.
3. Track "consistency groups": tuple `(set_of_recent_match_KFs, consistency_counter)`.
4. **Accept loop only when consistency_counter ≥ 3** (3 consecutive KFs all match the same place).

Without consistency groups: random ORB-SLAM3 runs blow up at first false positive. With it: extremely robust.

#### Essential Graph optim (after loop accepted)

Pose-only optimization (no MPs) — much faster than full BA. Edge types:
- Spanning tree (parent-child)
- Covisibility edges (high-overlap pairs)
- Loop closure edges (new)
- Map merging edges (Sim3)

For monocular: 7-DoF Sim3 nodes (scale ambiguity).
For stereo: 6-DoF SE3 nodes.
For IMU-inertial: 4-DoF `OptimizeEssentialGraph4DoF` (gravity fixes roll+pitch, only translation+yaw free).

#### IMU initialization (3 stages, super important)

- **VIBA1** (after ~1s + 5 KFs): estimate scale, gravity direction, gyro bias via constrained linear system.
- **VIBA2** (after ~10s): full inertial BA, refines accel bias + velocities.
- **VIBA refinement** (after 100s): re-run BA with locked scale.

This staged approach is needed because pose graph isn't observable for all parameters at once.

#### Atlas (multi-map)

Tracking lost > 5s → spawn new map. When place recognition matches KF in different map → MergeLocal() fuses via Sim3.

**For our drone case**: skip Atlas. Single map per session is fine.

---

### 1.6 OpenVINS MSCKF (reference for EKF backend)

**Reference:** Geneva et al. 2020 "OpenVINS: A Research Platform for Visual-Inertial Estimation".

#### State definition (much larger than loosely-coupled EKF)

$$\mathbf{x} = [\mathbf{x}_{IMU}^{15}, \mathbf{x}_{clones}^{7N}, \mathbf{x}_{SLAM}^{3M}, \mathbf{x}_{calib}^{?}]$$

- IMU 15 DoF (pose + vel + bias) — same as loosely-coupled.
- Sliding window of N=11 camera **clones** (7 DoF each: pose only, no vel/bias).
- M=25 SLAM features (3D each), persistent.
- Optional online calibration (cam intrinsics, IMU-cam extrinsic, time offset).

Total state size ~140 DoF — vs loosely-coupled EKF's 15. **20× bigger state → 20² = 400× more covariance entries to track. CPU cost is real.**

#### Propagation: RK4 or analytical integration

```
q_{k+1} = exp(0.5 * Δt * Ω(ω̄)) ⊗ q_k
p, v: integrate kinematics
```

State transition matrix Φ block structure (5×5 of 3×3 blocks). Process noise $Q_d = \int_0^{\Delta t} \Phi(\tau) G Q_c G^T \Phi(\tau)^T d\tau$ via Van Loan or closed-form.

#### MSCKF Update — Null-Space Projection (the secret sauce)

For feature $j$ observed in $M$ clones:
$$\mathbf{H}_f \in \mathbb{R}^{2M \times 3}, \quad \mathbf{H}_x \in \mathbb{R}^{2M \times n}$$

Apply Givens rotations to zero out $\mathbf{H}_f$ (project onto its left null space):
$$\tilde{\mathbf{H}}_x = \mathbf{H}_x[3:\text{end}, :], \quad \tilde{\mathbf{r}} = \mathbf{r}[3:\text{end}]$$

Result: $2M - 3$ constraints that DO NOT involve the 3D feature position. Feature marginalized out, but its observability transferred to camera poses.

**This is why MSCKF is more accurate than loosely-coupled**: each feature acts as a multi-pose constraint, not just a single-frame pose measurement.

#### SLAM Feature Update

Features that persist across many clones get promoted to "SLAM features": added to state vector permanently. Standard EKF update for these.

#### Chi-squared gating

$$\chi^2 = \tilde{\mathbf{r}}^T \mathbf{S}^{-1} \tilde{\mathbf{r}} \leq \chi^2_{0.95}(\text{dof})$$
Pre-computed lookup table.

#### Static + Dynamic initialization

- Static: detect stationary period (jerk threshold), align gravity, set bias = avg accel - R^T g.
- Dynamic: solve linear system for (velocity, gravity, features) → refine with Ceres → must have 45°+ rotation accumulated and ≥8 features with ≥2 obs.

#### First-Estimate Jacobians (FEJ)

Jacobians computed at first feature insertion, not re-linearization. Maintains observability properties (otherwise EKF becomes inconsistent over long runs — known unobservability issue).

#### Reported performance

- EuRoC stereo: 3-5 cm ATE
- ~20 ms per frame, single CPU core

---

## Part 2 — Critical findings & design decisions

### 2.1 Source-of-truth matrix (what's available where)

| Data | Live from depthai | rtabmap.db | Need to compute ourselves |
|---|---|---|---|
| VIO pose stream | ✅ `transform` | ❌ | ✅ if replacing BasaltVIO |
| SLAM pose stream (loop-corrected) | ✅ `transform` | ✅ `Node.pose` | ✅ |
| IMU samples | ✅ raw IMU queue | ❌ | — (input) |
| Stereo + depth frames | ✅ | ✅ `Data.image/depth` (compressed) | — (input) |
| Keyframes | ❌ | ✅ `Node` table | ✅ |
| Loop closure events | ❌ live | ✅ `Link` types 1-3 | ✅ |
| Loop closure score (BoW) | ❌ | ✅ `Statistics.hypothesis_ratio` | ✅ |
| Tracking lost events | ❌ (must derive from gaps) | ✅ `Statistics` | ✅ |
| ORB features per frame | ❌ | ✅ `Feature` table | ✅ |
| BoW vocabulary | ❌ | ✅ `Word` table | (use ORB-SLAM3 voc) |
| Local map landmarks | ❌ | partial via `Feature.depth_*` | ✅ |

**Implication for tooling roadmap:**

1. **Free wins from rtabmap.db** (no skyslam code needed):
   - Extend `extract_kf_from_db.py` to also dump `Feature` (ORB descriptors per KF) and `Statistics` (per-frame metrics).
   - Add `extract_neighbors_from_db.py` for `Link type=0` (KF-to-KF odometry edges).

2. **Skyslam output schema** should mirror RTABMap's where possible — easier comparison via `compare_sessions.py`.

### 2.2 Backend algorithm choice matrix

| Approach | Accuracy (EuRoC stereo ATE) | LOC (python est.) | Implementation risk |
|---|---|---|---|
| Loosely-coupled EKF (Pose-VIO fusion) | 10-20 cm | ~1500 | Low — well-understood, easy to debug |
| MSCKF (null-space projection) | 3-5 cm | ~3500 | Medium — Givens rotations + state management |
| SQRT sliding-window optim (Basalt-style) | 2-6 cm | ~6000 | High — QR factorization, marg, anchor management |
| ISAM2 incremental factor graph (gtsam wrapper) | similar to Basalt | ~2500 | Medium — gtsam handles math, we wire factors |

**Decision: 2-stage approach.**
- **Stage 1 (MVP)**: loosely-coupled EKF. Get pipeline plumbing complete, prove gold suite regression works, gain confidence.
- **Stage 2 (upgrade)**: if MVP ATE > 30 cm on `corridor_60s` → migrate to ISAM2 via gtsam. Skip MSCKF entirely (gtsam ISAM2 has better accuracy and we already need gtsam for loop closure).

**Why not MSCKF**: complexity overhead (null-space projection, FEJ, online calib) without commensurate benefit for our use case. ISAM2 gets similar accuracy via incremental optimization, cleaner code.

### 2.3 Frontend choice

| Approach | Pros | Cons |
|---|---|---|
| Pure OpenCV `calcOpticalFlowPyrLK` | trivial integration | translation-only, no patch warp, ~30% lower tracking lifetime |
| Patch-based KLT with SE(2) warp (Basalt) | best tracking, fast | implement Lucas-Kanade ourselves |
| ORB descriptors per frame (ORB-SLAM3 tracking) | rich descriptors enable relocalization | 5× slower than KLT |
| Hybrid: KLT for tracking + ORB at keyframes (our plan) | best of both | two feature pipelines to maintain |

**Decision: Hybrid.** KLT for per-frame frontend (drives VO + EKF), ORB extracted only at keyframes (drives loop closure + relocalization). Standard ORB-SLAM3 split.

Implementation: use `cv2.calcOpticalFlowPyrLK` for MVP (saves 2 weeks). Migrate to patch-based with SE(2) warp ONLY if KLT lifetime < 25 frames on `corridor_60s`.

### 2.4 Loop closure architecture

**Use ORB-SLAM3's pipeline verbatim** (it's the proven winner):

1. ORB at each keyframe → DBoW2 BoW vector.
2. Query DB → top-K candidates.
3. For each candidate: ORB descriptor matching → Sim3/SE3 RANSAC → reprojection match count.
4. **Consistency Groups**: only accept if same candidate's covisibility cluster matches 3+ consecutive KFs.
5. Essential Graph optim (pose-only).
6. Optionally launch Global BA in background thread.

**Do NOT** use RTABMap's Bayesian filter (graph-distance prior). It's elegant but harder to tune; ORB-SLAM3's Consistency Groups is simpler and equally robust.

### 2.5 Critical gotchas (from research)

1. **Quaternion convention.** Hamilton vs JPL is the #1 source of "everything works except orientation drifts" bugs. Stick with Hamilton (scalar-first, body→world) — matches numpy/scipy conventions, basalt, ORB-SLAM3, our existing code. NEVER mix with JPL (OpenVINS).

2. **Bias correction Jacobian for IMU preintegration.** Without it, every EKF update that changes bias estimate requires re-integrating all IMU samples in that interval. 10× slower. MUST implement Forster's Jacobian propagation.

3. **First-Estimate Jacobians (FEJ).** For EKF, after long runs the filter becomes inconsistent (over-confident) due to repeated re-linearization at updated points. Use FEJ: store Jacobian at first feature observation, reuse for all subsequent updates of that feature. Especially important for loosely-coupled EKF over corridor-length trajectories.

4. **Inverse-depth landmark parameterization.** Don't store landmarks as Euclidean XYZ (Basalt does inverse-depth for a reason: bounded scale at infinity, numerical stability). For our stereo case with depth map already available, we can get away with Euclidean for MVP — but switch to inverse-depth if BA diverges on distant features.

5. **Stereo init is immediate.** Don't write monocular-style 2-view triangulation bootstrap. First frame → all features with depth → 3D landmarks → done.

6. **IMU init must be staged.** Don't try to estimate scale + gravity + biases in one shot from cold start. ORB-SLAM3's VIBA1 (gyro bias + gravity + scale) → VIBA2 (full BA + accel bias + velocities) → re-refinement is the proven recipe. Even for stereo (where scale is known from baseline), gravity direction + biases still need staged init.

7. **Consistency Groups in loop closure.** Without this, ~10% of accepted loops are false positives → pose graph blows up. Always require 3+ consecutive consistent KFs.

8. **Pose graph optim "rejected if total error per node > threshold".** RTABMap's `Rtabmap/OptimizerMaxError=0.1` rule: after optim, if `total_chi2 / num_edges > 0.1`, reject the loop closure entirely. Saved them many times. Implement same safeguard.

9. **PCL is RTABMap-only.** depthai's BasaltVIO doesn't output pointcloud. When we replace RTABMapSLAM, we lose `obstaclePCL` + `groundPCL` streams unless we reimplement RTABMap's local grid mapper. **For drone use, we likely don't need PCL — depth map is enough.** Confirm before C6.

10. **Single-threaded SLAM (RTABMap blob).** This is why `setFreq()` defaults to 1 Hz. If we want multi-threaded skyslam: tracking on main, mapping + loop closing on workers. Python's GIL hurts here — use multiprocessing if true parallelism needed.

11. **Camera time offset.** Basalt and OpenVINS support online estimation but disable by default. OAK-D hardware-syncs camera + IMU via FSIN line → negligible offset. **Skip online time offset estimation.**

12. **Numerical precision.** Basalt uses float32 for state and float64 for covariance. SQRT marg requires this. For Python EKF, just use float64 throughout — only 2× slower, simpler to debug.

13. **Marginalization prior.** When old KF leaves window, its information must be turned into a prior on remaining variables. Naive Schur complement: invert dense matrix → numerically unstable. Basalt uses square-root QR. For our MVP, **just drop the old KF** (no marg prior). Accept the consistency loss; gain it back when migrating to ISAM2 (which handles marg correctly).

---

## Part 3 — Implementation plan v3 (revised based on research)

### Phase ordering rationale

Old plan was C0→C1→...→C5 linear. Revised plan reorders to:

1. Validate infrastructure FIRST (do we have what we need?).
2. Get a working end-to-end pipeline FAST with simplest possible components.
3. Iteratively upgrade each module based on measured weaknesses.

Each phase has hard **acceptance gate** before moving on. No upgrades to a module until next module proves it's the bottleneck.

---

### Phase 0 — Pre-flight & infrastructure (1-2 days)

**Goal**: confirm we have raw data + libraries needed. Avoid wasted weeks discovering missing data mid-implementation.

**Tasks**:
1. **Verify gold sessions have raw stereo frames**: open `sessions/gold/lab_static_10s/input/stereo/` — must have L_*.png, R_*.png, D_*.png. If only thumbnails: modify recorder + re-record gold (block ALL skyslam work until done).
2. **Install gtsam python**: `pip install gtsam`. Verify `import gtsam; gtsam.Pose3()` works. If it fails on macOS: fallback to a pure-python Gauss-Newton (300 LOC).
3. **Download ORB-SLAM3 vocabulary**: ~150 MB `ORBvoc.txt` from UZ-SLAMLab/ORB_SLAM3/Vocabulary → `assets/ORBvoc.txt`. Gitignore. Verify load with pyDBoW3.
4. **Install pyDBoW3**: `pip install pyDBoW3` (may need to build from source if no wheel). If fails: use minimal in-house BoW (~500 LOC, slower but works).
5. **Extend rtabmap extractor for new tables**: `baseline/tools/extract_kf_from_db.py` add dumps for `Feature` (per-KF ORB) + `Statistics` (per-frame metrics) + `Link type=0` (neighbor edges).
6. **Add baseline session #7** if needed: `corridor_with_imu_init` — 60s session that includes 3s static start + walk + return. Ensures we can validate IMU init.

**Acceptance gate**: All 6 tasks pass. If raw stereo missing or gtsam install impossible → halt, replan.

---

### Phase 1 — Replay harness + skyslam skeleton (2-3 days)

**Goal**: empty skyslam pipeline that consumes events from gold sessions, outputs pose stream, comparable via `compare_sessions.py`.

**Module structure**:
```
skyslam/
├── __init__.py        # SkySLAM facade class
├── types.py           # Pose, IMUSample, StereoFrame, Calib, Keyframe, Landmark
├── reader.py          # SessionReader: gold session → event stream
├── frontend.py        # KLT frontend (stub)
├── vo.py              # Stereo PnP VO (stub)
├── vio.py             # EKF VIO (stub)
├── mapping.py         # Keyframe + local BA (stub)
├── loop.py            # Loop closure (stub)
└── config.py          # All tunable params with defaults
tools/
└── replay_skyslam.py  # CLI: session → run skyslam → write outputs
```

**Acceptance gate**: 
- `replay_skyslam.py sessions/gold/lab_static_10s` produces `skyslam/vio_pose.jsonl` with N=199 rows of zero-pose.
- `compare_sessions.py basalt vs skyslam` runs without crashing (ATE huge, OK).
- Replay processes 5x real-time speed (skyslam stub is no-op, so this measures only I/O).

---

### Phase 2 — Stereo KLT frontend (2-3 days)

**Goal**: from each stereo+depth frame, output a list of tracked features with 3D position from depth.

**Algorithm** (per §1.3 + simplifications from §2.3):
1. Frame 1: `cv2.goodFeaturesToTrack(left, maxCorners=200, qualityLevel=0.01, minDistance=15)` on grid mask.
2. Frame N: `cv2.calcOpticalFlowPyrLK(prev_left, curr_left, prev_pts, winSize=(21,21), maxLevel=3)`.
3. Forward-backward check: drop tracks with recovered error > 1.0 px.
4. Refill if `len(active) < 100`: detect new features in mask excluding current track positions.
5. For each track, lookup depth via bilinear sample of depth map. Drop if depth ∉ [0.3, 8.0] m.
6. Emit `tracks.jsonl`: one line per frame `{ts_ns, frame_seq, tracks:[{id, u, v, age, depth_m}]}`.

**Acceptance gate** (per gold session):

| Session | Min avg_tracks/frame | Min median_age |
|---|---:|---:|
| lab_static_10s | 180 | 100 |
| lab_straight_20s | 150 | 25 |
| lab_loop_30s | 140 | 20 |
| corridor_60s | 100 | 15 |
| quick_motion_15s | 50 | 5 |
| loop_closure_45s | 130 | 18 |

**Visualization**: `tools/viz_tracks.py` — overlay tracks on frame timeline (age-colored: cyan→yellow→red).

---

### Phase 3 — Stereo PnP Visual Odometry (3-4 days)

**Goal**: from tracks + depth, output per-frame 6-DoF camera pose stream. No IMU yet.

**Algorithm** (per §1.3 backend + standard PnP):

State:
- `landmarks: dict[track_id, np.ndarray(3)]` — 3D position in world frame.
- `T_WC: np.ndarray(4,4)` — current camera pose.

Per-frame:
1. Get tracks from Phase 2.
2. Collect `(X_world, uv)` for tracks with existing landmarks. If ≥ 6 pairs:
   - `cv2.solvePnPRansac(SOLVEPNP_P3P, reproj_err=2.0, iters=100)` → `(R, t)`.
   - `cv2.solvePnPRefineLM` on inliers for sub-pixel pose.
   - If inliers ≥ 30: `T_WC = T_WC_new`, `tracking_ok=True`.
   - Else: keep last pose, `tracking_ok=False`.
3. For tracks WITHOUT landmarks: backproject from depth, transform to world, add to map.
4. Drop landmarks of dead tracks.
5. Emit pose: convert `T_WC → T_WB` via extrinsic, write `vo_pose.jsonl`.

**Acceptance gate**:

| Session | VO ATE vs basalt VIO | Tracking continuity |
|---|---:|---:|
| lab_static_10s | < 5 cm | 100% |
| lab_straight_20s | < 50 cm | ≥ 98% |
| lab_loop_30s | < 80 cm | ≥ 95% |
| corridor_60s | < 200 cm | ≥ 90% |
| quick_motion_15s | < 300 cm | ≥ 60% (motion blur expected) |
| loop_closure_45s | < 70 cm | ≥ 95% |

If `quick_motion_15s` < 60% → frontend too weak, must upgrade to patch-based KLT before proceeding.

---

### Phase 4 — Loosely-coupled EKF VIO (1-2 weeks)

**Goal**: fuse Phase 3 VO with IMU → smoother + more accurate pose stream, especially when visual fails.

**Implementation** (per §1.6 propagation + standard EKF update):

State (15 DoF, error-state parameterization):
$$\mathbf{x} = [\mathbf{p}_{WB}, \mathbf{v}_{WB}, \mathbf{q}_{WB}, \mathbf{b}_g, \mathbf{b}_a]$$
Covariance: 15×15.

Sub-tasks:

1. **`skyslam/imu_preintegration.py`** (Forster-style, per §1.3):
   - Class `IMUPreintegrator` accumulates `(ΔR, Δv, Δp)` over interval + bias Jacobians.
   - Methods: `integrate(sample)`, `predict(state)`, `correct_bias(δb_g, δb_a)`.
   - Test: integrate 1s of zero-noise IMU samples for static body → predictions return original state. (Critical sanity check.)

2. **`skyslam/ekf.py`**: 
   - Predict step: midpoint integrator per §1.3, covariance Φ update.
   - Update step: 6-DoF measurement (pos + small-angle quat), Joseph-form covariance update.
   - FEJ implementation: store Jacobians at first feature observation.

3. **`skyslam/vio.py`**: 
   - On IMU sample → `ekf.predict()`.
   - On Phase 3 VO pose → `ekf.update(vo_pose)`.
   - Emit fused state to `vio_pose.jsonl`.

4. **Initialization** (mini-VIBA1, per §1.5):
   - Wait until first 1s of IMU + 1 successful VO pose.
   - $b_g = \overline{\omega}_{1s}$, $b_a = \overline{a}_{1s} - R^T g$.
   - $v_0 = 0$ (assume stationary or use first 2 VO poses to estimate).
   - Strong prior on pose (matches VO), weak prior on biases.

5. **IMU noise calibration**:
   - Run `tools/imu_allan_variance.py sessions/gold/lab_static_10s` (write this tool) — compute Allan variance from 10s of static IMU.
   - Extract: `σ_g`, `σ_a` (white noise), `σ_bg`, `σ_ba` (random walk).
   - Update `skyslam/config.py` with measured values.

**Acceptance gate**:

| Session | VIO ATE vs basalt | vs Phase 3 VO |
|---|---:|---|
| lab_static_10s | < 3 cm | better (bias estimated) |
| lab_straight_20s | < 30 cm | better (smoother) |
| lab_loop_30s | < 50 cm | similar (no loop yet) |
| corridor_60s | < 150 cm | better |
| quick_motion_15s | < 100 cm | much better (IMU bridges visual gaps) |
| loop_closure_45s | < 50 cm | similar |

**Critical test**: plot `b_g(t), b_a(t)` for `lab_static_10s` — biases must plateau in ~3s. If they oscillate or drift unbounded → check noise params, Jacobian signs.

**Gate decision point**: if `corridor_60s` ATE > 200 cm → loosely-coupled EKF insufficient, migrate to ISAM2 backend before proceeding to mapping. Otherwise: continue.

---

### Phase 5 — Keyframes + ORB extraction (3-4 days)

**Goal**: from VIO stream + tracks, create persistent keyframe structure with rich ORB descriptors for loop closure.

**KF selection** (per §1.5):
- Δp > 0.3 m from last KF, OR
- Δθ > 15°, OR
- common-tracks ratio < 50% of last KF tracks

**Per KF**:
1. Snapshot pose from VIO.
2. Extract ORB: `cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)` → 500 keypoints + 32-byte descriptors.
3. Triangulate each ORB keypoint: lookup depth map → 3D position in world.
4. Store in keyframe DB: `kf_id, pose, ts_ns, orb_keypoints, orb_descriptors, landmark_positions`.
5. Emit `kf_events.jsonl` (matches our existing schema from C4).

**Acceptance gate**:
- `kf_events.jsonl` per gold session, KF count within 50%-150% of RTABMap's KF count for same session (from `extract_kf_from_db.py` output).
- ORB extraction time < 20 ms per KF.

---

### Phase 6 — BoW loop detection + Consistency Groups (1-2 weeks)

**Goal**: detect when current KF revisits an old location. CRITICAL: must include consistency groups to avoid false positives.

**Algorithm** (per §1.5):

1. **BoW indexing** (per KF in Phase 5):
   - Load ORB-SLAM3 vocab (Phase 0 asset).
   - Compute BoW vector + Feature vector.
   - Add to DB: `db.add(descriptors)` → `bow_id`.

2. **Query** (each new KF):
   - `db.query(descriptors, max_results=15+30)`.
   - Filter: skip last 30 KFs (avoid trivial), score ≥ 0.15.
   - Take top 3 candidates.

3. **Geometric verification** (each candidate):
   - Match ORB descriptors current vs candidate: Hamming distance + Lowe ratio 0.7.
   - 3D-2D PnP RANSAC: candidate's 3D landmarks against current 2D keypoints.
   - Accept if inliers ≥ 30.

4. **Consistency groups**:
   - Maintain `groups: list[tuple[set[kf_id], counter]]`.
   - New candidate match → expand existing group if overlap with covisibility, else new group.
   - Increment counter on consistent matches across consecutive KFs.
   - **ACCEPT loop closure only when counter ≥ 3**.

5. Emit `kf_loops.jsonl` (existing schema): `{from_kf, to_kf, transform, inliers, bow_score, consistency_count}`.

**Acceptance gate**:

| Session | Loop detected? | False positives |
|---|---:|---:|
| lab_static_10s | 0 (no motion) | 0 |
| lab_straight_20s | 0 expected | 0 |
| lab_loop_30s | ≥ 1 | 0 |
| corridor_60s | ≥ 1 | 0 |
| quick_motion_15s | 0 or 1 | 0 |
| loop_closure_45s | ≥ 3 | 0 |

**Critical test**: run on a session with NO loops (lab_straight_20s) — must detect 0 loops. If detects ≥1 → false positive, tune consistency threshold or score.

---

### Phase 7 — Pose graph optimization (Essential Graph) (3-4 days)

**Goal**: when loop accepted, snap pose graph to fix accumulated drift.

**Implementation** (per §1.5 + gtsam):

Pose graph structure:
- Nodes: all KF poses to date.
- Edges:
  - Odometry: $T_i^{-1} T_{i+1}$ between consecutive KFs (from Phase 5).
  - Loop closure: $T_{cq}$ from Phase 6 verification.

Solve via gtsam:
```
graph.add(PriorFactorPose3(X(0), kfs[0].pose, Constrained))
for i in 1..N: graph.add(BetweenFactorPose3(X(i-1), X(i), T_rel, odom_noise))
for loop in loops: graph.add(BetweenFactorPose3(X(loop.q), X(loop.c), T_qc, loop_noise))
result = LevenbergMarquardtOptimizer(graph, initial).optimize()
```

**Safeguard** (per §2.5 #8): after optim, compute `chi2 / n_edges`. If > 0.1 → reject loop closure, revert.

**Apply correction**:
- Update KF poses from result.
- Compute correction transform: `T_correction = T_new(last_kf) * T_old(last_kf)^-1`.
- Apply to subsequent frames: `T_new(t) = T_correction * T_old(t)` for `t > last_kf_ts`.
- Emit `odom_correction.jsonl` (matches basalt schema for viz tool compat).
- Final pose stream: `slam_pose.jsonl`.

**Acceptance gate**:

| Session | Final ATE | Loops |
|---|---:|---:|
| lab_loop_30s | < 30 cm | ≥ 1 |
| corridor_60s | < 50 cm | ≥ 1 |
| loop_closure_45s | < 15 cm | ≥ 3 |

Visualization check: viz_session.py shows trajectory snap back to start point on `loop_closure_45s`.

---

### Phase 8 — Local Bundle Adjustment (optional, 1-2 weeks)

**Goal**: refine recent KF poses + landmarks together for better local accuracy.

**Skip this phase IF** Phase 7 acceptance gate is already met. Local BA gives ~20% accuracy improvement but adds 2 weeks of complexity. For MVP, pose graph optim alone may suffice.

**If implementing**: per §1.5 LocalMapping section. Window of 10 covisible KFs + their landmarks. Fix oldest KF. gtsam `GenericProjectionFactor`. Huber loss.

---

### Phase 9 — Live promotion + A/B test (1 week)

**Goal**: integrate skyslam into `record_session.py` to run live alongside RTABMap, compare.

**Tasks**:
1. Refactor `SkySLAM` for streaming (currently designed for offline replay).
2. Add background thread in recorder: feed frames+IMU to skyslam in parallel with depthai.
3. Emit `skyslam/slam_pose.jsonl` live alongside `basalt/slam_pose.jsonl`.
4. New tool `tools/live_ab_compare.py`: real-time pyqtgraph window showing both trajectories side-by-side.
5. Record 5 new (not gold) sessions live. Compare. Document drift differences.

**Acceptance criteria for blob removal**:
- Skyslam ATE within 200% of basalt on ALL 6 gold sessions.
- Live skyslam keeps up with 20 FPS input (no frame drops > 5%).
- After 30 days of dev use with no critical bugs → remove `RTABMapSLAM` blob from pipeline.

---

## Part 4 — Math reference (centralized)

### Quaternion operations (Hamilton convention, scalar-first)

Multiplication: $q_1 \otimes q_2 = [w_1 w_2 - \mathbf{v}_1 \cdot \mathbf{v}_2,\ w_1 \mathbf{v}_2 + w_2 \mathbf{v}_1 + \mathbf{v}_1 \times \mathbf{v}_2]$

Inverse: $q^{-1} = [w, -\mathbf{v}] / \|q\|^2$

Exp map (small angle $\boldsymbol{\theta} \in \mathbb{R}^3$): $\exp_q(\boldsymbol{\theta}) \approx [1, \boldsymbol{\theta}/2]$ for $\|\boldsymbol{\theta}\| \ll 1$, exact via $[\cos(\|\boldsymbol{\theta}\|/2), \sin(\|\boldsymbol{\theta}\|/2) \boldsymbol{\theta} / \|\boldsymbol{\theta}\|]$.

Log map: $\log_q([w, \mathbf{v}]) = 2 \tan^{-1}(\|\mathbf{v}\|, w) \mathbf{v} / \|\mathbf{v}\|$.

Rotation matrix from quaternion ($q = [w, x, y, z]$):
$$R = \begin{pmatrix} 1-2y^2-2z^2 & 2xy-2wz & 2xz+2wy \\ 2xy+2wz & 1-2x^2-2z^2 & 2yz-2wx \\ 2xz-2wy & 2yz+2wx & 1-2x^2-2y^2 \end{pmatrix}$$

### IMU preintegration (Forster)

Continuous: $\dot{R} = R[\omega]_\times$, $\dot{v} = Ra + g$, $\dot{p} = v$.

Discrete delta accumulators between KF $i$ and $j$:
$$\Delta R_{ij} = \prod_{k=i}^{j-1} \exp((\omega_k - b_g) \Delta t)$$
$$\Delta v_{ij} = \sum_{k=i}^{j-1} \Delta R_{ik} (a_k - b_a) \Delta t$$
$$\Delta p_{ij} = \sum_{k=i}^{j-1} [\Delta v_{ik} \Delta t + \tfrac{1}{2} \Delta R_{ik} (a_k - b_a) \Delta t^2]$$

Bias Jacobians (linearized at nominal $\bar{b}_g, \bar{b}_a$, updated each integration step):
$$J_{\Delta R, b_g}, \quad J_{\Delta v, b_g}, J_{\Delta v, b_a}, \quad J_{\Delta p, b_g}, J_{\Delta p, b_a}$$

Bias-corrected residual:
$$r_{\Delta R} = \log\left[\left(\Delta \bar{R}_{ij} \exp(J_{\Delta R, b_g} \delta b_g)\right)^T R_i^T R_j\right]$$
(similar for $r_{\Delta v}, r_{\Delta p}$)

### Loosely-coupled EKF VIO update

Measurement (VO pose at $t_k$): $z = [p_{vo}, q_{vo}]$, model $h(x) = [p_{WB}, q_{WB}]$.

$$H = \begin{bmatrix} I_3 & 0 & 0 & 0 & 0 \\ 0 & 0 & I_3 & 0 & 0 \end{bmatrix} \in \mathbb{R}^{6 \times 15}$$

Innovation: $y_p = p_{vo} - p_{WB}$, $y_q = 2 \cdot \text{vec}(q_{vo} \otimes q_{WB}^{-1})$.

Joseph update:
$$P_+ = (I - KH) P_- (I - KH)^T + KRK^T$$

### Reprojection cost (for Local BA)

$$\text{Cost} = \sum_{ij} \rho_H\left( \left\| \pi(K, T_i^{-1} X_j) - u_{ij} \right\|^2_{\Sigma^{-1}} \right)$$

Pinhole projection: $\pi(K, [X, Y, Z]) = K [X/Z, Y/Z, 1]^T$.

Huber loss: $\rho_H(s) = s$ if $s < \delta^2$, else $2\delta\sqrt{s} - \delta^2$.

### Pose graph optimization (Essential Graph)

$$\min_{\{T_i\}} \sum_{(i,j) \in E} \left\| \log\left(T_{ij,\text{meas}}^{-1} T_i^{-1} T_j\right) \right\|^2_{\Sigma_{ij}}$$

SE(3) log map: 6-vector $(\rho, \theta)$ where $\rho \in \mathbb{R}^3$ (translation tangent), $\theta \in \mathbb{R}^3$ (rotation tangent).

---

## Part 5 — Risk register (revised based on research)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Gold sessions lack raw stereo frames | High | Block all skyslam work | Phase 0 first check. Re-record if needed. |
| gtsam pip install fails on macOS | Medium | Block Phase 7+ | Build from source. Or: in-house Gauss-Newton (~500 LOC). |
| ORB-SLAM3 vocab file inaccessible | Low | Block Phase 6 | Train custom vocab from gold sessions ORB descriptors. |
| EKF accuracy insufficient (>30 cm on corridor) | Medium | Need ISAM2 migration mid-project | Phase 4 gate decision. Have ISAM2 path designed in advance. |
| Loop closure false positives | High | Pose graph blowup | Consistency groups + post-optim chi² safeguard. NEVER skip these. |
| Patch-based KLT needed (lifetime too short) | Low | +2 weeks | OpenCV KLT first, upgrade only if measured weakness. |
| Live integration breaks RTABMap pipeline | Medium | Lose baseline | Run parallel, never replace until 30 days proven. |
| Bias Jacobian bug in preintegration | High | Silent slow drift | Unit test: zero-noise static IMU → zero drift. |
| Quaternion convention mix-up (Hamilton vs JPL) | High | Silent rotation drift | Document everywhere. Unit test: quaternion identity propagates as identity. |
| Total effort > 9 weeks | Medium | Drone deadline slip | Skip Phase 8 (Local BA). Phase 7 alone may be enough. |

---

## Part 6 — Open questions to resolve before starting

1. **Raw stereo frames**: do gold sessions have them, or only compressed thumbnails? (Phase 0 check)
2. **OAK-D IMU noise spec**: do we trust ICM-42688P datasheet values, or measure via Allan variance? (Phase 4)
3. **PCL replacement strategy**: do we need point cloud output post-skyslam, or is depth map sufficient for downstream drone code? (Phase 6 decision)
4. **Recording during skyslam dev**: should we record new gold sessions with `--keep-stereo --keep-imu-raw` flags? (Phase 0)
5. **Threading**: GIL constraints — use multiprocessing for parallel tracking/mapping, or async I/O with single-process? (Phase 9)
6. **Backend lib confirmation**: gtsam vs in-house? Decision after Phase 0 install test.

---

## Part 7 — Estimated effort & milestones

| Phase | Effort | Cumulative |
|---|---:|---:|
| 0. Pre-flight | 1-2 days | 2 days |
| 1. Replay + skeleton | 2-3 days | 5 days |
| 2. KLT frontend | 2-3 days | 8 days |
| 3. Stereo VO | 3-4 days | 12 days |
| 4. EKF VIO | 7-14 days | 26 days |
| 5. KF + ORB extraction | 3-4 days | 30 days |
| 6. BoW loop detection | 7-14 days | 44 days |
| 7. Pose graph optim | 3-4 days | 48 days |
| 8. Local BA (optional) | 7-14 days | 62 days |
| 9. Live promotion | 7 days | 69 days |

**Milestones:**
- **M1 (~12 days)**: VO-only, no IMU, runs offline. Gates infrastructure.
- **M2 (~30 days)**: VIO + KF extraction. Replaces "VIO half" of pipeline.
- **M3 (~48 days)**: Full SLAM with loop closure. Replaces RTABMapSLAM functionality.
- **M4 (~69 days)**: Live integration, blob removable.

If cutting Phase 8: M3 → ~40 days, M4 → ~55 days. Recommended cut if calendar pressure.

---

## Part 8 — Pre-implementation checklist (do this before writing any skyslam code)

- [ ] **Verify raw stereo frames** in `sessions/gold/lab_static_10s/input/stereo/` (block if missing).
- [ ] **Install gtsam**: `pip install gtsam && python -c "import gtsam"`.
- [ ] **Install pyDBoW3**: `pip install pyDBoW3 && python -c "import pyDBoW3"`.
- [ ] **Download ORB vocab** to `assets/ORBvoc.txt`.
- [ ] **Re-read Part 2** (decisions matrix) — confirm decisions still valid.
- [ ] **Confirm quaternion convention** in existing codebase (`oakd/recorder.py`, `baseline/tools/compare_sessions.py`) is Hamilton scalar-first. Document in `skyslam/types.py`.
- [ ] **Extend `extract_kf_from_db.py`** to dump `Feature` + `Statistics` tables (gives free comparison data).
- [ ] **Allan variance script** for IMU noise calibration on `lab_static_10s`.
- [ ] **Decide cut scope**: full plan (9 weeks) vs Phase-8-skipped (7 weeks)?
- [ ] **Schedule M1 (12-day) review**: at end of Phase 3, gut-check whether continuing is right.

---

**End of plan. Implementation can start tomorrow after Phase 0 checklist done.**
