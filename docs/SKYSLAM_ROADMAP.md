# SkySLAM — Comprehensive plan: autonomous hardware + SLAM software

> ⚠️ **SUPERSEDED for the software-plan section (2026-05-29)**
>
> The detailed software plan (stack, phases, math, acceptance gates) has been
> rewritten as **`docs/SKYSLAM_RESEARCH.md`** after a thorough read of the
> depthai-core, basalt, rtabmap, ORB-SLAM3 and OpenVINS source code. That
> plan v3 uses `numpy + opencv + gtsam + pyDBoW3` (Python-first), which is
> DIFFERENT from the C99-from-scratch plan kept here.
>
> This file is RETAINED as the **long-term hardware / system vision**
> (HW V1 IMX570 ToF, HW V2 custom MIPI board, FC↔SLAM link planning,
> drone integration). Do NOT read Section 3 "Software architecture" — it
> is obsolete.

**Authors**: Bao + Copilot
**Created**: 2026-05-27 (HW vision retained; SW plan superseded 2026-05-29)
**Status**: HW vision = long-term reference; SW plan = obsolete → see `SKYSLAM_RESEARCH.md`.
**End goal**: Autonomous drone with a fully self-designed SLAM stack (HW + SW), production-ready, with no Basalt / RTAB-Map / Luxonis dependency at runtime.

---

## 0. Table of contents

1. [Vision & design principles](#1-vision--design-principles)
2. [Hardware option comparison](#2-hardware-option-comparison)
3. [Software architecture](#3-software-architecture)
4. [Reference implementations to learn from](#4-reference-implementations-to-learn-from)
5. [Software roadmap — 6 phases](#5-software-roadmap--6-phases)
6. [Hardware roadmap — 4 stages](#6-hardware-roadmap--4-stages)
7. [Datasets & validation strategy](#7-datasets--validation-strategy)
8. [Repo & tooling layout](#8-repo--tooling-layout)
9. [Legal & licensing](#9-legal--licensing)
10. [Risks & mitigations](#10-risks--mitigations)
11. [Open decisions to settle later](#11-open-decisions-to-settle-later)

---

## 1. Vision & design principles

### Vision
- **100% self-owned**: from MIPI sensor driver → SLAM algorithm → FC link. No third-party runtime library we don't own.
- **Portable**: the same codebase runs on macmini (dev), RPi5 (prototype) and future custom SoCs (production).
- **Production-grade**: drone capable of autonomous indoor + outdoor flight, with loop closure, persistent map, FC integration.
- **Modular**: swapping the sensor (OAK-D → IMX570 ToF → custom MIPI camera) only changes one driver layer.

### Design principles
1. **C99 for the core** (math + algorithms): portable to MCU/custom SoC, no STL dependency.
2. **C++ is only a thin wrapper** for third-party hardware SDKs (depthai, libcamera, etc.).
3. **No heavy external libs at runtime**: no Eigen / Sophus / Ceres / g2o / OpenCV in the production binary. Roll our own.
4. **Sensor-agnostic backend**: VIO/SLAM only sees `sky_image_t`, `sky_imu_sample_t`, `sky_depth_t` — it has no idea about the hardware.
5. **Test-driven**: each module has unit tests with a numerical oracle (compared against Eigen/OpenVINS in dev, not shipped).
6. **Spec-first workflow**: write `spec.md` before coding each phase to avoid rewrites.
7. **Dataset-driven validation**: EuRoC + TUM-VI + TUM RGB-D + ICL-NUIM before any hardware test.

---

## 2. Hardware option comparison

### 2.1 OAK-D W (current, prototype)
- Stereo wide-FOV + BMI270 IMU + Myriad X VPU.
- Pros: plug-and-play, mature depthai SDK, proper IMU sync.
- Cons: closed-source firmware, limited VPU bandwidth (4 streams crash it), no real self-ownership.
- **Role**: prototype + dataset capture for phase 1–4 dev.

### 2.2 IMX570 ToF (hardware self-build V1)
- Sony DepthSense CAPD ToF 640×480, depth + IR amplitude, 30–120 fps.
- Range: 0.2–5 m indoor; poor under outdoor sunlight.
- Pros: high-quality depth on textureless surfaces (white walls); works in the dark; eliminates stereo matching.
- Cons: poor outdoors; short range; laser eats 3–5 W; multi-path artefacts; proprietary Sony SDK.
- **Role**: V1 hardware for an indoor inspection drone.

### 2.3 Custom stereo + ToF + IMU board (hardware self-build V2)
- 2× IMX296 global-shutter mono (stereo, feature tracking).
- IMX570 ToF (depth fusion, indoor mode).
- BMI088 or ICM-42688P IMU @ 1 kHz over SPI.
- STM32H7 or Cortex-R co-processor for timestamp sync + IMU integration.
- MIPI CSI-2 directly into the host SoC (RPi5 / Rockchip / Jetson Orin).
- **Role**: V2 production hardware, combining the advantages of stereo (outdoor) + ToF (indoor).

### 2.4 Sensor fusion strategy (for V2)
- **Outdoor / range > 5 m**: stereo VIO mode, ToF off (save power).
- **Indoor / range < 5 m / low light**: stereo + ToF fusion, ToF as a depth prior.
- **Total darkness**: ToF + IR amplitude (stereo mono is blind without light).

---

## 3. Software architecture

### 3.1 Layered diagram

```
┌─────────────────────────────────────────────────────────┐
│  Application: drone autonomous flight, mapping, viewer  │
├─────────────────────────────────────────────────────────┤
│  libskyslam   — keyframe DB, loop closure, pose graph   │
├─────────────────────────────────────────────────────────┤
│  libskyvio    — IMU preintegration, MSCKF EKF, SW opt   │
├─────────────────────────────────────────────────────────┤
│  libskyfront  — FAST, KLT, stereo/RGBD match, RANSAC    │
├─────────────────────────────────────────────────────────┤
│  libskysensors — sensor abstraction, dataset replay     │
├─────────────────────────────────────────────────────────┤
│  libskymath   — vec/mat/quat/SO3/SE3, Cholesky, LM      │
├─────────────────────────────────────────────────────────┤
│  drivers/     — OAK-D, IMX570, custom MIPI, MAVLink     │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Key data structures (sensor-agnostic)

```c
typedef struct {
    uint64_t ts_ns;
    uint16_t width, height, stride;
    uint8_t  format;         // GRAY8, RGB888, etc.
    uint8_t *pixels;
    const sky_camera_calib_t *calib;
} sky_image_t;

typedef struct {
    uint64_t ts_ns;
    uint16_t width, height;
    uint16_t *depth_mm;      // 0 = invalid
    uint16_t *confidence;    // optional, NULL if unavailable
    const sky_camera_calib_t *calib;
} sky_depth_t;

typedef struct {
    uint64_t ts_ns;
    float gyro_rad_s[3];
    float accel_m_s2[3];
    float temp_c;
} sky_imu_sample_t;

typedef enum {
    SKY_FRAME_MONO,
    SKY_FRAME_STEREO,
    SKY_FRAME_RGBD,
    SKY_FRAME_STEREO_PLUS_TOF,
} sky_frame_type_t;

typedef struct {
    sky_frame_type_t type;
    sky_image_t left, right;     // stereo / mono uses left only
    sky_depth_t depth;           // RGBD / fusion
} sky_visual_frame_t;
```

### 3.3 Threading model
- **Sensor thread**: drivers push raw frames into a lock-free SPSC queue.
- **Frontend thread**: pull frame, detect/track, output `sky_obs_packet_t`.
- **Backend thread**: IMU integration + VIO state update + (optional) SLAM optimisation.
- **Output thread**: pose → FC link + UI + logger.
- Separate IMU thread @ 1 kHz, low latency.

---

## 4. Reference implementations to learn from

| Project | License | Role | Note |
|---|---|---|---|
| **OpenVINS** (UDel RPNG) | BSD-3 | MSCKF VIO oracle | Cleanest code, well-documented, perfect to learn EKF VIO |
| **Basalt** (TUM) | BSD-3 | Sliding-window VIO + mapping reference | Currently runtime, replaced later |
| **RTAB-Map** (IntRoLab) | BSD-3 | SLAM + loop closure reference | Currently runtime, replaced later |
| **S-MSCKF** (UPenn) | BSD-3 | Stereo MSCKF reference | Compact, ~3k LOC, easy to port |
| **VINS-Fusion** (HKUST) | GPL-3 | Read for ideas, do NOT copy code | GPL → contaminates |
| **ORB-SLAM3** (UZ) | GPL-3 | Read for ideas, do NOT copy code | GPL → contaminates |
| **Kimera-VIO** (MIT) | BSD-2 | Reference for factor-graph approach | GTSAM-based |
| **DBoW3** | BSD-3 | Bag-of-words for loop closure | Can port or use as-is |
| **Sophus** (Strasdat) | MIT | SE3/SO3 oracle for testing | Dev-only, not shipped |
| **Eigen** | MPL-2 | Linear algebra oracle for testing | Dev-only, not shipped |
| **Ceres** (Google) | BSD-3 | NLLS solver reference | Can port the LM core |
| **Kalibr** (ETH) | BSD-3 | Camera-IMU calibration reference | Inspires the `sky_calibrate` tool |

**Code-copy rules**:
- BSD-3 / MIT / Apache-2: may be copied; keep the copyright notice in the copied file + add to `THIRD_PARTY_LICENSES.txt`.
- GPL / LGPL: do NOT copy. Read only for concepts, then rewrite from the idea.
- Patent-encumbered (SURF): avoid entirely.

---

## 5. Software roadmap — 6 phases

> **Estimate format**: `AI coding time | wall-clock time (including review + test + debug)`.

### Phase 0 — Foundation: `libskymath` (1–2 days | 3–5 days)

**Goal**: self-contained C99 math kernel, NEON-optimised, passing tests vs Eigen+Sophus.

**Modules**:
- `vec[2|3|4]`, `mat[3|4]`: dense ops, dot/cross/norm, transpose, inverse.
- `quat`: Hamilton convention, exp/log, slerp, ↔ rotation matrix.
- `so3`, `se3`: Lie group exp/log, adjoint, right/left Jacobians.
- `chol_dense`: dense Cholesky for 6×6 / 9×9 / 15×15 blocks.
- `chol_sparse_block`: block-sparse LDLT (or defer to Phase 3, use SuiteSparse temporarily).
- `lm_solver`: generic-callback Levenberg-Marquardt.
- `neon/`: ARM NEON intrinsics, scalar fallback on x86.

**Tests**:
- Random fuzz 10k samples vs Eigen, error < 1e-10.
- Property tests: `exp(log(R)) == R`, `Adj(T) * xi == log(T * exp(xi) * T^-1)`.
- Benchmark vs Eigen: target ≤ 1.5× slower for mat3/quat.

**Deliverable**: `libskymath.a` + `test_math` binary, CI green.

---

### Phase 1 — Sensor abstraction + dataset replay (0.5–1 day | 1–2 days)

**Goal**: define sensor types, replay EuRoC / TUM-VI / TUM RGB-D / ICL-NUIM.

**Modules**:
- `sky_sensor_types.h`: `sky_image_t`, `sky_imu_sample_t`, `sky_depth_t`, `sky_visual_frame_t`.
- `sky_camera_calib.c`: intrinsics struct + undistort (radial-tangential, Kannala-Brandt fisheye).
- `sky_dataset_euroc.c`: parse EuRoC MAV format.
- `sky_dataset_tumvi.c`: parse TUM-VI format.
- `sky_dataset_tum_rgbd.c`: parse TUM RGB-D format (for the ToF phase later).
- `sky_recorder.c`: record raw stream from live sensor to a binary file for replay.

**Tool**: `sky_replay <dataset> <path>` pushes via callback.

**Deliverable**: replay EuRoC MH_01 with timestamp + image checksum verification.

---

### Phase 2 — Frontend: feature tracking (2–3 days | 5–7 days)

**Goal**: feature detection + tracking + outlier rejection, NEON-optimised.

**Modules**:
- `fast_detector.c`: FAST-9, non-max suppression, grid distribution, NEON.
- `klt_tracker.c`: pyramidal Lucas-Kanade, 4 levels, sub-pixel refinement, NEON.
- `stereo_match.c`: epipolar match left↔right, ZNCC patch.
- `rgbd_lift.c`: 2D feature + depth lookup → 3D point + variance from confidence.
- `pnp_ransac.c`: P3P + RANSAC for 3D-2D outlier rejection.
- `essential_ransac.c`: 5-point + RANSAC for 2D-2D.

**Tool**: `sky_track_vis` — visualise tracks on EuRoC, measure tracking length + outlier rate.

**Success**: on EuRoC MH_01, median tracking length ≥ 30 frames, outlier rate < 10%.

---

### Phase 3a — MSCKF VIO (3–5 days code | 2–3 weeks debug)

**Goal**: working VIO on EuRoC, ATE < 0.2 m on MH_01.

**Modules**:
- `imu_preintegration.c`: Forster et al. 2017, midpoint integration, bias Jacobians.
- `state_msckf.c`: state `[p, q, v, bg, ba]` + cloned poses in a sliding window.
- `feature_init.c`: multi-view triangulation + Gauss-Newton inverse-depth refinement.
- `ekf_predict.c`: IMU propagation step.
- `ekf_update.c`: visual measurement update, EKF Kalman gain block-sparse.
- `chi2_test.c`: Mahalanobis distance outlier rejection.
- `vio_orchestrator.c`: frontend ↔ backend pipeline.

**Reference**: OpenVINS `MsckfManager`, S-MSCKF.

**Test on EuRoC** (compare with OpenVINS as oracle):
| Sequence | Target ATE (m) | Basalt ATE (m) |
|---|---|---|
| MH_01_easy | < 0.15 | 0.07 |
| MH_03_medium | < 0.25 | 0.12 |
| MH_05_difficult | < 0.40 | 0.20 |

**Critical path**: VIO bug debugging is extremely hard (a wrong Jacobian sign → unbounded drift). Mitigation:
- Compare intermediate IMU propagation output against OpenVINS logs every 100 samples.
- Unit-test every residual + Jacobian with a numerical diff (central difference).
- Visualise covariance ellipses along the trajectory.

---

### Phase 3b — Sliding-window optimiser (10–16 days code | 4–6 weeks) [OPTIONAL]

**Goal**: raise accuracy to match Basalt if Phase 3a is not enough.

**Modules**:
- `sw_optimizer.c`: fixed-lag smoother, 5–10 KF window.
- `marginalization.c`: Schur complement → prior factor.
- `factor_imu.c`, `factor_visual.c`, `factor_prior.c`: residual + Jacobian.
- Solver: use `lm_solver` + block-sparse Cholesky from Phase 0.

**Recommendation**: SKIP if Phase 3a reaches < 0.25 m mean ATE. MSCKF is enough for a production drone.

---

### Phase 4 — SLAM + loop closure: `libskyslam` (4–6 days code | 2–4 weeks)

**Goal**: persistent map, loop closure, drift correction.

**Modules**:
- `descriptor_orb.c`: ORB (FAST + BRIEF + orientation), NEON.
- `vocabulary_tree.c`: hierarchical k-means BoW, trained offline on the TUM dataset.
- `vocab_trainer.c` (offline tool): train vocab from ~10k images.
- `loop_detector.c`: query DB → top-K candidates → geometric verification (PnP RANSAC).
- `pose_graph.c`: keyframe graph + relative SE3 constraints.
- `pose_graph_optim.c`: Gauss-Newton on SE(3), block-sparse Cholesky.
- `map_db.c`: keyframe storage + persistent binary serialisation.
- `slam_orchestrator.c`: VIO odometry → KF selection → loop detect → optimise → correct.

**Reference**: RTAB-Map architecture, DBoW3.

**Test**:
- TUM-VI room1/2: loop detection rate > 80%, post-loop drift < 1% path length.
- Persistent DB: save / load / continue mapping across multiple sessions.

**Alternative**: use the DBoW3 library (BSD-3) instead of writing our own vocab tree → save ~1 week.

---

### Phase 5 — Hardware integration + deploy (2–3 days code | 1–2 weeks test)

**Goal**: real-time on OAK-D + RPi5, FC link via MAVLink.

**Modules**:
- `driver_oakd.cpp`: thin C++ wrapper around depthai 3.x → emit `sky_visual_frame_t` + `sky_imu_sample_t`.
- `fc_link_mavlink.c`: MAVLink `VISION_POSITION_ESTIMATE` over UART.
- `fc_link_msp.c`: MSP v2 (for Betaflight / INAV).
- `sky_daemon.c`: long-running service, systemd unit on RPi5.
- ROS2 bridge (optional): DDS publisher.

**Test on hardware**:
- macmini: ≥ 30 FPS real-time.
- RPi5: ≥ 20 FPS, CPU < 70%, RAM < 500 MB.
- Flight test: square 10 m indoor flight, post-loop drift < 0.3 m.

---

### Phase 6 — Production hardening (3–5 days | 1–2 weeks)

**Goal**: production-ready stack.

**Modules**:
- `sky_calibrate` tool: camera intrinsics (Zhang's method) + camera-IMU extrinsics (Kalibr-style).
- `failure_detector.c`: tracking lost / IMU saturation / depth invalid → recovery state.
- `sky_logger.c`: binary log format (sensor + state + estimate).
- `sky_log_replay` tool: replay logs offline.
- CI matrix: x86_64 + aarch64 build, run tests.
- Docs: math derivations PDF + architecture diagrams.

---

### Software summary

| Phase | LOC | AI code | Wall-clock |
|---|---|---|---|
| 0. Math | 5k | 1–2 days | 3–5 days |
| 1. Sensor | 2k | 0.5–1 day | 1–2 days |
| 2. Frontend | 3k | 2–3 days | 5–7 days |
| 3a. MSCKF | 4k | 3–5 days | 2–3 weeks |
| 3b. SW opt (opt) | 6k | 10–16 days | 4–6 weeks |
| 4. SLAM | 6k | 4–6 days | 2–4 weeks |
| 5. HW integ | 2k | 2–3 days | 1–2 weeks |
| 6. Hardening | 2k | 3–5 days | 1–2 weeks |
| **Total (no 3b)** | **~24k** | **~3 weeks** | **~3–4 months** |

---

## 6. Hardware roadmap — 4 stages

### Stage HW-0 — OAK-D W (now)
- COTS, used for software dev Phases 0–5.
- Driver: depthai 3.x.
- **End condition**: Phase 5 software done, ready to move to custom HW.

### Stage HW-1 — IMX570 ToF dev board (~2–3 months after HW-0 is stable)
- Buy a Lucid Helios2 (GigE) or Basler blaze-101 module.
- Connectivity: GigE Ethernet (Lucid) or USB3 (Basler).
- Pair with an IMX296 mono global-shutter over MIPI into the RPi5.
- BMI088 or ICM-42688P IMU over SPI into an STM32 → UART → host.
- **Additional software**: `driver_lucid.c` (GigE Vision), `driver_imx296.c` (libcamera), `driver_imu_stm32.c` (UART protocol).
- Goal: validate the hybrid stereo + ToF VIO concept.

### Stage HW-2 — Custom carrier board V1 (~6 months)
- Schematic + PCB designed in-house (KiCad).
- MIPI CSI-2 routing for 2× IMX296 + IMX570.
- STM32H7 co-processor: IMU + sensor timestamp sync.
- Power management: 5 V / 3.3 V / 1.8 V rails, laser driver for ToF.
- Connector: M.2 or Raspberry Pi HAT form factor.
- **Additional software**: kernel driver / libcamera IPA for the custom MIPI pipeline.

### Stage HW-3 — All-in-one module V2 (~12–18 months)
- Integrate the SoC (Rockchip RK3588 or Jetson Orin NX) with the sensors on one board.
- Form factor: 60 × 40 mm, < 50 g.
- Onboard SLAM stack runs autonomously, output over USB / Ethernet / UART.
- Optional IP67 enclosure.
- **Goal**: completely replace OAK-D for the production drone.

---

## 7. Datasets & validation strategy

### 7.1 Datasets

| Dataset | Sensor type | Role |
|---|---|---|
| **EuRoC MAV** | Stereo + IMU (MAV indoor) | Primary VIO benchmark |
| **TUM-VI** | Stereo + IMU (handheld, with loops) | SLAM loop closure benchmark |
| **TUM RGB-D** | RGB + depth (Kinect) | RGBD VIO/SLAM for the ToF phase |
| **ICL-NUIM** | Synthetic RGB-D | Absolute ground truth, debugging |
| **KITTI** | Stereo + GPS (outdoor car) | Outdoor stress test |
| **Self-recorded OAK-D** | Stereo + BMI270 | Real hardware test |
| **Self-recorded IMX570** | RGBD + IMU | Once HW-1 is ready |

### 7.2 Validation pyramid

```
        ┌───────────────────────┐
        │  Flight test (drone)  │  ← Stage 5-6, rare
        ├───────────────────────┤
        │  Hardware live replay │  ← OAK-D/IMX570 stream
        ├───────────────────────┤
        │  Dataset replay       │  ← EuRoC/TUM-VI, primary
        ├───────────────────────┤
        │  Oracle compare       │  ← vs OpenVINS / Basalt
        ├───────────────────────┤
        │  Unit + property test │  ← vs Eigen / Sophus
        └───────────────────────┘
```

### 7.3 Success metrics per phase

| Phase | Metric | Target |
|---|---|---|
| 0 | Math op error vs Eigen | < 1e-10 |
| 2 | EuRoC MH_01 tracking length | ≥ 30 frames median |
| 3a | EuRoC MH_01 ATE | < 0.15 m |
| 3a | EuRoC MH_05 ATE | < 0.40 m |
| 4 | TUM-VI loop detection rate | > 80% |
| 4 | Post-loop drift | < 1% path length |
| 5 | RPi5 real-time FPS | ≥ 20 |
| 5 | RPi5 CPU usage | < 70% |
| 5 | Indoor 10 m square flight drift | < 0.3 m |

---

## 8. Repo & tooling layout

### 8.1 Repo location
`/Users/bao/skydev/skyslam` (separate from `oak-d`).

### 8.2 Structure

```
skyslam/
├── README.md
├── LICENSE                          # BSD-3 or proprietary, your call
├── THIRD_PARTY_LICENSES.txt
├── CMakeLists.txt
├── cmake/
│   ├── toolchain-aarch64.cmake
│   └── neon.cmake
├── docs/
│   ├── architecture.md
│   ├── math/                        # derivations PDF
│   │   ├── lie_groups.tex
│   │   ├── imu_preintegration.tex
│   │   └── msckf_jacobians.tex
│   └── phases/
│       ├── phase0_spec.md
│       ├── phase1_spec.md
│       └── ...
├── libskymath/
│   ├── include/sky/math/
│   ├── src/
│   ├── neon/
│   └── test/
├── libskysensors/
│   ├── include/sky/sensors/
│   ├── src/
│   └── test/
├── libskyfront/
├── libskyvio/
├── libskyslam/
├── drivers/
│   ├── oakd/                        # C++ wrapper for depthai
│   ├── lucid_helios/                # GigE ToF
│   ├── libcamera_imx296/            # MIPI mono
│   ├── imu_stm32_uart/              # Custom IMU
│   └── fc_link/
│       ├── mavlink/
│       └── msp/
├── tools/
│   ├── sky_replay/
│   ├── sky_track_vis/
│   ├── sky_vio_replay/
│   ├── sky_slam_replay/
│   ├── sky_calibrate/
│   ├── sky_recorder/
│   ├── sky_log_replay/
│   └── sky_daemon/                  # systemd RPi5 service
├── third_party/                     # header-only deps only
│   ├── unity/                       # test framework
│   └── stb_image/
├── datasets/                        # symlinks
├── scripts/
│   ├── benchmark_vs_openvins.py
│   ├── plot_ate.py
│   └── train_vocab.py
└── .github/workflows/
    └── ci.yml                       # matrix x86 + aarch64
```

### 8.3 Build & toolchain

- **Build system**: CMake 3.20+, Ninja.
- **Compilers**: clang 16+ (macOS dev), gcc 12+ (Linux), gcc-aarch64-linux-gnu (cross).
- **Test**: Unity (Throw The Switch, MIT license).
- **CI**: GitHub Actions matrix x86_64 / aarch64.
- **Format**: clang-format Google style.
- **Lint**: clang-tidy + cppcheck.

---

## 9. Legal & licensing

### 9.1 Licenses we can use

- **BSD-3 / MIT / Apache-2**: copying code is OK, with attribution.
- **MPL-2 (Eigen)**: dev-only, do not ship at runtime.
- **LGPL**: avoid static linking; dynamic linking is OK but complex → avoid entirely.
- **GPL**: do NOT copy, do NOT link, only read for concepts.

### 9.2 Patent check
- **SIFT**: patent expired in 2020 — free.
- **SURF**: still patented → avoid.
- **ORB**: free (BSD in OpenCV).
- **FAST**: free.
- **BRIEF**: free.

### 9.3 SkySLAM's own license
- Proposal: **BSD-3-Clause** if open-sourcing, **proprietary closed** if keeping the commercial advantage.
- Decide after Phase 5 is working.

### 9.4 Compliance checklist per commit
- [ ] File copied from another project → keep the original header notice.
- [ ] `THIRD_PARTY_LICENSES.txt` updated.
- [ ] No GPL/LGPL linked into the production binary.

---

## 10. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Prolonged VIO debugging (wrong Jacobian sign) | High | High | Numerical-diff test every Jacobian; oracle compare with OpenVINS |
| NEON intrinsics hard to debug | Medium | Medium | Always keep a scalar fallback; test parity between both paths |
| Self-written block-sparse Cholesky bugs | High | High | Skip in Phase 0; use SuiteSparse temporarily; rewrite in Phase 3b if needed |
| EuRoC passes but OAK-D fails | Medium | High | Careful calibration, tune IMU noise model |
| RPi5 not real-time enough | Low | High | Profile early; Jetson Orin Nano as backup |
| OAK-D firmware still crashes | Medium | Medium | Workaround exists; dataset record as fallback |
| Sony IMX570 SDK is proprietary | High | Medium | Use Lucid Helios GigE instead (open protocol) |
| Custom PCB delay (HW-2) | High | Medium | Order via JLCPCB; keep a spare bare module as backup |
| Self-built hardware fails | Medium | Critical | Keep OAK-D path as production fallback |

---

## 11. Open decisions to settle later

1. **SkySLAM license**: BSD-3 or proprietary?
2. **Phase 3b**: do the sliding-window optimiser, or is MSCKF enough?
3. **DBoW3 vs self-written vocab tree**: time saved vs full self-ownership?
4. **SuiteSparse vs self-written block-sparse Cholesky**: same trade-off.
5. **HW-1 sensor pick**: Lucid Helios2 vs Basler blaze-101 vs custom IMX570 board?
6. **Production SoC**: RK3588 vs Jetson Orin NX vs custom?
7. **Primary FC protocol**: MAVLink (Ardupilot / PX4) or MSP (Betaflight / INAV)?
8. **How autonomous should the AI be**: code + commit, or code only + commit by hand?

---

## 12. Cross-references

- Current project using Basalt + RTAB-Map: `/Users/bao/skydev/oak-d`
- IMU rate / filter setting reference for FC: user memory `icm42688p-godr-vs-sample-rate.md`
- LPF design rule: user memory `lpf-time-constant-rule.md`

---

## 13. Change history

| Date | Change | Author |
|---|---|---|
| 2026-05-27 | First draft | Bao + Copilot |
| 2026-05-29 | Mark SW plan superseded by SKYSLAM_RESEARCH.md | Bao + Copilot |
| 2026-05-29 | Translate document to English | Bao + Copilot |

---

