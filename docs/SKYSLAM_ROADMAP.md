# SkySLAM — Kế hoạch toàn diện: Tự chủ phần cứng + phần mềm SLAM

> ⚠️ **SUPERSEDED cho phần software plan (2026-05-29)**
>
> Kế hoạch software chi tiết (stack, phases, math, acceptance gates) đã được
> rewrite thành **`docs/SKYSLAM_RESEARCH.md`** sau khi đọc thorough source
> code của depthai-core, basalt, rtabmap, ORB-SLAM3, OpenVINS. Plan v3 đó
> dùng `numpy + opencv + gtsam + pyDBoW3` (Python-first), KHÁC với plan
> C99-from-scratch ở đây.
>
> File này GIỮ LẠI làm **long-term hardware/system vision** (HW V1 IMX570
> ToF, HW V2 custom MIPI board, FC↔SLAM link planning, drone integration).
> Đừng đọc Section 3 "Software architecture" — đã obsolete.

**Tác giả**: Bảo + Copilot
**Ngày soạn**: 2026-05-27 (HW vision retained; SW plan superseded 2026-05-29)
**Trạng thái**: HW vision = long-term reference; SW plan = obsolete → xem `SKYSLAM_RESEARCH.md`.
**Mục tiêu cuối**: Drone autonomous với SLAM stack tự thiết kế hoàn toàn (HW + SW), production-ready, không phụ thuộc Basalt / RTAB-Map / Luxonis ở runtime.

---

## 0. Mục lục

1. [Tầm nhìn & nguyên tắc thiết kế](#1-tầm-nhìn--nguyên-tắc-thiết-kế)
2. [So sánh hardware options](#2-so-sánh-hardware-options)
3. [Software architecture](#3-software-architecture)
4. [Reference implementations để học](#4-reference-implementations-để-học)
5. [Roadmap phần mềm — 6 phases](#5-roadmap-phần-mềm--6-phases)
6. [Roadmap phần cứng — 4 stages](#6-roadmap-phần-cứng--4-stages)
7. [Datasets & validation strategy](#7-datasets--validation-strategy)
8. [Repo & tooling layout](#8-repo--tooling-layout)
9. [Legal & licensing](#9-legal--licensing)
10. [Risks & mitigations](#10-risks--mitigations)
11. [Quyết định mở cần chốt sau](#11-quyết-định-mở-cần-chốt-sau)

---

## 1. Tầm nhìn & nguyên tắc thiết kế

### Tầm nhìn
- **Tự chủ 100%**: từ MIPI driver sensor → SLAM algorithm → FC link. Không lib runtime nào không thuộc về mình.
- **Portable**: cùng codebase chạy được trên macmini (dev), RPi5 (prototype), custom SoC (production tương lai).
- **Production-grade**: drone bay được autonomous indoor + outdoor, có loop closure, persistent map, FC integration.
- **Modular**: thay sensor (OAK-D → IMX570 ToF → custom MIPI camera) chỉ cần đổi 1 lớp driver.

### Nguyên tắc thiết kế
1. **C99 cho core** (math + algorithms): portable sang MCU/custom SoC, no STL dependency.
2. **C++ chỉ làm thin wrapper** cho hardware SDK của bên thứ 3 (depthai, libcamera, v.v.).
3. **No heavy external libs ở runtime**: không Eigen / Sophus / Ceres / g2o / OpenCV trong production binary. Tự viết.
4. **Sensor-agnostic backend**: VIO/SLAM chỉ thấy `sky_image_t`, `sky_imu_sample_t`, `sky_depth_t` — không biết hardware nào.
5. **Test-driven**: mỗi module có unit test với numerical oracle (so với Eigen/OpenVINS trong dev, không ship).
6. **Spec-first workflow**: viết spec.md trước khi code mỗi phase, để tránh rewrite.
7. **Dataset-driven validation**: dùng EuRoC + TUM-VI + TUM RGB-D + ICL-NUIM trước khi hardware test.

---

## 2. So sánh hardware options

### 2.1 OAK-D W (hiện tại, prototype)
- Stereo wide-FOV + BMI270 IMU + Myriad X VPU.
- Pros: plug-and-play, depthai SDK chín, IMU sync chuẩn.
- Cons: closed-source firmware, VPU bandwidth giới hạn (4 streams crash), không thể tự chủ.
- **Vai trò**: prototype + dataset capture cho phase 1-4 dev.

### 2.2 IMX570 ToF (hardware tự làm V1)
- Sony DepthSense CAPD ToF 640×480, depth + IR amplitude, 30-120fps.
- Range: 0.2-5m indoor; kém ngoài trời nắng.
- Pros: depth chất lượng cao trên surface không texture (tường trắng); hoạt động trong tối; bỏ stereo matching.
- Cons: outdoor kém; range ngắn; laser ăn 3-5W; multi-path artifacts; SDK Sony proprietary.
- **Vai trò**: hardware V1 cho indoor inspection drone.

### 2.3 Custom stereo + ToF + IMU board (hardware tự làm V2)
- IMX296 global shutter mono × 2 (stereo, feature tracking).
- IMX570 ToF (depth fusion, indoor mode).
- BMI088 hoặc ICM-42688P IMU @ 1kHz qua SPI.
- STM32H7 hoặc Cortex-R co-processor sync timestamp + IMU integration.
- MIPI CSI-2 trực tiếp vào host SoC (RPi5 / Rockchip / Jetson Orin).
- **Vai trò**: production hardware V2, kết hợp ưu điểm stereo (outdoor) + ToF (indoor).

### 2.4 Sensor fusion strategy (cho V2)
- **Outdoor / range >5m**: stereo VIO mode, ToF off (tiết kiệm power).
- **Indoor / range <5m / low-light**: stereo + ToF fusion, ToF làm prior cho depth.
- **Tối hoàn toàn**: ToF + IR amplitude (mono stereo không có ánh sáng).

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
- **Sensor thread**: drivers push raw frames vào lock-free SPSC queue.
- **Frontend thread**: pull frame, detect/track, output `sky_obs_packet_t`.
- **Backend thread**: IMU integration + VIO state update + (optional) SLAM optimize.
- **Output thread**: pose → FC link + UI + logger.
- IMU thread riêng @1kHz, low latency.

---

## 4. Reference implementations để học

| Project | License | Vai trò | Note |
|---|---|---|---|
| **OpenVINS** (UDel RPNG) | BSD-3 | MSCKF VIO oracle | Code cleanest, well-documented, perfect cho học EKF VIO |
| **Basalt** (TUM) | BSD-3 | Sliding-window VIO + mapping reference | Đang dùng runtime, sau này thay |
| **RTAB-Map** (IntRoLab) | BSD-3 | SLAM + loop closure reference | Đang dùng runtime, sau này thay |
| **S-MSCKF** (UPenn) | BSD-3 | Stereo MSCKF reference | Compact, ~3k LOC, dễ port |
| **VINS-Fusion** (HKUST) | GPL-3 | Đọc ý tưởng, KHÔNG copy code | GPL → contaminate |
| **ORB-SLAM3** (UZ) | GPL-3 | Đọc ý tưởng, KHÔNG copy code | GPL → contaminate |
| **Kimera-VIO** (MIT) | BSD-2 | Reference cho factor graph approach | GTSAM-based |
| **DBoW3** | BSD-3 | Bag-of-words cho loop closure | Có thể port hoặc dùng tạm |
| **Sophus** (Strasdat) | MIT | SE3/SO3 oracle cho test | Dev-only, không ship |
| **Eigen** | MPL-2 | Linear algebra oracle cho test | Dev-only, không ship |
| **Ceres** (Google) | BSD-3 | NLLS solver reference | Có thể port LM core |
| **Kalibr** (ETH) | BSD-3 | Camera-IMU calibration reference | Inspire cho `sky_calibrate` tool |

**Quy tắc copy code**:
- BSD-3 / MIT / Apache-2: được copy, giữ copyright notice ở file copy + `THIRD_PARTY_LICENSES.txt`.
- GPL / LGPL: KHÔNG copy. Chỉ được đọc để học khái niệm, viết lại từ ý tưởng.
- Patent-encumbered (SURF): tránh hoàn toàn.

---

## 5. Roadmap phần mềm — 6 phases

> **Estimate format**: `AI coding time | wall-clock time (gồm review + test + debug)`.

### Phase 0 — Foundation: `libskymath` (1-2 ngày | 3-5 ngày)

**Goal**: math kernel C99 tự lực, NEON-optimized, pass test vs Eigen+Sophus.

**Modules**:
- `vec[2|3|4]`, `mat[3|4]`: dense ops, dot/cross/norm, transpose, inverse.
- `quat`: Hamilton convention, exp/log, slerp, ↔ rotation matrix.
- `so3`, `se3`: Lie group exp/log, adjoint, right/left Jacobians.
- `chol_dense`: dense Cholesky cho block 6×6 / 9×9 / 15×15.
- `chol_sparse_block`: block-sparse LDLT (or defer to Phase 3, dùng SuiteSparse tạm).
- `lm_solver`: Levenberg-Marquardt generic callback.
- `neon/`: ARM NEON intrinsics, fallback scalar khi x86.

**Tests**:
- Random fuzz 10k samples vs Eigen, sai số < 1e-10.
- Property test: `exp(log(R)) == R`, `Adj(T) * xi == log(T * exp(xi) * T^-1)`.
- Benchmark vs Eigen: target ≤ 1.5× chậm hơn cho mat3/quat.

**Deliverable**: `libskymath.a` + `test_math` binary, CI green.

---

### Phase 1 — Sensor abstraction + dataset replay (0.5-1 ngày | 1-2 ngày)

**Goal**: định nghĩa sensor types, replay EuRoC / TUM-VI / TUM RGB-D / ICL-NUIM.

**Modules**:
- `sky_sensor_types.h`: `sky_image_t`, `sky_imu_sample_t`, `sky_depth_t`, `sky_visual_frame_t`.
- `sky_camera_calib.c`: intrinsics struct + undistort (radial-tangential, Kannala-Brandt fisheye).
- `sky_dataset_euroc.c`: parse EuRoC MAV format.
- `sky_dataset_tumvi.c`: parse TUM-VI format.
- `sky_dataset_tum_rgbd.c`: parse TUM RGB-D format (cho ToF dev sau này).
- `sky_recorder.c`: ghi raw stream từ live sensor ra binary file → replay.

**Tool**: `sky_replay <dataset> <path>` push qua callback.

**Deliverable**: replay EuRoC MH_01 verify timestamps + image checksums.

---

### Phase 2 — Frontend: feature tracking (2-3 ngày | 5-7 ngày)

**Goal**: feature detection + tracking + outlier reject, NEON-optimized.

**Modules**:
- `fast_detector.c`: FAST-9, non-max suppression, grid distribution, NEON.
- `klt_tracker.c`: pyramidal Lucas-Kanade, 4 levels, sub-pixel refine, NEON.
- `stereo_match.c`: epipolar match left↔right, ZNCC patch.
- `rgbd_lift.c`: 2D feature + depth lookup → 3D point + variance từ confidence.
- `pnp_ransac.c`: P3P + RANSAC cho 3D-2D outlier reject.
- `essential_ransac.c`: 5-point + RANSAC cho 2D-2D.

**Tool**: `sky_track_vis` — visualize tracks trên EuRoC, đo tracking length + outlier rate.

**Success**: trên EuRoC MH_01, median tracking length ≥ 30 frames, outlier rate < 10%.

---

### Phase 3a — MSCKF VIO (3-5 ngày code | 2-3 tuần debug)

**Goal**: working VIO trên EuRoC, ATE < 0.2m trên MH_01.

**Modules**:
- `imu_preintegration.c`: Forster et al. 2017, midpoint integration, Jacobians wrt bias.
- `state_msckf.c`: state `[p, q, v, bg, ba]` + cloned poses sliding window.
- `feature_init.c`: multi-view triangulation + Gauss-Newton inverse-depth refine.
- `ekf_predict.c`: IMU propagation step.
- `ekf_update.c`: visual measurement update, EKF Kalman gain block-sparse.
- `chi2_test.c`: Mahalanobis distance outlier reject.
- `vio_orchestrator.c`: frontend ↔ backend pipeline.

**Reference**: OpenVINS `MsckfManager`, S-MSCKF.

**Test on EuRoC** (compare with OpenVINS as oracle):
| Sequence | Target ATE (m) | Basalt ATE (m) |
|---|---|---|
| MH_01_easy | < 0.15 | 0.07 |
| MH_03_medium | < 0.25 | 0.12 |
| MH_05_difficult | < 0.40 | 0.20 |

**Critical path**: VIO bug debug cực khó (sai dấu Jacobian → drift vô hạn). Mitigation:
- Compare from intermediate IMU prop output với OpenVINS log mỗi 100 samples.
- Unit test mỗi residual + Jacobian với numerical diff (central difference).
- Visualize covariance ellipse trên trajectory.

---

### Phase 3b — Sliding-window optimizer (10-16 ngày code | 4-6 tuần) [OPTIONAL]

**Goal**: nâng accuracy ngang Basalt nếu Phase 3a không đủ.

**Modules**:
- `sw_optimizer.c`: fixed-lag smoother, window 5-10 KFs.
- `marginalization.c`: Schur complement → prior factor.
- `factor_imu.c`, `factor_visual.c`, `factor_prior.c`: residual + Jacobian.
- Solver: dùng `lm_solver` + block-sparse Cholesky từ Phase 0.

**Khuyến nghị**: SKIP nếu Phase 3a đạt < 0.25m ATE trung bình. MSCKF đủ cho drone production.

---

### Phase 4 — SLAM + loop closure: `libskyslam` (4-6 ngày code | 2-4 tuần)

**Goal**: persistent map, loop closure, drift correction.

**Modules**:
- `descriptor_orb.c`: ORB (FAST + BRIEF + orientation), NEON.
- `vocabulary_tree.c`: hierarchical k-means BoW, train offline trên TUM dataset.
- `vocab_trainer.c` (offline tool): train vocab từ ~10k images.
- `loop_detector.c`: query DB → top-K candidates → geometric verify (PnP RANSAC).
- `pose_graph.c`: keyframe graph + relative SE3 constraints.
- `pose_graph_optim.c`: Gauss-Newton on SE(3), block-sparse Cholesky.
- `map_db.c`: keyframe storage + persistent binary serialization.
- `slam_orchestrator.c`: VIO odometry → KF selection → loop detect → optimize → correct.

**Reference**: RTAB-Map architecture, DBoW3.

**Test**:
- TUM-VI room1/2: loop detection rate > 80%, drift sau loop < 1% path length.
- Persistent DB: save / load / continue mapping multi-session.

**Alternative**: dùng DBoW3 library (BSD-3) thay vì tự viết vocab tree → tiết kiệm 1 tuần.

---

### Phase 5 — Hardware integration + deploy (2-3 ngày code | 1-2 tuần test)

**Goal**: chạy realtime trên OAK-D + RPi5, FC link MAVLink.

**Modules**:
- `driver_oakd.cpp`: thin C++ wrapper depthai 3.x → emit `sky_visual_frame_t` + `sky_imu_sample_t`.
- `fc_link_mavlink.c`: MAVLink `VISION_POSITION_ESTIMATE` qua UART.
- `fc_link_msp.c`: MSP v2 (cho Betaflight/INAV).
- `sky_daemon.c`: long-running service, systemd unit RPi5.
- ROS2 bridge (optional): DDS publisher.

**Test on hardware**:
- macmini: ≥ 30 FPS realtime.
- RPi5: ≥ 20 FPS, CPU < 70%, RAM < 500MB.
- Flight test: bay vuông 10m indoor, drift sau loop < 0.3m.

---

### Phase 6 — Production hardening (3-5 ngày | 1-2 tuần)

**Goal**: production-ready stack.

**Modules**:
- `sky_calibrate` tool: camera intrinsics (Zhang's method) + camera-IMU extrinsics (Kalibr-style).
- `failure_detector.c`: track lost / IMU saturate / depth invalid → recovery state.
- `sky_logger.c`: binary log format (sensor + state + estimate).
- `sky_log_replay` tool: replay log offline.
- CI matrix: x86_64 + aarch64 build, run tests.
- Docs: math derivations PDF + architecture diagrams.

---

### Tổng kết phần mềm

| Phase | LOC | AI code | Wall-clock |
|---|---|---|---|
| 0. Math | 5k | 1-2 ngày | 3-5 ngày |
| 1. Sensor | 2k | 0.5-1 ngày | 1-2 ngày |
| 2. Frontend | 3k | 2-3 ngày | 5-7 ngày |
| 3a. MSCKF | 4k | 3-5 ngày | 2-3 tuần |
| 3b. SW opt (opt) | 6k | 10-16 ngày | 4-6 tuần |
| 4. SLAM | 6k | 4-6 ngày | 2-4 tuần |
| 5. HW integ | 2k | 2-3 ngày | 1-2 tuần |
| 6. Hardening | 2k | 3-5 ngày | 1-2 tuần |
| **Total (no 3b)** | **~24k** | **~3 tuần** | **~3-4 tháng** |

---

## 6. Roadmap phần cứng — 4 stages

### Stage HW-0 — OAK-D W (now)
- COTS, dùng cho phần mềm dev Phase 0-5.
- Driver: depthai 3.x.
- **Khi nào kết thúc**: Phase 5 software done, ready chuyển sang custom HW.

### Stage HW-1 — IMX570 ToF dev board (~2-3 tháng sau khi HW-0 stable)
- Mua module Lucid Helios2 (GigE) hoặc Basler blaze-101.
- Kết nối: GigE Ethernet (Lucid) hoặc USB3 (Basler).
- Pair với IMX296 mono global shutter qua MIPI vào RPi5.
- BMI088 hoặc ICM-42688P IMU qua SPI vào STM32 → UART → host.
- **Phần mềm cần thêm**: `driver_lucid.c` (GigE Vision), `driver_imx296.c` (libcamera), `driver_imu_stm32.c` (UART protocol).
- Goal: validate hybrid stereo + ToF VIO concept.

### Stage HW-2 — Custom carrier board V1 (~6 tháng)
- Schematic + PCB tự thiết kế (KiCad).
- MIPI CSI-2 routing cho IMX296 × 2 + IMX570.
- STM32H7 co-processor: IMU + sensor timestamp sync.
- Power management: 5V/3.3V/1.8V rails, laser driver cho ToF.
- Connector: M.2 hoặc Raspberry Pi HAT form factor.
- **Phần mềm cần thêm**: kernel driver / libcamera IPA cho custom MIPI pipeline.

### Stage HW-3 — All-in-one module V2 (~12-18 tháng)
- Tích hợp SoC (Rockchip RK3588 hoặc Jetson Orin NX) cùng sensor trên 1 board.
- Form factor: 60×40mm, < 50g.
- Onboard SLAM stack chạy autonomous, output qua USB / Ethernet / UART.
- IP67 enclosure tùy chọn.
- **Goal**: thay thế hoàn toàn OAK-D cho production drone.

---

## 7. Datasets & validation strategy

### 7.1 Datasets

| Dataset | Sensor type | Vai trò |
|---|---|---|
| **EuRoC MAV** | Stereo + IMU (MAV indoor) | VIO benchmark chính |
| **TUM-VI** | Stereo + IMU (handheld, có loop) | SLAM loop closure benchmark |
| **TUM RGB-D** | RGB + depth (Kinect) | RGBD VIO/SLAM cho phase ToF |
| **ICL-NUIM** | Synthetic RGB-D | Ground truth tuyệt đối, debug |
| **KITTI** | Stereo + GPS (outdoor car) | Outdoor stress test |
| **Self-recorded OAK-D** | Stereo + BMI270 | Real hardware test |
| **Self-recorded IMX570** | RGBD + IMU | Khi có HW-1 |

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
| 3a | EuRoC MH_01 ATE | < 0.15m |
| 3a | EuRoC MH_05 ATE | < 0.40m |
| 4 | TUM-VI loop detection rate | > 80% |
| 4 | Post-loop drift | < 1% path length |
| 5 | RPi5 realtime FPS | ≥ 20 |
| 5 | RPi5 CPU usage | < 70% |
| 5 | Indoor flight 10m square drift | < 0.3m |

---

## 8. Repo & tooling layout

### 8.1 Repo location
`/Users/bao/skydev/skyslam` (separate từ `oak-d`).

### 8.2 Structure

```
skyslam/
├── README.md
├── LICENSE                          # BSD-3 hoặc proprietary, anh quyết
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
│   ├── oakd/                        # C++ wrapper depthai
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

### 9.1 Licenses anh có thể dùng

- **BSD-3 / MIT / Apache-2**: copy code OK, kèm attribution.
- **MPL-2 (Eigen)**: dev-only, không ship runtime.
- **LGPL**: tránh tĩnh-link, OK dynamic-link nhưng phức tạp → tránh hoàn toàn.
- **GPL**: KHÔNG copy, KHÔNG link, chỉ đọc concept.

### 9.2 Patent check
- **SIFT**: patent hết hạn 2020 — free.
- **SURF**: vẫn còn patent → tránh.
- **ORB**: free (BSD trong OpenCV).
- **FAST**: free.
- **BRIEF**: free.

### 9.3 SkySLAM's own license
- Đề xuất: **BSD-3-Clause** nếu anh muốn open-source, **proprietary closed** nếu giữ commercial advantage.
- Quyết định sau khi có Phase 5 working.

### 9.4 Compliance checklist mỗi commit
- [ ] File copy từ project khác → header notice nguyên gốc.
- [ ] `THIRD_PARTY_LICENSES.txt` cập nhật.
- [ ] Không link GPL/LGPL trong production binary.

---

## 10. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| VIO debug kéo dài (Jacobian sai dấu) | High | High | Numerical diff test mọi Jacobian; oracle compare với OpenVINS |
| NEON intrinsics khó debug | Medium | Medium | Scalar fallback luôn có, test parity hai paths |
| Block-sparse Cholesky tự viết bug | High | High | Phase 0 skip; dùng SuiteSparse tạm; viết lại ở Phase 3b nếu cần |
| EuRoC pass nhưng OAK-D fail | Medium | High | Calibration cẩn thận, IMU noise model tune |
| RPi5 không đủ realtime | Low | High | Profile sớm; có Jetson Orin Nano backup |
| OAK-D firmware vẫn crash | Medium | Medium | Đã workaround; có dataset record fallback |
| IMX570 SDK Sony proprietary | High | Medium | Dùng Lucid Helios GigE thay thế (open protocol) |
| Custom PCB delay (HW-2) | High | Medium | Đặt JLCPCB; có spare bare module backup |
| Hardware tự làm fail | Medium | Critical | Vẫn giữ OAK-D path là fallback production |

---

## 11. Quyết định mở cần chốt sau

1. **License SkySLAM**: BSD-3 hay proprietary?
2. **Phase 3b**: có làm sliding-window optimizer không, hay MSCKF đủ?
3. **DBoW3 vs self-write vocab tree**: tiết kiệm thời gian vs full tự chủ?
4. **SuiteSparse vs self-write block-sparse Cholesky**: tương tự.
5. **HW-1 sensor lựa chọn**: Lucid Helios2 vs Basler blaze-101 vs custom IMX570 board?
6. **HW production SoC**: RK3588 vs Jetson Orin NX vs custom?
7. **FC protocol primary**: MAVLink (Ardupilot/PX4) hay MSP (Betaflight/INAV)?
8. **Anh muốn AI làm autonomous tới đâu**: code + commit, hay chỉ code + anh commit manual?

---

## 12. Cross-references

- Project hiện tại sử dụng Basalt + RTAB-Map: `/Users/bao/skydev/oak-d`
- Tham khảo IMU rate / filter setting cho FC: user memory `icm42688p-godr-vs-sample-rate.md`
- LPF design rule: user memory `lpf-time-constant-rule.md`

---

## 13. Lịch sử thay đổi

| Ngày | Thay đổi | Tác giả |
|---|---|---|
| 2026-05-27 | Bản draft đầu tiên | Bảo + Copilot |

---

**Khi nào quay lại file này**: trước khi bắt đầu Phase 0, anh nên đọc lại mục 5, 7, 11 để xác nhận direction. Update mục 11 khi có quyết định mới.
