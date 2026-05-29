# SkySLAM — Pipeline Checkpoints for Future Comparison

**Mục đích**: Định nghĩa các điểm chốt (checkpoint) trong pipeline hiện tại
(BasaltVIO + RTABMapSLAM) để khi tự viết `skyslam`, mình có thể so sánh
output từng giai đoạn với baseline để debug và đo accuracy.

**Ngày soạn**: 2026-05-29 (rewrite: nguyên tắc "chỉ show dữ liệu thật")
**Pipeline tham chiếu**: `oakd/sources/depthai_slam.py` (live) + `depthai_vo.py` (VIO-only)
**Target so sánh**: `skyslam/` (chưa viết — kế hoạch chi tiết tại `docs/SKYSLAM_RESEARCH.md`)

---

## 0. Nguyên tắc tối thượng: HONEST VISUALIZATION

> **Recorder + viewer CHỈ được show dữ liệu thật từ pipeline đang chạy.**
> Tuyệt đối KHÔNG tạo dữ liệu phụ (FeatureTracker song song, depth lookup
> tự chế, sparse map ước lượng…) rồi gắn nhãn như là output của
> Basalt/RTABMap.

Vì sao quan trọng: chiến lược dài hạn là **thay thế từng module** của
pipeline depthai bằng code mình viết. Nếu viewer hiện cả dữ liệu fake,
khi mình thay module sẽ không biết kết quả mới đúng hay sai (đang so
với chính dữ liệu fake cũ → vô nghĩa).

**Quy tắc cứng:**
- Mỗi cái UI hiện ra phải truy ngược được về **một depthai Output**
  thực tế đã subscribe.
- Cấm dùng node phụ (`FeatureTracker`, `ImageManip`…) cho mục đích
  visualize "vờ như" của blob khác.
- Internals của blob (Basalt features, RTABMap keyframes/BoW…) ⇒ chỉ
  expose được khi mình tự viết module đó. Trong lúc chưa viết: **để trống**.

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

**Cái GHẾ TRỐNG (depthai KHÔNG expose):**

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

## 2. Nguyên tắc tổ chức

1. **Black box những gì không sửa được**: Basalt và RTABMap là binary blob.
   KHÔNG can thiệp internals. Chỉ log được input + output streams.
2. **JSONL cho metadata, binary cho dense data**: timestamps + pose ra JSONL
   (dễ diff/grep); ảnh + depth + cloud ra raw binary.
3. **Timestamp là vàng**: mọi record có `ts_ns` (nanosecond, monotonic từ
   pipeline start). Đây là khóa join giữa các stream và giữa 2 pipelines.
4. **Reproducible**: record session đầy đủ → replay được vào cả pipeline cũ
   (sanity) và pipeline mới (skyslam). Cùng input → mới so sánh được output.
5. **Observability levels** (chỉ những gì có sẵn từ depthai):
   - L0: input streams (camera, IMU) — luôn ghi được
   - L1: final pose output — luôn ghi được
   - L2: pose-graph correction (odomCorrection) — RTABMap expose
   - L2: dense reconstruction (obstacle/ground PCL) — RTABMap expose
   - L3+: internal features/landmarks/keyframes — **KHÔNG expose, để trống**

---

## 3. Checkpoint matrix (status tại 2026-05-29)

| # | Tên checkpoint | Status | Source | Notes |
|---|---|---|---|---|
| C0 | Input: stereo frame (L+R+D) | ✅ live | `StereoDepth.rectifiedLeft/syncedRight/depth` | PNG + raw16 |
| C1 | Input: IMU sample | ✅ live | `IMU.out` | 200 Hz |
| C2 | VIO pose (Basalt) | ✅ live | `BasaltVIO.transform` | FLU world |
| C3 | SLAM pose (RTABMap, loop-corrected) | ✅ live | `RTABMapSLAM.transform` | FLU world |
| C4 | Keyframe selected | ✅ offline | `rtabmap.db` SQLite (`Node` table) | `tools/extract_kf_from_db.py` — auto-run sau record |
| C4+ | Loop closure links (rich) | ✅ offline | `rtabmap.db` SQLite (`Link` types 1/2/3) | cùng tool, → `kf_loops.jsonl` |
| C5 | Loop closure detected (derived) | ⚠️ derived | jump > thr trong `odomCorrection` stream | precision OK, không có BoW score live |
| C6 | Tracking lost/recovered | ⚠️ derived | gap > thr trong SLAM pose stream | đủ để vẽ timeline |
| C7 | Frontend features per frame | ❌ blob hidden | **chỉ skyslam mới có** | KHÔNG fake bằng FeatureTracker |
| C8 | IMU preintegration delta | ❌ blob hidden | chỉ skyslam | |
| C9 | Optimizer state (Jacobians) | ❌ blob hidden | chỉ skyslam | |
| — | Odom correction (raw) | ✅ live | `RTABMapSLAM.odomCorrection` | input cho C5 derivation |
| — | Dense point cloud | ✅ live | `RTABMapSLAM.obstaclePCL + groundPCL` | dense reconstruction, KHÔNG phải sparse map |

Ký hiệu: ✅ = stream thật từ pipeline, ⚠️ = derive từ stream thật, ❌ = blob không expose → chưa làm.

---

## 4. Migration strategy: thay từng blob

Mục tiêu: thay dần `BasaltVIO` rồi `RTABMapSLAM` bằng module skyslam tự viết.
Mỗi lần thay 1 cục, sẽ UNLOCK thêm checkpoint mới (vì module mình viết sẽ
expose internals).

### Phase A (hiện tại) — record honest baseline
Record session với BasaltVIO + RTABMapSLAM, có C0, C1, C2, C3, C5, C6, PCL.
KHÔNG có C4/C7/C8/C9. Đây là **ground truth pipeline** để so sánh.

### Phase B — thay backend RTABMap bằng `skyslam_backend` (pose graph + loop)
- Input: vẫn dùng `BasaltVIO.transform` (C2) làm odom.
- Output: pose mới (C3 thay thế), thêm:
  - C4 (keyframe selection): khi mình tự chọn KF
  - C5 đầy đủ (kf_query/kf_match/BoW score/inliers): khi mình tự detect loop
  - Sparse pose graph topology
- Validation: so sánh `skyslam_backend.transform` vs `RTABMapSLAM.transform`
  trên cùng input C2 → ATE/RPE phải <= baseline.

### Phase C — thay frontend Basalt bằng `skyslam_frontend` (VIO)
- Input: C0 + C1 (stereo + IMU).
- Output: pose (C2 thay thế), thêm:
  - C7 (tracked features 2D)
  - C8 (IMU preintegration deltas)
  - Sparse 3D landmarks ← **đây mới là "feature point cloud" thật**, vài
    trăm điểm, khác hẳn dense PCL của RTABMap.
- Validation: chạy song song `skyslam_frontend` + `BasaltVIO`, so sánh
  pose stream + xem features overlay có tương quan không.

### Phase D — full skyslam stack
- `BasaltVIO` và `RTABMapSLAM` chỉ còn dùng để regression test.
- Toàn bộ C0…C9 đều là dữ liệu skyslam, viewer overlay được mọi thứ.

---

## 3. Schemas chi tiết

Mọi record có 2 field bắt buộc: `ts_ns` (uint64) + `seq` (uint32, monotonic).

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
- Lưu intrinsics 1 lần ở `calib.json`, schema chỉ ref ID nếu muốn gọn.

### C1 — imu

```jsonl
{"ts_ns": 1716950123450000000, "seq": 8421,
 "gyro": [0.001, -0.002, 0.0005],
 "accel": [0.12, -0.05, 9.81],
 "temp_c": 32.5}
```

200 Hz stream. Field tên fix cứng để diff dễ.

### C2 — pose6dof (VIO raw, FLU world frame as Basalt outputs)

```jsonl
{"ts_ns": 1716950123456789000, "seq": 42,
 "frame_id": "flu_world",
 "pos": [0.123, -0.456, 0.789],
 "quat_wxyz": [0.9998, 0.001, -0.002, 0.005],
 "vel": [0.05, 0.01, 0.0],
 "tracking_ok": true,
 "source": "basalt_vio"}
```

- `pos` mét, `quat` đơn vị, `vel` m/s.
- `source` enum: `basalt_vio` | `rtabmap_slam` | `sky_vio` | `sky_slam`.
- Lưu **FLU world** (frame gốc của Basalt), KHÔNG convert NED — để so sánh
  apples-to-apples giữa Basalt và skyvio. NED conversion là post-processing.

### C3 — pose6dof (SLAM loop-corrected)

Same schema as C2, `source = rtabmap_slam` hoặc `sky_slam`.

### C4 — kf_event ✅ DONE (offline)

```jsonl
{"ts_ns": 2402326697, "kf_id": 1, "weight": 0,
 "pos": [0.0007, 0.0022, 0.0707],
 "quat_wxyz": [0.6513, -0.0671, -0.7532, -0.0629]}
```

- Source: SQLite `rtabmap.db` table `Node`, pose là BLOB 48-byte (3x4 float32
  row-major `[R|t]`), stamp giây từ device boot.
- Extract tool: `tools/extract_kf_from_db.py` — auto-chạy ở cuối
  `record_session.py` (xem dòng cuối log: `kf: N keyframes, M loop links`).
- Loop links phụ (C4+) ghi song song vào `kf_loops.jsonl` từ `Link` table
  với type ∈ {1=global, 2=local_space, 3=local_time}:

```jsonl
{"ts_ns": ..., "from_kf": 12, "to_kf": 3, "type": 1, "type_name": "global",
 "transform_pos": [...], "transform_quat_wxyz": [...]}
```

- `weight=-9` là intermediate node của RTABMap (graph reduction); `weight=0`
  là keyframe bình thường.
- Viewer (`viz_session.py`) render KF = chấm vàng + loop closure = đường
  vàng giữa 2 KF trong tab Pose 3D.

### C5 — loop_event

```jsonl
{"ts_ns": ..., "event": "loop_closure",
 "pos_jump_m": 0.18, "rot_jump_deg": 3.4,
 "correction_pos": [...], "correction_quat_wxyz": [...]}
```

- Detect bằng cách tap output `RTABMapSLAM.odomCorrection` (map←odom transform)
  và đo delta giữa hai sample liên tiếp. Jump lớn (default
  `pos > 10 cm` hoặc `rot > 5°`) = loop closure đã apply correction.
- Bỏ qua `LOOP_WARMUP_S` (default 3s) đầu tiên vì RTABMap publish correction
  bất định khi map đang khởi tạo (sẽ gây hàng chục false-positive).
- Schema BoW (`kf_query`, `score`, `inliers`) chỉ available qua SQLite —
  `tools/extract_kf_from_db.py` đã extract `Link` table (loop edges với
  transform + type) ra `kf_loops.jsonl`. Score + inlier count thì cột
  `information` của `Link` table có nhưng chưa parse (TODO nếu cần).
- Raw stream được dump song song ở `basalt/odom_correction.jsonl` để debug.

### C6 — track_event

```jsonl
{"ts_ns": ..., "event": "tracking_lost", "last_pose_seq": 142, "gap_s": 0.72}
{"ts_ns": ..., "event": "tracking_recovered", "first_pose_seq": 158}
```

- Derive bằng cách scan SLAM pose stream tìm gap > `TRACK_GAP_S` (default
  0.5s). Done trong `SessionRecorder.close()`.

### C7 — features (tracked corners) — ❌ NOT IMPLEMENTED

Depthai `BasaltVIO` blob KHÔNG expose internal feature tracks. Đã từng
thử dùng `dai.node.FeatureTracker` song song để giả vờ — **bỏ vì sai
nguyên tắc Section 0** (FeatureTracker dùng Harris/Shi-Tomasi khác hẳn
KLT internal của Basalt, overlay sẽ misrepresent những gì VIO thật sự
tracking).

Sẽ implement ở Phase C (xem Section 4) khi viết `skyslam_frontend`.

### Point cloud (RTABMap dense reconstruction)

Index `basalt/pointcloud.jsonl`:
```jsonl
{"ts_ns": ..., "seq": 0, "kind": "obstacle", "n_points": 5421,
 "path": "pointcloud/000000_obstacle.f32"}
```
- Binary file: raw `Nx3 float32` little-endian, world (FLU map) frame.
- RTABMap publish lại TOÀN BỘ cloud mỗi keyframe → viewer chỉ load
  emission gần nhất per kind (tránh trùng).
- **Đây là DENSE reconstruction từ depth map**, KHÔNG phải sparse map
  của VIO. Mỗi pixel depth hợp lệ trở thành 1 điểm 3D sau voxel filter
  (default 5cm). Hàng trăm ngàn điểm cho 1 phòng nhỏ.
- Disable với `record_session --no-pcl`.

### C8-C9 — chỉ implement khi skyslam tự viết

---

## 4. Cấu trúc session folder

```
sessions/
└── 2026-05-29_oak_lab_loop1/
    ├── calib.json              # intrinsics + extrinsics, ghi 1 lần
    ├── meta.json               # pipeline version, params, host info
    ├── input/
    │   ├── imu.jsonl           # C1
    │   ├── frames.jsonl        # C0 metadata
    │   └── img/                # PNG + raw16, ~50 MB / phút
    ├── basalt/                 # output từ pipeline hiện tại (REAL streams)
    │   ├── vio_pose.jsonl         # C2 — BasaltVIO.transform
    │   ├── slam_pose.jsonl        # C3 — RTABMapSLAM.transform
    │   ├── odom_correction.jsonl  # raw RTABMapSLAM.odomCorrection
    │   ├── loop_events.jsonl      # C5 (derived from odom_correction jumps)
    │   ├── track_events.jsonl    # C6 (derived from slam pose gaps)
    │   ├── pointcloud.jsonl      # index of dense PCL emissions
    │   └── pointcloud/NNNN_*.f32  # RTABMap obstacle/ground PCL, Nx3 f32
    └── skyslam/                # output từ skyslam tương lai
        ├── vio_pose.jsonl         # C2 thay thế (Phase C)
        ├── slam_pose.jsonl        # C3 thay thế (Phase B)
        ├── kf_events.jsonl       # C4 (UNLOCKED khi skyslam backend)
        ├── features.jsonl        # C7 (UNLOCKED khi skyslam frontend)
        ├── landmarks.jsonl       # sparse 3D landmarks (UNLOCKED khi frontend)
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

## 5. Cách record từ pipeline hiện tại

Cần thêm 1 module `oakd/recorder.py`:

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

Wire vào `OakBasaltSlamSource`: thêm tap để mọi output queue đều fan-out
vào recorder. CLI flag mới: `--record sessions/<name>`.

Phải nghiên cứu RTABMap nodes API để biết queue nào emit KF/loop events
(có thể `slam.passthroughRect` + statistics output hoặc query database
file sau khi chạy).

---

## 6. Replay protocol cho skyslam tương lai

Khi skyslam có Phase 1 (sensor abstraction):

1. `sky_replay --session sessions/2026-05-29_oak_lab_loop1` push input C0+C1
   vào skyvio/skyslam.
2. Skyslam emit C2+C3+C4+C5+C6 ra `sessions/.../skyslam/`.
3. Tool `compare_sessions.py basalt/ skyslam/` tính metrics.

→ Cùng 1 input, 2 pipeline ra 2 set output, đo difference khách quan.

---

## 7. Comparison metrics

### 7.1 Pose accuracy (C2, C3)

- **ATE** (Absolute Trajectory Error): sau khi align trajectory (Umeyama SE3),
  RMSE position error. Đơn vị mét.
- **RPE** (Relative Pose Error): error trong cửa sổ trượt 1m / 1s. Tách
  translation + rotation component.
- Implementation reference: `evo` toolbox (Python, MIT), hoặc tự viết
  `compare_pose.py` ~200 LOC.

### 7.2 Keyframe agreement (C4)

- **KF count ratio**: `n_kf_skyslam / n_kf_basalt`. Target 0.7-1.5.
- **KF temporal overlap**: % của Basalt KFs có 1 skyslam KF trong ±0.5s.

### 7.3 Loop closure (C5)

- **Recall**: % loop của Basalt mà skyslam cũng detect.
- **Precision**: % loop của skyslam đúng (so với Basalt làm ground truth).
- **Drift correction**: ATE trước vs sau loop, đo tại các điểm "passed by
  previously visited place".

### 7.4 Robustness (C6)

- **Tracking uptime**: % thời gian `tracking_ok=true`.
- **Recovery time**: trung bình giây từ `lost` đến `recovered`.

### 7.5 Performance (không phải checkpoint, đo song song)

- **FPS** trung bình + p99 latency.
- **CPU %** trung bình.
- **RAM** peak.

Lưu vào `meta.json` của mỗi run.

---

## 8. Baseline gold standard

Trước khi build skyslam, cần record 3-5 session "gold" làm baseline cố định:

| Session | Mô tả | Mục đích |
|---|---|---|
| `lab_loop1_120s` | Bàn → đi 1 vòng phòng → quay về | Loop closure basic |
| `lab_figure8_180s` | Số 8 trong phòng, 2 loops | Multi-loop |
| `hallway_oneway_60s` | Đi thẳng hành lang, không loop | Pure VIO, no SLAM correction |
| `static_30s` | Đặt im trên bàn 30 giây | Drift baseline, IMU bias |
| `aggressive_motion_60s` | Lắc nhanh, xoay đột ngột | Robustness stress |

Mỗi session record full C0+C1+C2+C3+C4+C5+C6 từ pipeline Basalt+RTABMap.
Lưu vào `sessions/` (gitignore lớn, dùng external drive hoặc Git LFS).

Khi skyslam đạt từng phase, replay 5 sessions này → tính metrics →
plot regression chart "skyslam vs basalt" theo thời gian.

---

## 9. Implementation TODO

Pre-skyslam infra ✅ DONE:

- [x] `oakd/recorder.py` — fan-out tap mọi queue
- [x] `tools/record_session.py` — CLI với `--duration`, `--no-pcl`, `-f`
- [x] `tools/compare_sessions.py` — ATE/RPE giữa 2 pose streams
- [x] Record 6 gold sessions (xem `docs/GOLD_SESSIONS.md`)
- [x] `tools/extract_kf_from_db.py` — extract KF + loop từ rtabmap.db
- [x] `tools/baseline_report.py` — emit Markdown baseline
- [x] Frozen baseline (`docs/GOLD_BASELINE.md`)

Skyslam work — xem **`docs/SKYSLAM_RESEARCH.md`** Part 3 cho plan v3
(9 phases với acceptance gates).

---

## 10. Cross-references

- **Skyslam plan v3 (research-backed)**: `docs/SKYSLAM_RESEARCH.md`
- Gold regression suite: `docs/GOLD_SESSIONS.md` + `docs/GOLD_BASELINE.md`
- Long-term hardware/FC vision: `docs/SKYSLAM_ROADMAP.md`
- Current SLAM source: `oakd/sources/depthai_slam.py`
- Current VIO source: `oakd/sources/depthai_vo.py`
- Pose data structure: `oakd/pose.py`

---

## 11. Lịch sử

| Ngày | Thay đổi | Tác giả |
|---|---|---|
| 2026-05-29 | Draft đầu | Bảo + Copilot |
| 2026-05-29 | Honest pipeline rewrite (drop fake FeatureTracker overlay) | Bảo + Copilot |
| 2026-05-29 | C4 done: KF + loop links từ rtabmap.db SQLite | Bảo + Copilot |
| 2026-05-29 | Gold suite mở rộng → 6 sessions (thêm loop_closure_45s) | Bảo + Copilot |
