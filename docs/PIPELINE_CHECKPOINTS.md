# SkySLAM — Pipeline Checkpoints for Future Comparison

**Mục đích**: Định nghĩa các điểm chốt (checkpoint) trong pipeline hiện tại
(BasaltVIO + RTABMapSLAM) để khi tự viết `skyslam`, mình có thể so sánh
output từng giai đoạn với baseline để debug và đo accuracy.

**Ngày soạn**: 2026-05-29
**Pipeline tham chiếu**: `oakd/sources/depthai_slam.py` (live) + `depthai_vo.py` (VIO-only)
**Target so sánh**: `skyslam/` (chưa viết — xem `docs/SKYSLAM_ROADMAP.md`)

---

## 1. Nguyên tắc

1. **Black box những gì không sửa được**: Basalt và RTABMap là binary blob trong
   depthai. Không can thiệp internals. Chỉ log được input + output streams.
2. **JSONL cho metadata, binary cho dense data**: timestamps + pose ra JSONL
   (dễ diff/grep); ảnh + depth ra raw binary (PNG/EXR/raw16).
3. **Timestamp là vàng**: mọi record phải có `ts_ns` (nanosecond, monotonic
   từ pipeline start). Đây là khóa join giữa các stream và giữa 2 pipelines.
4. **Reproducible**: record session đầy đủ → replay được vào cả pipeline cũ
   (sanity) và pipeline mới (skyslam). Cùng input → mới so sánh được output.
5. **5 levels of observability**, từ dễ ghi nhất đến cần modify source code:
   - L0: input streams (camera, IMU) — luôn ghi được
   - L1: final pose output — luôn ghi được
   - L2: keyframe + loop closure events — RTABMap expose qua message ports
   - L3: frontend features (FAST keypoints, tracks) — phải patch hoặc bypass
   - L4: internal optimizer state (Jacobians, covariance) — chỉ làm được khi
     skyslam tự viết

---

## 2. Checkpoint matrix

| # | Tên checkpoint | Level | Stream | Schema | Mục đích so sánh |
|---|---|---|---|---|---|
| C0 | Input: stereo frame | L0 | binary + jsonl | image_pair | Giống nhau input → mới so output |
| C1 | Input: IMU sample | L0 | jsonl | imu | Như trên, IMU @ 200Hz |
| C2 | VIO raw pose (Basalt) | L1 | jsonl | pose6dof | Compare ATE, RPE giữa Basalt vs skyvio |
| C3 | SLAM loop-corrected pose | L1 | jsonl | pose6dof | Compare ATE giữa RTABMap vs skyslam |
| C4 | Keyframe selected | L2 | jsonl | kf_event | Compare KF selection policy |
| C5 | Loop closure detected | L2 | jsonl | loop_event | Compare loop detection recall/precision |
| C6 | Tracking lost / recovered | L2 | jsonl | track_event | Compare robustness |
| C7 | Frontend features per frame | L3 | jsonl | features | (chỉ khi skyslam, không record từ Basalt) |
| C8 | IMU preintegration delta | L3 | jsonl | imu_preint | (chỉ skyslam) |
| C9 | Optimizer state | L4 | binary | optim_state | (chỉ skyslam) |

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

### C4 — kf_event

```jsonl
{"ts_ns": ..., "seq": 42, "event": "keyframe_added",
 "kf_id": 17, "frame_seq": 42,
 "pos": [...], "quat_wxyz": [...],
 "reason": "translation_threshold"}
```

- RTABMap expose `dataOutputQueue` với keyframe data; phải subscribe để bắt event.
- `reason` enum: `first_frame` | `translation_threshold` | `rotation_threshold`
  | `feature_drop` | `time_threshold`.

### C5 — loop_event

```jsonl
{"ts_ns": ..., "kf_query": 142, "kf_match": 17,
 "score": 0.87, "inliers": 64,
 "relative_pose": {"pos": [...], "quat_wxyz": [...]}}
```

- `score` từ BoW similarity 0..1.
- `inliers` số RANSAC inlier khi geometric verify.
- Trong RTABMap đọc qua `Statistics` event hoặc database after run.

### C6 — track_event

```jsonl
{"ts_ns": ..., "event": "tracking_lost", "last_pose_seq": 142}
{"ts_ns": ..., "event": "tracking_recovered", "first_pose_seq": 158}
```

### C7-C9 — chỉ implement khi skyslam tự viết

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
    ├── basalt/                 # output từ pipeline hiện tại
    │   ├── vio_pose.jsonl      # C2
    │   ├── slam_pose.jsonl     # C3
    │   ├── kf_events.jsonl     # C4
    │   ├── loop_events.jsonl   # C5
    │   └── track_events.jsonl  # C6
    └── skyslam/                # output từ skyslam tương lai
        ├── vio_pose.jsonl
        ├── slam_pose.jsonl
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

Cần làm trước khi skyslam start:

- [ ] `oakd/recorder.py` — fan-out tap mọi queue của depthai pipeline
- [ ] `tools/record_session.py` — CLI wrapper `--source slam --record <path>`
- [ ] `tools/compare_sessions.py` — load 2 sessions, tính ATE/RPE/KF/loop metrics
- [ ] Record 5 gold sessions
- [ ] Document RTABMap event subscription cách subscribe KF + loop events
  (cần thử nghiệm; nếu RTABMap node không expose, phải đọc từ database file
  sau khi run xong)

Estimate: ~2-3 ngày AI coding cho recorder + compare tool.

---

## 10. Cross-references

- Long-term plan: `docs/SKYSLAM_ROADMAP.md`
- Current SLAM source: `oakd/sources/depthai_slam.py`
- Current VIO source: `oakd/sources/depthai_vo.py`
- Pose data structure: `oakd/pose.py`

---

## 11. Lịch sử

| Ngày | Thay đổi | Tác giả |
|---|---|---|
| 2026-05-29 | Draft đầu | Bảo + Copilot |
