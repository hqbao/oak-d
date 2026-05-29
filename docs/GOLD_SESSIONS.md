# Gold Sessions — Regression Suite for SkySLAM

**Mục đích**: Một bộ session "vàng" cố định để mọi version pipeline tương lai
(BasaltVIO+RTABMapSLAM hôm nay, skyslam_backend ngày mai, skyslam_frontend
ngày kia) đều phải vượt qua. Đây là **single source of truth** cho regression.

**Nguyên tắc**:
- Record 1 lần, dùng mãi mãi. KHÔNG re-record trừ khi calibration thay đổi.
- Folder `sessions/gold/` gitignored (dữ liệu nặng) — backup tay sang ổ riêng.
- Baseline numbers được freeze trong `GOLD_BASELINE.md` (gen bằng
  `tools/baseline_report.py` sau khi record xong).
- Khi viết module mới → chạy lại trên toàn bộ gold, so với baseline.
  Regression = thấy số tệ hơn baseline → fail.

---

## Scenarios

6 scenario được chọn để cover các "axis" thử thách khác nhau của VIO/SLAM.

| ID | Tên | Thời lượng | Mục đích test |
|---|---|---|---|
| 1 | `lab_static_10s` | 10s | bias drift, IMU noise floor — phải đứng yên |
| 2 | `lab_straight_20s` | 20s | translation scale, IMU-visual sync |
| 3 | `lab_loop_30s` | 30s | loop closure (vòng kín nhỏ, ~3m radius) |
| 4 | `corridor_60s` | 60s | long-range drift, loop closure xa (~15m loop) |
| 5 | `quick_motion_15s` | 15s | tracking robustness (lắc + quay đầu nhanh) |
| 6 | `loop_closure_45s` | 45s | C4 keyframes + loop closure DB persistence |

### Cách thực hiện từng scenario

#### 1. `lab_static_10s` — đứng yên
Đặt camera lên tripod hoặc bàn, KHÔNG chạm. Mục đích: pose phải ~constant,
bất cứ drift nào = sai IMU bias.

```bash
.venv/bin/python tools/record_session.py sessions/gold/lab_static_10s \
    --duration 10 -f
```

Expected: |pos drift| < 5cm, |rot drift| < 1°.

#### 2. `lab_straight_20s` — đi thẳng
Cầm camera bằng tay, đi thẳng ~5m, dừng, quay 180°, đi về điểm xuất phát.
Mục đích: scale + tracking liên tục, KHÔNG có loop closure (chưa quay lại
gần đủ để trigger).

```bash
.venv/bin/python tools/record_session.py sessions/gold/lab_straight_20s \
    --duration 20 -f
```

Expected: ATE VIO-vs-SLAM < 10cm (vì chưa có loop, gần như equal).

#### 3. `lab_loop_30s` — vòng kín nhỏ
Đi 1 vòng quanh bàn / desk (~3m radius), quay về đúng điểm xuất phát.
Mục đích: trigger loop closure → odom_correction phải có 1 jump rõ rệt.

```bash
.venv/bin/python tools/record_session.py sessions/gold/lab_loop_30s \
    --duration 30 -f
```

Expected: ≥1 loop event trong `loop_events.jsonl`, ATE SLAM < ATE VIO.

#### 4. `corridor_60s` — hành lang dài
Đi dọc hành lang ~10-15m, quay đầu, đi về. Mục đích: drift tích lũy xa,
loop closure ở khoảng cách lớn.

```bash
.venv/bin/python tools/record_session.py sessions/gold/corridor_60s \
    --duration 60 -f
```

Expected: VIO drift đáng kể (~30cm-1m), SLAM correct về <20cm sau loop.

#### 5. `quick_motion_15s` — lắc mạnh
Cầm camera, đi vòng tròn nhanh + quay đầu liên tục + chuyển hướng đột ngột.
Mục đích: test feature tracker khi motion blur + IMU saturation.

```bash
.venv/bin/python tools/record_session.py sessions/gold/quick_motion_15s \
    --duration 15 -f
```

Expected: không tracking_lost (nếu có = pipeline yếu, document số gap).

#### 6. `loop_closure_45s` — vòng kín có loop closure (C4 test)
Đứng yên 5s ngắm 1 mốc rõ (góc bàn, poster), đi vòng nhỏ ~2-3m radius
trong 30s, về **đúng** vị trí + **đúng** hướng ban đầu, đứng yên 5-10s
cho RTABMap chốt loop. Mục đích: verify `rtabmap.db` persist được +
`extract_kf_from_db.py` tạo ≥1 loop link.

```bash
.venv/bin/python tools/record_session.py sessions/gold/loop_closure_45s \
    --duration 45 -f
```

Expected: ≥1 `kf_loops` (Link type 1/2/3 trong rtabmap.db). Nếu = 0 →
record lại (góc về sai, BoW không match).

---

## Sau khi record xong cả 6

1. Backup folder `sessions/gold/` sang ổ ngoài (USB/cloud). Không có
   git tracking, mất là mất.
2. Chạy báo cáo baseline:
   ```bash
   .venv/bin/python tools/baseline_report.py sessions/gold/ \
       > docs/GOLD_BASELINE.md
   ```
3. Commit `docs/GOLD_BASELINE.md` (file nhẹ, chỉ số metric).

Từ giờ trở đi, mỗi khi viết module skyslam mới:
```bash
# Re-process gold sessions với skyslam
.venv/bin/python tools/replay_skyslam.py sessions/gold/lab_loop_30s
# So sánh với baseline
.venv/bin/python tools/compare_sessions.py \
    sessions/gold/lab_loop_30s/basalt/slam_pose.jsonl \
    sessions/gold/lab_loop_30s/skyslam/slam_pose.jsonl
```

(`replay_skyslam.py` chưa viết — sẽ làm ở Phase B/C.)
