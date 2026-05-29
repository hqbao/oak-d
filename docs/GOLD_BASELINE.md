# Gold Baseline Report

**Generated**: 2026-05-29 12:39:46
**Source**: `/Users/bao/skydev/oak-d/sessions/gold`
**Pipeline**: BasaltVIO + RTABMapSLAM (depthai 3.6.1)
**RPE window**: 1.0s

ATE/RPE compare **SLAM (ref) vs VIO (test)** — they measure how much loop closure correction RTABMap adds on top of pure VIO. Higher numbers = more correction (i.e. VIO drifted more).

| Session | Dur | Frm | VIO | SLAM | Lp | Trk | ATE rmse cm | ATE max cm | RPE t cm | RPE r deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `corridor_60s` |  61.5s | 1022 | 1021 | 1021 |  1 |  2 |  83.43 | 137.92 | 30.55 | 25.43 |
| `lab_loop_30s` |  30.1s |  599 |  598 |  598 |  1 |  0 |  60.22 |  77.85 | 27.57 | 33.38 |
| `lab_static_10s` |  11.6s |  199 |  198 |  198 |  0 |  0 |   3.14 |  16.64 |  3.51 | 63.06 |
| `lab_straight_20s` |  22.1s |  399 |  398 |  398 |  1 |  0 |  12.32 |  40.38 | 10.97 | 42.05 |
| `quick_motion_15s` |  15.1s |  299 |  298 |  298 |  0 |  0 |  20.92 | 114.17 | 104.87 | 38.83 |

---

## How to regenerate

```bash
.venv/bin/python tools/baseline_report.py sessions/gold \
    > docs/GOLD_BASELINE.md
```

## How to interpret

- **ATE rmse**: average position error between VIO trajectory and SLAM (loop-corrected) trajectory, after SE(3) alignment.
- **ATE max**: worst-case offset (usually right before a loop closure).
- **RPE t / r**: per-second drift in metres and degrees.
- **Lp** / **Trk**: number of loop closures + tracking events detected.

When `skyslam` is implemented, re-run on the same sessions and expect ATE numbers within ~20% of these (or better).
