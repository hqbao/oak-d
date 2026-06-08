# Resolution tuning (run lighter at lower frame size)

The from-scratch VIO/SLAM pipeline was tuned at the **640×400** baseline. The
cheapest way to save CPU is to capture at a **lower resolution** (cost scales
with the pixel count), but lowering the resolution shrinks every *pixel-unit*
threshold in the pipeline — corner spacing, the KLT window, the PnP
reprojection gate, the stereo disparity range, the ORB budget — so the baseline
numbers become too coarse and feature tracking / depth / pose quality degrade.

`comms/lib/config/resolution.py` (`ResolutionProfile`, the data-only profile) is
the single place that scales those parameters from the baseline to the live
`(width, height)`; the math-coupled builders that turn it into the live
frontend/odometry/BA/loop configs live in each project's
`mathlib/resolution_build.py` (`vio/mathlib/resolution_build.py`,
`slam/mathlib/resolution_build.py`, `imu_camera/mathlib/resolution_build.py`). The
capture process builds one from `--width/--height` and auto-scales; any knob can be
overridden at runtime for the co-tuning workflow below.

## How to run lighter

```bash
./run.sh --width 320 --height 200                                 # half res (1/4 the pixels)
./run.sh --session sessions/gold/<name> --width 320 --height 200  # half res, replay
./run.sh --width 480 --height 300                                 # 0.75x
```

The startup log prints the active profile, e.g.:

```
[ours] resolution profile: 320x200 (s=0.50): corners=200 min_dist=6.0px
           blk=5px klt=11px/2lvl reproj=1.0px ndisp=48 orb=400
```

At a low ToF resolution the `blk` field shows `+bucket`, e.g. the VL53L9CX sim:

```
vio: frontend profile -> 54x42 (s=0.08): corners=80 min_dist=4.0px
           blk=3px+bucket klt=7px/1lvl reproj=1.0px ndisp=32 orb=200
```

> Keep the aspect ratio at **16:10** (640:400) when picking a size — the scale
> factor is `s = width / 640` and the intrinsics depthai returns are rescaled to
> the requested resolution, so a proportional downscale keeps `fx`, `cx`, the
> disparity↔depth mapping and the field of view consistent.

## The runtime knobs

All seven default to **auto** (scaled from the 640×400 baseline). Pass a flag to
override just that one; the rest stay auto. Metric parameters (depths in metres,
`max_translation_speed` in m/s, the gyro-fusion gates in degrees) are
resolution-independent and are **not** scaled.

| Flag | Config field | Unit | Auto scale (`s = width/640`) | What it does / why it matters at low res |
|---|---|---|---|---|
| `--max-corners` | `FrontendConfig.max_corners` | count | `max(80, round(400·s))` | Shi-Tomasi corner budget. Fewer pixels hold fewer distinct corners; too high wastes time on weak corners, too low starves PnP. |
| `--min-distance` | `FrontendConfig.min_distance` | px | `max(4, 12·s)` | Min spacing between corners. Must shrink with the image or all corners collapse into a few clusters. |
| `--klt-win` | `FrontendConfig.win_size` | px (odd) | `odd(21·s)`, ≥7 | KLT tracking window. At low res a 21 px window spans too much of the frame; smaller keeps the local-flow assumption valid. |
| `--klt-levels` | `FrontendConfig.max_level` | count | 3 at 640, −1 per halving | KLT pyramid depth. A tiny image needs a shallow pyramid; extra levels just blur it to mush. |
| `--reproj-px` | `OdometryConfig.ransac_reproj_px` | px | `max(1, 2·s)` | PnP RANSAC inlier gate. A fixed 2 px is a bigger fraction of a small image; scaling keeps the inlier/outlier split sane. |
| `--num-disparities` | `SGMConfig.num_disparities` | px | `max(32, even(96·s))` | Stereo disparity search range. Disparity is a pixel distance → halves at half width. Keeps the near-depth bound `fx·B/ndisp` roughly constant. |
| `--orb-features` | `LoopConfig.orb_features` | count | `max(200, round(800·s))` | ORB budget for loop closure (`ours-slam`). Fewer pixels → fewer reliable ORB keypoints. |

Auto-scaled but **not** individually flagged (derived from `s`/`width` inside the
profile): `min_inliers_for_translation` (`max(6, round(12·s))`), the loop
epipolar/PnP thresholds (`max(1, 2·s)`), the BA Huber scale (`max(1, 2·s)`), and
the two **low-resolution corner-detection levers** below. Add a flag if
co-tuning shows one of these needs it.

| Profile field | Config field | Auto rule (`width`) | What it does / why it matters at low res |
|---|---|---|---|
| `block_size` | `FrontendConfig.block_size` | 7 at `s≥1`; 5 for `160<width<640`; **3 for `width≤160`** | Shi-Tomasi structure-tensor window (a *pixel* footprint). A 7 px window over a ~54 px-wide ToF frame over-smooths and **halves** the corner count; 3 px roughly doubles it. Pinned to 7 at the 640 baseline so that detect path is **byte-identical**. |
| `bucketed` | `FrontendConfig.bucketed` | `True` only when `width≤160`, else `False` | Per-cell grid detection (≈6×5 cells, ≤2 corners/cell, per-cell relative quality threshold). Forces **even spatial coverage** so corners don't cluster — clustered corners give degenerate PnP geometry, which is what makes the tracker flicker `LOST↔OK`. Off at 640 → the original global detect path (byte-identical). |

> Both levers are **resolution-gated**: at 640×400 they are `7 / False`, so the
> corner detector runs its original global path and the offline byte-parity
> oracle (`verification/oracle_replay_selftest.py`) stays `gap=0`.

## Co-tuning workflow

Start from auto, watch the trajectory in the viewer against a slow hand path,
and adjust one knob at a time. Symptoms → first knob to try:

- **Tracking drops / pose freezes often** → raise `--max-corners`, lower
  `--min-distance`, raise `--klt-win`.
- **Path wobbles / jitters** → raise `--reproj-px` slightly, or lower
  `--max-corners` (dropping weak corners).
- **Depth looks sparse / holes** → raise `--num-disparities` (covers nearer
  objects), check the near range is still reachable.
- **Loops never close (`ours-slam`)** → raise `--orb-features`.
- **Still too slow** → drop resolution further, or lower `--max-corners` /
  `--klt-win` / `--klt-levels`.

## Verified configs per resolution

Auto-scaled values are the **starting point**, not a measured optimum. Record
co-tested values here as we find them.

| Resolution | s | Status | `--max-corners` | `--min-distance` | `--klt-win` | `--klt-levels` | `--reproj-px` | `--num-disparities` | `--orb-features` | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 640×400 | 1.00 | baseline (tuned) | 400 | 12 | 21 | 3 | 2.0 | 96 | 800 | reference; gold ATE corridor 0.61% |
| 480×300 | 0.75 | auto (untested) | 300 | 9.0 | 15 | 3 | 1.5 | 72 | 600 | — |
| 320×200 | 0.50 | auto (untested) | 200 | 6.0 | 11 | 2 | 1.0 | 48 | 400 | — |
| 160×100 | 0.25 | auto (untested) | 100 | 4.0 | 7 | 1 | 1.0 | 32 | 200 | floors hit; recommended low-res **floor** for VIO |

Update the **Status** column to `co-tested` and fill the **Notes** with the
measured ATE / observations once we run each on device.

## How small can it go? (measured)

Offline probe on the `corridor_60s` gold stereo (downscale the real frames,
rescale the calib, run the live SGM preset + the scaled Shi-Tomasi corners) plus
a device run of the live preview at 54×42:

| Resolution | s | Depth coverage | Corners | Verdict |
|---|---|---|---|---|
| 640×400 | 1.00 | ~45% | 146 | reference |
| 320×200 | 0.50 | ~54% | 104 | healthy |
| 160×100 | 0.25 | ~58% | 75 | healthy — **recommended floor** |
| 96×60 | 0.15 | ~50% | 38 | edge; usable, fewer tracks |
| 54×42 (`block_size=3` + bucketed) | ~0.08 | ~30% | **~50** | runs (device-confirmed) — with the low-res corner levers the tracker is now **consistent** (see below); still less accurate than higher res, but no longer flickers |

Notes:

- The SGM `live()` preset uses `downscale=2`, which internally **halves**
  `num_disparities`, so stereo stays geometrically valid (internal ndisp <
  compute width) even at 54×42 — the tiny sizes get *less accurate*, not broken.
- **54×42 corner-detection levers (`block_size=3` + `bucketed`).** A 7 px
  Shi-Tomasi window over a 54 px-wide frame over-smooths, and the strongest
  corners cluster in one region — so PnP saw ~7 clustered tracks and alternated
  0↔9 inliers (degenerate geometry → constant `LOST↔OK` flicker). Measured on the
  `lab_loop_30s` gold ToF replay (`--vl53l9cx`, 598 frames, device-free), the
  block-size + bucketed levers change this decisively:

  | Detector at 54×42 | mean tracks | grid coverage | mean PnP inliers (median) | LOST frames | OK↔LOST flips |
  |---|---|---|---|---|---|
  | before (`blk7`, `min_dist=12` default) | 7.2 | 24% | 0.1 (0) | **590 / 598 (98.7%)** | 12 |
  | scaled spacing only (`blk7`, `min_dist=4`) | 33.7 | 61% | 25.2 (27) | 9 / 598 (1.5%) | 1 |
  | + `block_size=3` (lever 1) | 42.8 | 69% | 31.6 (33) | 9 / 598 (1.5%) | 1 |
  | + `block_size=3` **+ bucketed** (both levers) | **49.6** | **85%** | **35.7 (37)** | **9 / 598 (1.5%)** | **1** |

  Reproduced on `corridor_60s` (before 6.2 tracks / 97.7% LOST → after 50.7
  tracks / 85% coverage / 2.5% LOST). The bucketed lever's signature is the jump
  in spatial coverage (61% → 85%), which is what removes the PnP degeneracy.
- **160×100 is the practical floor** for VIO (≈58% coverage, ≈75 corners): much
  lighter than 640×400 while still tracking well. **96×60** is the edge if you
  need maximum lightness.
- Preview any size live first (matches the VIO depth path exactly): run
  `./run.sh --width 160 --height 100` and open the UI's **Visualize → Camera +
  Depth + IMU (triplet)** window (the capture resolution flows through to it).
