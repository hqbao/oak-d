# Resolution tuning (run lighter at lower frame size)

The from-scratch VIO/SLAM pipeline was tuned at the **640×400** baseline. The
cheapest way to save CPU is to capture at a **lower resolution** (cost scales
with the pixel count), but lowering the resolution shrinks every *pixel-unit*
threshold in the pipeline — corner spacing, the KLT window, the PnP
reprojection gate, the stereo disparity range, the ORB budget — so the baseline
numbers become too coarse and feature tracking / depth / pose quality degrade.

`oakd/vio/resolution.py` (`ResolutionProfile`) is the single place that scales
those parameters from the baseline to the live `(width, height)`. Every
ours-\* source builds one from `--width/--height` and auto-scales; any knob can
be overridden at runtime for the co-tuning workflow below.

## How to run lighter

```bash
./run.sh --source ours --width 320 --height 200      # half res (1/4 the pixels)
./run.sh --source ours-slam --width 320 --height 200 # SLAM at half res
./run.sh --source ours --width 480 --height 300      # 0.75x
```

The startup log prints the active profile, e.g.:

```
[ours-vio] resolution profile: 320x200 (s=0.50): corners=200 min_dist=6.0px
           klt=11px/2lvl reproj=1.0px ndisp=48 orb=400
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

Auto-scaled but **not** individually flagged (derived from `s` inside the
profile): `min_inliers_for_translation` (`max(6, round(12·s))`), the loop
epipolar/PnP thresholds (`max(1, 2·s)`), and the BA Huber scale (`max(1, 2·s)`).
Add a flag if co-tuning shows one of these needs it.

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
| 160×100 | 0.25 | auto (untested) | 100 | 4.0 | 7 | 1 | 1.0 | 32 | 200 | floors hit; likely too small for SLAM |

Update the **Status** column to `co-tested` and fill the **Notes** with the
measured ATE / observations once we run each on device.
