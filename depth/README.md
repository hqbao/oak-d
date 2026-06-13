# `depth/` — the stereo-depth project (SOURCE-OF-TRUTH for the SGM math)

The **second** of the five split projects (`imu_camera`, `depth`, `vio`, `slam`,
`ui`), built by replicating the **proven `imu_camera` template**. `depth` owns the
from-scratch SGM dense-stereo matcher + the two depth steps
(`compute_depth` → `publish_depth`).

> **`depth/` is the canonical home of the stereo math.** The capture project
> (`imu_camera`) vendors a **byte-identical copy** because depth runs **INLINE**
> on the capture process's `imu_cam` thread in the live topology today — so the
> launcher never spawns a depth process. A `diff -r` gate keeps the two copies in
> lock-step, and this tree is where the stereo math is edited and where a future
> "depth as its own process" promotion would graduate from.

`depth.main` is the **standalone harness** that proves the source tree already
runs as its own independent project: it subscribes to raw `cam.sync` over IPC,
computes metric depth with the SGM matcher, and publishes `frame.depth` on its
own endpoint.

```
imu_camera.main ──(oak.capture: cam.sync raw L/R + calib.bundle)──▶ depth.main ──(oak.depth: frame.depth)──▶ consumers
  capture proc                    IPC                                depth proc            IPC
```

## Layers

| Package | Role | Source |
|---------|------|--------|
| `depth/comms/` | the **FROZEN** vendored comms contract | copied **bit-identically** from `imu_camera/comms` |
| `sky/depth/stereo.py` | the SGM matcher + rectifiers (shared, **one canonical copy**) | the top-level `sky.depth` library; imported by both `depth` and `imu_camera` |
| `depth/io/` | recorded-session reading (used **only** for the full stereo calibration) | re-rooted copy of `imu_camera/io` |
| `depth/modules/` | the `compute_depth` + `publish_depth` **functions** | same SGM math as `imu_camera/modules/{compute_depth,publish_depth}.py`, but flattened from `Step` classes to plain functions (matcher passed explicitly) |
| `depth/main.py` | the standalone depth process (**procedural** shell) | new (runs the depth flow straight from the `cam.sync` IPC callback; `vio.main` IPC topology) |
| `depth/tests/` | the SGM-vs-chip-depth regression self-test | re-rooted copy of `imu_camera/tests/stereo_sgm_selftest.py` |
| `depth/tools/` | standalone learning/diagnostic tools (e.g. the SGM cost-volume explorer) | new; offline, opt-in, never on the data path |

### `depth/comms/` — byte-identical, do not hand-edit

`depth/comms` is **copied bit-identically** from `imu_camera/comms`. A gate runs
`diff -r --exclude=__pycache__ depth/comms imu_camera/comms` and it must be empty.
All its internal imports are RELATIVE, so the copy works as `depth.comms`
unchanged. **Never hand-edit it** — change `imu_camera/comms` and re-vendor.

Public API depth uses: `LocalPubSub`, `IPCPubSub(role="server"|"client")`,
`IPCPublisher`, `IPCSubscriber`, `RingRegistry`, `topics`,
`messages.{DepthFrame,CamSync,END}`, `converters`, and `wire.WireCalibBundle`.
depth's process shell is **plain procedural** (straight-line functions, no
reactive `Module` / `Step` graph — see `depth.main` below); `Module` / `Step`
still ship in `comms` for the other projects + the byte-parity diff.

### `sky.depth.stereo` — the shared canonical SGM math

The SGM matcher + rectifiers live in the top-level shared library at
`sky/depth/stereo.py` (numpy + numba, fully self-contained — no top-level cv2 /
project imports). It used to be vendored byte-identically in both
`depth/mathlib/stereo` and `imu_camera/mathlib/stereo` under a `diff -r`
lock-step gate; it has since been consolidated into the single shared
`sky.depth.stereo`, so that gate is **retired** — there is one copy, imported by
both projects. The standalone depth process therefore runs the **numerically
identical** matcher the capture process computes inline (proven —
`depth.tests.stereo_sgm_selftest` reports the same numbers as
`imu_camera.tests.stereo_sgm_selftest` line-for-line).

* `SGMStereoMatcher` + `SGMConfig` — semi-global block matching with built-in
  left/right rectification.
* `StereoMatcher` + `StereoConfig` — the sparse block matcher the self-test uses.
* `sgm_disparity_capture` + `SGMStereoMatcher.dense_disparity_capture` — an
  **opt-in** hook that runs the SAME SGM math but *keeps* the internal cost
  volumes (`C` = raw census-Hamming, `S` = N-path aggregated) instead of
  discarding them, returning `(disparity, C, S)` (and, on the matcher, the
  rectified pair too). It forces `downscale=1` and is used ONLY by the cost-volume
  explorer tool; `sgm_disparity` / `dense_depth` are bit-for-bit unchanged
  (verified: the captured disparity is byte-identical to the production path).

#### Density-preserving disparity denoise (live preset only)

The raw SGM disparity carries salt-pepper mismatches + isolated "flying" blobs
that survive the L/R / uniqueness gates and make the 3D map look exploded.
`SGMConfig` exposes two **post-filters** that clean the disparity map *after* the
WTA/uniqueness/LR gates — so they never reject more matches (keypoint depth
density is preserved; the rejection thresholds — `uniqueness` / `lr_max_diff` /
census — are left untouched, which would otherwise starve PnP):

* `median_disp` — `cv2.medianBlur` aperture on the disparity (e.g. `3` for 3×3;
  `0` = off). Kills salt-pepper without shifting edges; a median over a
  mostly-valid window stays valid, so a hole only opens where the neighbourhood
  was already mostly invalid.
* `speckle_window` / `speckle_range` + `speckle_cv2` — small-blob removal. With
  `speckle_cv2=True` it uses `cv2.filterSpeckles` (fast C, quantised int16 grid
  used only to GROUP blobs — survivors keep their float sub-pixel disparity);
  otherwise it falls back to the numba `_speckle_filter` flood fill.

Both run at the **computed** (post-downscale) resolution, where the map is small,
so the measured per-frame cost is a fraction of a millisecond. They are **OFF by
default** (`SGMConfig()` is byte-identical to before) and **ON only in
`SGMConfig.live()`** (`median_disp=3`, `speckle_window=20`, `speckle_cv2=True`),
i.e. the live / replay-preview depth — the path that feeds the live 3D map.
Gate: `python -m imu_camera.tests.sgm_denoise_bench` (latency increase bounded,
keypoint density not dropped, speckle proxy ≥30% lower).

### Why `depth/io/` (the calibration the wire bundle doesn't carry)

The matcher's rectifiers (`RightRectifier` / `LeftRectifier`) need the **full
per-camera stereo calibration** — `K_left` / `K_right`, the per-camera distortion,
and the `T_left_right` rigid transform. That calibration is **NOT** on the wire
`calib.bundle` (`WireCalibBundle` broadcasts only the rectified-left intrinsic +
the IMU extrinsics — everything VIO/SLAM need, since they never recompute depth).

So — exactly as the capture project builds its matcher from `reader.calib`
(replay) / `cal.calib` (live) — `depth.main` builds the matcher from the recorded
session's `calib.json` (`--session`). The raw stereo frames themselves still
arrive **over IPC** on `cam.sync`; the session is read **only** for the
calibration. The wire bundle is used as the readiness barrier + frame sizing, and
is re-broadcast on depth's endpoint so a `frame.depth` consumer that connects there
boots with the bundle cached.

## `depth.main` — the standalone depth process

The shell is **procedural**: there is no reactive module graph. The raw stereo
arrives over IPC on the `IPCSubscriber` recv thread (single-threaded, strictly
FIFO), and the subscribed callback runs the depth flow straight through —
`compute_depth(matcher, msg) → publish_depth(bus, frame)` — one published
`frame.depth` per consumed `cam.sync`, in order. The compute runs **on the recv
thread on purpose**: it back-pressures the socket (the next `recv_bytes` waits for
this frame's depth to publish), so a slow SGM never drops a frame.

1. Open a **calib client** on the capture endpoint; block until the retained
   `calib.bundle` arrives (readiness barrier + frame size).
2. Build the `SGMStereoMatcher` from the session's full `StereoCalib`.
3. Attach capture's rings (consumer-side) so the subscriber bridge can read
   `cam.sync`'s raw left/right out of shared memory; create depth's **own** rings
   for the `frame.depth` output.
4. Open depth's **output** `IPCPubSub` server (`blocking=False`) + an
   `IPCPublisher` for `frame.depth`; re-broadcast the retained `calib.bundle`.
5. Subscribe a plain callback on the `LocalPubSub` for `cam.sync`: it runs
   `compute_depth(matcher, msg)` then `publish_depth(bus, frame)` — the **same two
   stages** the capture project runs inline (as `ComputeDepthStep` /
   `PublishDepthStep`) in `imu_camera.modules.pipeline.ImuCamModule`, here as plain
   functions with the matcher passed explicitly. A sibling branch forwards `END`
   from `cam.sync` onto `frame.depth`.
6. Open the **input** `IPCPubSub` client + `IPCSubscriber` bridge for `cam.sync`.
7. Run until capture sends `END` on `cam.sync`, the `--max-frames` cap is hit, or
   SIGTERM / Ctrl-C; then stop the input bridge (after which every delivered frame
   is already published — the compute ran inline, so there is no worker inbox to
   drain) → flush the publisher → close server → unlink rings (mirrors the
   `imu_camera.main` / `vio.main` shutdown lifecycle, with `os._exit` so no
   lingering thread holds the process open).

The **process-level gate** is `depth.tests.proc_depth_smoke_selftest` (spawns
`imu_camera.main` → `depth.main` over IPC, asserts both rc=0, the 1:1 FIFO
contract via gap-free observed `frame.depth` seqs + the producer count, the
`END` forwarding, and clean shutdown).

CLI: `--capture-endpoint` (default `oak.capture`), `--endpoint` (default
`oak.depth`), `--session`, `--max-frames`, `--depth-fast`, `--calib-timeout`.

## `depth.tools.sgm_cost_explorer` — the SGM cost-volume explorer (learning tool)

A standalone, offline tool that makes the dense matcher's internals visible and
explains **why textureless surfaces give noisy depth**. It loads ONE frame from a
recorded gold session, re-rectifies the recorded raw right frame via
`RightRectifier` + the session calibration, runs the SGM with the opt-in
volume-capture hook, and lets you inspect a single pixel's matching-cost curves:

* `C(d)` — the **raw** census-Hamming cost over disparity (the evidence *before*
  the smoothness prior),
* `S(d)` — the **N-path SGM-aggregated** cost (the evidence *after* it),

with the winner-take-all minimum, the sub-pixel parabola offset, and the
uniqueness second-best band marked. A **textured** corner has a sharp single
`C(d)` valley (one clear match → reliable depth); a **textureless** flat region
has a flat / multi-valley `C(d)` (ambiguous → wrong disparity → noise), and only
`S(d)` after global aggregation develops a clearer minimum — the whole point of
SGM, made visible.

It is purely a consumer of the depth math; it never touches the data path or the
production depth output, and the capture hook leaves `sgm_disparity` unchanged.

```bash
# interactive (needs a display): click the left image / depth map -> plot C,S
.venv/bin/python -m depth.tools.sgm_cost_explorer --session sessions/gold/corridor_60s

# headless: write the textured-vs-textureless 2x2 curve PNG (numpy -> cv2, no GUI)
.venv/bin/python -m depth.tools.sgm_cost_explorer \
    --session sessions/gold/corridor_60s --frame 40 --render /tmp/sgm_cost.png
```

## `depth.tools.epipolar_explorer` — the stereo-rectification epipolar explorer (learning tool)

A standalone, offline tool that answers **"is my rectification correct?"**. SGM only
works if a 3D point lands on the **same row** in both images; raw cameras don't satisfy
this (distortion + inter-camera rotation push matches onto different rows). It loads ONE
gold frame, rectifies the recorded raw right via `RightRectifier` + the session
calibration, and draws a **2-row figure**: top = BEFORE (raw rows drift), bottom = AFTER
(rows snap onto the same scanline). The same ~13 horizontal scanlines are drawn across
both left|right panels; a handful of strong Shi-Tomasi corners detected in the left are
located in the right by a same-row-band block search, and each corner's **vertical
row-mismatch** is annotated (GREEN = on-row, RED = off-row) with the per-row median
reported — collapsing from raw → rectified.

**Honest about the data:** a gold session stores the chip's *already-rectified* LEFT
(`rectifiedLeft`) + a *raw* RIGHT (`syncedRight`), so the only genuinely raw image is the
right one. The BEFORE row is therefore labelled `chip-rectified LEFT | RAW right` (the
row-mismatch lives in the right), and the AFTER right is `RightRectifier.rectify(raw_right)`;
the left is kept as the common rectified grid (not re-warped through `LeftRectifier`,
which expects a raw left), exactly as the replay depth path does
(`from_calib(..., rectify_left=False)`). It is purely a consumer of the rectifier math —
it never touches the data path, comms, or oracle.

```bash
# headless: write the before/after scanline PNG (numpy -> cv2, no GUI)
.venv/bin/python -m depth.tools.epipolar_explorer \
    --session sessions/gold/lab_loop_30s --frame 40 --render /tmp/epipolar.png
```

## Run

```bash
# the SGM-vs-chip-depth regression self-test (no device; offline gold sessions)
.venv/bin/python -m depth.tests.stereo_sgm_selftest

# the epipolar-explorer self-test: asserts the render is non-blank AND rectification
# reduces the corner row-mismatch on gold frames (offline; no device)
.venv/bin/python -m depth.tests.epipolar_explorer_selftest

# depth-as-a-process pair (replay; no device needed). Capture publishes cam.sync
# on its endpoint; depth consumes it and publishes frame.depth on its own.
.venv/bin/python -m imu_camera.main --session sessions/gold/lab_loop_30s \
    --endpoint oak.capture
.venv/bin/python -m depth.main --capture-endpoint oak.capture \
    --endpoint oak.depth --session sessions/gold/lab_loop_30s --max-frames 20
```

## Gates (run from the repo root)

```bash
diff -r --exclude=__pycache__ depth/comms imu_camera/comms          # must be empty
.venv/bin/python -c "import depth.main, depth.modules.compute_depth"  # imports OK
.venv/bin/python -m depth.tests.stereo_sgm_selftest                  # byte-parity vs ours stereo
```
