# `vio/` — the visual-inertial odometry project (Phase 3 of the split)

The **third** of the five split projects (`imu_camera`, `depth`, `vio`, `slam`,
`ui`), built by replicating the **proven `imu_camera` template**. `vio` subscribes
to the capture process over IPC, runs the RGB-D visual odometry (+ gyro prior)
and the sliding-window bundle adjustment, and republishes its results on its own
IPC endpoint for SLAM / UI / tools.

```
imu_camera.main  ──(oak.capture)──▶  vio.main  ──(oak.vio)──▶  slam / ui / tools
   capture proc        IPC            VIO proc       IPC
```

It was ported **VERBATIM** from the reference oracle (`ours/`): only import roots
were re-rooted and Flow/Task/Bus classes were renamed. **No algorithm changed**,
so the numerical output is byte-identical to the oracle — proved by
`vio.tests.vio_ba_selftest` (its numbers match `ours/tools/vio_ba_selftest.py`
line-for-line).

## Layers

| Package | Role | Source it was ported from |
|---------|------|---------------------------|
| `vio/comms/` | the **FROZEN** vendored comms contract | copied **bit-identically** from `imu_camera/comms` |
| `vio/engine/` | the swappable in-process / subprocess runners for the heavy keyframe solve (the algorithm lives in shared `sky`) | `ours/lib/engine` |
| `vio/resolution_build.py`, `vio/warmup.py` | the math-coupled config builders + JIT warmup VIO owns at the project root | `ResolutionProfile.{frontend,odometry,ba_huber_px}` + `ours/lib` warmup |
| `vio/modules/` | the odometry + backend pipeline (**procedural** step functions + two plain worker threads) | `ours/flows/{odometry,backend}` |
| `vio/main.py` | the VIO process | `ours/proc/vio.py` |
| `vio/tests/` | regression self-tests | `ours/tools/{klt,vio_ba}_selftest.py` |

### `vio/comms/` — byte-identical, do not hand-edit

`vio/comms` is **copied bit-identically** from `imu_camera/comms`. A CI gate runs
`diff -r vio/comms imu_camera/comms` and it must be empty (build caches —
`__pycache__`, `*.pyc`, `*.nbc/.nbi` — are git-ignored and excluded). All its
internal imports are RELATIVE, so the copy works as `vio.comms` unchanged. **Never
hand-edit it** — change `imu_camera/comms` and re-vendor.

Public API the VIO process uses: `LocalPubSub`, `IPCPubSub(role="server"|"client")`,
`IPCPublisher`, `IPCSubscriber`, `RingRegistry`, `topics`, `encode`/`decode`, and
`wire.WireCalibBundle`. (The reactive `Module` / pub-sub `Step` classes are still
**defined** in the vendored comms — other processes use them — but `vio/modules/`
no longer imports them: its pipeline is plain procedural Python, see below.
`ModuleContext` is the one comms type still used, as a plain `(bus, name, state)`
state holder for the odometry worker.)

### `vio/engine/` — the swappable solve runners

After the `sky.*` consolidation the VIO algorithm itself (frontend KLT, RGB-D
odometry, windowed BA, the tight VIO window, IMU/SO(3) helpers) lives in the shared
`sky` leaf library; the misnamed grab-bag `vio/mathlib/` has been **dissolved by
concern**. What VIO still owns is the *execution* glue:

- `vio/engine/` — the swappable in-process / subprocess engines that drive the
  heavy keyframe solve. `make_ba_engine` wraps `sky.backend.windowed.WindowedBAMap`
  and `make_vi_engine` wraps `sky.vio.window.WindowedVIOMap`; `worker=True` ships
  each keyframe to a child process so the solve never holds the camera read loop's
  GIL. The engines know nothing about the bus — pure machinery called by the
  module steps.

**ARCHITECTURE RULE.** The math-coupled config builders and the JIT warmup live at
the **project root**, **not** in the generic, bit-identical `vio/comms/`:

- `vio/resolution_build.py` — `frontend_config(res, *, numba)`,
  `odometry_config(res, **guards)`, `ba_huber_px(res)` (ported verbatim from the
  pre-split `ResolutionProfile.frontend` / `.odometry` / `.ba_huber_px`). They
  import VIO's own (now `sky`) math; the profile in `vio.comms.lib.config.resolution`
  stays data-only and headless.
- `vio/warmup.py` — `warmup_klt(klt_cfg=None)` warms **only** the KLT numba
  kernel. VIO consumes `frame.depth` from capture, so it does **not** run SGM (that
  is `imu_camera`, which warms its own SGM kernel).

### `vio/modules/` — the procedural pipeline (no reactive `Module` / `Step`)

The class-heavy Step/Module reactive framework was flattened to plain procedural
Python: every step is a function with explicit args (no `ctx.state` lookups), and
each reactive module became a plain `threading.Thread` worker that owns its inbox,
coalescing, END handling, and downstream-END forward explicitly.

The files are grouped by **role in the data flow** (the package `__init__.py` carries
the full module-map): `pipeline.py` (read first — the workers that orchestrate),
`carriers.py` (the per-frame dataclass records), `frontend.py` (sparse visual VO),
`imu_prior.py` (IMU prior + gravity + tilt), `backend.py` (keyframe + windowed BA),
`publishers.py` (emit results on topics), plus `propagate_imu.py` (the `--tight` live
nav), `direct_odometry.py` (the `--direct` alternative front-end), and `loop_inbox.py`
(SLAM loop-correction feedback). Flow: `frame → frontend → imu_prior → backend →
publishers`.

`OdometryWorker` joins `imucam.sample` (IMU prior, `process_imucam`) +
`frame.depth` (KLT track → RGB-D PnP → gyro fusion → pose, `process_frame`) and
publishes `pose.odom`, `keyframe`, `frame.tracks`, `frame.inliers` (+ `pose.vo`
when the live builder enables it). It owns the **2-input multi-END join**
explicitly: one inbox carries `(topic, msg)` tuples, the loop routes each by topic
to the right step chain, and END is forwarded downstream + `done` set only once
**both** inputs have ENDed (`expected_ends == 2`) — the load-bearing concurrency
the old `Module` gave for free. `BackendWorker` consumes `keyframe`
(`process_kf`), runs windowed BA behind a swappable engine (`worker=True` runs it
in a subprocess; `tight=True` switches to the VIO map), and publishes
`pose.refined`. `OdometryModule` / `BackendModule` are kept as aliases (vio.main +
the selftests import them). The internal carriers (`Step` / `Primed` / `Tracked`, in
`carriers.py`) thread one frame's state through the chain; they never go on the bus.

The odometry worker holds a `ModuleContext` (a plain `(bus, name, state)` holder,
NOT the reactive substrate) so the per-run state the step functions thread through
(`vo` / `priors` / `imu_segs` / the live `live_nav` / `loop_inbox` …) lives in one
place — and the selftests that reach into `odom.ctx.state` keep working unchanged.

> Naming note: the carrier dataclass is named `Step` (`vio/modules/carriers.py`) — a
> real per-frame data record (`estimate_motion` → downstream), **kept as a
> dataclass**. The framework `Step` base class (the old per-step superclass) is
> gone from `vio/modules/` now that the steps are plain functions, so the old
> `Step` / `StepBase` import collision is gone too.

#### TIGHT live pose — `propagate_imu` (`--tight` only)

On `--tight` (`retain_imu=True`) the live `pose.odom` is **IMU forward-propagated**
between vision solves (Basalt-like `predictState`), so it reacts instantly to motion
and keeps moving through a covered camera / textureless wall instead of freezing. The
step (`vio/modules/propagate_imu.py`) owns a body→world nav-state `(R, p, v)` and on
every frame:

1. **Gap-free integration.** The retained per-frame IMU block is integrated forward
   under gravity (`imu.predict_state`). The previous block's last sample is prepended
   so the interval is exactly `(prev_block_last_ts, this_block_last_ts]` with **no
   dropped boundary segment** — a fast push registers at full magnitude (the naive
   per-block cut `(prev_ts, ts]` shares no sample and silently drops ~1-of-N
   inter-sample segments → only ~60 % of the displacement).
2. **Velocity-gated ZUPT.** A Zero-Velocity Update freezes translation only when
   *genuinely* at rest = accel ≈ g **and** gyro ≈ 0 **and** `|v|` small (sustained, with
   hysteresis). Accel+gyro alone cannot tell rest from a constant-velocity cruise (both
   read `|accel|≈g`, `|gyro|≈0`), so the velocity gate is what stops the old mid-push
   *pause*; at-rest drift is still held to ~0.
3. **Smooth complementary vision correction.** Every frame whose PnP solve is valid
   (`step.info["ok"]` + enough inliers), the nav-state is nudged a **bounded fraction**
   toward the fresh vision pose (`imu.complementary_correct`: position + velocity +
   attitude error-state feedback), replacing the old hard `p = p_vis` re-anchor +
   `v = displacement/dt` injection — so vision pulls the drift back *continuously* with
   no snap and no overshoot. On a failed/covered frame the correction is skipped and the
   pose pure-dead-reckons.

LOOSE path (`retain_imu=False`) is a **pure pass-through no-op** — `pose.odom` stays the
vision-only odometry pose, so the byte-parity oracle is untouched. Gates:
`vio.tests.imu_push_response_selftest` (the push-profile gate), `imu_propagate_selftest`,
`tight_live_pose_selftest`.

#### DIRECT odometry mode — `process_frame_direct` (`--direct` only)

`--direct` selects a **third** odometry mode (alongside loose-default and `--tight`):
dense **direct** RGB-D visual odometry, for the 54×42 VL53-class ToF target where the
sparse corner/KLT front-end scale-collapses (Sim3 scale 0.23–0.63) from feature
starvation. It is opt-in and **byte-identical-off**: with no flag the loose/tight
path is unchanged and the byte-parity oracle stays gap=0 (the oracle never passes
`--direct` and runs its own in-process harness, not this worker).

On `--direct` the `frame.depth` edge routes to `process_frame_direct` (not the sparse
`process_frame`), which drives `DirectOdometryEngine` (`vio/modules/direct_odometry.py`)
— the live port of the offline-proven loop in `verification/direct_vo_bench.py`. Per
frame:

1. **Dense direct frame-to-keyframe alignment.** `sky.front.direct.estimate_pose_direct`
   (the LEAF estimator, reused verbatim) aligns every gradient pixel by photometric
   Gauss-Newton against the current keyframe, reading metric scale straight from the
   accurate per-pixel ToF depth (geometric point-to-plane term OFF by default — the
   ablation showed it redundant at 54×42; available via `DirectConfig.geo_weight`).
2. **Live IMU 6-DoF seed (reused, not rebuilt).** The GN `init_T` is the keyframe→cur
   relative pose from an IMU dead-reckon nav-state propagated with the SAME
   `sky.vio.imu.predict_state` the live tight path runs, gravity-levelled once with
   `gravity_aligned_R0` (seeded from the **bundle's** `accel_align` startup reference —
   the per-frame prior is empty on the no-IMU startup frames), and pulled toward each
   accepted fix with `complementary_correct` (same gains as the tight path). The
   per-frame raw-IMU block comes from the SAME `preintegrate_prior` retention the tight
   path uses (`--direct` forces it on, independent of `retain_imu`).
3. **Divergence guard.** A frame's VO pose is rejected — replaced by the IMU
   dead-reckon, which is also what the dead-reckoner is then corrected toward (so the
   seed velocity is not poisoned) — when the estimator flags `diverged` OR the VO
   keyframe-relative step ≫ the IMU-predicted step (ratio gate with a floor). This is
   the lever that kills the fast-motion divergence.

Keyframes are emitted on a **natural** cadence (trans ≥ 0.1 m / rot ≥ 6° / overlap
drop / divergence), not the fixed `kf_every` count. The published topics are the SAME
as the other modes (`pose.odom`, `pose.vo`, `keyframe`, and empty-but-ticking
`frame.tracks` / `frame.inliers`), so the UI + SLAM + comms are untouched (no new IPC
topic). `--direct` is independent of `--tight` and is meant to pair with `--vl53l9cx`:
the live recipe is `./run.sh --vl53l9cx --direct`. Gate: `vio.tests.direct_smoke_selftest`
(live worker smoke + Sim3-scale sanity vs Basalt); launcher forwarding:
`launcher.tests.direct_forward_selftest`.

### `vio/main.py` — the VIO process

Two-client startup against the capture endpoint (a **calib client** that blocks on
the retained `calib.bundle`, then a **data client** for `imucam.sample` +
`frame.depth`), builds the local odometry/backend graph with the **live** config
(`level_tilt=True`, `OdometryConfig(gyro_fuse=use_gyro)`, `publish_vo=True`), and
mirrors its outputs onto its own `IPCPubSub` server with an `IPCPublisher`,
re-broadcasting the retained `calib.bundle` as a readiness barrier. The
worker-engine subprocess boundary (`--worker`) stays on stdlib pickle
(`multiprocessing.Queue`, same-project classes) — it is **not** routed through the
cross-process codec. Same SIGTERM / drain / `os._exit` lifecycle as the template.

## Run

```bash
# capture (replay) serves oak.capture; vio subscribes + serves oak.vio
python -m imu_camera.main --session sessions/gold/lab_loop_30s &
python -m vio.main --capture-endpoint oak.capture --endpoint oak.vio
```

## Verify

```bash
cd /Users/bao/skydev/oak-d

# 1. comms byte-identical (build caches excluded; they are git-ignored)
diff -r -x '__pycache__' -x '*.pyc' -x '*.nbc' -x '*.nbi' \
     vio/comms imu_camera/comms && echo "COMMS BYTE-IDENTICAL"

# 2. import smoke
.venv/bin/python -c "import vio.main, vio.modules.pipeline; print('VIO IMPORT OK')"

# 3. math byte-parity vs the oracle + KLT correctness
.venv/bin/python -m vio.tests.vio_ba_selftest      # PASS (== ours numbers)
.venv/bin/python -m vio.tests.odometry_selftest    # PASS

# 4. PAIR smoke: capture (replay) + vio over IPC on a gold session
#    expect ~60 pose.odom (dense), ~12 keyframe, ~12 pose.refined, clean exit.
```
