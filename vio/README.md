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
| `vio/mathlib/` | the math VIO owns (frontend / odometry / backend / engine / imu) | `ours/lib/{frontend,odometry,backend,engine,imu}` |
| `vio/modules/` | the odometry + backend reactive modules | `ours/flows/{odometry,backend}` |
| `vio/main.py` | the VIO process | `ours/proc/vio.py` |
| `vio/tests/` | regression self-tests | `ours/tools/{klt,vio_ba}_selftest.py` |

### `vio/comms/` — byte-identical, do not hand-edit

`vio/comms` is **copied bit-identically** from `imu_camera/comms`. A CI gate runs
`diff -r vio/comms imu_camera/comms` and it must be empty (build caches —
`__pycache__`, `*.pyc`, `*.nbc/.nbi` — are git-ignored and excluded). All its
internal imports are RELATIVE, so the copy works as `vio.comms` unchanged. **Never
hand-edit it** — change `imu_camera/comms` and re-vendor.

Public API the VIO process uses: `LocalPubSub`, `IPCPubSub(role="server"|"client")`,
`IPCPublisher`, `IPCSubscriber`, `Module`, `Step`, `RingRegistry`, `topics`,
`encode`/`decode`, and `wire.WireCalibBundle`.

### `vio/mathlib/` — the math VIO owns + the architecture rule

The math sub-packages (`frontend`, `odometry`, `backend`, `engine`, `imu`) are the
verbatim port. **`imu` is vendored too** because the odometry / backend / pnp math
(and `vio_ba_selftest`) depend on `imu.imu` (SO(3) helpers + IMU preintegration);
it is numpy-only and self-contained, mirroring how `imu_camera` vendors `imu`
under its own `mathlib`.

**ARCHITECTURE RULE.** The math-coupled config builders and the JIT warmup live in
`vio/mathlib/`, **not** in the generic, bit-identical `vio/comms/`:

- `vio/mathlib/resolution_build.py` — `frontend_config(res, *, numba)`,
  `odometry_config(res, **guards)`, `ba_huber_px(res)` (ported verbatim from the
  pre-split `ResolutionProfile.frontend` / `.odometry` / `.ba_huber_px`). They
  import VIO's own math; the profile in `vio.comms.lib.config.resolution` stays
  data-only and headless.
- `vio/mathlib/warmup.py` — `warmup_klt(klt_cfg=None)` warms **only** the KLT
  numba kernel. VIO consumes `frame.depth` from capture, so it does **not** run
  SGM (that is `imu_camera`, which warms its own SGM kernel).

### `vio/modules/` — the reactive pipeline (Flow → Module, Task → Step)

`OdometryModule` joins `imucam.sample` (IMU prior) + `frame.depth` (KLT track →
RGB-D PnP → gyro fusion → pose) and publishes `pose.odom`, `keyframe`,
`frame.tracks`, `frame.inliers` (+ `pose.vo` when the live builder enables it).
`BackendModule` consumes `keyframe`, runs windowed BA behind a swappable engine,
and publishes `pose.refined`. The internal carriers (`step` / `primed` /
`tracked`) thread one frame's state through the chain; they never go on the bus.

> Naming note: the carrier dataclass is named `Step` (`vio/modules/step.py`) and
> the pub/sub base class is **also** named `Step` (renamed from `Task`, exported
> from `vio.comms`). In the six step files that need both, the base is imported as
> `from vio.comms import Step as StepBase` so the carrier keeps the plain `Step`
> name. This collision is unique to `vio` — `imu_camera` has no such carriers.

#### TIGHT live pose — `PropagateImu` (`--tight` only)

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
