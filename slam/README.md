# `slam/` — the loop-closure SLAM project (Phase 4 of the split)

The **fourth** of the five split projects (`imu_camera`, `depth`, `vio`, `slam`,
`ui`), built by replicating the **proven `imu_camera` / `vio` template**. `slam`
subscribes to the VIO process over IPC, runs ORB loop closure + SE(3) pose-graph
optimisation over the keyframe stream, and republishes its results on its own IPC
endpoint for the UI / tools.

```
imu_camera.main ──(oak.capture)──▶ vio.main ──(oak.vio)──▶ slam.main ──(oak.slam)──▶ ui / tools
   capture proc        IPC          VIO proc      IPC         SLAM proc      IPC
```

It owns the **SLAM map** (ORB feature index + pose graph). The VIO map (windowed
BA) lives in the VIO process; the two maps are **independent by design**. The
correction stream is **one-way**: SLAM publishes `loop.correction` for the UI but
**never closes the loop back into VIO** — behaviour unchanged from the pre-split
`ours.proc.slam`.

It was ported **VERBATIM** from the reference oracle (`ours/`): only import roots
were re-rooted and Flow/Task/Bus classes were renamed (Flow → Module, Task → Step,
Bus → LocalPubSub, Ipc*Bus/Flow → IPCPubSub/IPCPublisher/IPCSubscriber). **No
algorithm changed**, so the numerical output is byte-identical to the oracle —
proved by `slam.tests.loop_closure_selftest` (its numbers match
`ours/tools/posegraph_selftest.py` line-for-line) and by the 3-process smoke
matching the oracle loop count.

## Layers

| Package | Role | Source it was ported from |
|---------|------|---------------------------|
| `slam/comms/` | the **FROZEN** vendored comms contract | copied **bit-identically** from `imu_camera/comms` |
| `slam/engine/` | the swappable in-process / subprocess runner SLAM owns; the loop-closure math now comes from `sky.slam` | `ours/lib/engine` (loop algorithms consolidated into `sky.slam`) |
| `slam/resolution_build.py` | the math-coupled config builder SLAM owns at the project root | `ResolutionProfile.loop` |
| `slam/modules/` | the loop-closure pipeline (**procedural** functions + a plain worker thread) | `ours/flows/slam` |
| `slam/main.py` | the SLAM process | `ours/proc/slam.py` |
| `slam/tests/` | regression self-tests | `ours/tools/posegraph_selftest.py` |

### `slam/comms/` — byte-identical, do not hand-edit

`slam/comms` is **copied bit-identically** from `imu_camera/comms`. A gate runs
`diff -r slam/comms imu_camera/comms` and it must be empty (build caches —
`__pycache__`, `*.pyc`, `*.nbc/.nbi` — are git-ignored and excluded). All its
internal imports are RELATIVE, so the copy works as `slam.comms` unchanged. **Never
hand-edit it** — change `imu_camera/comms` and re-vendor.

Public API the SLAM process uses: `LocalPubSub`, `IPCPubSub(role="server"|"client")`,
`IPCPublisher`, `IPCSubscriber`, `RingRegistry`, `topics`, and
`wire.WireCalibBundle`. (The reactive `Module` / `Step` classes are still **defined**
in the vendored comms — other processes use them — but `slam/` no longer imports them:
its pipeline is plain procedural Python, see below.) The `keyframe`, `loop.correction`,
`slam.map`, `slam.loop`
topics and the `Keyframe` / `LoopCorrection` / `SlamOverlay` / `LoopMatch` messages
already live in the shared comms contract. `slam.loop` (LIVE-only, additive) is the
per-loop-CANDIDATE match funnel for the UI's "Loop Closure" window — the matched ORB
pixel pairs + per-match verification stage (appearance/epipolar/PnP) + funnel counts +
rotation-gate verdict, published for EVERY verified candidate (confirmed OR rejected).
It carries NO keyframe images (SLAM keeps only descriptors); the UI joins it by seq to
the `keyframe` grays it buffers.

### `slam/engine/` — the runner SLAM owns; the loop math lives in `sky.slam`

The loop-closure algorithms have been **consolidated into the shared `sky/`
library**: SLAM no longer carries a private `loop/` package nor any forced-vendor
copies — it **imports** them from `sky.*`:

- `sky.slam.orb` — from-scratch oriented-FAST + rotated-BRIEF + Hamming matcher +
  fundamental-matrix RANSAC (**no cv2**).
- `sky.slam.loopclosure` — appearance gate + geometric verification → metric
  `T_cur_old` (its `LoopConfig` is what `slam/resolution_build.py` tunes).
- `sky.slam.posegraph` — SE(3) Gauss-Newton/LM PGO with a Huber kernel on loop
  edges.
- `sky.slam.slam` — `SlamMap` / `SlamConfig`, the persistent-keyframe
  orchestrator (`slam/main.py` and `slam/modules/pipeline.py` import these).

The transitive math those use is also shared, not vendored: the loop verifier's
PnP is `sky.front.pnp` (`solve_pnp_ransac`), and the SE(3)/SO(3) Lie helpers that
drive the PGO are `sky.math` (`se3_exp`, `skew`, `so3_exp`) — the old
forced-vendor copies (the old `slam/mathlib/odometry/pnp.py`,
`slam/mathlib/imu/imu.py`, `slam/mathlib/backend/bundle.py`) and the
`slam/mathlib/loop/` package are **gone**, and the misnamed `slam/mathlib/`
grab-bag itself has been **dissolved by concern**. `sky.*` is a `numpy`-only leaf
library (no process / `comms` / `io` imports), so importing it keeps SLAM portable.

What SLAM still owns (now at the project root, not under a `mathlib`):

- `slam/engine/` — SLAM's **own** copy of the swappable in-process / subprocess
  runners (`worker=False` is byte-identical offline, `worker=True` runs the solve
  in a child process so it never holds the read loop's GIL). The worker
  lazy-imports the heavy map from `sky.slam.slam`.

The **windowed BA** is *not* part of SLAM: SLAM only ever calls
`make_slam_engine` (loop closure), so the engine carries **only** the
loop-closure path (`make_slam_engine` / `slam_step` / `_slam_worker_main`). The
never-fired `make_ba_engine` / `_ba_worker_main` path the byte-copied engine used
to carry (a lazy import of the windowed-BA backend that SLAM never resolved) has
been removed — there is no BA backend under `slam/engine/`.

**ARCHITECTURE RULE.** The math-coupled config builder lives at the **project
root**, **not** in the generic, bit-identical `slam/comms/`:

- `slam/resolution_build.py` — `loop_config(res)` (ported verbatim from
  the pre-split `ResolutionProfile.loop`), which imports the shared
  `sky.slam.loopclosure.LoopConfig`. The profile in
  `slam.comms.lib.config.resolution` stays data-only and headless.

> **No `warmup.py`.** Unlike `vio` (which warms its KLT numba kernel), SLAM has
> **no numba JIT** to pre-compile — its ORB frontend is pure NumPy — so no warmup
> module exists.

### `slam/modules/` — the procedural pipeline (no reactive `Module` / `Step`)

The pipeline is **plain procedural Python**. The per-keyframe work is one function,
`process_keyframe(engine, bus, kf, *, publish_map)`, which calls the single-purpose
step functions in order — the data flow reads as straight-line code, not a framework
step chain:

- `slam_submit(engine, kf)` — `engine.submit` + `engine.poll`; returns a
  `LoopCorrection` **on a confirmed loop**, else `None`.
- `publish_correction(bus, msg)` — emit it on `loop.correction` (called only when
  `slam_submit` returned non-`None`; the None-guard is now explicit at the call site).
- (LIVE only) `publish_loops(engine, bus)` — poll the per-candidate match funnel →
  `slam.loop`; `publish_slam_map(engine, bus)` — poll the cheap overlay → `slam.map`.
  Both poll **independent** engine channels **after** the submit above.

`SlamWorker` (a plain `threading.Thread`, exported also under the legacy name
`SlamModule`) drains a keyframe inbox and runs `process_keyframe`. It is the
procedural replacement for the old reactive `Module` — it owns the inbox, the
optional coalescing, END handling, and the downstream-END forwarding as **explicit
code**, not framework hooks.

Two key behaviours are preserved byte-for-byte from the oracle:

- **`publish_map` flag** (LIVE-only, defaults `False`). When on, the worker emits a
  continuous `slam.map` overlay so the UI draws keyframe dots **every** keyframe
  instead of only after a loop closes, AND captures the loop-closure match funnel
  (`make_slam_engine(capture_loops=True)`) to publish a `LoopMatch` on `slam.loop`
  for every verified candidate (for the UI's "Loop Closure" window). The offline
  path (flag off) is byte-identical: `process_keyframe` reduces to `slam_submit` +
  conditional `publish_correction`, the engine never captures, and the deterministic
  `loop.correction` scoring path runs the byte-frozen `verify` with no extra work.
  (The old `_RunCorrectionChain` wrapper existed only to force "submit-then-overlay"
  ordering inside the single reactive route; procedurally that order is just the
  call order in `process_keyframe`, so the wrapper is gone.)
- **`latest_only` coalescing — LOAD-BEARING, kept explicit.** SLAM consumes
  `keyframe` with `latest_only=True` on the LIVE path: the ORB + pose-graph solve
  cannot keep up under real-time load, so the inbox must **drop intermediate
  keyframes and always solve the freshest one** (a strict-FIFO inbox would back up
  without bound and the `slam.map` overlay would lag seconds behind). `SlamWorker`
  replicates this with a single-slot "latest keyframe" holder its bus feeder
  overwrites + a wake-up-token inbox the worker drains — byte-for-byte the old
  `Module._coalesce` / `Module.run`, specialised to the one keyframe topic. `END` is
  never coalesced, so clean shutdown still propagates. This is **why the confirmed-loop
  count in `proc3_smoke_selftest` is non-deterministic-but-bounded** (a variable
  number of keyframes is dropped each run). The OFFLINE / oracle callers build the
  worker with `latest_only=False` for a strict-FIFO inbox (the deterministic path
  must process every keyframe).

> **`SlamWorker` has two in-process consumers besides `slam.main`:**
> `vio/tests/closed_loop_drift_selftest.py` and `verification/loop_teleport_diag.py`
> both construct it directly (as `SlamModule`, `latest_only=False`, `worker=False`,
> `publish_map=False`) over a `LocalPubSub` bus to collect real `loop.correction`s.
> The legacy `SlamModule` name is kept as an alias of `SlamWorker` so they are
> unchanged.

### `slam/main.py` — the SLAM process

A single-client startup against the **VIO** endpoint: a **calib client** blocks on
the retained `calib.bundle` (VIO re-broadcasts it after allocating its `kf_*`
rings, so its arrival proves VIO is up, intrinsics are known, and the keyframe
rings exist), then SLAM attaches to VIO's keyframe rings and builds the local
graph with the **live** config:
`SlamConfig(loop_max_odom_rot_deg=30.0, kf_min_trans_m=0.1, kf_min_rot_deg=5.0)`,
`latest_only=True`, `publish_map=True`. It mirrors `loop.correction` + `slam.map`
onto its own `IPCPubSub` server with an `IPCPublisher`, re-broadcasting the
retained `calib.bundle` as a readiness barrier. The worker-engine subprocess
boundary (`--worker`) stays on stdlib pickle (`multiprocessing.Queue`,
same-project classes) — it is **not** routed through the cross-process codec. Same
SIGTERM / drain / `os._exit` lifecycle as the template.

## Run

```bash
# capture (replay) serves oak.capture; vio subscribes + serves oak.vio;
# slam subscribes oak.vio + serves oak.slam.
python -m imu_camera.main --session sessions/gold/lab_loop_30s &
python -m vio.main  --capture-endpoint oak.capture --endpoint oak.vio &
python -m slam.main --vio-endpoint oak.vio --endpoint oak.slam
```

## Verify

```bash
cd /Users/bao/skydev/oak-d

# 1. comms byte-identical (build caches excluded; they are git-ignored)
diff -r --exclude=__pycache__ slam/comms imu_camera/comms && echo "COMMS BYTE-IDENTICAL"

# 2. import smoke
.venv/bin/python -c "import slam.main, slam.modules.pipeline; print('SLAM IMPORT OK')"

# 3. math byte-parity vs the oracle (== ours/tools/posegraph_selftest.py numbers)
.venv/bin/python -m slam.tests.loop_closure_selftest

# 4. 3-PROC smoke: imu_camera (replay) + vio + slam over a gold loop session.
#    Asserts all 3 procs rc=0, slam.map advances (kf dots), and loop.correction
#    n_loops matches the oracle (4 on lab_loop_30s).
.venv/bin/python -m slam.tests.proc3_smoke_selftest \
    --session sessions/gold/lab_loop_30s --expect-loops 4
```
