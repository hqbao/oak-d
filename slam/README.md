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
| `slam/mathlib/` | the engine SLAM owns (`engine/` + `resolution_build`); the loop-closure math now comes from `sky.slam` | `ours/lib/engine` (loop algorithms consolidated into `sky.slam`) |
| `slam/modules/` | the loop-closure reactive module | `ours/flows/slam` |
| `slam/main.py` | the SLAM process | `ours/proc/slam.py` |
| `slam/tests/` | regression self-tests | `ours/tools/posegraph_selftest.py` |

### `slam/comms/` — byte-identical, do not hand-edit

`slam/comms` is **copied bit-identically** from `imu_camera/comms`. A gate runs
`diff -r slam/comms imu_camera/comms` and it must be empty (build caches —
`__pycache__`, `*.pyc`, `*.nbc/.nbi` — are git-ignored and excluded). All its
internal imports are RELATIVE, so the copy works as `slam.comms` unchanged. **Never
hand-edit it** — change `imu_camera/comms` and re-vendor.

Public API the SLAM process uses: `LocalPubSub`, `IPCPubSub(role="server"|"client")`,
`IPCPublisher`, `IPCSubscriber`, `Module`, `Step`, `RingRegistry`, `topics`, and
`wire.WireCalibBundle`. The `keyframe`, `loop.correction`, `slam.map`, `slam.loop`
topics and the `Keyframe` / `LoopCorrection` / `SlamOverlay` / `LoopMatch` messages
already live in the shared comms contract. `slam.loop` (LIVE-only, additive) is the
per-loop-CANDIDATE match funnel for the UI's "Loop Closure" window — the matched ORB
pixel pairs + per-match verification stage (appearance/epipolar/PnP) + funnel counts +
rotation-gate verdict, published for EVERY verified candidate (confirmed OR rejected).
It carries NO keyframe images (SLAM keeps only descriptors); the UI joins it by seq to
the `keyframe` grays it buffers.

### `slam/mathlib/` — the engine SLAM owns; the loop math lives in `sky.slam`

The loop-closure algorithms have been **consolidated into the shared `sky/`
library**: SLAM no longer carries a private `loop/` package nor any forced-vendor
copies — it **imports** them from `sky.*`:

- `sky.slam.orb` — from-scratch oriented-FAST + rotated-BRIEF + Hamming matcher +
  fundamental-matrix RANSAC (**no cv2**).
- `sky.slam.loopclosure` — appearance gate + geometric verification → metric
  `T_cur_old` (its `LoopConfig` is what `slam/mathlib/resolution_build.py` tunes).
- `sky.slam.posegraph` — SE(3) Gauss-Newton/LM PGO with a Huber kernel on loop
  edges.
- `sky.slam.slam` — `SlamMap` / `SlamConfig`, the persistent-keyframe
  orchestrator (`slam/main.py` and `slam/modules/pipeline.py` import these).

The transitive math those use is also shared, not vendored: the loop verifier's
PnP is `sky.front.pnp` (`solve_pnp_ransac`), and the SE(3)/SO(3) Lie helpers that
drive the PGO are `sky.math` (`se3_exp`, `skew`, `so3_exp`) — the old
forced-vendor copies (`slam/mathlib/odometry/pnp.py`, `slam/mathlib/imu/imu.py`,
`slam/mathlib/backend/bundle.py`) and the `slam/mathlib/loop/` package are
**gone**. `sky.*` is a `numpy`-only leaf library (no process / `comms` / `io`
imports), so importing it keeps SLAM portable.

What SLAM's `mathlib/` still owns:

- `slam/mathlib/engine/` — SLAM's **own** copy of the swappable in-process /
  subprocess runners (`worker=False` is byte-identical offline, `worker=True`
  runs the solve in a child process so it never holds the read loop's GIL). The
  worker lazy-imports the heavy map from `sky.slam.slam`.

The **windowed BA** is *not* part of SLAM: SLAM only ever calls
`make_slam_engine` (loop closure), so the engine carries **only** the
loop-closure path (`make_slam_engine` / `slam_step` / `_slam_worker_main`). The
never-fired `make_ba_engine` / `_ba_worker_main` path the byte-copied engine used
to carry (a lazy import of the windowed-BA backend that SLAM never resolved) has
been removed — there is no BA backend under `slam/mathlib/`.

**ARCHITECTURE RULE.** The math-coupled config builder lives in `slam/mathlib/`,
**not** in the generic, bit-identical `slam/comms/`:

- `slam/mathlib/resolution_build.py` — `loop_config(res)` (ported verbatim from
  the pre-split `ResolutionProfile.loop`), which imports the shared
  `sky.slam.loopclosure.LoopConfig`. The profile in
  `slam.comms.lib.config.resolution` stays data-only and headless.

> **No `warmup.py`.** Unlike `vio` (which warms its KLT numba kernel), SLAM has
> **no numba JIT** to pre-compile — its ORB frontend is pure NumPy — so no warmup
> module exists.

### `slam/modules/` — the reactive pipeline (Flow → Module, Task → Step)

`SlamModule` subscribes `keyframe` and publishes `loop.correction`. It wraps
`SlamMap` behind a swappable engine; every keyframe is submitted (the map's own
motion gate may skip redundant ones), and on a confirmed loop the pose graph is
optimised and the rewritten keyframe poses are published as a correction.

The single-purpose steps each own one responsibility: `SlamStep` (submit + poll
the engine → `LoopCorrection` on a loop), `PublishCorrection` (emit on
`loop.correction`), `PublishSlamMap` (poll the cheap overlay → `slam.map`),
`PublishLoops` (poll the per-candidate match funnel → `slam.loop`, LIVE-only).

Two key behaviours are preserved verbatim from the oracle:

- **`publish_map` flag** (LIVE-only, defaults `False`). When on, SlamModule emits
  a continuous `slam.map` overlay so the UI draws keyframe dots **every** keyframe
  instead of only after a loop closes, AND captures the loop-closure match funnel
  (`make_slam_engine(capture_loops=True)`) to publish a `LoopMatch` on `slam.loop`
  for every verified candidate (for the UI's "Loop Closure" window). The offline
  path (flag off) is byte-identical: the engine never captures, so the deterministic
  `loop.correction` scoring path runs the byte-frozen `verify` with no extra work.
- **`_RunCorrectionChain`** — `Module.on` keeps **one** step list per topic, and
  `SlamStep` returns `None` on every non-loop keyframe (which short-circuits the
  chain). So the live path wraps `[SlamStep(), PublishCorrection()]` in one step
  that always returns the keyframe, letting the outer chain continue to
  `PublishLoops` (passes the keyframe through) and `PublishSlamMap` (terminal, polls
  the overlay **after** the submit). One combined chain, correct order, zero impact
  on the `loop.correction` semantics.

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
