# Algorithm Consolidation Plan — duplicated `*/mathlib/` → shared `sky.*`

Status: **IN PROGRESS — S0 + S1 DONE** (`sky/` scaffolded, `skymath` re-homed as
`sky.math`, SGM stereo deduped into `sky.depth.stereo`; oracle `gap=0` preserved).
S2–S6 still pending; S7 deferred until Phase 4 freezes. Goal: pull the DUPLICATED
algorithm code out of the 5 projects into one shared `sky.*` library so each process
is a thin shell (IPC wiring + calls into `sky.*`). Builds on the completed `skymath/`
(now `sky.math`, Step 1). Same discipline: byte-identical numerics, **`gap=0` gated at
every step**, divergent behavior preserved under distinct names (never silently
unified). Maps 1:1 onto the C `libsky*` layering (`docs/C_PORT_PLAN.md`) → this is the
Python precursor to the port.

## Measured drift map (diffed this session — better than feared)
| Target | Copies | Diff | Verdict |
|---|---|---|---|
| SGM `stereo.py` | imu_camera, depth | 3 hunks, **docstring-only** (code-identical; a `diff -r` gate already locks them) | **clean dedup** |
| PnP `odometry/pnp.py` | vio, slam | **0 diff** (byte-identical) | **clean dedup** |
| BA `backend/bundle.py` | vio, slam | 5 lines, **comment-only** (optimizer core identical; VIO-vs-loop factors live in the CALLERS, not here) | **clean dedup** (validate factor location first) |
| IMU calib `{accel_calib,calib_collect,calib_store,imu_calib}.py` | imu_camera, ui | accel/collect 0 diff; store/imu_calib docstring-path only | **clean dedup** |
| IMU preint `imu/imu.py` | imu_camera, slam, **vio** | imu_camera **== slam (0 diff)**; vio = 707-line tight-VIO superset (438 diff) | **dedup imu_camera+slam; DEFER vio (Phase 4)** |
| Engine `engine/{base,inprocess,steps,subprocess}.py` | vio, slam | base/inprocess small; steps 43 / subprocess 77 diff — genuine structural divergence | **extract-common-keep-variant (hardest; last)** |
| `resolution_build.py`, `warmup.py` | 2–3 | each builds a DIFFERENT config for the lib that project owns (deliberate) | **NOT a dedup target — leave per-project** |

## Target structure: ONE `sky/` package at repo root, with sub-packages
The shared library is a SINGLE top-level package `sky/` (importable as `import sky`);
each domain is a sub-package under it. `skymath` was re-homed INTO it as `sky.math`
(one library, not a sibling). Done so far: `sky.math` (=skymath) + `sky.depth` (SGM).
`sky.math` (DONE) · `sky.depth` (SGM, DONE) · `sky.front` (PnP, later KLT/corners) ·
`sky.sensors` (IMU calib) · `sky.imu` (loose preint) · `sky.backend` (bundle/marginalize/
windowed, loose) · `sky.engine` (Module/Step glue) · `sky.slam` (loop, re-home) ·
**`sky.vio` (tight VIO) = DEFERRED**. `sky.*` MUST stay free of process/comms/ui/io
imports (numpy + already-used cv2/numba only) so it's movable — enforce with an import-lint.

## Sequenced rollout (one domain/step, each `gap=0`-gated; dev → tester(gap=0 + ./run.sh replay) → docs)
- **S0 ✅ DONE** scaffolded `sky/` as one package; re-homed `skymath/` → `sky/math/` via the clean repoint (no shim — moved the 3 files, repointed all 27 `skymath` import sites to `sky.math`, deleted old `skymath/`). Added `sky.assert_import_clean()` (the import-lint: `sky.*` must pull NO process/comms/io module). Oracle stayed `gap=0`.
- **S1 ✅ DONE** `sky.depth` (SGM): moved imu_camera's canonical `stereo.py` → `sky/depth/stereo.py`, generalized the 3 docstring refs (`io.reader.StereoCalib`), repointed all 15 call-sites in both projects, deleted BOTH old `mathlib/stereo/` copies, and retired the prose `diff -r` lock-step gate (it was documentation in `depth/__init__.py` / `depth/mathlib/__init__.py` / the stereo `__init__.py` files — no executable gate existed). No SGM numerics touched; oracle stayed `gap=0`; both stereo selftests PASS; full 4-process replay clean.
- **S2** `sky.front.pnp` — byte-identical, tiny blast radius (vio+slam).
- **S3** `sky.sensors` (IMU calib) — clean dedup; touches `ui` (offscreen Qt selftests); **coordinate with in-flight calib work**.
- **S4** `sky.imu` (loose preint, **imu_camera+slam ONLY**; vio's superset untouched) — medium-risk (oracle-feeding), but the 0-diff makes it safe.
- **S5** `sky.backend.bundle` — **first confirm** the VIO-vs-loop factors live in the callers (windowed.py/loopclosure.py), not bundle.py; if so → clean dedup, else downgrade to extract-common.
- **S6** `sky.engine` (extract-common-keep-variant) — hardest stable target, LOW port-value (C replaces engine with the IPC process model) — do last / droppable.
- **S7 DEFERRED** `sky.vio`/`sky.slam` (tight VIO `vio_window.py` + vio's `imu.py` superset + `propagate_imu.py`; slam loop re-home) — **blocked until Phase 4 freezes** (`phase4_tight_diverge_diag.py` clean). Hard rule: stabilize-in-Python-then-consolidate.

## Stays per-project (not consolidated)
All `*/comms/` (vendored wire contract); `resolution_build.py`/`warmup.py` (per-project
config builders); each process `main.py`/`modules/`/`io/` (orchestration + IPC + drivers);
vio's tight-VIO surface (Phase 4); `baseline/` (Basalt ref); `ui/` Qt/viz (only its calib
math joins `sky.sensors`).

## Effort + risks
~**3–5 weeks** for the stable set S0–S6 (S7 unschedulable until Phase 4). S0–S3 ~1 wk
(mechanical, low-risk — most are docstring/0-diff so `gap=0` is near-automatic); S4 ~3-4d
(oracle-feeding); S5 ~3-5d (the factor-location investigation is the risk); S6 ~1 wk (only
genuine extract-common). Risks: per-step byte-parity (primary gate); engine drift
reconciliation; import-cycle/movability (lint `sky.*`); collision with in-flight calib
work (S3); the bundle factor-location assumption (S5). Honest note: this is INFRASTRUCTURE
(thinner processes, no drift) — it does NOT advance the algorithm/Phase 4; it makes future
algorithm work easier + is the port precursor.

## Recommended first step
**S0 + S1** (scaffold `sky/` then dedup SGM → `sky.depth`): safest, oracle-covered, and
the direct precursor to the C port's first real process. Establishes the reusable pattern
(move → repoint/shim → retire redundant gate → gap=0) that S2–S6 follow mechanically.
