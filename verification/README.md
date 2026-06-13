# `verification/` — byte-parity harness for the 5-project split

This directory PROVES that splitting the monolithic pre-split `ours/` codebase into
five projects (`imu_camera` + `vio` + `slam` + `ui` + `launcher`) preserved the
numerical behaviour **byte-for-byte**.

It only **imports** the projects. It never modifies `ours/` (the reference oracle)
or any project directory — `ours/` is kept INTACT as the gold oracle.

## The core idea

The **live** pipeline is now separate OS processes talking over IPC. Process
scheduling is nondeterministic, so the live path **cannot** give byte-parity. So
the parity oracle is an **in-process** harness that imports each project's
verbatim-ported MATH directly (single `LocalPubSub`-equivalent, no `IPCPubSub`)
and reproduces the pre-split deterministic scoring loop exactly.

Two independent things are proven:

1. **End-to-end math parity** — the in-process oracle drives the SPLIT projects'
   math through the SAME ATE/Sim3 scoring loop `ours/tools/vio_run.py` uses, and
   reproduces the pre-split ATE scores. The component math was ported verbatim
   (only import roots + docstrings re-rooted), so the end-to-end score is
   bit-identical.
2. **IPC contract parity** — the vendored `comms/` package is byte-identical
   across all 5 project copies, every copy's codec produces identical bytes for a
   fixed test-vector set, and the bridge + shared-memory rings round-trip a
   message intact.

## Files

| File | Purpose |
|---|---|
| `baseline_metrics.json` | The FROZEN pre-split baseline (rounded display mm + full-precision f64 metres), captured from `ours/tools/vio_run.py`. |
| `oracle_replay.py` | The in-process replay oracle: SPLIT-project math (`imu_camera`/`vio`/`slam`) through the EXACT `vio_run` ATE/Sim3 scoring (`umeyama`/`ate` copied verbatim). NO `IPCPubSub`. |
| `vio_oracle_runner.py` | CLI mirror of `ours/tools/vio_run.py` (`--session`/`--backend`/`--max-frames`/`--all`) driving the oracle. Prints the same ATE block. |
| `oracle_replay_selftest.py` | Byte-parity gate: for each baseline entry, asserts new-oracle == baseline within `TOL_MM=1e-6` mm AND bit-for-bit == the LIVE old oracle (`ours.tools.vio_run.score_session`). Fails loudly with the exact gap. |
| `ipc_comms_selftest.py` | Cross-project `comms` parity: dir-diff of all 5 copies, 5-copy codec sha256 digest + cross-decode, `SharedArrayRing` round-trip, full bridge round-trip over a real Unix socket. |
| `loose_vs_tight_bench.py` | LOOSE-vs-TIGHT ATE benchmark over the gold suite at full-res + 54x42 ToF (read-only; imports the frozen math + scoring). |
| `direct_vo_bench.py` | **Research harness** (Stage-1 hypothesis test + Stage-2a IMU seed). Frame-to-keyframe DENSE DIRECT RGB-D VO (`sky.front.direct`) over the gold suite at 54x42 ToF, scored with the SAME columns as `loose_vs_tight_bench` and printed SIDE-BY-SIDE with the measured sparse baseline. Tests whether dense direct + accurate ToF depth fixes the sparse VIO's scale collapse @ 54x42. `--seed {none,gyro,imu}` picks the Gauss-Newton `init_T`: `gyro` (Stage-1, rotation prior only) or `imu` (Stage-2a, full 6-DoF IMU dead-reckoned seed via `sky.vio.imu.predict_state` + `complementary_correct`). Read-only; touches no frozen path -> oracle stays gap=0. |

## How to run

```sh
cd /Users/bao/skydev/oak-d

# 1) Head-to-head single run (must print the baseline 75.9 mm)
.venv/bin/python verification/vio_oracle_runner.py \
    --session sessions/gold/lab_loop_30s --backend vio --max-frames 20

# 2) Byte-parity gate over every baseline entry (+ live old-oracle bit check)
.venv/bin/python verification/oracle_replay_selftest.py

# 3) 5-copy comms + codec parity + ring + bridge round-trip
.venv/bin/python verification/ipc_comms_selftest.py
```

`oracle_replay_selftest.py` accepts `--no-live-old` to skip importing `ours`
(baseline-JSON-only mode).

## What "byte-parity" means here

Each metric is compared at full float64 `repr`. The observed agreement is EXACT
(gap `0.000e+00`); the `1e-6` mm tolerance is a guard so a future
numerically-insignificant change is still flagged loud. **The tolerance is never
weakened to force a pass** — a divergence is a release VETO.

The gates are not vacuous: each has a verified negative control (a poisoned
baseline / a tampered codec copy both correctly produce a FAIL + non-zero exit).

## Backends

`f2f` (frame-to-frame VO), `ba` (windowed bundle adjustment), `slam` (ba + loop
closure + pose graph), `vio` (tight-coupled visual-inertial window). All four were
verified bit-identical vs the old oracle, including the `depth=ours` from-scratch
SGM stereo path.
