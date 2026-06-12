# `imu_camera/tools/` — pre-flight diagnostics (calibration owner)

`imu_camera` owns and publishes the calibration contract (`calib.bundle`), so the
calibration sanity gate lives here. These tools are **standalone and additive** —
they only *import* the project's own loader (`imu_camera.io.reader`); they never
modify a runtime path, the comms contract, or any recorded session on disk.

## Files

| File | Purpose |
|---|---|
| `calib_check.py` | Pre-flight / CI gate that validates a session's parsed `StereoCalib` (intrinsics, stereo extrinsic, IMU↔cam extrinsic, recorded-data consistency) against physical sanity bands and flags malformed / implausible values **before** a run. |
| `gravity_sphere.py` | Offline learning view (`ALGORITHMS.md` §1.3): a 3D scatter that makes accelerometer **bias / scale / misalignment** visible — the six raw face means form a tilted, off-centre ellipsoid that **snaps** onto the g-sphere after `T(a−b)`, with `residual_g` shown as leftover scatter. Headless `--render PNG` (matplotlib Agg). |

## `calib_check.py`

Validates the **parsed** calib (the exact object the live pipeline consumes via
`StereoCalib.from_json`) — so what it checks is byte-identical to what VIO/depth
will use. In particular the loader converts the `T_left_right` translation from
**centimetres to metres**, so the tool validates the metres value (OAK-D baseline
≈ 0.075 m) and specifically catches a skipped or doubled cm→m conversion.

```sh
cd /Users/bao/skydev/oak-d

# Validate a recorded session (primary — also checks recorded-frame consistency)
.venv/bin/python -m imu_camera.tools.calib_check --session sessions/gold/lab_loop_30s

# Validate a bare calib.json (secondary — no recorded-data checks)
.venv/bin/python -m imu_camera.tools.calib_check --calib path/to/calib.json

# Strict CI gate: treat WARN as failure for the exit code
.venv/bin/python -m imu_camera.tools.calib_check --session <dir> --strict
```

**Output** is an aligned `CHECK | MEASURED | EXPECTED | STATUS` table (status ∈
`{PASS, WARN, FAIL, INFO}`), a one-line explanation per non-`PASS` row, and a
`N pass / M warn / K fail / J info` summary.

**Exit code** — `0` when no `FAIL` (`WARN` allowed); nonzero on any `FAIL`, or
(under `--strict`) any `WARN`. Suitable as a pre-run / CI gate.

### Checks

* **Intrinsics** (left + right): `fx,fy>0`; pixel aspect `|fx-fy|/fx`; principal
  point inside image + near centre; `K` consistent with `fx,fy,cx,cy`; image size
  `>0` and equal L/R; horizontal FOV in a sane band; distortion finite, a known
  model length, sane magnitude.
* **Stereo extrinsic** `T_left_right` (metres): rotation ∈ SO(3) (`‖RRᵀ−I‖`,
  `det≈+1`); inter-camera angle small (parallel rig); baseline in `0.02–0.30 m`
  (with a cm→m hint when out of band); baseline dominantly along camera-X.
* **IMU↔camera**: if `T_imu_left` present, rotation ∈ SO(3) + small lever-arm;
  otherwise **INFO** "no IMU extrinsics → gyro prior disabled" (a valid state, not
  a failure). `imu_noise` densities finite/positive when present, else INFO.
* **Recorded-data consistency** (`--session` only): calib resolution == recorded
  frame shape; median recorded depth in an indoor band (skips stereo warm-up
  frames that pin a handful of pixels at the far disparity rail).

## `gravity_sphere.py`

Makes the accelerometer's three error sources (`sky.sensors.accel_calib`)
*visible* in one 3D picture (`ALGORITHMS.md` §1.3 calls this "the most illuminating
one for this stage"): the translucent **g-sphere** (radius `g`), the six **raw**
face means (RED) forming a tilted, off-centre ellipsoid (centre = **bias**,
per-axis half-axes = **scale**, tilt = **misalignment**), and the **calibrated**
vectors `T(a−b)` (GREEN) **snapping** onto the sphere. `bias`, per-axis scale,
max off-diagonal misalignment and `residual_g` are annotated; grey connectors draw
each raw→calibrated "snap".

```sh
cd /Users/bao/skydev/oak-d

# REAL stored calib (auto-picks the only device in .cache/imu_calib.json)
.venv/bin/python -m imu_camera.tools.gravity_sphere --render /tmp/gravity_sphere.png

# A specific device, or the synthetic demonstration (no device / disk needed)
.venv/bin/python -m imu_camera.tools.gravity_sphere --device <id> --render out.png
.venv/bin/python -m imu_camera.tools.gravity_sphere --demo --render /tmp/demo.png
```

**Data source — be honest about it.** The calib store
(`sky.sensors.calib_store`) persists only the **solved** `AccelCalibration`
(`T, bias, residual_g, g`), *not* the six raw face vectors `SixFaceCollector`
captured (it solves and discards them), and no raw accel stream is recorded on
disk. So the default mode loads the **real stored** calib and **reconstructs** the
six raw faces by inverting the stored model exactly
(`a_raw_k = b + T⁻¹(g·dir_k)`) — the genuine "before" dots for *this device's own*
calibration (feeding them back through `cal.apply` lands on `g·dir_k`, so the green
dots sit on the sphere and the annotated `residual_g` is the true stored fit
residual). `--demo` instead synthesizes from a **known** injected model + noise and
shows the production `solve_accel_calibration` recovering it (residual scatter
visible). The figure title and stderr always state which source was used. The tool
is offline: it only *imports* `sky.*` read-only — gap=0 is unaffected.

**Shared render core — also LIVE in the accel wizard.** The figure-drawing is
factored into `render_gravity_sphere(raw_vecs, calib_or_None, …) -> RGB image`, and
the **live** six-face accelerometer wizard (`ui.qt.calib_dialogs.AccelCalibDialog`)
reuses that *same* routine: as the operator tumbles the device the captured raw
vectors land on the sphere in real time (RED), and after the 6th face + solve they
**snap** onto the g-sphere (GREEN) — the exact offline payoff, live during capture.
The wizard feeds the renderer the six raw face means + the solved
`AccelCalibration` it already holds **locally** (no comms/IPC change), imports
`render_gravity_sphere` **lazily** (only when the dialog opens), and shows the
returned image as a Qt pixmap (matplotlib-Agg → numpy → `QPixmap`, no
`FigureCanvasQTAgg` dependency). One figure, two callers (DRY); `render_sphere_png`
is now a thin disk-writing wrapper over the core.

## Self-tests

```sh
.venv/bin/python -m imu_camera.tests.calib_check_selftest
.venv/bin/python -m imu_camera.tests.gravity_sphere_selftest
```

`calib_check_selftest` asserts a real gold calib produces **no FAIL** (exit 0, and
`--strict`-clean), and that five injected faults — non-orthonormal stereo R, absurd
baseline, L/R size mismatch, K/scalars desync, NaN distortion — each FAIL with a
nonzero exit. The broken cases are built **in memory** (deep-copied `StereoCalib`);
no session on disk is touched.

`gravity_sphere_selftest` proves the teaching claim for both sources: the
calibrated faces sit **closer to `|g|`** than the raw faces (the snap is real), the
synthetic solver recovers the injected model to a tight residual, the stored-calib
reconstruction round-trips onto `g·dir` to numerical precision, and the headless
render writes a **non-blank** PNG (size + pixel-variance checked).

The **live** wizard reuse is covered by an offscreen-Qt selftest:

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m ui.tests.accel_calib_sphere_selftest
```

It feeds `AccelCalibDialog` synthetic six-face captures (a known distorted sensor)
and asserts the sphere pixmap is blank before capture, fills (non-blank + changing
content) as faces land, shows the post-solve **snap** after the 6th face, and is
re-rendered **only** on a face-set/solve change (a no-capture tick does not redraw).
