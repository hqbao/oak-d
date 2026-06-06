# OURS vs BasaltVIO — Implementation Differences & Path to Parity

**Purpose**: A standalone, memory-independent record of *how our from-scratch
VIO differs from the BasaltVIO black box we are replacing*, so that future work
to close the gap (or to deliberately keep a difference) has a concrete basis and
does not have to re-derive it.

**Last updated**: 2026-06-07
**Our code**: `ours/flows/` (live flow pipeline) + `ours/ui/live_source.py` (viewer bridge) + `ours/lib/engine/` (out-of-process BA/SLAM)
**Basalt (reference)**: `dai.node.BasaltVIO` consumed in `baseline/depthai_vio.py`
**Scoring**: `ours/tools/vio_run.py` (offline, vs recorded Basalt poses) + `ours/tools/live_replay.py`

> TL;DR. Basalt is a **tight-coupled stereo-inertial sliding-window optimiser**.
> Ours is a **loosely-coupled RGB-D + gyro frontend**: the gyro owns rotation,
> vision (PnP on a stereo depth map) owns translation, and the accelerometer is
> used *only* to level roll/pitch — it does **not** constrain translation/scale.
> That single architectural choice (no accel in translation) is the root of
> every accuracy gap below, and also the root of every scenario where we *beat*
> Basalt. Everything else is secondary.

---

## 1. Side-by-side architecture

| Aspect | **BasaltVIO** (reference) | **Ours** (`--source ours`) |
|---|---|---|
| Coupling | **Tight** — IMU preintegration factors + visual factors in ONE joint optimisation | **Loose** — gyro corrects rotation; vision alone gives translation; accel only levels tilt |
| Visual input | Raw **stereo** (left+right), multi-view triangulation | **RGB-D**: left gray + the chip's stereo **depth map** (SGBM blob output) |
| Scale source | IMU (metric) + stereo baseline, jointly | **Depth map** per-frame (blur-biased on fast motion) |
| Rotation | Joint with everything | **Gyro preintegration** owns it; vision corrects, weighted by inlier confidence + a gyro-disagreement gate |
| Translation | Joint, IMU-constrained | Frame-to-frame **own PnP** (`ours/vio/pnp.py`: RANSAC DLT + LM, library-free, default) with depth → metric `t`; smoothed by `InertialTranslationFilter` |
| Accelerometer | Constrains velocity/position/scale (preint factor) | **Tilt leveling only** (complementary filter at rest); feed-forward to position is **OFF** by default |
| Optimiser | Sliding-window **bundle adjustment + marginalisation prior** | Production `ours`: **none** (pure f2f). `ours-ba`: analytic Schur BA window. `ours-slam`: + ORB loop + SE(3) pose graph |
| Bias estimation | Online gyro+accel bias in the window | Gyro bias from a startup static window only (no online accel bias on the production path) |
| Gravity direction | State in the optimiser | Fixed world-down `+y`; roll/pitch tracked by an EMA complementary filter, **not** a state |
| Keyframes / marginalisation | Yes (Basalt KF management + marginalisation) | No marginalisation; `ours-ba` keeps a fixed-size window, `ours-slam` keeps a pose graph |

### Code map (ours)
- Transparent time-synced input (`image, depth, IMU`) building block:
  `ours/vio/synced.py` (`iter_synced`, `SyncedSample`, `slice_imu`) — groups the
  IMU samples that fall in each frame interval `(t_prev, t_cur]` on the one
  recorder `ts_ns` clock; the inspector `ours/tools/synced_view.py` shows the triplet
  (replay + `--live`): image | depth | gyro angular-velocity line chart + 3D
  accel vector. Honest-only: every panel traces to a real recorded stream.
- Frontend (own pure-NumPy KLT + Shi-Tomasi, forward-backward check):
  `ours/vio/frontend.py`, `klt.py`, `klt_numba.py`, `corners.py`
- Frame-to-frame RGB-D PnP + gyro fusion: `ours/vio/odometry.py`
  (`RGBDVisualOdometry`, `OdometryConfig`); own library-free PnP solver in
  `ours/vio/pnp.py` (RANSAC DLT + robust-LM seed rescue + LM refine, default;
  `OAKD_OWN_PNP=0` switches to the cv2 oracle for the dev A/B only)
- Position smoother: `ours/vio/inertial_filter.py`
  (`InertialTranslationFilter`, `use_accel_prediction=False`)
- IMU preintegration (Forster): `ours/vio/imu.py`
- Windowed BA (depth-anchored, analytic Schur): `ours/vio/bundle.py`, `windowed.py`
- **Experimental tight-coupled window** (the Basalt-style path): `ours/vio/vio_window.py`
  (`optimize_vio`, `WindowedVIOMap`, `VioConfig`) — see §4
- Loop closure + pose graph: `ours/vio/loopclosure.py`, `posegraph.py`, `slam.py`

---

## 2. The differences that COST us accuracy (the gap to close)

These are the deltas to attack if the goal is "be Basalt".

1. **No accelerometer in translation (THE big one).** Vision under fast motion
   loses parallax/tracks and depth blurs, so our metric translation undershoots
   (fast-push tracks ~0.85–0.93 of Basalt scale). Basalt's IMU preint pins
   velocity/scale through those frames. We have **no metric anchor beyond
   vision**, so:
   - fast straight push under-travels vs Basalt;
   - a fast in-place yaw injects a *phantom* translation that vision cannot
     distinguish from a real one (measured: yaw-phantom per-frame |t| 26 mm vs
     real-translate 58–63 mm — **they overlap**, so no vision-only discriminator
     exists). Our guards in §3 are band-aids around this, not a fix.

   *Update:* on the **windowed-BA** path the worst of this (BA collapsing the
   forward baseline far below even the frame-to-frame VO — offline default window
   gave Sim3 scale 0.30–0.39 vs f2f 0.90–0.98) is now fixed by the **front-end
   relative-translation prior** (`BAConfig.use_vo_trans_prior`): the metric f2f
   PnP inter-keyframe translation is fed back as a soft scale anchor, the same
   role IMU preintegration plays for Basalt but sourced from our own VO
   (push_straight 0.39→0.97, push_fwdback 0.30→0.78). This anchors `ours-ba`/
   `ours-slam` to the VO scale; closing the remaining gap to Basalt still needs
   true IMU-in-the-estimator (the tight-coupled path: `ours/lib/backend/vio_window.py`,
   offline `vio_run.py --backend vio` — see §4.1).

2. **RGB-D depth map instead of raw stereo triangulation.** We consume the
   chip's SGBM depth blob; it is blur-biased and noisy at range (capped at 8 m
   in `OdometryConfig`). Basalt triangulates from raw stereo across the window,
   which is sharper and self-consistent with its BA.

3. **No sliding-window joint optimisation on the production path.** `ours` is
   pure frame-to-frame; error is never re-linearised against older frames.
   `ours-ba` adds a window but with **no marginalisation prior**, so it cannot
   carry information forward the way Basalt does. (It does carry a front-end
   relative-translation prior — see §2.1 — which anchors scale but not the full
   pose information a marginalisation prior would.)

4. **No online gravity-direction / accel-bias state.** Roll/pitch come from an
   EMA complementary filter, not an estimated state, so a slow gravity/bias
   drift cannot be corrected (shows up as corridor scale ~1.15 on the `vio`
   path).

---

## 3. Differences that are DELIBERATE (where ours BEATS Basalt — keep these)

Because position is vision-only and there is **no accel double-integration**,
ours does **not** drift when there is no real motion. Confirmed on device:

- **Covered camera / darkness** → ours stays put; Basalt's IMU integration
  drifts the pose. **Ours wins.**
- **Textureless white wall, stationary** → ours stays put. **Ours wins.**
- **Still / on a desk** → ours drift max ~107 mm over 15 s; no runaway.

If we add tight accel (§2.1) we **risk losing this** — Basalt drifts here
*because* it trusts the accel. Any tight-coupling work must preserve a
zero-velocity / no-motion behaviour (e.g. ZUPT when accel net ≈ 0 and vision
agrees) or we trade one win for the fast-push loss.

### The three loose-coupling guards (live `ours`, opt-in, gold byte-identical)
All in `OdometryConfig`, all **off by default** so offline gold scoring is
unchanged; the live source enables them. Each tuned by **measurement**, not
guessing:

| Guard | Live value | What it fixes | Measured |
|---|---|---|---|
| `max_translation_speed` | 4.0 m/s | Phantom per-frame jumps under shake/yaw (roller-coaster wobble) — clamps to a physical hand-speed bound | decimate=3 fast-push jitter 36.7→26.9, scale preserved; full-rate ~no-op |
| `min_inliers_for_translation` | 12 | White-wall garbage: KLT fills 400 corners but PnP keeps ~0–11 inliers → freeze translation (gyro still turns) | white-wall path-jitter 4.3→2.0; fast-push ATE 2.14%→1.82% (unharmed; p25=33 inliers) |
| `resolve_translation_on_disagree` | **off** | (kept available) re-solve t with gyro rotation locked when vision disagrees | **ineffective** on `push_shake_20s` (fires ~8% frames, never zeroed t) → left off |

Key insight that drove the white-wall guard: **`n_tracks` is NOT a usable
signal** (KLT fills its corner budget with garbage on a blank wall). The honest
discriminator is **`n_inliers`** (white-wall median 0 vs real motion ≥ ~33).

---

## 4. The Basalt-style path we already have: `--backend vio`

`ours/vio/vio_window.py` is the **tight-coupled** experiment — the seed for
parity. It folds IMU preint (rotation + velocity + position increments) and
visual reprojection + depth into ONE window solve for pose + velocity +
gyro/accel bias + landmarks (true Basalt style).

Current state (measured): on healthy motion it **regresses vs `ba`** and shows
slow accel/gravity drift (corridor scale ~1.15); it **collapses scale on fast
motion** (push_fwdback scale ~0.35). Why:
- the solver is **finite-difference** (rough) vs the analytic Schur BA in
  `bundle.py`;
- **no marginalisation prior** (window forgets);
- gravity direction is not a free state.

**Recent change (2026-06-04):** added `lock_tilt` to `VioConfig` — each pose has
**4 DoF (3 translation + yaw about world-vertical)** instead of 6; roll/pitch
are held to the accel-levelled gravity. This stops gravity leaking into a
horizontal translation and tightened the IMU vel/pos sigmas (0.15→0.03). Verified
by `ours/tools/vio_ba_selftest.py` scenario C (tilt-locked yaw+pos recovery) and
`ours/tools/vio_scale_probe.py --measure-b`. **`vio` is still opt-in / experimental
and touches no production path.**

### 4.1 The tight-coupled path today (offline) + what a live rebuild inherits

The tight-coupled window solver lives in `ours/lib/backend/vio_window.py`
(`optimize_vio`, `WindowedVIOMap`, `VioConfig`) and is exercised **offline** by
`ours/tools/vio_run.py --backend vio` and validated by `ours/tools/vio_ba_selftest.py`.

> The old live `--source ours-vio` viewer rode the legacy monolith, which has been
> removed; a live tight-coupled viewer would be rebuilt on the flow pipeline exactly
> like `ours-ba` / `ours-slam` (an `EngineFlow` + a `FlowPoseSource` mode), reusing
> the out-of-process engine so the solve never holds the camera read-loop GIL.

Two properties measured on the legacy live experiment are **already inherited by the
flow pipeline**, so a rebuild starts from them:

1. **Fully portable depth (no VPU / no `StereoDepth`).** The acquisition front-end
   taps the two **RAW** mono cameras and does everything itself: our own
   `LeftRectifier`/`RightRectifier` (library-free, ~1e-7 vs `cv2.stereoRectify`) →
   our own dense **SGM** matcher (`ours/lib/stereo/stereo.py`, census + N-path
   Hirschmüller, numba) → metric depth. The chip stereo engine is never used, so the
   front-end ports to any 2-camera + CPU target. (In the flow pipeline this is the
   `imu_cam` flow's `ComputeDepth` task.) **Grid invariant:** depth is on the
   RECTIFIED-left grid, so tracking runs on the rectified left too — feeding the raw
   left to PnP read depth at the wrong pixel (median 27 px off) → PnP churn.

2. **The window solve does not starve the camera loop.** `optimize_vio` was
   vectorised (batched numpy: einsum + `np.add.at` scatter) so it releases the GIL,
   **numerically identical** to the old scalar version (`vio_ba_selftest` A/B/C
   PASS). On the rebuilt path the heavy solve runs **out-of-process**
   (`ours/lib/engine/subprocess.py`), which removes GIL contention entirely — the
   same mechanism that fixed the `ours-ba`/`ours-slam` fast-push undershoot.

---

## 5. Concrete roadmap to "be Basalt" (ordered, each independently testable)

Do these on the `vio` path (`vio_window.py`) so production `ours` stays safe.

1. **Online gravity-direction state** (STILL TODO #1). Add the world-down
   direction (2 DoF) as an optimisation variable seeded from the accel level.
   This is the prerequisite that unblocks the scale collapse. Gate: corridor
   scale → ~1.0, no regression on `lab_*`.
2. **Marginalisation prior.** Carry a Gaussian prior over the oldest pose/vel/
   bias when it leaves the window (Schur-complement marginalisation) instead of
   dropping it. Gate: long-session drift stops compounding.
3. **Analytic Jacobians.** Replace the finite-difference `optimize_vio` inner
   loop with analytic reprojection + IMU Jacobians (reuse the structure in
   `bundle.py`). Gate: matches/beats `ba` on healthy gold + ~10× faster.
   - *Done so far (2026-06-05):* the FD assembly is **vectorised** (batched
     numpy, GIL released) — this gave the real-time live worker (§4.1) but is
     still finite-difference. Analytic Jacobians remain TODO for the accuracy +
     further speed gate.
4. **Raw-stereo triangulation** instead of the depth-map blob, so scale and
   landmarks are self-consistent with the BA. Gate: fast-push scale → ~1.0.
5. **Keyframe management** (Basalt-style selection + window of KFs, not every
   frame). Gate: real-time at full fps with windowed solve.
6. **Promote `vio` to the production translation path** only after 1–4 hold,
   and **add ZUPT / no-motion handling** so we keep the §3 static-drift win.

### Honest limits to keep stating (don't re-discover these)
- Normal/fast push is **already Basalt-grade** (ATE ~2%); the visible gap is
  only the *extreme* regimes.
- Extreme ~376°/s in-place yaw is **unsolvable by vision-only — and Basalt
  itself diverges there** (its recorded pose is a phantom on `yaw_inplace_15s`),
  so it is not usable as ground truth.
- The `max_translation_speed` clamp helps wobble at **low fps**; at full rate
  the `InertialTranslationFilter` already absorbs it (near no-op).
- Tight accel is the only real fix for fast-push undershoot, but it **risks the
  static-drift win** — treat §5.6 (ZUPT) as mandatory, not optional.

---

## 6. Run / score commands (for the next session)

```bash
# Live (device): our loosely-coupled VIO (bare f2f marker)
./run.sh --source ours --fps 60
# Live (device): + out-of-process windowed BA / loop-closure SLAM refining the map
./run.sh --source ours-ba                  # cyan BA-refined trajectory behind the marker
./run.sh --source ours-slam                # keyframe dots + loop-closure flash

# Tight-coupled window is offline-only today (no live viewer; rebuild on flows when needed):
.venv/bin/python ours/tools/vio_run.py --backend vio --depth ours --depth-fast

# Offline scoring vs Basalt
.venv/bin/python ours/tools/vio_run.py --all --backend f2f    # production frontend
.venv/bin/python ours/tools/vio_run.py --all --backend ba     # + windowed BA
.venv/bin/python ours/tools/vio_run.py --all --backend vio    # tight-coupled (experimental)
.venv/bin/python ours/tools/vio_run.py --all --backend vio --depth ours --depth-fast  # + our SGM depth

# Our from-scratch depth vs the chip depth (oracle)
.venv/bin/python ours/tools/stereo_selftest.py               # match rate + rel err on gold
.venv/bin/python ours/tools/synced_view.py --session sessions/gold/<name>  # eyeball replay (image|depth|IMU)

# Live-path replay without a device (reproduces the display pipeline frame-for-frame)
.venv/bin/python ours/tools/live_replay.py --session sessions/gold/<name> \
    --clamp 4 --min-inliers 12

# Tight-coupled diagnostics
.venv/bin/python ours/tools/vio_scale_probe.py --measure-b   # OLD(loose,6DoF) vs NEW(tight,lock_tilt)

# Self-tests (must pass before/after touching VIO)
.venv/bin/python ours/tools/imu_preint_selftest.py
.venv/bin/python ours/tools/vio_ba_selftest.py               # incl. scenario C (tilt-lock)
```

Record a new gold session (textureless wall, fast push, etc.):
```bash
.venv/bin/python baseline/tools/record_session.py sessions/gold/<name> --duration 15 --fps 20
```
