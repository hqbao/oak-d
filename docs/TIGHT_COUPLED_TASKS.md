# Tight-coupled VIO rebuild — micro-task plan (every step has a picture)

> Companion to `docs/BASALT_REBUILD_PLAN.md`. That doc says *what* the 5 Basalt
> blocks are. **This** doc breaks the tight-coupled path into the smallest
> possible tasks, where **each task has a concrete INPUT and a VISUAL OUTPUT you
> can look at to judge pass/fail** — no "big bang" steps, nothing you have to
> take on faith.
>
> The golden rule here: **one task = one new idea + one plot/overlay + one
> number gate.** If a task can't be visualised, it's too big — split it.
>
> Vietnamese gloss ("Ý nghĩa") is added per task so the *why* is clear.

> **Status (2026-06-10):** The phase-level rollup lives in
> `docs/TIGHT_COUPLED_PLAN.md`. Phases 0–2 are **DONE**: the tight IMU
> preintegration covariance + `Ω_I` weight (Tracks A5/A6 below) are wired into
> `optimize_vio`, AND the tight estimator is now a **selectable live backend**
> via `--tight` (`vio.main`/`launcher.main`/`./run.sh`) — `make_vi_engine` +
> `vio_step`/`_vio_worker_main`, the `Keyframe` carrier superset (`ts_ns` +
> inter-KF `imu_seg`), and `BackendModule(tight=…, imu_info_weight=True)`. The
> LOOSE backend stays the default and **byte-identical** (oracle `gap = 0`,
> comms untouched). Phase-2 RUNS gate proved by
> `vio/tests/tight_smoke_selftest.py` (tight produces a finite/sane trajectory on
> gold) + a live `launcher.main --tight` replay (`pose.refined` flows). The
> loose-vs-tight **ATE benchmark is Phase 3** (Track C2/F1 gates).

---

## How to read a task

```
### <id> — <title>
How it works   : the algorithm in 2–3 sentences.
Input          : the file/stream it consumes (something you can already inspect).
Visual output  : the exact artifact you LOOK AT (a plot, an image overlay, a 3D
                 trajectory) + the command/tool that produces it.
Pass gate      : the number / shape that means "this step is correct".
Ý nghĩa        : một câu tiếng Việt — tại sao bước này quan trọng.
Builds         : new files this task adds (kept tiny + self-tested).
```

Every visual tool writes a PNG to `/tmp/skyviz/<id>.png` (or opens the existing
Qt 3D viewer). Every task ships a `*_selftest.py` that must stay green.

---

## Track 0 — Prerequisites (so we can actually reproduce the bug)

### T0.1 — Record a super-fast-push session
How it works   : the recorder dumps stereo C0/C1 + IMU + the Basalt/RTABMap pose
                 (our ground-truth reference) at 20 fps, exactly the live rate.
Input          : the OAK-D on the bench (quit `./run.sh` first — single access).
Visual output  : `baseline/tools/viz_session.py sessions/fast_push_15s` — scrub the
                 frames; the Basalt pose path is drawn so you see the true motion.
Pass gate      : the session has the fast push that fails live (net ≥ ~1.0 m,
                 peak speed ≥ ~1.0 m/s — read from the Basalt pose).
Ý nghĩa        : gold hiện tại quá hiền (net ~0.56 m) nên KHÔNG tái hiện được
                 "ì lại". Cần đúng cú đẩy siêu nhanh anh làm trên bàn.
Builds         : nothing (uses `baseline/tools/record_session.py`).

### T0.2 — Reproduce the stall offline (prove the mechanism BEFORE coding)
How it works   : replay the recorded session through the offline oracle in 3 modes
                 — raw VO tip (= VIO), filtered tip (EMA), filtered + rate-
                 limited BA correction (= VIO-BA).
Input          : `sessions/fast_push_15s`.
Visual output  : `verification/vio_oracle_runner.py … --plot` → one figure, three
                 "distance travelled vs time" curves overlaid on the Basalt curve.
Pass gate      : the VIO-BA curve visibly **stalls then crawls at ~0.3 m/s**
                 while the raw-tip curve tracks Basalt. (Confirms the rate cap is
                 the culprit, not KLT.)
Ý nghĩa        : nhìn tận mắt cái "ì lại" sinh ra từ EMA + giới hạn 0.3 m/s, chứ
                 không phải KLT. Đây là bằng chứng để chốt, trước khi sửa gì.
Builds         : `--plot` flag on the oracle runner (distance-vs-time curves).

> Tracks A–F below are the actual rebuild. They are ordered so each track is
> useful on its own and every gate is offline. Stop after any track and you still
> have something measurable.

---

## Track A — IMU preintegration (Basalt Block 2), no estimator yet

Goal of the track: a NumPy `IntegratedImuMeasurement` we fully trust, validated
piece by piece. This is pure math with no vision — perfect for visual unit tests.

### A1 — IMU timeline & dt
How it works   : load `imu.jsonl`, align IMU samples to camera frame timestamps,
                 compute per-segment `dt`.
Input          : `sessions/<s>/input/imu.jsonl`, `frames.jsonl`.
Visual output  : `imu_preint_viz.py --stage timeline` → plot accel.xyz & gyro.xyz
                 vs time with vertical lines at each camera frame.
Pass gate      : timestamps strictly increasing; IMU ≈ 200 Hz; 0 gaps > 2× median.
Ý nghĩa        : nền móng — sai đồng hồ/dt thì mọi tích phân sau đều sai.
Builds         : `vio/tools/imu_preint_viz.py` (grows each task), `vio/mathlib/imu/preintegration.py` (skeleton).

### A2 — Gyro-only ΔR per segment
How it works   : integrate gyro between two frames → `ΔR` (SO3 exp), the rotation
                 part of preintegration only.
Input          : two consecutive frames' IMU segment.
Visual output  : `--stage dR` → plot the preintegrated `ΔR` angle vs the VO
                 rotation angle between the same two frames, over the session.
Pass gate      : median |ΔR_preint − ΔR_vo| < 1.0° on a gold session.
Ý nghĩa        : xác nhận khâu xoay đúng (ta đã có gyro prior, đây là bản tích phân).
Builds         : `propagate ΔR` in `preintegration.py` + `imu_preint_selftest.py`.

### A3 — Full midpoint propagate (ΔR, Δv, Δp)
How it works   : Basalt midpoint: `R_mid=R·exp(½dt·ω)`, `a_w=R_mid·a`,
                 `v+=a_w·dt`, `p+=v·dt+½a_w·dt²`. Subtract the linearisation-point
                 bias first.
Input          : an IMU segment + a starting bias guess.
Visual output  : `--stage dp` → two panels: (1) a STILL session's |Δp| vs time
                 (should hug 0), (2) the fast-push session's |Δp| vs the Basalt
                 displacement (should be the right order of magnitude).
Pass gate      : still-session drift < 5 cm/s; push |Δp| within 2× of Basalt over
                 0.5 s windows.
Ý nghĩa        : đây là khâu mà gia tốc kế **dự đoán tịnh tiến** — chính cái
                 `ours-ba` đang thiếu khi đẩy nhanh.
Builds         : `propagateState` + extends selftest.

### A4 — Jacobians F / A / G (finite-difference check)
How it works   : analytic state-transition `F`, accel `A`, gyro `G` Jacobians vs
                 numerical perturbation.
Input          : a random valid segment (synthetic).
Visual output  : `--stage jac` → bar chart of max |analytic − numeric| per block
                 (F, A, G).
Pass gate      : every bar < 1e-5.
Ý nghĩa        : Jacobian sai thì bộ tối ưu (Track C) hội tụ bậy. Phải khoá bằng số.
Builds         : `F/A/G` in `preintegration.py` + jacobian selftest (mirrors
                 the existing `_vt_jac_check` pattern in `bundle.py`).

### A5 — Covariance & sqrt information   · DONE (2026-06-10)
How it works   : propagate `cov = A·cov·Aᵀ + B·(Q/dt)·Bᵀ` per IMU segment, where
                 A/B (the plan's F + the A/G noise inputs combined) use the SAME
                 midpoint sample + dp-before-dv ordering as the delta update, so
                 the weight matches the residual exactly; expose `sqrt_info`
                 (Cholesky/LDLT of the information matrix) to whiten the residual.
Input          : a segment + accel/gyro noise densities from `ImuNoise`
                 (`calib.json` carries no IMU noise yet, so it is a config knob;
                 defaults are BMI270-class). NOT additive to the deltas.
Visual output  : Monte-Carlo gate (analytic Σ vs empirical covariance of 4000
                 noisy re-integrations) instead of a single still-session plot —
                 a stronger correctness check than the 1σ-vs-time curve.
Pass gate      : MET — full-matrix relative Frobenius 2.6 %; Σ-trace
                 monotonically increasing; position 1σ 0.25 cm over 0.25 s (sane).
Ý nghĩa        : cho bộ tối ưu biết "tin IMU tới mức nào" — trọng số của factor IMU.
Builds         : covariance accumulation in `vio/mathlib/imu/imu.py` (additive,
                 deltas bit-unchanged) + `vio/tests/imu_preint_cov_selftest.py`.

### A6 — Residual + bias correction (the IMU factor)   · DONE (2026-06-10)
How it works   : `_imu_residual(state_i, g, state_j, pre)` = 9-D [δφ; δv; δp] error
                 with first-order bias correction (`pre.corrected`), whitened by
                 the per-edge information square root `Ω_I = pre.sqrt_info`
                 (`sqrt_info @ r`, ordering matched to the Phase-0 Σ_ij) when
                 `VioConfig.imu_info_weight=True`; the bias random-walk row stays a
                 separate Gaussian factor. A per-edge cache (`_ImuEdge`) integrates
                 once and relinearises only when the host-KF bias drifts past
                 threshold (first-order `corrected` handles small bias changes).
                 FD Jacobians (the validated `vio_window` default) rather than
                 analytic — analytic is the Phase-4 C-port refinement.
Input          : state_i, state_j, the cached preintegration `pre` (carries Σ_ij).
Visual output  : `vio/tests/vio_ba_selftest.py` scenarios D/E/F — recover a
                 synthetic VI window with `Ω_I = sqrt_info` weighting; printed
                 cost/residual + per-state errors vs ground truth.
Pass gate      : MET — residual → ~0 at the optimum; recovery pos ≤ 0.19 mm,
                 rot ≤ 0.002° (incl. estimated bias) with the covariance weight;
                 graceful degradation under inflated Σ. (Forster FD-Jacobian
                 correctness is exercised end-to-end by the recovery, not a
                 standalone FD-bar check; that lands with analytic Jacobians in
                 Phase 4.)
Ý nghĩa        : đây chính là "sợi dây" IMU nối 2 trạng thái trong cửa sổ tối ưu —
                 nay được cân đúng bằng hiệp phương sai thật Σ_ij⁻¹, không phải
                 sigma cố định.
Builds         : `Ω_I` whitening + `_ImuEdge` per-edge cache in
                 `vio/mathlib/backend/vio_window.py` + scenarios D/E/F in
                 `vio/tests/vio_ba_selftest.py`.
                 **Track A done = Block 2 done.**

> Byte-parity note (matches PLAN §4): the new `Ω_I` weight is **opt-in**
> (`VioConfig.imu_info_weight`, default False) because the frozen byte-parity
> oracle has `backend="vio"` entries that run this exact residual. The oracle uses
> the default (fixed sigmas) and stays `gap = 0`; the tight path turns it on.

### A7 — Velocity-divergence stabilisation @ 54×42   · DONE (2026-06-11)
How it works   : two opt-in `VioConfig` terms cure the rank-6 velocity deficiency
                 that lets the IMU difference-tie compound a drifting seed at
                 feature-starved 54×42. (A) `vel_cv_prior` — constant-velocity
                 smoothness `r_cv=(v_j−v_i)/σ_cv` APPENDED to the stacked
                 `_imu_eval` residual (NOT into `_imu_residual`, which would
                 desync the 9×9 `sqrt_info`); the existing FD loop fills its
                 columns. (B) `vel_zupt` — analytic, excitation-gated absolute
                 anchor `r_zupt=v_i/σ_z` (`H[v_i,v_i]+=I/σ_z²`). Gate is
                 GRAVITY-AWARE: `a_exc=|‖pre.dv‖/dt−|g||` (specific force still
                 carries gravity) + `w_exc=‖log(pre.dR)‖/dt`. `stabilize_velocity`
                 on `WindowedVIOConfig` flips both via `replace` in `run_ba`.
Input          : `imu_factors` (per-edge `pre`), the window nav states.
Visual output  : `verification/phase4_velocity_bisect.py --vel-cv-prior [--vel-zupt]`
                 per-KF `|v_opt|` ramp; `verification/phase4_bench_velprior.py`
                 OFF/CV/CVZ ATE+scale A/B (harness unmodified, flags via `replace`).
Pass gate      : MET — oracle `gap=0` flags OFF (incl. `backend="vio"`); FD/analytic
                 unit checks pass (`vio/tests/phase4_velprior_selftest.py`). 54×42
                 ATE: shake 1554→832 (−46 %), push-fast 249→104 (−58 %), straight
                 38.9→33.0 (−15 %), lab-loop 73→63 cm (−14 %); full-res ±1 % (no
                 regression). ZUPT anchors rest (still maxstep ↓, scale →1) without
                 crushing dynamic forward speed. HONEST LIMIT: shake runaway HALVED
                 not flattened (the IMU seed itself ramps); scale stays <1 at 54×42.
Ý nghĩa        : tiêm "thông tin vận tốc tuyệt đối" mà factor IMU đơn lẻ thiếu —
                 hãm đà phân kỳ vận tốc ở 54×42 mà KHÔNG đụng đường loose/oracle.
Builds         : `vel_cv_prior`/`vel_zupt` + gravity-aware gate in
                 `vio/mathlib/backend/vio_window.py`; `stabilize_velocity` knob;
                 `vio/tests/phase4_velprior_selftest.py`;
                 `verification/phase4_bench_velprior.py` (+ `--vel-*` flags on
                 `verification/phase4_velocity_bisect.py`).

> Byte-parity note: all five velocity-stabilisation fields default OFF and EVERY
> new code path is guarded by `if cfg.vel_cv_prior:` / `if cfg.vel_zupt:`, so the
> OFF path is byte-identical and the frozen `backend="vio"` oracle stays `gap=0`.

---

## Track B — IMU-only dead-reckoning demo (the "aha", 1 task)

### B1 — Chain `predictState` across the whole session (no vision)
How it works   : start from the Basalt initial state, integrate IMU only, frame to
                 frame, with the bias from `calib`.
Input          : `sessions/fast_push_15s` IMU.
Visual output  : the `ui` Viewer3D overlay — IMU-only trajectory vs Basalt vs raw
                 VO, all three in one 3D view.
Pass gate      : qualitative — during the **first ~1 s of the fast push** the
                 IMU-only path follows the push shape (it WILL drift after; that's
                 expected and is exactly what vision will correct).
Ý nghĩa        : đây là khoảnh khắc "thấy tận mắt" tại sao tight-coupling chữa
                 được đẩy nhanh — IMU một mình đã bắt được cú đẩy mà loose path
                 phải bò theo. Không viết optimiser vẫn thấy được giá trị.
Builds         : `vio/tools/imu_only_odom.py` (thin driver over Track A).

---

## Track C — Tight visual-inertial solve (Basalt Block 3)

Build the optimiser smallest-first: 2 states → N states → with bias. Reuses the
existing `bundle.py` vision sqrt-Schur; adds the IMU factor from Track A.

### C1 — Two-frame visual-inertial bundle
How it works   : 2 camera states + 1 IMU factor (A6) + the shared vision
                 reprojection block. Solve with the LM we already have in `bundle.py`.
Input          : synthetic: known relative pose + landmarks + a matching IMU segment.
Visual output  : `vi_bundle_viz.py --case two_frame` → before/after reprojection
                 overlay on both frames + a printed recovered-vs-true transform.
Pass gate      : recovers the known transform to < 1 mm / 0.1°.
Ý nghĩa        : đơn vị nhỏ nhất của bộ tối ưu tight — vision + IMU cùng 1 hệ.
Builds         : `vio/mathlib/backend/vio_solve.py` (2-state core) + selftest.

### C2 — N-state sliding window (vision + IMU, no marginalisation yet)
How it works   : extend C1 to `vio_max_states` recent pose-vel-bias states; IMU
                 factors chain consecutive states; old states just dropped (lossy,
                 fixed in Track E).
Input          : `sessions/<motion>` (gold) replayed.
Visual output  : two artifacts — (a) per-iteration cost curve (vision / IMU /
                 total) per frame; (b) `view_pose3d` window trajectory vs Basalt.
Pass gate      : `vio_run.py --backend vio2` ATE ≤ `--backend ba` on every gold
                 motion session (today's `vio` regresses — this must beat it).
Ý nghĩa        : cửa sổ tight thực sự. Cổng so trực tiếp với BA hiện tại.
Builds         : window assembly in `vio_solve.py`; `--backend vio2` in `vio_run.py`.

### C3 — Estimate IMU biases in the window
How it works   : add accel/gyro bias states + random-walk priors
                 (`bias_sqrt_weight`); the bias Jacobians (A4/A6) couple them.
Input          : a long gold session (bias needs time to observe).
Visual output  : `vi_bundle_viz.py --stage bias` → estimated `ba`,`bg` vs time;
                 should converge to a small steady value (not wander).
Pass gate      : bias bounded (|ba| < 0.5 m/s², |bg| < 2°/s) and steady on a still
                 segment.
Ý nghĩa        : bias là lý do loose filter không dám dùng gia tốc kế. Ước lượng
                 được bias = dùng được IMU để dự đoán tịnh tiến một cách trung thực.
Builds         : bias states + priors + selftest.

### C4 — Fast-push gate (the target)
How it works   : run C2+C3 on the recorded fast-push session.
Input          : `sessions/fast_push_15s`.
Visual output  : `view_pose3d` overlay: `ours-vio` (tight) vs `ours` vs Basalt +
                 the distance-vs-time curve from T0.2.
Pass gate      : tight path Sim3 scale ≥ 0.95 **and** distance-vs-time has **no
                 stall/crawl** (it should match `ours`, not the old `ours-ba`).
Ý nghĩa        : đây là mục tiêu cuối của anh — đẩy siêu nhanh mà vẫn full đoạn,
                 không ì. Chứng minh bằng đúng session đã làm `ours-ba` fail.
Builds         : nothing new (it's the gate that retires the bug).

---

## Track D — Keyframe management + landmarks (Basalt Block 4)

### D1 — Connection-ratio keyframe trigger
How it works   : new KF when `connected/(connected+new) < 0.7` AND
                 `frames_after_kf > 5` (Basalt's rule), replacing our time-based
                 `kf_every`.
Input          : a gold session.
Visual output  : `kf_viz.py` → timeline: camera speed curve with a marker at each
                 KF + the connection ratio that triggered it.
Pass gate      : KFs cluster at low-overlap (fast-motion / turn) moments, not at
                 fixed intervals; same/better ATE with **fewer** KFs.
Ý nghĩa        : đặt keyframe đúng lúc cảnh đổi nhiều — vừa nhẹ vừa vững hơn.
Builds         : KF trigger in `vio_solve.py` + viz.

### D2 — Anchored inverse-depth landmarks
How it works   : store each landmark as (host KF, bearing via StereographicParam,
                 `inv_dist`) instead of world XYZ; seed `inv_dist` from SGM depth.
Input          : a gold session.
Visual output  : `lm_viz.py` → (a) landmarks reprojected on the image coloured by
                 inverse depth; (b) scatter of SGM depth vs triangulated depth.
Pass gate      : scatter sits on the diagonal (no systematic scale offset);
                 reprojection RMSE ≤ the XYZ version.
Ý nghĩa        : inverse-depth ổn định hướng "đẩy thẳng" (ít parallax) — đúng trục
                 mà BA hay sập scale. Đây là bản nguyên lý của thí nghiệm
                 depth-host từng thất bại khi làm thô.
Builds         : inverse-depth landmark type + triangulation + selftest.

### D3 — Triangulation baseline gate
How it works   : only triangulate a track when the baseline ≥
                 `min_triangulation_dist` (0.05 m) and `0 < inv_dist < 3`.
Input          : a gold session.
Visual output  : `lm_viz.py --stage gate` → histogram of accepted vs rejected
                 baselines; rejected (tiny-baseline, ill-conditioned) ones drop out.
Pass gate      : no landmark with inv_dist outside (0,3); fewer wild depths.
Ý nghĩa        : chặn điểm 3D "ảo" do baseline quá nhỏ — nguồn nhiễu scale.
Builds         : the gate + selftest.

---

## Track E — Square-root marginalisation (Basalt Block 5)

### E1 — sqrt prior carry (keep the past instead of dropping it)
How it works   : when a state leaves the window, marginalise it into a sqrt
                 information prior (QR) instead of deleting it; demote old full
                 states to pose-only (drop vel+bias).
Input          : a long gold session.
Visual output  : `marg_viz.py` → (a) heatmap of the sqrt prior `H` before/after a
                 KF leaves; (b) trajectory continuity — zoom on the moment a KF is
                 marginalised, there must be **no jump**.
Pass gate      : trajectory continuous across marginalisation; ATE improves vs
                 Track C (C had no memory of dropped states).
Ý nghĩa        : giữ thông tin quá khứ → quỹ đạo mượt, không giật khi keyframe rời
                 cửa sổ. Đây là thứ làm Basalt drift cực thấp.
Builds         : `MargHelper` sqrt marginalisation in `vio/mathlib/backend/marginalize.py` (extend) + selftest.

### E2 — FEJ (first-estimates Jacobian) null-space check
How it works   : freeze the linearisation point of marginalised states; convert
                 the prior to delta-independent form (`b -= H·delta`).
Input          : a gold session.
Visual output  : `marg_viz.py --stage nullspace` → the 4-DoF gauge null-space
                 energy vs time (Basalt's `checkMargNullspace`); must stay ≈ 0.
Pass gate      : null-space energy < threshold for the whole run (no spurious
                 information injected into the unobservable gauge).
Ý nghĩa        : tránh "bơm" thông tin giả vào hướng tự do (yaw/translation gauge)
                 — lỗi kinh điển làm VIO tự tin sai rồi drift.
Builds         : FEJ handling + null-space selftest.

---

## Track F — The payoff (1 task)

### F1 — Three-way live comparison on the fast push
How it works   : run `ours`, `ours-ba` (current loose), `ours-vio` (the rebuild)
                 offline on the recorded fast-push session.
Input          : `sessions/fast_push_15s`.
Visual output  : one figure — 3 distance-vs-time curves over Basalt + a 3D overlay;
                 plus a table (scale, ATE, "stall?" yes/no).
Pass gate      : `ours-vio` ≥ 0.95 scale, ATE ≤ `ours-ba`, **no stall**, and it
                 matches `ours` on the push while ALSO being drift-corrected on the
                 loop sessions (where `ours` drifts).
Ý nghĩa        : một hình duy nhất chứng minh tight-coupled vừa đẩy-nhanh-full
                 (như ours) vừa hết drift (như Basalt). Kết thúc bài toán.
Builds         : nothing new (composes T0.2 `--plot` + `view_pose3d`).

---

## One-screen task → visual → gate table

| Task | Input | You LOOK AT | Pass gate |
|------|-------|-------------|-----------|
| T0.1 | OAK-D | viz_session frames+Basalt path | fast push captured (net ≥1 m) |
| T0.2 | fast_push | 3 distance-vs-time curves | ours-ba stalls @0.3 m/s |
| A1 | imu.jsonl | accel/gyro vs time | dt sane, 200 Hz |
| A2 | IMU seg | ΔR_preint vs ΔR_vo | <1° median |
| A3 | IMU seg | \|Δp\| still vs push | still <5 cm/s |
| A4 | synthetic | F/A/G FD bars | <1e-5 |
| A5 | seg+noise | pos 1σ vs time | monotonic, cm-scale |
| A6 ✓ | states | recovery w/ Ω_I=sqrt_info (D/E/F) | pos≤0.19mm, rot≤0.002° |
| B1 | fast_push IMU | 3D IMU-only vs Basalt vs ours | follows push first ~1 s |
| C1 | synthetic | reproj overlay 2 frames | recover <1 mm/0.1° |
| C2 | gold motion | cost curve + 3D window | vio2 ATE ≤ ba |
| C3 | long gold | bias vs time | bounded + steady |
| C4 | fast_push | overlay + distance curve | scale ≥0.95, no stall |
| D1 | gold | KF markers on speed curve | KFs at low-overlap |
| D2 | gold | inv-depth reproj + scatter | scatter on diagonal |
| D3 | gold | baseline histogram | inv_dist ∈ (0,3) |
| E1 | long gold | sqrt-H heatmap + continuity | no jump, ATE↓ |
| E2 | gold | null-space energy vs time | ≈0 throughout |
| F1 | fast_push | 3-way curves + table | vio ≥0.95, no stall |

---

## What gets built (all tiny, all self-tested)
```
vio/mathlib/imu/preintegration.py     # Track A (Block 2)
vio/mathlib/backend/vio_solve.py      # Track C (Block 3) — window solve
vio/mathlib/backend/marginalize.py    # Track E (Block 5) — extend existing
vio/tools/imu_preint_viz.py           # A1–A6 plots
vio/tools/imu_only_odom.py            # B1
vio/tools/vi_bundle_viz.py            # C1, C3
vio/tools/kf_viz.py  lm_viz.py  marg_viz.py    # D1–D3, E1–E2
vio/tests/*_selftest.py               # one per Track A/C/D/E unit
# verification/vio_oracle_runner.py gains --backend vio2 ; the oracle runner gains --plot
```

No task depends on hardware except T0.1 (the one recording). Everything else is
offline on recorded sessions, so progress is fully visible and reversible.
