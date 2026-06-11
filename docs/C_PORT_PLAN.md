# C99 Port Plan — oak-d (Python) → `skyslam` (C)

Status: **PLAN (approved-pending) — no code yet.** Synthesized from an architecture
review + a cited research pass (2026-06-11). Supersedes the single-binary framing in
`docs/SKYSLAM_ROADMAP.md` §3/§5 (that doc's library layering + repo skeleton §8 are
still reusable; the IPC-first strategy below replaces its sequencing).

## Central reframing — IPC is the port boundary, not a library layer
The system is **5 OS processes glued by a wire contract** (`imu_camera`, `depth`,
`vio`, `slam`, `ui`). You do NOT "port the SLAM stack"; you **replace one Python
process at a time with a C process that speaks the identical IPC wire**, the other
four staying Python. Unit of migration = a **process**. Prerequisite for ANY process
migration = a C library that reproduces the wire byte-for-byte.

## FIRST deliverable: `libskycomms` (C IPC), NOT `libskymath`
A C math kernel produces nothing runnable in the live graph; the IPC layer is the
gate. Three nested layers to reproduce in C (specs all in `imu_camera/comms/`):
1. **Unix-domain socket + path** (`ipc.py`): `$TMPDIR/ours_ipc/<endpoint>.sock`, dir 0700.
2. **`multiprocessing.connection` framing**: 4-byte **big-endian** `!i` length prefix
   then payload (`-1` sentinel + `!Q` for >2GB, which we never write); JSON handshake
   `{"role","topics"}` first frame; `b"BYE"` end sentinel.
3. **`codec.py`** (the easy, language-neutral part): big-endian, LEB128 varints,
   zig-zag signed, IEEE754-BE floats, 10 type tags, ndarray `[dtype][ndim][dims][C-bytes]`,
   messages encoded **positionally by @dataclass field order** keyed by `wire.TOPIC_WIRE`.
4. **shm rings** (`shared_array.py`/`ring_registry.py`): single segment, slot i =
   `[i*nbytes:(i+1)*nbytes]`, attach `track=False`.
C TUs: `sky_mpconn.c`, `sky_ipc.c`, `sky_codec.c`, `sky_wire.h`, `sky_shm_ring.c`, `sky_topics.h`.
**Milestone M1 = a C no-op process joins the live graph** (subscribes `imucam.sample`,
decodes real msgs) — proves wire parity end-to-end, de-risks everything week 1.

## What is port-safe NOW vs must WAIT
- **Port-safe (the LOOSE/default path = the frozen `gap=0` oracle):** SGM stereo,
  KLT+corners frontend (already Numba-JIT'd hot loops), RGB-D PnP odometry, loose
  windowed BA, loose IMU preintegration, SLAM loop closure + pose graph, IMU calib.
- **DO NOT PORT YET (unstable, actively churning):** the **tight VIO** (`--tight`,
  Phase 4 — still exploding @ 54×42, see `verification/phase4_tight_diverge_diag.py`),
  `propagate_imu.py`, closed-loop SLAM→VIO feedback, and the not-yet-existing 54×42
  VL53-ToF target algorithm. **Hard rule: stabilize-in-Python-then-port.** Porting an
  unstable algorithm = debugging the math AND the language port at once, no oracle.
- **UI stays Python/Qt permanently** (ground-control station, not flight runtime).

## Module/layer order (mapped to real files) + per-process migration
```
libskycomms (FIRST)  imu_camera/comms/*           ← gates every C process
libskymath           in-tree `sky/math/` (SO3/SE3 kernel) consolidated — Step 1 DONE;
                     C port = Step 2 (vec/mat/quat/Cholesky/LM grow here later)
libskysensors        imu_camera/mathlib/device/live_calib.py + imu/*calib*
libskydepth (SGM)    in-tree `sky/depth/stereo.py` (shared; consolidated — Step 1 DONE)
libskyfront          vio/mathlib/frontend/{klt,corners,frontend}.py + odometry/pnp.py
libskyvio (LOOSE)    vio/mathlib/{imu/imu.py loose, odometry/odometry.py, backend/*}
libskyslam           slam/mathlib/loop/{orb,loopclosure,posegraph,slam}.py
```
**Process migration order (safest first):** `depth` (isolated, stateless, SGM-oracle
already chip-safe) → `imu_camera` (device driver, reactive, no optimizer) →
`vio` LOOSE (hard numerical parity) → `slam` (depends on C vio) → `ui` = never.

## Test-oracle discipline (the keystone — extends the repo's `gap=0`)
- **Tier A — wire byte-parity** (gates `libskycomms`): C↔Python cross-decode +
  frozen sha256 per `Wire*` (mirror `codec_roundtrip_selftest` / `ipc_comms_selftest`)
  + shm `memcmp`. Codec: `<`/big-endian explicit, **field-by-field serialize, NEVER
  `memcpy` a C struct** (padding), canonical-by-construction (fixed field order, no
  default elision), exclude NaN from golden bytes.
- **Tier B — numeric golden vectors** (gates each math lib): Python dumps (inputs,
  float64 outputs) as binary fixtures; C asserts within a **per-module tolerance**:
  pure LA (exp/log/adjoint/Cholesky) **≤1e-10**; float32/transcendental/iterative
  (BA/PGO/PnP/KLT/SGM) **tolerance-only ~1e-6, validate on the END metric (ATE)** not
  intermediates (float summation order + RANSAC RNG differ — seed PRNG identically).
- **Tier C — end-to-end ATE parity** (gates each process migration): replay a gold
  session through the MIXED graph (1 C + 4 Python), score vs `baseline_metrics.json`
  within `TOL_MM`.
- **Jacobian numdiff in C** (non-negotiable, #1 drift-bug defense): central-difference
  check on every residual/Jacobian < 1e-6, independent of Python.

## Float-environment rules for parity TUs (researcher-grounded)
No `-ffast-math`; **`-ffp-contract=off`** (FMA changes the last bit, inlining-dependent);
explicit `fma()` only if wanted on both sides; NumPy uses **pairwise summation** —
reimplement it in C for contiguous reductions or declare them tolerance-only; build
parity tests under ASan/UBSan (uninit reads masquerade as tolerance failures).

## Repo + build
Separate repo `/Users/bao/skydev/skyslam` (do NOT disturb running oak-d). CMake≥3.20
+ Ninja, C99 `-Wall -Wextra -Werror`, no STL/Eigen/Ceres/OpenCV in shipped binaries
(dev-time oracles only). aarch64 cross-compile (RPi5), Unity (MIT) tests, CI matrix
{x86_64-clang, x86_64-gcc, aarch64-cross}. NEON kernels ALWAYS ship a scalar fallback
+ a NEON↔scalar parity test; develop/debug scalar on mac, NEON last. `scripts/dump_fixtures.py`
(in skyslam) imports oak-d's comms+mathlib to generate golden fixtures (read-only
cross-repo dep; oak-d never depends on skyslam). LDLT (not Cholesky) for near-singular
VIO info matrices.

## License map (copy-safe vs read-only)
Copy-safe (BSD/MIT/Apache): **GTSAM** (IMU preint reference), **Basalt** (sliding-window
VIO), **DBoW2** (BoW), **OpenCV calib3d/SGBM** (Apache). Read-only / clean-room (do not
copy code): **OpenVINS** (GPL-3), **ORB-SLAM2/3** (GPL-3), **S-MSCKF/msckf_vio** (Penn
non-commercial). Patents: SIFT free (expired 2020), ORB free, SURF avoid; verify the
specific SGM aggregation tricks vs live DLR patents.

## Honest effort + top risks
~**4–6 months** wall-clock for capture+depth+vio(loose)+slam in C on the Pi, EXCLUDING
the tight VIO (unschedulable until it converges in Python). M5 (vio loose) dominates.
Risks: **(#1) shared-memory segment NAME mismatch** — CPython `shm_open` name mangling
differs mac↔Linux↔SoC; **1-week spike (strace/dtruss) BEFORE committing the design**,
it gates all ring-bearing topics. Then: mp.connection framing edge cases; Jacobian
sign bugs (numdiff defense); float summation-order divergence (validate on ATE); NEON
debug slowness (scalar fallback); porting-an-unstable-algorithm (hard rule: don't);
oak-d drift while porting (freeze a tagged commit as the oracle).

## Recommended first step
**M0 = `libskycomms` + the shared-memory-name spike**, then **M1 = a C no-op process
joins the live graph**. Only after that does `libskymath` (M2) and the first real C
process (`depth`, M3) follow.
