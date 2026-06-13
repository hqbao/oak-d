"""``ui.calib`` -- the ``ui``-COUPLED half of camera calibration.

The device-free calibration MATH (corner detect, diversity-gated collector, stereo
solve, calib.json writer, and the pure checkerboard generator) was RELOCATED into
the shared :mod:`sky.calib` leaf library (R5) -- ``ui`` was its only consumer.
What stays here is only the part that genuinely needs the ``ui`` process and so
CANNOT be a leaf (it would import ``ui.comms`` / PyQt6):

* :mod:`ui.calib.checkerboard` -- the thin I/O WRAPPER around the shared
  generator. It re-exports the pure :func:`sky.calib.checkerboard.make_checkerboard`
  / :func:`~sky.calib.checkerboard.square_px_from_mm`, and adds ``save_checkerboard``
  (writes the board via the project's pure-Python PNG codec
  :mod:`ui.comms.lib.misc.pngio`), an optional lazy-Qt ``--show`` fullscreen
  preview, and the operator CLI. The ``ui.comms`` / PyQt6 edges are exactly why
  this half stays per-project (see ``docs/CONSOLIDATION_PLAN.md``).

The shared math now lives in :mod:`sky.calib` (cross-referenced below so the flow
still reads end-to-end):

* :mod:`sky.calib.checkerboard` -- the PURE (numpy-only) board generator.
* :mod:`sky.calib.detect` -- ``detect_corners`` finds + subpixel-refines the
  inner checkerboard corners in one grayscale frame (lazy cv2); ``reconcile_lr``
  resolves the 180-degree L<->R corner-ORDER ambiguity so a stereo pair's left and
  right corners name the SAME board points before the solve (the PRIMARY real-device
  garbage fix -- a reversed right order diverges the baseline to ~1 m).
* :mod:`sky.calib.collector` -- ``StereoCheckerboardCollector``, a pure,
  hardware-agnostic capture state machine that accepts only DIVERSE stereo views,
  ``reconcile_lr``-corrects each accepted pair's L<->R corner order, and refuses to
  ``complete`` on an all-fronto-parallel set (it guards focal-length observability by
  requiring genuine tilt coverage).
* :mod:`sky.calib.solve` -- ``solve_stereo`` recovers both intrinsics +
  distortion and the LEFT->RIGHT extrinsic (the project ``T_left_right`` convention),
  via ``cv2.calibrateCameraExtended`` x2 + ``cv2.stereoCalibrate``
  (``CALIB_FIX_INTRINSIC``); lazy cv2. Fits the WIDE-FOV ``CALIB_RATIONAL_MODEL``
  (8-coeff) with a seeded intrinsic guess (``CALIB_USE_INTRINSIC_GUESS``) for the OAK-D
  **W** fisheye (falling back to the standard 5-coeff model on a mild lens), rejects
  per-view reprojection outliers with a MAD gate (preserving high-tilt views), and
  flags an implausible solve ``ok=False`` (focal/baseline/inter-camera-rotation/stereo-
  RMS out of physical bounds) instead of saving garbage.
* :mod:`sky.calib.writer` -- ``write_calib_json`` emits a calib.json that
  :meth:`imu_camera.io.reader.StereoCalib.from_json` consumes; pure-Python (no cv2).
  The translation is written in CENTIMETRES so the loader's ``*0.01`` round-trips it
  back to metres.

Phases 2 + 4 live OUTSIDE this package (kept here only so the flow reads end-to-end):
Phase 2 is the RAW stereo IPC source
(:class:`ui.modules.ipc_sources.IpcStereoRawSource`, capture's ``imucam.sample``
left+right pair); Phase 4 is the Qt wizard
(:class:`ui.qt.camera_calib_dialog.CameraCalibWizard`) that shows the board, drives
the collector, runs the off-thread solve, grades it with ``calib_check``, and writes
the calib.json.

cv2 POLICY -- the flight runtime (VIO / SLAM / depth) is cv2-free and STAYS so. The
calibration wizard is an operator/dev tool, so OpenCV is acceptable in Phases 1 + 3,
but every module that needs cv2 (``detect``, ``solve``) LAZY-imports it inside its
functions; ``checkerboard`` and ``writer`` are cv2-free outright. Importing this
package (or any of its submodules) therefore never loads OpenCV. The submodules are
NOT eagerly imported here, so ``import ui.calib`` stays import-light.
"""
