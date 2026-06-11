"""``ui.mathlib.calib`` -- camera-calibration math the Calibration menu needs.

Parallel to :mod:`sky.sensors` (the shared IMU-calibration collectors + store):
this subpackage owns the **camera** side of calibration -- the printable/displayable
target, the corner detector, the diversity-gated collector, the stereo solve, and
the calib.json writer. The wizard that drives them is a four-phase operator flow;
this package holds Phases 1 + 3 (the device-free math), with the data path and UI
living outside it (cross-referenced below).

Phase 1 -- the target the operator points the OAK-D at:

* :mod:`ui.mathlib.calib.checkerboard` -- a pure (no OpenCV / no Qt) generator that
  renders a standard, ``cv2.findChessboardCorners``-detectable checkerboard to a
  ``uint8`` grayscale image and saves it via the project's own pure-Python PNG codec
  (:mod:`ui.comms.lib.misc.pngio`); a lazy-Qt ``--show`` opens it fullscreen for the
  shine-on-screen workflow. ``(cols, rows)`` are INNER corners, not squares.
  Multi-chip-generic: just an image, with no OAK-D / depthai specifics.

Phase 3 -- the calibration math core (OFFLINE-testable, no device):

* :mod:`ui.mathlib.calib.detect` -- ``detect_corners`` finds + subpixel-refines the
  inner checkerboard corners in one grayscale frame (lazy cv2); ``reconcile_lr``
  resolves the 180-degree L<->R corner-ORDER ambiguity so a stereo pair's left and
  right corners name the SAME board points before the solve (the PRIMARY real-device
  garbage fix -- a reversed right order diverges the baseline to ~1 m).
* :mod:`ui.mathlib.calib.collector` -- ``StereoCheckerboardCollector``, a pure,
  hardware-agnostic capture state machine that accepts only DIVERSE stereo views,
  ``reconcile_lr``-corrects each accepted pair's L<->R corner order, and refuses to
  ``complete`` on an all-fronto-parallel set (it guards focal-length observability by
  requiring genuine tilt coverage).
* :mod:`ui.mathlib.calib.solve` -- ``solve_stereo`` recovers both intrinsics +
  distortion and the LEFT->RIGHT extrinsic (the project ``T_left_right`` convention),
  via ``cv2.calibrateCameraExtended`` x2 + ``cv2.stereoCalibrate``
  (``CALIB_FIX_INTRINSIC``); lazy cv2. Fits the WIDE-FOV ``CALIB_RATIONAL_MODEL``
  (8-coeff) with a seeded intrinsic guess (``CALIB_USE_INTRINSIC_GUESS``) for the OAK-D
  **W** fisheye (falling back to the standard 5-coeff model on a mild lens), rejects
  per-view reprojection outliers with a MAD gate (preserving high-tilt views), and
  flags an implausible solve ``ok=False`` (focal/baseline/inter-camera-rotation/stereo-
  RMS out of physical bounds) instead of saving garbage.
* :mod:`ui.mathlib.calib.writer` -- ``write_calib_json`` emits a calib.json that
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
NOT eagerly imported here, so ``import ui.mathlib.calib`` stays import-light.
"""
