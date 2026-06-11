"""``sky.calib`` -- the shared, device-free CAMERA-calibration math.

This is the ONE canonical home for the camera-side calibration math the
calibration wizard needs. It used to live in ``ui/mathlib/calib/``; ``ui`` was the
only consumer, so the device-free modules were RELOCATED here (R5) to make the UI
process a thin shell that just calls into :mod:`sky.calib`. It is the camera
counterpart to :mod:`sky.sensors` (the shared IMU-calibration collectors + store).

* :mod:`sky.calib.checkerboard` -- the PURE board generator:
  :func:`~sky.calib.checkerboard.make_checkerboard` renders a standard,
  ``cv2.findChessboardCorners``-detectable checkerboard to a ``uint8`` grayscale
  array; :func:`~sky.calib.checkerboard.square_px_from_mm` sizes squares for
  printing. ``(cols, rows)`` are INNER corners, not squares. (Only the PURE half
  is here: the PNG-save / Qt-preview / CLI wrapper stays per-project in
  ``ui.mathlib.calib.checkerboard`` because it imports ``ui.comms`` / PyQt6.)
* :mod:`sky.calib.detect` -- ``detect_corners`` finds + subpixel-refines the inner
  checkerboard corners in one grayscale frame (lazy cv2); ``reconcile_lr`` resolves
  the 180-degree L<->R corner-ORDER ambiguity so a stereo pair's left and right
  corners name the SAME board points before the solve.
* :mod:`sky.calib.collector` -- ``StereoCheckerboardCollector``, a pure,
  hardware-agnostic capture state machine that accepts only DIVERSE stereo views,
  ``reconcile_lr``-corrects each accepted pair, and refuses to ``complete`` on an
  all-fronto-parallel set (guarding focal-length observability).
* :mod:`sky.calib.solve` -- ``solve_stereo`` recovers both intrinsics + distortion
  and the LEFT->RIGHT extrinsic (the project ``T_left_right`` convention) via
  ``cv2.calibrateCameraExtended`` x2 + ``cv2.stereoCalibrate`` (lazy cv2), fits the
  wide-FOV ``CALIB_RATIONAL_MODEL`` with a seeded guess, MAD-rejects outlier views,
  and flags an implausible solve ``ok=False`` instead of saving garbage.
* :mod:`sky.calib.writer` -- ``write_calib_json`` emits a calib.json that
  :meth:`imu_camera.io.reader.StereoCalib.from_json` consumes; pure-Python (no cv2),
  translation written in CENTIMETRES so the loader's ``*0.01`` round-trips to metres.

cv2 POLICY: the flight runtime (VIO / SLAM / depth) is cv2-free and STAYS so. The
calibration modules that need cv2 (``detect``, ``solve``) LAZY-import it inside
their functions, so merely importing :mod:`sky.calib` never loads OpenCV. cv2 is
on the ALLOWED set for ``sky`` (these modules already used it), so the package
stays a leaf -- it imports only ``numpy`` / lazy ``cv2`` / other :mod:`sky.*`,
never any process / comms / io module (maps onto the C ``libskycalib`` layer in
``docs/C_PORT_PLAN.md``). The submodules are NOT eagerly imported here, so
``import sky.calib`` stays import-light.
"""
