"""Read the live OAK-D calibration + startup IMU references for the VIO graph.

The unified front-end (``cam`` + ``imu_cam`` off ONE
:class:`~imu_camera.mathlib.device.oak_live.SharedLiveDevice`) needs the same boot-time facts the
old monolithic capture flow read in ``LiveCaptureFlow.open()``:

* camera intrinsics ``K`` and the stereo :class:`~imu_camera.io.reader.StereoCalib`
  (so the SGM matcher can rectify the raw cameras),
* the IMU->camera rotation ``R_imu_cam`` (gyro prior conjugation),
* a per-device gyro bias (cached, calibrated once -- only that first calibration
  needs the device held still) and a startup gravity-align accelerometer seed
  (measured each boot; once the bias is cached this is a quick non-gated read, since
  the odometry flow's continuous ``CorrectTilt`` re-levels roll/pitch at rest).

This module is the single place that turns an acquired shared device into those
references, so the live graph and any future tool apply identical maths. It is
hardware-only: validated on the bench, never in the offline test harness (the
offline path never imports depthai). The still-gate + cache logic mirrors the
proven ``flows/capture/live.py`` so behaviour is unchanged by the front-end split.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

from sky.sensors.accel_calib import AccelCalibration
from sky.sensors.calib_store import (
    load_accel_calib, load_gyro_bias, save_gyro_bias)
from sky.sensors.imu_calib import ImuCalibration

from imu_camera.comms.lib.config.resolution import ResolutionProfile
from imu_camera.mathlib.resolution_build import sgm_config
from imu_camera.mathlib.imu.decode import decode_imu_packets
from imu_camera.io.reader import StereoCalib
from .camera_calib_store import load_camera_calib
from .oak_live import SharedLiveDevice

# Startup stillness gates (identical to the legacy capture path): the gravity
# level and any measured gyro bias are means, so ANY motion during the window
# poisons them -> reject the sample and restart the window when moving.
_STILL_GYRO = 0.15      # rad/s
_STILL_ACCEL = 0.6      # m/s^2 deviation from the window mean


@dataclass(frozen=True)
class LiveFrontEndCalib:
    """Everything the live VIO graph needs, read once from the shared device."""

    K: np.ndarray
    calib: StereoCalib
    R_imu_cam: np.ndarray | None
    sgm_cfg: object
    res: ResolutionProfile
    accel_align: np.ndarray | None
    imu_calibration: ImuCalibration | None
    # Per-device key for the IMU calib store -- the SAME id used by
    # load_gyro_bias / load_accel_calib below, so a UI-saved calibration keys
    # identically and takes effect on the next capture start. ``None`` if the
    # device exposes no id.
    device_id: str | None = None


def _read_stereo_calib(ch, width: int, height: int):
    """Read ``(K, StereoCalib, R_imu_cam)`` from a depthai calibration handler."""
    import depthai as dai

    left_socket = dai.CameraBoardSocket.CAM_B
    right_socket = dai.CameraBoardSocket.CAM_C

    K = np.array(ch.getCameraIntrinsics(left_socket, width, height),
                 dtype=np.float64)

    def _intr(sock):
        Ki = np.array(ch.getCameraIntrinsics(sock, width, height),
                      dtype=np.float64)
        dist = list(ch.getDistortionCoefficients(sock))
        return {"fx": float(Ki[0, 0]), "fy": float(Ki[1, 1]),
                "cx": float(Ki[0, 2]), "cy": float(Ki[1, 2]),
                "dist": [float(x) for x in dist],
                "width": int(width), "height": int(height)}

    T_lr = np.array(ch.getCameraExtrinsics(left_socket, right_socket),
                    dtype=np.float64).reshape(4, 4)
    calib = StereoCalib.from_json({
        "intrinsics_left": _intr(left_socket),
        "intrinsics_right": _intr(right_socket),
        "T_left_right": T_lr.tolist(),
    })

    try:
        R_imu_cam = np.array(ch.getImuToCameraExtrinsics(left_socket),
                             dtype=np.float64)[:3, :3]
    except Exception:
        R_imu_cam = None
    return K, calib, R_imu_cam


def _collect_startup(device: SharedLiveDevice, R_imu_cam, accel_cal,
                     *, estimate_bias: bool, gate: bool = True,
                     window_s: float = 0.4, timeout_s: float = 6.0):
    """Mean startup accel (gravity-align, cam frame). Returns ``(accel_align, gyro_bias)``.

    Two modes:

    * ``gate=True`` (used when the per-device gyro **bias** must be measured -- the
      first-ever run or ``--recalibrate-bias``): a sample is accepted only while the
      device is at rest (``|gyro| < _STILL_GYRO`` and accel within ``_STILL_ACCEL``
      of the window mean); any motion clears the buffer and restarts the window, so
      the bias mean is a true still-window. This is the only path that asks the
      operator to hold still, and it runs at most once per device.
    * ``gate=False`` (bias already cached -- the normal Start): collect over
      ``window_s`` with NO motion gate, a quick *rough* gravity seed. It need not be
      accurate because the odometry flow's continuous ``CorrectTilt`` re-levels
      roll/pitch on any at-rest frame -- so the operator does not hold still at Start.
    """
    q = device.q_imu
    accel: list[np.ndarray] = []
    gyro: list[np.ndarray] = []
    win_start: float | None = None
    t_start = time.monotonic()

    def _level(samples):
        a = np.mean(samples, axis=0)
        a = accel_cal.apply(a) if accel_cal is not None else a
        return (R_imu_cam @ a) if R_imu_cam is not None else a

    while time.monotonic() - t_start < timeout_s:
        msg = q.tryGet() if q is not None else None
        if msg is None:
            time.sleep(0.005)
            continue
        for w, v, _ in decode_imu_packets(msg):
            if not (np.all(np.isfinite(v)) and np.all(np.isfinite(w))):
                continue
            if gate:
                moving = float(np.linalg.norm(w)) > _STILL_GYRO
                if accel and float(np.linalg.norm(
                        v - np.mean(accel, axis=0))) > _STILL_ACCEL:
                    moving = True
                if moving:
                    accel.clear()
                    gyro.clear()
                    win_start = None
                    continue
            accel.append(v)
            gyro.append(w)
            now = time.monotonic()
            if win_start is None:
                win_start = now
            elif now - win_start >= window_s and len(gyro) >= 10:
                bias = np.mean(gyro, axis=0) if estimate_bias else None
                return _level(accel), bias
    # Never settled: rough level from whatever we saw, no measured bias.
    if estimate_bias:
        print("[live] WARNING: camera kept moving during startup calibration; "
              "gyro bias not estimated. Hold still and restart.", file=sys.stderr)
    return (_level(accel) if accel else None), None


def select_camera_calib(dev_id: str | None, factory_K: np.ndarray,
                        factory_calib: StereoCalib,
                        user_calib: StereoCalib | None,
                        *, use_camera_calib: bool = False
                        ) -> tuple[np.ndarray, StereoCalib]:
    """Choose the LIVE stereo calibration. FACTORY is the default; the user's saved
    calib is applied ONLY when explicitly opted into.

    Factored out of :func:`read_live_calibration` so the decision is unit-testable
    HEADLESS (no OAK-D, no depthai) -- the only device-dependent input is the
    already-read factory ``(K, StereoCalib)``; ``user_calib`` is whatever
    :func:`~imu_camera.mathlib.device.camera_calib_store.load_camera_calib` returned
    (the caller is expected to pass ``None`` when ``use_camera_calib`` is off, so the
    store is never even read in the default path).

    Behaviour (factory is the trusted metrology reference, hence the default):

    * ``use_camera_calib=False`` (DEFAULT) -> use the factory ``(K, StereoCalib)``.
      No warning is emitted -- factory is the intended default, not an error state.
    * ``use_camera_calib=True`` and ``user_calib`` present -> OVERRIDE with the user's
      ``K`` (= left intrinsics) and their full :class:`StereoCalib`
      (intrinsics_left/right + ``T_left_right``), logged prominently. The IMU<->cam
      extrinsic ``R_imu_cam`` is NOT part of the wizard solve, so the caller keeps the
      factory ``R_imu_cam`` -- it is intentionally untouched here.
    * ``use_camera_calib=True`` but ``user_calib`` absent -> the operator asked for
      their calib but none is saved: emit ONE prominent WARNING and fall back to
      factory.

    Returns ``(K, StereoCalib)`` for the live graph to use.
    """
    if not use_camera_calib:
        # Default path: factory is the trusted reference. No store read, no warning --
        # this is the intended configuration, not a missing-calibration error.
        return factory_K, factory_calib
    if user_calib is not None:
        # The user's left intrinsics ARE the live K (the same K StereoCalib exposes
        # as ``calib.left.K``); rebuild it from the parsed calib so the two never
        # disagree, rather than trusting a separately-passed array.
        K = np.asarray(user_calib.left.K, dtype=np.float64)
        baseline_mm = user_calib.baseline_m * 1000.0
        print(f"[live] using SAVED camera calibration for device {dev_id} "
              f"(baseline {baseline_mm:.1f} mm) -- factory calib overridden",
              file=sys.stderr)
        return K, user_calib
    # Asked for the user calib but none is saved: fall back to factory, but tell the
    # operator their request could not be honoured so they can run the wizard.
    print(f"[live] WARNING: asked for user camera calib (--use-camera-calib) but "
          f"none saved for {dev_id} -- using factory; run the Camera (stereo) "
          f"calibration wizard.", file=sys.stderr)
    return factory_K, factory_calib


def read_live_calibration(device: SharedLiveDevice, *, width: int, height: int,
                          use_gyro: bool, depth_fast: bool,
                          recalibrate_bias: bool = False,
                          use_camera_calib: bool = False) -> LiveFrontEndCalib:
    """Acquire the shared device and read all VIO boot references.

    The device is :meth:`~imu_camera.mathlib.device.oak_live.SharedLiveDevice.acquire`-d here and
    kept open (the caller releases it when the run ends); the camera/IMU sources
    attach to the same reference-counted device when they start.
    """
    device.acquire()
    ch = device.read_calibration()
    K, calib, R_imu_cam = _read_stereo_calib(ch, width, height)
    res = ResolutionProfile.for_resolution(width, height)
    sgm_cfg = sgm_config(res, fast=depth_fast)

    # Read the per-device id up front so the calib bundle carries it regardless
    # of whether the gyro/IMU branch runs (the UI keys saved IMU calib by it).
    dev_id = device.device_id

    # FACTORY calib is the default trusted metrology reference. The operator's OWN
    # saved stereo calibration is applied ONLY when explicitly opted into via
    # ``--use-camera-calib`` -- in which case we read the per-device store (keyed by
    # the SAME id the wizard saved under) and override K + the StereoCalib
    # (intrinsics + L->R extrinsic); R_imu_cam stays factory (the wizard does not
    # calibrate it). In the default path we do NOT even read the store. This is
    # purely the LIVE device path -- the replay/oracle path reads its calib from the
    # recorded session and never reaches here.
    user_calib = load_camera_calib(dev_id) if use_camera_calib else None
    K, calib = select_camera_calib(dev_id, K, calib, user_calib,
                                   use_camera_calib=use_camera_calib)

    accel_align = None
    imu_calibration = None
    if use_gyro:
        cached = None if recalibrate_bias else load_gyro_bias(dev_id)
        accel_cal: AccelCalibration | None = load_accel_calib(dev_id)
        # Hold-still ONLY when the gyro bias must be measured (first run /
        # --recalibrate-bias). Once cached, take a quick non-gated gravity seed --
        # the continuous CorrectTilt in the odometry flow re-levels at rest, so the
        # operator never has to hold the camera still at Start.
        need_bias = cached is None
        accel_align, measured = _collect_startup(
            device, R_imu_cam, accel_cal, estimate_bias=need_bias,
            gate=need_bias, window_s=0.4 if need_bias else 0.2)
        gyro_bias = cached if cached is not None else measured
        if measured is not None and cached is None:
            try:
                p = save_gyro_bias(dev_id, measured, 0)
                print(f"[live] gyro bias calibrated {measured.round(5).tolist()} "
                      f"rad/s -> saved to {p}", file=sys.stderr)
            except OSError as e:
                print(f"[live] WARNING: could not save gyro bias: {e}",
                      file=sys.stderr)
        if gyro_bias is not None or accel_cal is not None:
            imu_calibration = ImuCalibration(gyro_bias=gyro_bias, accel=accel_cal)

    return LiveFrontEndCalib(K=K, calib=calib, R_imu_cam=R_imu_cam,
                             sgm_cfg=sgm_cfg, res=res, accel_align=accel_align,
                             imu_calibration=imu_calibration, device_id=dev_id)
