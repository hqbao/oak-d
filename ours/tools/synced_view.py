#!/usr/bin/env python3
"""Visualise the time-synced (image, depth, IMU) triplet from a recorded session.

This is the eyeball companion to :mod:`ours.vio.synced`: it replays a gold
session as a sequence of :class:`ours.vio.SyncedSample` bundles and shows, for
every frame, the three inputs the from-scratch VIO consumes -- all aligned on the
same device clock::

    [ rectified-left image | depth colormap | IMU panel ]

The IMU panel shows, derived ONLY from the real IMU samples that fall in this
frame's inter-frame interval ``(t_prev, t_cur]``:

  * **Gyro -> angular-velocity line chart.** The per-frame mean angular velocity
    (bias-subtracted, ``deg/s``) of all three axes scrolls as a 3-line chart
    (wx red, wy green, wz blue). "Integrated" here means *aggregated over the
    frame* -- the mean of the ~10 raw samples in the interval -- not raw
    per-sample noise. The gyro rate is a direct measurement, so this is honest
    raw signal (no dead-reckoning drift, unlike integrating it to an angle).
  * **Accel -> 3D vector.** The accelerometer samples in the interval are
    **averaged** into one specific-force vector (m/s^2) and drawn in a small 3D
    coordinate box (isometric) with the optical axes for reference, plus a true
    vertical line and the tilt-from-vertical angle; at rest it points along
    gravity with |a| ~ 9.8.

Frames with the camera extrinsics in ``calib.json`` express both in the camera
optical frame (x right, y down, z forward); older sessions without extrinsics
fall back to the raw IMU frame (the panel says which).

cv2 here is only a dev-tool display dependency (windowing + colormap), exactly
like the other ``tools/*`` viewers -- nothing here is in a production path.

Usage::

    python ours/tools/synced_view.py                                   # default gold
    python ours/tools/synced_view.py --session sessions/gold/lab_loop_30s
    python ours/tools/synced_view.py --scale 1.5 --no-bias             # raw gyro bias

    python ours/tools/synced_view.py --live                            # live OAK-D
    python ours/tools/synced_view.py --live --width 320 --height 200   # lighter live

Keys: SPACE pause/resume, ``n`` step one frame (paused), ``r`` clear the gyro
chart, ``q`` / ESC quit. (Live has no pause/step; ``r`` clear + ``q`` quit only.)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib import (  # noqa: E402
    SessionReader, SGMConfig, SGMStereoMatcher, StereoCalib, slice_imu,
)
from ours.lib.config.resolution import ResolutionProfile  # noqa: E402
from ours.lib.viz.depth_render import colorize_depth  # noqa: E402,F401

_G = 9.80665  # m/s^2, only used to scale the accel arrow to ~unit at rest


def _label(img: np.ndarray, text: str, y: int = 22) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _gray_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# Per-axis colours, shared by the gyro chart and the 3D accel reference axes.
# x = right (red), y = down (green), z = forward (blue)  -- BGR tuples.
_AX_X = (80, 80, 255)
_AX_Y = (80, 255, 80)
_AX_Z = (255, 120, 120)
_ISO_C = 0.8660254037844387  # cos(30 deg)


def _iso(v) -> tuple[float, float]:
    """Camera-optical 3D (x right, y down, z forward) -> isometric 2D offset.

    Standard 2:1-ish isometric: x and z are the two ground axes receding at
    +-30 deg, optical y (down) maps straight to screen-down. Gives a clean,
    readable 3D box without perspective foreshortening.
    """
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    sx = (x - z) * _ISO_C
    sy = (x + z) * 0.5 + y
    return sx, sy


def _arrow3d(canvas, center, vec3, length, color, label=None, thick=2):
    """Draw an isometric-projected 3D vector as an arrow from ``center``."""
    sx, sy = _iso(np.asarray(vec3, float))
    tip = (int(center[0] + sx * length), int(center[1] + sy * length))
    cv2.arrowedLine(canvas, center, tip, color, thick, cv2.LINE_AA,
                    tipLength=0.18)
    if label:
        cv2.putText(canvas, label, (tip[0] + 3, tip[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_gyro_chart(panel, x0, y0, w, h, hist, y_max):
    """Scrolling 3-line chart of per-frame mean angular velocity (deg/s).

    ``hist`` is an ``(T, 3)`` array of [wx, wy, wz] in deg/s (newest last); rows
    that are all-NaN (frames with no IMU samples) break the lines honestly.
    ``y_max`` sets the symmetric vertical scale (deg/s).
    """
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1)
    yz = y0 + h // 2
    cv2.line(panel, (x0, yz), (x0 + w, yz), (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(panel, f"+{y_max:.0f}", (x0 + 3, y0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(panel, f"-{y_max:.0f}", (x0 + 3, y0 + h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(panel, "deg/s", (x0 + w - 42, y0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (110, 110, 110), 1, cv2.LINE_AA)

    if hist is not None and len(hist) >= 1:
        hist = np.asarray(hist, float)
        T = len(hist)
        half = h / 2.0

        def _x(t):
            return int(x0 + (t / (T - 1) * w if T > 1 else w * 0.5))

        def _y(v):
            yy = yz - np.clip(v / y_max, -1.0, 1.0) * half
            return int(yy)

        for ax, color in ((0, _AX_X), (1, _AX_Y), (2, _AX_Z)):
            prev = None
            for t in range(T):
                v = hist[t, ax]
                if not np.isfinite(v):
                    prev = None  # break the line over a no-IMU frame
                    continue
                pt = (_x(t), _y(v))
                if prev is not None:
                    cv2.line(panel, prev, pt, color, 1, cv2.LINE_AA)
                prev = pt

        # current (latest finite) value per axis, as a legend.
        last = np.full(3, np.nan)
        for t in range(T - 1, -1, -1):
            if np.isfinite(hist[t]).all():
                last = hist[t]
                break
        for k, (name, color) in enumerate(
                (("wx", _AX_X), ("wy", _AX_Y), ("wz", _AX_Z))):
            val = f"{last[k]:+6.1f}" if np.isfinite(last[k]) else "  --  "
            cv2.putText(panel, f"{name} {val}", (x0 + 6 + k * 86, y0 + h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _draw_accel_3d(panel, cx, cy, scale_px, accel_vec):
    """Draw the averaged accel as a 3D vector in an isometric reference box."""
    # Faint reference axes (optical x/y/z) so the vector's direction is legible.
    ref = scale_px * 0.7
    _arrow3d(panel, (cx, cy), (1, 0, 0), ref, (60, 60, 110), "x", 1)
    _arrow3d(panel, (cx, cy), (0, 1, 0), ref, (60, 110, 60), "y", 1)
    _arrow3d(panel, (cx, cy), (0, 0, 1), ref, (110, 60, 60), "z", 1)
    # True vertical (anti-gravity = optical -y): at rest the accel sits here.
    upx, upy = _iso((0, -1, 0))
    cv2.line(panel, (cx, cy),
             (int(cx + upx * scale_px), int(cy + upy * scale_px)),
             (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(panel, "up", (int(cx + upx * scale_px) + 3,
                              int(cy + upy * scale_px)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1, cv2.LINE_AA)
    mag = float(np.linalg.norm(accel_vec))
    if np.isfinite(accel_vec).all() and mag > 1e-6:
        _arrow3d(panel, (cx, cy), accel_vec / _G, scale_px, (60, 220, 255),
                 "a", 3)


def render_imu_panel(size: int, gyro_hist: np.ndarray, accel_vec: np.ndarray,
                     n_imu: int, frame_str: str,
                     y_max: float | None = None) -> np.ndarray:
    """Render the IMU panel: gyro angular-velocity chart + 3D accel vector.

    ``gyro_hist`` is an ``(T, 3)`` rolling history of per-frame mean angular
    velocity (deg/s); ``accel_vec`` the averaged specific force (m/s^2) for the
    current frame. Both are real IMU-derived values -- the gyro rate is a direct
    measurement (no dead-reckoning drift) and the accel is the plain average.
    """
    s = size
    panel = np.full((s, s, 3), 24, dtype=np.uint8)

    # --- top half: gyro angular-velocity line chart --------------------------
    cv2.putText(panel, "GYRO -> angular velocity (deg/s)", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cx0, cy0 = 8, 24
    cw, chh = s - 16, s // 2 - 34
    if y_max is None:
        if gyro_hist is not None and len(gyro_hist) and \
                np.isfinite(gyro_hist).any():
            peak = float(np.nanmax(np.abs(gyro_hist)))
        else:
            peak = 0.0
        # round up to a tidy scale, floor of 20 deg/s so noise looks small.
        y_max = max(20.0, float(np.ceil(peak / 20.0) * 20.0))
    _draw_gyro_chart(panel, cx0, cy0, cw, chh, gyro_hist, y_max)

    # --- bottom half: 3D accel vector ----------------------------------------
    cv2.putText(panel, "ACCEL -> 3D vector (m/s^2)", (8, s // 2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    _draw_accel_3d(panel, s // 2, 3 * s // 4 - 12, s * 0.16, accel_vec)

    mag = float(np.linalg.norm(accel_vec))
    if np.isfinite(accel_vec).all() and mag > 1e-6:
        tilt = float(np.degrees(np.arccos(
            np.clip(-accel_vec[1] / mag, -1.0, 1.0))))
        cv2.putText(panel, f"tilt {tilt:4.1f} deg from vertical",
                    (8, s - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (60, 220, 255), 1, cv2.LINE_AA)
        atxt = (f"a [{accel_vec[0]:+5.2f} {accel_vec[1]:+5.2f} "
                f"{accel_vec[2]:+5.2f}]  |a| {mag:5.2f}")
    else:
        atxt = "a  (no IMU samples this frame)"
    cv2.putText(panel, atxt, (8, s - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{frame_str}   imu_n {n_imu}", (8, s - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 200, 150), 1, cv2.LINE_AA)
    return panel


def precompute_imu(reader: SessionReader, use_bias: bool):
    """Per-frame mean angular velocity (deg/s) + averaged accel, in the cam frame.

    For each frame interval ``(t_prev, t_cur]`` the gyro samples are averaged
    (bias-subtracted) into one angular-velocity vector and the accel samples into
    one specific-force vector, both rotated into the camera optical frame (or the
    raw IMU frame if the session has no extrinsics). The gyro bias is the mean of
    the first ~1 s (the static-startup assumption); ``--no-bias`` keeps it raw.
    Returns ``(ang_vel_dps, accels, counts, frame)`` arrays of length
    ``len(reader)`` plus a human-readable frame label.
    """
    imu = reader.load_imu()
    ts_i, gyro, accel = imu["ts_ns"], imu["gyro"], imu["accel"]

    T_imu_cam = reader.calib.T_imu_left
    if T_imu_cam is not None:
        R_imu_cam = np.asarray(T_imu_cam, float)[:3, :3]
        frame = "cam frame"
    else:
        R_imu_cam = np.eye(3)
        frame = "IMU frame"

    gyro_bias = np.zeros(3)
    if use_bias and len(ts_i):
        win = ts_i <= (ts_i[0] + int(1e9))  # first 1 s
        if win.any():
            gyro_bias = gyro[win].mean(axis=0)

    frame_ts = [int(rec["ts_ns"]) for rec in reader._frames]
    ang_vel, accels, counts = [], [], []
    for i, t in enumerate(frame_ts):
        t0 = frame_ts[0] if i == 0 else frame_ts[i - 1]
        seg = slice_imu(ts_i, gyro, accel, t0, t, bracket=False)
        counts.append(len(seg))
        if len(seg):
            w_cam = R_imu_cam @ (seg.gyro.mean(axis=0) - gyro_bias)
            ang_vel.append(np.degrees(w_cam))
            accels.append(R_imu_cam @ seg.accel.mean(axis=0))
        else:
            ang_vel.append(np.full(3, np.nan))
            accels.append(np.full(3, np.nan))
    return (np.array(ang_vel), np.array(accels),
            np.array(counts, dtype=int), frame)


def run(session_dir: Path, fps: float, scale: float, use_bias: bool) -> int:
    reader = SessionReader(session_dir)
    if len(reader) == 0:
        print(f"no frames in {session_dir}")
        return 1
    ang_vel, accels, counts, frame = precompute_imu(reader, use_bias)
    print(f"session {reader.dir.name}: {len(reader)} frames, "
          f"IMU in {frame}, "
          f"gyro bias {'startup-estimate' if use_bias else 'OFF (raw)'}")
    print("keys: SPACE pause | n step | r clear chart | q quit")

    win = "synced_view  [ image | depth | IMU ]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    chart_w = 120          # rolling gyro-chart window (frames)
    i = 0
    chart_start = 0        # advanced by the "clear chart" key
    paused = False
    period = 1.0 / max(fps, 1e-3)

    while 0 <= i < len(reader):
        t0 = time.perf_counter()
        fr = reader.load_frame(i)
        H, W = fr.gray_left.shape

        lo = max(chart_start, i - chart_w + 1)
        hist = ang_vel[lo:i + 1]
        panel_imu = render_imu_panel(
            H, hist, accels[i], n_imu=int(counts[i]),
            frame_str=f"seq {fr.seq}  t {fr.ts_s:6.2f}s  {frame}")

        left = _label(_gray_bgr(fr.gray_left), "image (rectified-left)")
        valid = float((fr.depth_m > 1e-6).mean()) * 100.0
        depth = _label(colorize_depth(fr.depth_m), f"depth  valid {valid:.0f}%")
        panel = np.hstack([left, depth, panel_imu])
        if scale != 1.0:
            panel = cv2.resize(panel, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)
        cv2.imshow(win, panel)

        wait = max(1, int((period - (time.perf_counter() - t0)) * 1000)) \
            if not paused else 0
        key = cv2.waitKey(0 if paused else wait) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("r"):
            chart_start = i  # clear the chart history from here
            continue
        if paused and key == ord("n"):
            i += 1
            continue
        if not paused:
            i += 1
    cv2.destroyAllWindows()
    return 0


def run_live(width: int, height: int, fps: int, scale: float,
             use_bias: bool, fast: bool, bias_window_s: float = 1.0) -> int:
    """Live (image, depth, IMU) triplet from a connected OAK-D.

    Mirrors the VPU-free live VIO input exactly (ours.legacy.depthai_ours_vio):
    taps the two RAW cameras + the IMU, rectifies BOTH frames and runs our SGM
    ourselves (no chip StereoDepth), and integrates/averages the IMU the same way
    the VIO does -- so the triplet shown here is the real pipeline input.

    The gyro samples drained each display frame are **averaged** into one
    angular-velocity vector (bias-subtracted, camera frame) and scrolled on a
    3-axis line chart; the accel samples are averaged into one specific-force
    vector shown in 3D -- so the triplet shown here is the real pipeline input.
    """
    import depthai as dai  # lazy: replay mode works without depthai
    from collections import deque

    left_socket = dai.CameraBoardSocket.CAM_B
    right_socket = dai.CameraBoardSocket.CAM_C
    res = ResolutionProfile.for_resolution(width, height)
    cfg = res.sgm(fast=fast)
    win = "synced_view LIVE  [ image | depth | IMU ]"

    with dai.Pipeline() as p:
        left = p.create(dai.node.Camera).build(left_socket, sensorFps=fps)
        right = p.create(dai.node.Camera).build(right_socket, sensorFps=fps)
        imu = p.create(dai.node.IMU)
        imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW,
                             dai.IMUSensor.GYROSCOPE_RAW], 200)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)

        left_out = left.requestOutput((width, height))
        right_out = right.requestOutput((width, height))
        q_left = left_out.createOutputQueue(maxSize=4, blocking=False)
        q_right = right_out.createOutputQueue(maxSize=4, blocking=False)
        q_imu = imu.out.createOutputQueue(maxSize=50, blocking=False)
        p.start()

        ch = p.getDefaultDevice().readCalibration()

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
        matcher = SGMStereoMatcher.from_calib(calib, cfg, rectify_left=True)

        # IMU->left-camera rotation: brings gyro/accel into the camera optical
        # frame so the triad/quaternion match the VIO convention. Identity (and
        # an "IMU frame" label) if the device has no extrinsics.
        try:
            R_imu_cam = np.array(
                ch.getImuToCameraExtrinsics(left_socket), dtype=np.float64
            )[:3, :3]
            frame = "cam frame"
        except Exception:
            R_imu_cam = np.eye(3)
            frame = "IMU frame"

        print(f"live {width}x{height}@{fps}  SGM ndisp={cfg.num_disparities} "
              f"downscale={cfg.downscale}  IMU in {frame}")
        print(f"compiling SGM kernels (one-time JIT)...", flush=True)
        dummy = np.zeros((height, width), np.uint8)
        matcher.dense_depth(dummy, dummy)
        print("ready", flush=True)

        # --- estimate the gyro zero-rate bias from a short static window -------
        # Same assumption the VIO makes: the device is held still at startup, so
        # the mean gyro over the first second is its bias. Skipped with --no-bias.
        gyro_bias = np.zeros(3)
        if use_bias:
            print(f"hold STILL ~{bias_window_s:.0f}s for gyro bias...",
                  flush=True)
            buf, t_start = [], time.monotonic()
            while time.monotonic() - t_start < bias_window_s:
                msg = q_imu.tryGet()
                if msg is None:
                    cv2.waitKey(1)
                    continue
                for pkt in msg.packets:
                    g = pkt.gyroscope
                    w = np.array([g.x, g.y, g.z], np.float64)
                    if np.all(np.isfinite(w)):
                        buf.append(w)
            if buf:
                gyro_bias = np.mean(buf, axis=0)
            print(f"gyro bias = {gyro_bias.round(5)} rad/s "
                  f"({len(buf)} samples)", flush=True)

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.startWindowThread()
        print("keys: r clear chart | q quit")

        def _as_gray(msg):
            g = msg.getCvFrame()
            if g.ndim == 3:
                g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
            return g

        pend_l: dict[int, np.ndarray] = {}
        pend_r: dict[int, np.ndarray] = {}
        gyro_hist: deque = deque(maxlen=120)  # rolling angular velocity (deg/s)
        shown_a_frame = False
        t_run0 = time.monotonic()

        while p.isRunning():
            got = False
            while True:
                m = q_left.tryGet()
                if m is None:
                    break
                pend_l[m.getSequenceNum()] = _as_gray(m)
                got = True
            while True:
                m = q_right.tryGet()
                if m is None:
                    break
                pend_r[m.getSequenceNum()] = _as_gray(m)
                got = True

            # Drain the IMU EVERY iteration: average the gyro into one angular
            # velocity (bias-subtracted, rotated to the camera frame) and the
            # accel into one specific force over this display frame.
            gyro_sum = np.zeros(3)
            gyro_n = 0
            acc_sum = np.zeros(3)
            acc_n = 0
            msg = q_imu.tryGet()
            while msg is not None:
                for pkt in msg.packets:
                    a = pkt.acceleroMeter
                    v = np.array([a.x, a.y, a.z], np.float64)
                    if np.all(np.isfinite(v)):
                        acc_sum += v
                        acc_n += 1
                    g = pkt.gyroscope
                    w = np.array([g.x, g.y, g.z], np.float64)
                    if np.all(np.isfinite(w)):
                        gyro_sum += w
                        gyro_n += 1
                msg = q_imu.tryGet()
            # Mean angular velocity (deg/s) and mean specific force in cam frame.
            if gyro_n > 0:
                ang_vel = np.degrees(R_imu_cam @ (gyro_sum / gyro_n - gyro_bias))
            else:
                ang_vel = np.full(3, np.nan)
            accel_cam = (R_imu_cam @ (acc_sum / acc_n)
                         if acc_n else np.full(3, np.nan))

            common = pend_l.keys() & pend_r.keys()
            if common:
                seq = max(common)
                gl, gr = pend_l[seq], pend_r[seq]
                pend_l = {k: v for k, v in pend_l.items() if k > seq}
                pend_r = {k: v for k, v in pend_r.items() if k > seq}
                rect_left, ours = matcher.dense_depth_rectified_left(gl, gr)
                disp_left = np.clip(rect_left, 0, 255).astype(np.uint8)

                gyro_hist.append(ang_vel)
                panel_imu = render_imu_panel(
                    height, np.array(gyro_hist), accel_cam, n_imu=acc_n,
                    frame_str=f"seq {seq}  t {time.monotonic()-t_run0:6.1f}s  "
                              f"{frame}")
                valid = float((ours > 1e-6).mean()) * 100.0
                left_lbl = _label(_gray_bgr(disp_left), "image (rect-left)")
                depth_lbl = _label(colorize_depth(ours),
                                   f"depth  valid {valid:.0f}%")
                panel = np.hstack([left_lbl, depth_lbl, panel_imu])
                if scale != 1.0:
                    panel = cv2.resize(panel, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_NEAREST)
                cv2.imshow(win, panel)
                shown_a_frame = True
            elif not shown_a_frame:
                placeholder = np.zeros((height, width * 2 + height, 3),
                                       dtype=np.uint8)
                cv2.imshow(win, _label(placeholder, "waiting for camera..."))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                gyro_hist.clear()
            if not got:
                time.sleep(0.002)
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path,
                    default=Path("sessions/gold/corridor_60s"),
                    help="recorded session directory to replay")
    ap.add_argument("--live", action="store_true",
                    help="pull from a connected OAK-D instead of replaying")
    ap.add_argument("--fast", action="store_true",
                    help="(live) use the half-res SGM preset -- faster")
    ap.add_argument("--width", type=int, default=640,
                    help="(live) capture width [640]")
    ap.add_argument("--height", type=int, default=400,
                    help="(live) capture height [400]")
    ap.add_argument("--fps", type=float, default=20.0,
                    help="replay speed / live frame-rate cap (frames/second)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="upscale the whole panel for visibility")
    ap.add_argument("--no-bias", action="store_true",
                    help="do NOT subtract the startup gyro bias (show raw drift)")
    args = ap.parse_args()
    if args.live:
        return run_live(args.width, args.height, int(args.fps), args.scale,
                        use_bias=not args.no_bias, fast=args.fast)
    return run(args.session, args.fps, args.scale, use_bias=not args.no_bias)


if __name__ == "__main__":
    raise SystemExit(main())
