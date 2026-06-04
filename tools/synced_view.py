#!/usr/bin/env python3
"""Visualise the time-synced (image, depth, IMU) triplet from a recorded session.

This is the eyeball companion to :mod:`oakd.vio.synced`: it replays a gold
session as a sequence of :class:`oakd.vio.SyncedSample` bundles and shows, for
every frame, the three inputs the from-scratch VIO consumes -- all aligned on the
same device clock::

    [ rectified-left image | depth colormap | IMU panel ]

The IMU panel shows exactly what the prompt asked for, derived ONLY from the real
IMU samples that fall in this frame's inter-frame interval ``(t_prev, t_cur]``:

  * **Gyro -> integrated quaternion.** The gyro is integrated (the same
    :class:`oakd.vio.GyroPreintegrator` the VIO uses) into a *running* attitude
    since frame 0, shown as a unit quaternion ``(w,x,y,z)`` + roll/pitch/yaw and a
    little 3-axis triad you can watch rotate. This is **gyro-only dead reckoning**
    (no accel/vision fusion) so it WILL slowly drift -- that is the honest raw
    signal, labelled as such.
  * **Accel -> averaged vector.** The accelerometer samples in the interval are
    **averaged** into one specific-force vector (m/s^2), drawn as an arrow and
    printed numerically; at rest it points along gravity with |a| ~ 9.8.

Frames with the camera extrinsics in ``calib.json`` express both in the camera
optical frame (x right, y down, z forward); older sessions without extrinsics
fall back to the raw IMU frame (the panel says which).

cv2 here is only a dev-tool display dependency (windowing + colormap), exactly
like the other ``tools/*`` viewers -- nothing here is in a production path.

Usage::

    python tools/synced_view.py                                   # default gold
    python tools/synced_view.py --session sessions/gold/lab_loop_30s
    python tools/synced_view.py --scale 1.5 --no-bias             # raw gyro bias

    python tools/synced_view.py --live                            # live OAK-D
    python tools/synced_view.py --live --width 320 --height 200   # lighter live

Keys: SPACE pause/resume, ``n`` step one frame (paused), ``r`` reset attitude,
``q`` / ESC quit. (Live has no pause/step; ``r`` reset + ``q`` quit only.)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oakd.frames import quat_to_rot, quat_to_rpy, rot_to_quat  # noqa: E402
from oakd.vio import (  # noqa: E402
    GyroPreintegrator, SessionReader, SGMConfig, SGMStereoMatcher,
    StereoCalib, slice_imu,
)
from oakd.vio.resolution import ResolutionProfile  # noqa: E402

# Fixed depth range (metres) for the colormap so colours are stable across
# frames (a per-frame autoscale makes the scene "breathe" and hides drift).
_D_MIN = 0.3
_D_MAX = 8.0
_G = 9.80665  # m/s^2, only used to scale the accel arrow to ~unit at rest


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """Metric depth (m, 0 == invalid) -> BGR turbo image (near = red)."""
    valid = depth_m > 1e-6
    norm = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid.any():
        z = np.clip(depth_m, _D_MIN, _D_MAX)
        t = 1.0 - (z - _D_MIN) / (_D_MAX - _D_MIN)  # near = hot
        norm[valid] = (t[valid] * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _label(img: np.ndarray, text: str, y: int = 22) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _gray_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _project(v: np.ndarray) -> tuple[float, float]:
    """Camera-optical 3D (x right, y down, z forward) -> 2D screen offset.

    Simple oblique projection: forward (+z) recedes up-and-left so the triad
    reads as 3D. Screen y already grows downward, matching the optical +y (down).
    """
    sx = v[0] - 0.45 * v[2]
    sy = v[1] - 0.45 * v[2]
    return float(sx), float(sy)


def _arrow(canvas, center, vec3, length, color, label=None, thick=2):
    """Draw an oblique-projected 3D vector as an arrow from ``center``."""
    sx, sy = _project(np.asarray(vec3, float))
    tip = (int(center[0] + sx * length), int(center[1] + sy * length))
    cv2.arrowedLine(canvas, center, tip, color, thick, cv2.LINE_AA,
                    tipLength=0.18)
    if label:
        cv2.putText(canvas, label, (tip[0] + 3, tip[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def render_imu_panel(size: int, R_cum: np.ndarray, q_wxyz: np.ndarray,
                     delta_deg: float, accel_vec: np.ndarray,
                     n_imu: int, frame_str: str) -> np.ndarray:
    """Render the IMU panel: integrated-gyro triad/quaternion + averaged accel.

    ``R_cum`` is the running gyro-integrated attitude (its columns are the body
    axes); ``q_wxyz`` its quaternion; ``delta_deg`` the rotation magnitude over
    THIS frame's interval; ``accel_vec`` the averaged specific force (m/s^2) in
    the same frame as the triad. All quantities are real IMU-derived values.
    """
    s = size
    panel = np.full((s, s, 3), 24, dtype=np.uint8)

    # Two stacked viz cells: top = attitude triad, bottom = accel arrow.
    top_c = (s // 4 + 30, s // 4 + 6)
    bot_c = (3 * s // 4 - 10, 3 * s // 4 - 30)
    axis_len = s * 0.16

    # --- top: gyro-integrated attitude triad (body axes in the world) ---------
    cv2.putText(panel, "GYRO -> integrated quaternion (gyro-only, drifts)",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1,
                cv2.LINE_AA)
    # Faint reference triad (identity) so drift/rotation is visible against it.
    for col, color in ((0, (60, 60, 120)), (1, (60, 120, 60)),
                       (2, (120, 60, 60))):
        e = np.zeros(3)
        e[col] = 1.0
        _arrow(panel, top_c, e, axis_len, color, thick=1)
    # Live integrated triad (bright): columns of R_cum = body x/y/z in world.
    _arrow(panel, top_c, R_cum[:, 0], axis_len, (80, 80, 255), "x", 2)
    _arrow(panel, top_c, R_cum[:, 1], axis_len, (80, 255, 80), "y", 2)
    _arrow(panel, top_c, R_cum[:, 2], axis_len, (255, 120, 120), "z", 2)

    roll, pitch, yaw = np.degrees(quat_to_rpy(q_wxyz))
    lines_top = [
        f"q w {q_wxyz[0]:+.3f}",
        f"  x {q_wxyz[1]:+.3f}",
        f"  y {q_wxyz[2]:+.3f}",
        f"  z {q_wxyz[3]:+.3f}",
        f"rpy {roll:+6.1f} {pitch:+6.1f} {yaw:+6.1f}",
        f"dframe {delta_deg:5.2f} deg",
    ]
    for k, ln in enumerate(lines_top):
        cv2.putText(panel, ln, (8, 40 + k * 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (230, 230, 230), 1, cv2.LINE_AA)

    # --- bottom: averaged accel vector ---------------------------------------
    cv2.putText(panel, "ACCEL -> averaged vector (m/s^2)", (8, s // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    mag = float(np.linalg.norm(accel_vec))
    if np.isfinite(accel_vec).all() and mag > 1e-6:
        _arrow(panel, bot_c, accel_vec / _G, s * 0.18, (60, 220, 255),
               "a", 3)
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
    """Per-frame gyro-integrated attitude + averaged accel, in the triad frame.

    Reuses :class:`GyroPreintegrator` (camera-frame inter-frame rotation, with the
    startup-window bias estimate the VIO uses) so the displayed attitude matches
    the pipeline's own convention. Returns ``(quats, deltas_deg, accels, frame)``
    arrays of length ``len(reader)`` plus a human-readable frame label.
    """
    imu = reader.load_imu()
    ts_i, gyro, accel = imu["ts_ns"], imu["gyro"], imu["accel"]

    T_imu_cam = reader.calib.T_imu_left
    if T_imu_cam is not None:
        R_imu_cam = np.asarray(T_imu_cam, float)[:3, :3]
        frame = "cam frame"
    else:
        T_imu_cam = np.eye(4)
        R_imu_cam = np.eye(3)
        frame = "IMU frame"

    preint = GyroPreintegrator(
        ts_i, gyro, T_imu_cam,
        gyro_bias=None if use_bias else np.zeros(3),
        estimate_bias_window_s=1.0 if use_bias else 0.0)

    frame_ts = [int(rec["ts_ns"]) for rec in reader._frames]
    R = np.eye(3)
    quats, deltas, accels, counts = [], [], [], []
    for i, t in enumerate(frame_ts):
        t0 = frame_ts[0] if i == 0 else frame_ts[i - 1]
        dR = preint.delta_rotation(t0, t)
        R = R @ dR
        # numerical hygiene: re-orthonormalise so long runs never leave SO(3)
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        quats.append(rot_to_quat(R))
        ang = float(np.degrees(np.linalg.norm(
            cv2.Rodrigues(dR)[0].ravel())))
        deltas.append(ang)
        seg = slice_imu(ts_i, gyro, accel, t0, t, bracket=False)
        counts.append(len(seg))
        if len(seg):
            accels.append(R_imu_cam @ seg.accel.mean(axis=0))
        else:
            accels.append(np.full(3, np.nan))
    return (np.array(quats), np.array(deltas), np.array(accels),
            np.array(counts, dtype=int), frame)


def run(session_dir: Path, fps: float, scale: float, use_bias: bool) -> int:
    reader = SessionReader(session_dir)
    if len(reader) == 0:
        print(f"no frames in {session_dir}")
        return 1
    quats, deltas, accels, counts, frame = precompute_imu(reader, use_bias)
    print(f"session {reader.dir.name}: {len(reader)} frames, "
          f"IMU triad in {frame}, "
          f"gyro bias {'startup-estimate' if use_bias else 'OFF (raw)'}")
    print("keys: SPACE pause | n step | r reset attitude | q quit")

    win = "synced_view  [ image | depth | IMU ]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    i = 0
    paused = False
    period = 1.0 / max(fps, 1e-3)
    q0 = quats[0].copy()  # reference for the "reset attitude" key

    while 0 <= i < len(reader):
        t0 = time.perf_counter()
        fr = reader.load_frame(i)
        H, W = fr.gray_left.shape

        # attitude relative to the (optionally reset) reference frame
        R_cum = quat_to_rot(q0).T @ quat_to_rot(quats[i])
        q_show = rot_to_quat(R_cum)
        panel_imu = render_imu_panel(
            H, R_cum, q_show, float(deltas[i]), accels[i],
            n_imu=int(counts[i]),
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
            q0 = quats[i].copy()
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

    Mirrors the VPU-free live VIO input exactly (oakd.sources.depthai_ours_vio):
    taps the two RAW cameras + the IMU, rectifies BOTH frames and runs our SGM
    ourselves (no chip StereoDepth), and integrates/averages the IMU the same way
    the VIO does -- so the triplet shown here is the real pipeline input.

    The gyro is integrated **live** into a running camera-frame attitude (the
    same ``so3_exp`` recursion the VIO uses), shown as a quaternion + triad; the
    accel samples drained each frame are averaged into one specific-force vector.
    """
    import depthai as dai  # lazy: replay mode works without depthai
    from oakd.vio.imu import so3_exp

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
              f"downscale={cfg.downscale}  IMU triad in {frame}")
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
        print("keys: r reset attitude | q quit")

        def _as_gray(msg):
            g = msg.getCvFrame()
            if g.ndim == 3:
                g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
            return g

        pend_l: dict[int, np.ndarray] = {}
        pend_r: dict[int, np.ndarray] = {}
        R_cum = np.eye(3)            # running gyro-integrated attitude (cam frame)
        R_ref = np.eye(3)            # reference for the "reset attitude" key
        gyro_last_ts: float | None = None
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

            # Drain the IMU EVERY iteration: integrate gyro into the running
            # attitude (per-sample dt, bias-subtracted, conjugated to the camera
            # frame) and accumulate accel to average over this display frame.
            R_imu_step = np.eye(3)
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
                        try:
                            ts = g.getTimestampDevice().total_seconds()
                        except Exception:
                            ts = None
                        if ts is not None:
                            if gyro_last_ts is not None:
                                dt = ts - gyro_last_ts
                                if 0.0 < dt < 0.1:
                                    R_imu_step = R_imu_step @ so3_exp(
                                        (w - gyro_bias) * dt)
                                    gyro_n += 1
                            gyro_last_ts = ts
                msg = q_imu.tryGet()
            # Conjugate the IMU-frame increment into the camera frame, advance
            # the running attitude.
            if gyro_n > 0:
                R_cum = R_cum @ (R_imu_cam @ R_imu_step @ R_imu_cam.T)
                U, _, Vt = np.linalg.svd(R_cum)
                R_cum = U @ Vt
            delta_deg = float(np.degrees(np.linalg.norm(
                cv2.Rodrigues(R_imu_step)[0].ravel()))) if gyro_n else 0.0
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

                R_show = R_ref.T @ R_cum
                q_show = rot_to_quat(R_show)
                panel_imu = render_imu_panel(
                    height, R_show, q_show, delta_deg, accel_cam,
                    n_imu=acc_n,
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
                R_ref = R_cum.copy()
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
