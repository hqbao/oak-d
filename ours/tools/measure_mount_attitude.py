#!/usr/bin/env python3
"""Measure the camera's mounted (static) attitude to anchor it permanently.

The camera is held still in its final mount pose (e.g. bolted to the drone).
This tool reads the IMU for a few seconds and reports the gravity-leveled
initial attitude (roll/pitch; yaw is unobservable without a magnetometer).

It prefers the IMU's on-chip **fused** outputs when available, because they are
far cleaner than raw accelerometer samples:

  * ``ROTATION_VECTOR`` / ``GAME_ROTATION_VECTOR`` -> an absolute quaternion
    (gravity-referenced tilt, drift-free). BNO08x-class IMUs expose this.
  * ``GRAVITY`` -> a pre-filtered gravity vector.

If the IMU is a raw-only part (e.g. BMI270, common on some OAK-D models) we fall
back to **averaging the raw accelerometer** over the whole window. For a static
measurement the noise is zero-mean, so averaging thousands of samples gives a
clean gravity estimate anyway.

Output: the gravity direction in the camera optical frame, the resulting
roll/pitch, and the 3x3 ``R0`` (camera->world) you can hard-code to anchor the
mount. Run it with the camera held in the exact mount orientation.

Usage::

    python ours/tools/measure_mount_attitude.py            # ~4 s capture
    python ours/tools/measure_mount_attitude.py --secs 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.frames import quat_to_rpy  # noqa: E402
from ours.lib.imu.imu import gravity_aligned_R0  # noqa: E402


def _quat_to_rot(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--secs", type=float, default=4.0,
                    help="static capture duration in seconds")
    ap.add_argument("--rate", type=int, default=200, help="IMU report rate (Hz)")
    args = ap.parse_args()

    import depthai as dai

    left_socket = dai.CameraBoardSocket.CAM_B

    # Sensor combos to try, best (fused + raw for cross-check) first. The device
    # rejects some combinations (e.g. GRAVITY+ACCEL), so we fall back in order.
    def S(name):
        return getattr(dai.IMUSensor, name, None)

    combos = [
        ["ROTATION_VECTOR", "ACCELEROMETER_RAW"],
        ["GAME_ROTATION_VECTOR", "ACCELEROMETER_RAW"],
        ["ROTATION_VECTOR"],
        ["GAME_ROTATION_VECTOR"],
        ["ACCELEROMETER_RAW"],
    ]

    accel_samples: list[np.ndarray] = []
    rot_grav_samples: list[np.ndarray] = []
    seen: set[str] = set()
    R_imu_cam = np.eye(3)

    captured = False
    for names in combos:
        sensors = [S(n) for n in names if S(n) is not None]
        if not sensors:
            continue
        try:
            with dai.Pipeline() as p:
                imu = p.create(dai.node.IMU)
                imu.enableIMUSensor(sensors, args.rate)
                imu.setBatchReportThreshold(1)
                imu.setMaxBatchReports(20)
                q_imu = imu.out.createOutputQueue(maxSize=100, blocking=False)
                p.start()

                ch = p.getDefaultDevice().readCalibration()
                try:
                    R_imu_cam = np.array(
                        ch.getImuToCameraExtrinsics(left_socket), dtype=np.float64
                    )[:3, :3]
                except Exception:
                    R_imu_cam = np.eye(3)
                    print("!! no IMU->cam extrinsics; assuming identity")

                print(f"Using IMU sensors: {names}")
                print(f"Hold the camera STILL in its mount pose. "
                      f"Capturing {args.secs:.0f} s ...")
                t_start = time.monotonic()
                while time.monotonic() - t_start < args.secs:
                    msg = q_imu.tryGet()
                    if msg is None:
                        time.sleep(0.003)
                        continue
                    for pkt in msg.packets:
                        a = getattr(pkt, "acceleroMeter", None)
                        if a is not None and (a.x or a.y or a.z):
                            accel_samples.append(
                                np.array([a.x, a.y, a.z], dtype=np.float64))
                            seen.add("accel")
                        rv = getattr(pkt, "rotationVector", None)
                        if rv is not None and (rv.i or rv.j or rv.k or rv.real):
                            R = _quat_to_rot(rv.real, rv.i, rv.j, rv.k)
                            g_world_up = np.array([0.0, 0.0, 1.0])
                            g_imu = R.T @ (-g_world_up)
                            rot_grav_samples.append(R_imu_cam @ g_imu)
                            seen.add("rotvec")
            captured = True
            break
        except RuntimeError as e:
            print(f"  (combo {names} rejected: {e})")
            continue

    if not captured:
        print("!! Could not start the IMU with any sensor combo.")
        return 1

    print(f"\nSensors that produced data: {sorted(seen) or 'NONE'}")

    # Choose the cleanest available gravity-in-cam estimate.
    def report(label, g_cam, extra=""):
        g_cam = np.asarray(g_cam, dtype=np.float64)
        R0 = gravity_aligned_R0(g_cam)
        # express startup attitude in NED for human-readable roll/pitch/yaw
        M = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0]])
        P = np.array([[0, 1, 0.0], [0, 0, 1], [1, 0, 0]])
        Rned = M @ R0 @ P
        # matrix -> quat -> rpy
        from ours.depthai_ours_vio import _rot_to_quat_wxyz
        r, pi, y = np.degrees(quat_to_rpy(_rot_to_quat_wxyz(Rned)))
        gdir = g_cam / (np.linalg.norm(g_cam) + 1e-12)
        print(f"\n[{label}] {extra}")
        print(f"  gravity-in-cam (optical, unit) = [{gdir[0]:+.3f} {gdir[1]:+.3f} {gdir[2]:+.3f}]")
        print(f"  startup attitude  roll={r:+6.1f}  pitch={pi:+6.1f}  yaw={y:+6.1f} (deg)")
        print("  R0 (camera->world) to anchor:")
        for row in R0:
            print("    [{:+.6f}, {:+.6f}, {:+.6f}],".format(*row))
        return R0

    results = {}
    if accel_samples:
        A = np.array(accel_samples)
        g_cam = R_imu_cam @ A.mean(axis=0)
        std = A.std(axis=0)
        results["accel"] = report(
            "AVERAGED RAW ACCEL", g_cam,
            extra=f"(n={len(A)}, |g|={np.linalg.norm(A.mean(0)):.3f}, "
                  f"per-axis std={std[0]:.3f}/{std[1]:.3f}/{std[2]:.3f} m/s^2)")
    if rot_grav_samples:
        Rg = np.array(rot_grav_samples)
        g_cam = Rg.mean(axis=0)
        results["rotvec"] = report(
            "FUSED ROTATION_VECTOR", g_cam, extra=f"(n={len(Rg)}, on-chip fusion)")

    if not results:
        print("\n!! No usable IMU data. Is the IMU enabled / device connected?")
        return 1

    # Cross-check agreement between methods, if more than one is available.
    if len(results) > 1:
        print("\n--- agreement between methods (angle between gravity dirs) ---")
        keys = list(results.keys())
        # recompute gravity dirs from R0 (world down = R0 @ [0,1,0]... simpler:
        # re-derive from each R0's second world axis). Instead compare R0s.
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                Ri, Rj = results[keys[i]], results[keys[j]]
                dR = Ri @ Rj.T
                ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
                print(f"  {keys[i]:8s} vs {keys[j]:8s}: {ang:.2f} deg")

    print("\nRecommended source: "
          + ("FUSED ROTATION_VECTOR" if "rotvec" in results
             else "FUSED GRAVITY" if "gravity" in results
             else "AVERAGED RAW ACCEL (only option; fine for a static anchor)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
