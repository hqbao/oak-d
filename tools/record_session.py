#!/usr/bin/env python3
"""Record a baseline session: dump C0/C1/C2/C3 streams from the OAK-D.

Same pipeline as ``oakd/sources/depthai_slam.py`` (BasaltVIO + RTABMapSLAM),
but every relevant queue is also fanned out to a :class:`SessionRecorder`.

Usage::

    ./tools/record_session.py sessions/2026-05-29_loop1 --duration 60
    ./tools/record_session.py sessions/lab_static_30s --duration 30 --fps 20
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oakd.recorder import SessionRecorder  # noqa: E402


def _read_calib(device, width: int, height: int, left_socket, right_socket) -> dict:
    """Pull intrinsics/extrinsics from device, return JSON-friendly dict."""
    try:
        ch = device.readCalibration()
    except Exception as e:
        return {"error": f"readCalibration failed: {e}"}

    def cam_intr(sock):
        try:
            K = np.array(ch.getCameraIntrinsics(sock, width, height))
            dist = list(ch.getDistortionCoefficients(sock))
            return {
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "K": K.tolist(),
                "dist": [float(x) for x in dist],
                "width": int(width), "height": int(height),
            }
        except Exception as e:
            return {"error": str(e)}

    def extr(src, dst):
        try:
            T = np.array(ch.getCameraExtrinsics(src, dst))
            return T.tolist()
        except Exception as e:
            return {"error": str(e)}

    return {
        "left_socket": str(left_socket),
        "right_socket": str(right_socket),
        "intrinsics_left": cam_intr(left_socket),
        "intrinsics_right": cam_intr(right_socket),
        "T_left_right": extr(left_socket, right_socket),
    }


def _quat_from_transform(td) -> tuple[list[float], list[float]]:
    """Extract (pos_xyz, quat_wxyz) from a depthai TransformData message."""
    tr = td.getTranslation()
    qf = td.getQuaternion()
    return ([float(tr.x), float(tr.y), float(tr.z)],
            [float(qf.qw), float(qf.qx), float(qf.qy), float(qf.qz)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", help="session folder to create")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="seconds to record (Ctrl+C to stop earlier)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--imu-rate", type=int, default=200)
    ap.add_argument("--every-n", type=int, default=1,
                    help="record every Nth stereo frame (1 = keep all)")
    ap.add_argument("-f", "--force", action="store_true",
                    help="delete out_dir if it already exists")
    ap.add_argument("--no-pcl", action="store_true",
                    help="disable RTABMap point cloud publishing (saves disk)")
    args = ap.parse_args()

    import depthai as dai  # lazy

    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        if args.force:
            import shutil
            shutil.rmtree(out)
            print(f"[record] wiped existing folder: {out}", file=sys.stderr)
        else:
            print(f"refusing to record into non-empty folder: {out} "
                  f"(use -f to overwrite)", file=sys.stderr)
            return 2

    rec: SessionRecorder | None = None
    stop = {"flag": False}

    def _sigint(_sig, _frm):
        stop["flag"] = True
        print("\n[record] stopping...", file=sys.stderr)

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    left_socket = dai.CameraBoardSocket.CAM_B
    right_socket = dai.CameraBoardSocket.CAM_C

    with dai.Pipeline() as p:
        left = p.create(dai.node.Camera).build(left_socket, sensorFps=args.fps)
        right = p.create(dai.node.Camera).build(right_socket, sensorFps=args.fps)
        imu = p.create(dai.node.IMU)
        stereo = p.create(dai.node.StereoDepth)
        vio = p.create(dai.node.BasaltVIO)
        slam = p.create(dai.node.RTABMapSLAM)

        imu.enableIMUSensor(
            [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
            args.imu_rate,
        )
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)
        vio.setImuUpdateRate(args.imu_rate)

        stereo.setExtendedDisparity(False)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setRectifyEdgeFillColor(0)
        stereo.enableDistortionCorrection(True)
        stereo.initialConfig.setLeftRightCheckThreshold(10)
        stereo.setDepthAlign(left_socket)

        slam.setParams({
            "RGBD/CreateOccupancyGrid": "true",
            "Grid/3D": "true",
            "Rtabmap/DetectionRate": "1",
            "Rtabmap/SaveWMState": "true",
            "Mem/IncrementalMemory": "true",
        })
        slam.setPublishGrid(False)
        slam.setPublishObstacleCloud(not args.no_pcl)
        slam.setPublishGroundCloud(not args.no_pcl)

        # Persist RTABMap database so we can extract keyframes (C4) and
        # rich loop-closure metadata (C5 with BoW score + inliers) offline.
        (out / "basalt").mkdir(parents=True, exist_ok=True)
        slam.setDatabasePath(str(out / "basalt" / "rtabmap.db"))
        slam.setSaveDatabaseOnClose(True)

        left.requestOutput((args.width, args.height)).link(stereo.left)
        right.requestOutput((args.width, args.height)).link(stereo.right)
        stereo.syncedLeft.link(vio.left)
        stereo.syncedRight.link(vio.right)
        imu.out.link(vio.imu)
        stereo.depth.link(slam.depth)
        stereo.rectifiedLeft.link(slam.rect)
        vio.transform.link(slam.odom)

        # Fan-out: tap every stream we need for C0/C1/C2/C3.
        q_left = stereo.rectifiedLeft.createOutputQueue()
        q_right = stereo.syncedRight.createOutputQueue()
        q_depth = stereo.depth.createOutputQueue()
        q_imu = imu.out.createOutputQueue()
        q_vio = vio.transform.createOutputQueue()
        q_slam = slam.transform.createOutputQueue()
        q_corr = slam.odomCorrection.createOutputQueue()
        q_obs_pcl = None if args.no_pcl else slam.obstaclePCL.createOutputQueue()
        q_gnd_pcl = None if args.no_pcl else slam.groundPCL.createOutputQueue()

        p.start()

        # Pipeline is live — safe to create the recorder + write calib.
        rec = SessionRecorder(
            out_dir=out,
            params={
                "width": args.width, "height": args.height, "fps": args.fps,
                "imu_rate_hz": args.imu_rate, "every_n": args.every_n,
            },
        )
        try:
            calib = _read_calib(p.getDefaultDevice(), args.width, args.height,
                                left_socket, right_socket)
            rec.write_calib(calib)
        except Exception as e:
            rec.write_calib({"error": f"calib read failed: {e}"})

        t_start = time.monotonic()
        # Latest-frame buffers so we can record a synced stereo triplet
        # (left, right, depth) keyed by sequence.
        pending: dict[int, dict] = {}
        max_pending = 32
        frame_count = 0

        def _consume_synced(seq_key: int) -> None:
            nonlocal frame_count
            entry = pending.get(seq_key)
            if not entry or not {"L", "R", "D"}.issubset(entry):
                return
            pending.pop(seq_key, None)
            if frame_count % args.every_n == 0:
                rec.on_stereo(
                    entry["L"], entry["R"], entry["D"], ts_ns=entry["ts"],
                )
            frame_count += 1
            # drop stale partial triplets
            for k in [k for k in pending if k < seq_key - max_pending]:
                pending.pop(k, None)

        def _stash(kind: str, msg) -> None:
            try:
                seq = int(msg.getSequenceNum())
            except Exception:
                seq = 0
            ts = rec.now_ns()
            entry = pending.setdefault(seq, {"ts": ts})
            if kind == "L":
                entry["L"] = msg.getCvFrame()
            elif kind == "R":
                entry["R"] = msg.getCvFrame()
            elif kind == "D":
                entry["D"] = msg.getFrame()  # uint16 depth
            entry["ts"] = ts
            _consume_synced(seq)

        last_log = t_start

        while not stop["flag"] and p.isRunning():
            now = time.monotonic()
            if now - t_start >= args.duration:
                break

            got_any = False

            msg = q_left.tryGet()
            if msg is not None:
                _stash("L", msg); got_any = True
            msg = q_right.tryGet()
            if msg is not None:
                _stash("R", msg); got_any = True
            msg = q_depth.tryGet()
            if msg is not None:
                _stash("D", msg); got_any = True

            # IMU: each message has N packets
            imu_msg = q_imu.tryGet()
            if imu_msg is not None:
                got_any = True
                ts_ns = rec.now_ns()
                for pkt in imu_msg.packets:
                    try:
                        a = pkt.acceleroMeter
                        g = pkt.gyroscope
                        rec.on_imu(
                            gyro_xyz=(g.x, g.y, g.z),
                            accel_xyz=(a.x, a.y, a.z),
                            ts_ns=ts_ns,
                        )
                    except Exception:
                        continue

            vio_msg = q_vio.tryGet()
            if vio_msg is not None:
                got_any = True
                pos, quat = _quat_from_transform(vio_msg)
                rec.on_vio_pose(pos, quat, ts_ns=rec.now_ns())

            slam_msg = q_slam.tryGet()
            if slam_msg is not None:
                got_any = True
                pos, quat = _quat_from_transform(slam_msg)
                rec.on_slam_pose(pos, quat, ts_ns=rec.now_ns())

            corr_msg = q_corr.tryGet()
            if corr_msg is not None:
                got_any = True
                pos, quat = _quat_from_transform(corr_msg)
                rec.on_odom_correction(pos, quat, ts_ns=rec.now_ns())

            for q, kind in (
                (q_obs_pcl, "obstacle"),
                (q_gnd_pcl, "ground"),
            ):
                if q is None:
                    continue
                pcl_msg = q.tryGet()
                if pcl_msg is not None:
                    got_any = True
                    try:
                        pts = pcl_msg.getPoints()  # Nx3 float
                        if pts is not None and len(pts):
                            rec.on_pointcloud(np.asarray(pts), kind=kind,
                                              ts_ns=rec.now_ns())
                    except Exception:
                        pass

            if not got_any:
                time.sleep(0.002)

            if now - last_log >= 2.0:
                last_log = now
                print(
                    f"[record] t={now-t_start:5.1f}s  "
                    f"frames={rec._frame_seq}  imu={rec._imu_seq}  "
                    f"vio={rec._vio_seq}  slam={rec._slam_seq}  "
                    f"corr={rec._corr_seq}  pcl={rec._pcl_seq}",
                    flush=True,
                )

    rec.close()
    print(f"[record] done -> {out}")
    print(f"  frames: {rec._frame_seq}")
    print(f"  imu:    {rec._imu_seq}")
    print(f"  vio:    {rec._vio_seq}")
    print(f"  slam:   {rec._slam_seq}")
    print(f"  corr:   {rec._corr_seq}")
    print(f"  pcl:    {rec._pcl_seq}")

    # C4: auto-extract keyframes + loop closures from rtabmap.db
    db_path = out / "basalt" / "rtabmap.db"
    if db_path.exists():
        try:
            from tools.extract_kf_from_db import extract as _extract_kf
            n_kf, n_lp = _extract_kf(out)
            print(f"  kf:     {n_kf} keyframes, {n_lp} loop links")
        except Exception as e:
            print(f"  [warn] kf extract failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
