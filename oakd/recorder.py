"""Session recorder: dump checkpoint streams (C0/C1/C2/C3) to disk.

Folder layout follows ``docs/PIPELINE_CHECKPOINTS.md``::

    <out_dir>/
      calib.json
      meta.json                       (written by close())
      input/
        imu.jsonl                     (C1)
        frames.jsonl                  (C0 metadata)
        img/000000_L.png 000000_R.png 000000_D.raw16 ...
      basalt/
        vio_pose.jsonl                (C2: Basalt VIO, FLU world)
        slam_pose.jsonl               (C3: RTABMap SLAM, FLU world)
        odom_correction.jsonl         (raw map<-odom transform stream)
        loop_events.jsonl             (C5: derived in close() from odom_correction)
        track_events.jsonl            (C6: derived in close() from slam pose gaps)
        features.jsonl                (C7: 2D tracked features per frame)
        pointcloud.jsonl              (index of point cloud emissions)
        pointcloud/000000.f32 ...     (Nx3 float32, world frame)

All ``ts_ns`` values are host-monotonic nanoseconds from the recorder's t0.
All poses are stored in the **FLU world** frame as emitted by Basalt /
RTABMap — NED conversion is a viewer-side concern and would lose
information for comparison purposes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

import cv2
import numpy as np


class SessionRecorder:
    def __init__(
        self,
        out_dir: str | Path,
        sensor_name: str = "OAK-D W",
        pipeline_name: str = "basalt_vio + rtabmap_slam",
        params: dict[str, Any] | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.img_dir = self.out_dir / "input" / "img"
        self.basalt_dir = self.out_dir / "basalt"
        (self.out_dir / "input").mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.basalt_dir.mkdir(parents=True, exist_ok=True)

        self._lock = Lock()
        self._t0_ns = time.monotonic_ns()
        self._sensor = sensor_name
        self._pipeline = pipeline_name
        self._params = dict(params or {})

        # line-buffered JSONL writers
        self._f_imu = (self.out_dir / "input" / "imu.jsonl").open("w", buffering=1)
        self._f_frames = (self.out_dir / "input" / "frames.jsonl").open("w", buffering=1)
        self._f_vio = (self.basalt_dir / "vio_pose.jsonl").open("w", buffering=1)
        self._f_slam = (self.basalt_dir / "slam_pose.jsonl").open("w", buffering=1)
        self._f_corr = (self.basalt_dir / "odom_correction.jsonl").open("w", buffering=1)
        self._f_feat = (self.basalt_dir / "features.jsonl").open("w", buffering=1)
        self._f_pcl_idx = (self.basalt_dir / "pointcloud.jsonl").open("w", buffering=1)
        self.pcl_dir = self.basalt_dir / "pointcloud"
        self.pcl_dir.mkdir(exist_ok=True)

        self._frame_seq = 0
        self._imu_seq = 0
        self._vio_seq = 0
        self._slam_seq = 0
        self._corr_seq = 0
        self._feat_seq = 0
        self._pcl_seq = 0
        self._closed = False

        # In-memory snapshots used for post-process event derivation in close().
        self._slam_pose_log: list[tuple[int, np.ndarray, np.ndarray]] = []
        self._corr_log: list[tuple[int, np.ndarray, np.ndarray]] = []

    # ---------------- timing ----------------

    def now_ns(self) -> int:
        return time.monotonic_ns() - self._t0_ns

    # ---------------- calibration ----------------

    def write_calib(self, calib: dict[str, Any]) -> None:
        with (self.out_dir / "calib.json").open("w") as f:
            json.dump(calib, f, indent=2)

    # ---------------- C0: stereo frame ----------------

    def on_stereo(
        self,
        left_u8: np.ndarray,
        right_u8: np.ndarray,
        depth_u16: np.ndarray,
        ts_ns: int | None = None,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            seq = self._frame_seq
            self._frame_seq += 1
            ts = self.now_ns() if ts_ns is None else int(ts_ns)
        base = f"{seq:06d}"
        cv2.imwrite(str(self.img_dir / f"{base}_L.png"), left_u8)
        cv2.imwrite(str(self.img_dir / f"{base}_R.png"), right_u8)
        depth_u16.astype("<u2").tofile(self.img_dir / f"{base}_D.raw16")
        h, w = depth_u16.shape[:2]
        rec = {
            "ts_ns": ts,
            "seq": seq,
            "type": "stereo",
            "left_path": f"img/{base}_L.png",
            "right_path": f"img/{base}_R.png",
            "depth_path": f"img/{base}_D.raw16",
            "width": int(w),
            "height": int(h),
        }
        self._f_frames.write(json.dumps(rec) + "\n")

    # ---------------- C1: IMU sample ----------------

    def on_imu(
        self,
        gyro_xyz: Sequence[float],
        accel_xyz: Sequence[float],
        temp_c: float | None = None,
        ts_ns: int | None = None,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            seq = self._imu_seq
            self._imu_seq += 1
            ts = self.now_ns() if ts_ns is None else int(ts_ns)
        rec = {
            "ts_ns": ts,
            "seq": seq,
            "gyro": [float(x) for x in gyro_xyz],
            "accel": [float(x) for x in accel_xyz],
        }
        if temp_c is not None:
            rec["temp_c"] = float(temp_c)
        self._f_imu.write(json.dumps(rec) + "\n")

    # ---------------- C2 / C3: poses ----------------

    def _write_pose(
        self,
        fp,
        seq: int,
        ts_ns: int,
        pos: Sequence[float],
        quat_wxyz: Sequence[float],
        source: str,
        tracking_ok: bool,
    ) -> None:
        rec = {
            "ts_ns": ts_ns,
            "seq": seq,
            "frame_id": "flu_world",
            "pos": [float(x) for x in pos],
            "quat_wxyz": [float(x) for x in quat_wxyz],
            "tracking_ok": bool(tracking_ok),
            "source": source,
        }
        fp.write(json.dumps(rec) + "\n")

    def on_vio_pose(
        self,
        pos_flu: Sequence[float],
        quat_wxyz: Sequence[float],
        ts_ns: int | None = None,
        tracking_ok: bool = True,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            seq = self._vio_seq
            self._vio_seq += 1
            ts = self.now_ns() if ts_ns is None else int(ts_ns)
        self._write_pose(self._f_vio, seq, ts, pos_flu, quat_wxyz,
                         "basalt_vio", tracking_ok)

    def on_slam_pose(
        self,
        pos_flu: Sequence[float],
        quat_wxyz: Sequence[float],
        ts_ns: int | None = None,
        tracking_ok: bool = True,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            seq = self._slam_seq
            self._slam_seq += 1
            ts = self.now_ns() if ts_ns is None else int(ts_ns)
            self._slam_pose_log.append((
                ts, np.asarray(pos_flu, dtype=np.float64),
                np.asarray(quat_wxyz, dtype=np.float64),
            ))
        self._write_pose(self._f_slam, seq, ts, pos_flu, quat_wxyz,
                         "rtabmap_slam", tracking_ok)

    # ---------------- odom correction (raw stream for C5 derivation) ----------------

    def on_odom_correction(
        self,
        pos_flu: Sequence[float],
        quat_wxyz: Sequence[float],
        ts_ns: int | None = None,
    ) -> None:
        """map<-odom correction transform from RTABMap. Big jumps = loop closure."""
        with self._lock:
            if self._closed:
                return
            seq = self._corr_seq
            self._corr_seq += 1
            ts = self.now_ns() if ts_ns is None else int(ts_ns)
            p = np.asarray(pos_flu, dtype=np.float64)
            q = np.asarray(quat_wxyz, dtype=np.float64)
            self._corr_log.append((ts, p, q))
        rec = {
            "ts_ns": ts,
            "seq": seq,
            "pos": [float(x) for x in p],
            "quat_wxyz": [float(x) for x in q],
        }
        self._f_corr.write(json.dumps(rec) + "\n")

    # ---------------- C7: tracked features (2D, on rectified left) ----------------

    def on_features(
        self,
        feats: Sequence[tuple[float, float, int, int]],
        ts_ns: int | None = None,
    ) -> None:
        """feats = list of (x_px, y_px, track_id, age). One record per frame."""
        with self._lock:
            if self._closed:
                return
            seq = self._feat_seq
            self._feat_seq += 1
            ts = self.now_ns() if ts_ns is None else int(ts_ns)
        rec = {
            "ts_ns": ts,
            "seq": seq,
            "n": len(feats),
            "pts": [[float(x), float(y), int(i), int(a)] for x, y, i, a in feats],
        }
        self._f_feat.write(json.dumps(rec) + "\n")

    # ---------------- point cloud (RTABMap obstacle/ground) ----------------

    def on_pointcloud(
        self,
        xyz: np.ndarray,
        kind: str = "obstacle",
        ts_ns: int | None = None,
    ) -> None:
        """Dump Nx3 float32 point cloud as raw binary + index entry."""
        if xyz.size == 0:
            return
        with self._lock:
            if self._closed:
                return
            seq = self._pcl_seq
            self._pcl_seq += 1
            ts = self.now_ns() if ts_ns is None else int(ts_ns)
        rel_path = f"pointcloud/{seq:06d}_{kind}.f32"
        arr = np.ascontiguousarray(xyz.reshape(-1, 3).astype(np.float32))
        arr.tofile(self.basalt_dir / rel_path)
        rec = {
            "ts_ns": ts,
            "seq": seq,
            "kind": kind,
            "n_points": int(arr.shape[0]),
            "path": rel_path,
        }
        self._f_pcl_idx.write(json.dumps(rec) + "\n")

    # ---------------- shutdown ----------------

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            duration_s = self.now_ns() / 1e9
            for fp in (self._f_imu, self._f_frames, self._f_vio,
                       self._f_slam, self._f_corr, self._f_feat,
                       self._f_pcl_idx):
                try:
                    fp.flush()
                    fp.close()
                except Exception:
                    pass

        # Derive C5 (loop closure) + C6 (tracking) events post-hoc.
        loop_n = self._derive_loop_events()
        track_n = self._derive_tracking_events()

        meta = {
            "session_id": self.out_dir.name,
            "pipeline": self._pipeline,
            "sensor": self._sensor,
            "duration_s": round(duration_s, 3),
            "counts": {
                "frames": self._frame_seq,
                "imu_samples": self._imu_seq,
                "vio_poses": self._vio_seq,
                "slam_poses": self._slam_seq,
                "odom_corrections": self._corr_seq,
                "loop_events": loop_n,
                "tracking_events": track_n,
                "feature_frames": self._feat_seq,
                "pointclouds": self._pcl_seq,
            },
            "params": self._params,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with (self.out_dir / "meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

    # ---------------- event derivation ----------------

    # Threshold above which a jump in odomCorrection is treated as a loop closure.
    LOOP_POS_JUMP_M = 0.10
    LOOP_ROT_JUMP_DEG = 5.0
    # Ignore the first N seconds of odom_correction: RTABMap publishes a
    # rapidly-changing map<-odom transform while it builds the initial map,
    # which would otherwise show up as a swarm of false loop closures.
    LOOP_WARMUP_S = 3.0
    # If two consecutive SLAM poses are further apart than this in time, the
    # tracking is considered lost between them.
    TRACK_GAP_S = 0.5

    def _derive_loop_events(self) -> int:
        """Scan odomCorrection stream; emit a loop_event whenever the
        correction jumps more than the threshold from the previous sample.
        Samples within ``LOOP_WARMUP_S`` of recorder start are skipped because
        RTABMap's initial map<-odom transform is unstable during map init.
        """
        path = self.basalt_dir / "loop_events.jsonl"
        n_events = 0
        prev_p, prev_q = None, None
        warmup_ns = int(self.LOOP_WARMUP_S * 1e9)
        with path.open("w") as f:
            for ts, p, q in self._corr_log:
                if ts < warmup_ns:
                    prev_p, prev_q = p, q
                    continue
                if prev_p is None:
                    prev_p, prev_q = p, q
                    continue
                d_pos = float(np.linalg.norm(p - prev_p))
                # quaternion angular distance (smallest)
                dot = float(np.clip(abs(np.dot(prev_q, q)), -1.0, 1.0))
                d_rot_deg = float(np.degrees(2.0 * np.arccos(dot)))
                if d_pos > self.LOOP_POS_JUMP_M or d_rot_deg > self.LOOP_ROT_JUMP_DEG:
                    rec = {
                        "ts_ns": ts,
                        "event": "loop_closure",
                        "pos_jump_m": d_pos,
                        "rot_jump_deg": d_rot_deg,
                        "correction_pos": [float(x) for x in p],
                        "correction_quat_wxyz": [float(x) for x in q],
                    }
                    f.write(json.dumps(rec) + "\n")
                    n_events += 1
                prev_p, prev_q = p, q
        return n_events

    def _derive_tracking_events(self) -> int:
        """Scan SLAM pose stream; emit lost/recovered events around gaps."""
        path = self.basalt_dir / "track_events.jsonl"
        n_events = 0
        prev_ts: int | None = None
        prev_seq: int | None = None
        lost = False
        with path.open("w") as f:
            for seq, (ts, _p, _q) in enumerate(self._slam_pose_log):
                if prev_ts is not None:
                    dt_s = (ts - prev_ts) / 1e9
                    if not lost and dt_s > self.TRACK_GAP_S:
                        f.write(json.dumps({
                            "ts_ns": prev_ts,
                            "event": "tracking_lost",
                            "gap_s": dt_s,
                            "last_pose_seq": prev_seq,
                        }) + "\n")
                        n_events += 1
                        lost = True
                    elif lost and dt_s <= self.TRACK_GAP_S:
                        f.write(json.dumps({
                            "ts_ns": ts,
                            "event": "tracking_recovered",
                            "first_pose_seq": seq,
                        }) + "\n")
                        n_events += 1
                        lost = False
                prev_ts = ts
                prev_seq = seq
        return n_events
