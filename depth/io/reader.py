"""Reader for recorded gold sessions (the honest, offline data source).

A recorded session on disk looks like::

    sessions/gold/lab_loop_30s/
        meta.json            # session metadata (fps, counts, params)
        calib.json           # stereo intrinsics + left->right extrinsics
        input/
            frames.jsonl     # one line per stereo frame (paths + ts)
            imu.jsonl        # one line per IMU sample (gyro rad/s, accel m/s^2)
            img/
                000000_L.png     # rectified left, uint8 grayscale (H, W)
                000000_R.png     # synced RIGHT (raw, NOT rectified), uint8 (H, W)
                000000_D.raw16   # depth, uint16 millimetres, row-major (H, W)

Everything the from-scratch VIO needs (image, depth, IMU, calibration) is here,
so the whole pipeline can be developed and validated without an OAK-D attached.

Note: the recorder saves ``stereo.syncedRight`` as ``*_R.png``, which is the
right frame *synced* to the left but **not** rectified (only ``rectifiedLeft`` and
``depth`` are rectified by the chip). To block-match it against the rectified
left you must rectify it first -- see :class:`sky.depth.stereo.RightRectifier`.

Conventions
-----------
* Images are OpenCV layout ``(H, W)``.
* Camera optical frame: +x right, +y down, +z forward.
* Depth is returned in **metres** (``float32``); invalid pixels are ``0.0``.
* ``T_left_right`` maps points from the left camera frame to the right camera
  frame and is stored in **metres**.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from depth.comms.lib.misc.pngio import imread_gray


@dataclass
class CameraCalib:
    """Pinhole intrinsics for one rectified camera."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: np.ndarray  # distortion coefficients as recorded (rectified imgs => ~0)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_json(cls, d: dict) -> "CameraCalib":
        return cls(
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            width=int(d["width"]),
            height=int(d["height"]),
            dist=np.asarray(d.get("dist", []), dtype=np.float64),
        )


@dataclass
class StereoCalib:
    """Left + right intrinsics and the left->right rigid transform."""

    left: CameraCalib
    right: CameraCalib
    T_left_right: np.ndarray  # 4x4, metres
    # IMU<->camera extrinsics (4x4, metres) and IMU noise, when recorded.
    # ``T_imu_left`` maps points from the IMU frame to the left-camera frame;
    # the left camera is where depth is aligned, so this is the transform a VIO
    # uses to bring IMU measurements into the visual frame. ``None`` for older
    # sessions recorded before IMU extrinsics were captured.
    T_imu_left: np.ndarray | None = None
    T_imu_right: np.ndarray | None = None
    imu_noise: dict | None = None

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.T_left_right[:3, 3]))

    @property
    def has_imu_extrinsics(self) -> bool:
        return self.T_imu_left is not None

    @staticmethod
    def _parse_T(value) -> np.ndarray | None:
        """Parse a 4x4 transform (already in metres) from calib JSON, or None."""
        if not isinstance(value, list):
            return None
        return np.asarray(value, dtype=np.float64).reshape(4, 4)

    @classmethod
    def from_json(cls, d: dict) -> "StereoCalib":
        # depthai reports extrinsic translation in centimetres -> convert to m.
        T = np.asarray(d["T_left_right"], dtype=np.float64).reshape(4, 4).copy()
        T[:3, 3] *= 0.01
        noise = d.get("imu_noise")
        if isinstance(noise, dict) and "error" in noise:
            noise = None
        return cls(
            left=CameraCalib.from_json(d["intrinsics_left"]),
            right=CameraCalib.from_json(d["intrinsics_right"]),
            T_left_right=T,
            T_imu_left=cls._parse_T(d.get("T_imu_left")),
            T_imu_right=cls._parse_T(d.get("T_imu_right")),
            imu_noise=noise,
        )


@dataclass
class Frame:
    """One recorded stereo+depth frame, fully decoded into arrays."""

    seq: int
    ts_ns: int
    gray_left: np.ndarray  # (H, W) uint8
    gray_right: np.ndarray | None  # (H, W) uint8 or None if not loaded
    depth_m: np.ndarray  # (H, W) float32, metres, 0 == invalid
    K: np.ndarray  # 3x3 left intrinsics

    @property
    def ts_s(self) -> float:
        return self.ts_ns * 1e-9


class SessionReader:
    """Random-access + iterable reader over a recorded session's input/."""

    def __init__(self, session_dir: str | Path):
        self.dir = Path(session_dir)
        if not self.dir.exists():
            raise FileNotFoundError(self.dir)
        self.input_dir = self.dir / "input"
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.calib = StereoCalib.from_json(
            json.loads((self.dir / "calib.json").read_text())
        )
        self._frames = [
            json.loads(line)
            for line in (self.input_dir / "frames.jsonl").read_text().splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def K(self) -> np.ndarray:
        return self.calib.left.K

    def load_frame(self, index: int, load_right: bool = False) -> Frame:
        rec = self._frames[index]
        left_path = self.input_dir / rec["left_path"]
        if not left_path.exists():
            raise FileNotFoundError(left_path)
        gray_left = imread_gray(left_path)

        gray_right = None
        if load_right and rec.get("right_path"):
            gray_right = imread_gray(self.input_dir / rec["right_path"])

        h, w = int(rec["height"]), int(rec["width"])
        depth_mm = np.fromfile(
            self.input_dir / rec["depth_path"], dtype=np.uint16
        ).reshape(h, w)
        depth_m = depth_mm.astype(np.float32) * 1e-3

        return Frame(
            seq=int(rec["seq"]),
            ts_ns=int(rec["ts_ns"]),
            gray_left=gray_left,
            gray_right=gray_right,
            depth_m=depth_m,
            K=self.K,
        )

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self.load_frame(i)

    def load_imu(self) -> dict[str, np.ndarray]:
        """Load all IMU samples as arrays: ts_ns (N,), gyro (N,3), accel (N,3)."""
        ts, gyro, accel = [], [], []
        for line in (self.input_dir / "imu.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            ts.append(s["ts_ns"])
            gyro.append(s["gyro"])
            accel.append(s["accel"])
        return {
            "ts_ns": np.asarray(ts, dtype=np.int64),
            "gyro": np.asarray(gyro, dtype=np.float64),
            "accel": np.asarray(accel, dtype=np.float64),
        }
