"""Frame conventions and rigid-body transforms.

World frame:   NED   (X=North, Y=East, Z=Down)
Body frame:    FRD   (X=Forward, Y=Right, Z=Down)
Camera frame:  OpenCV (Xc=right, Yc=down, Zc=forward)

Mount on this drone (USB up, WIDE facing forward) gives R_body_cam = I, so a
pose expressed in the camera frame is already in the body frame.

The viewer renders ENU (X=East, Y=North, Z=Up) for natural pilot view. Use
``ned_to_enu()`` only when feeding the GL scene; keep stored state in NED.
"""
from __future__ import annotations

import numpy as np


# Camera (OpenCV) -> Body (FRD)   on this mount: identity
R_BODY_CAM: np.ndarray = np.eye(3, dtype=np.float64)


def ned_to_enu(p_ned: np.ndarray) -> np.ndarray:
    """Convert a position (or array of positions) from NED to ENU."""
    p = np.asarray(p_ned, dtype=np.float64)
    out = np.empty_like(p)
    # (N, E, D) -> (E, N, -D)
    out[..., 0] = p[..., 1]
    out[..., 1] = p[..., 0]
    out[..., 2] = -p[..., 2]
    return out


def quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_to_rpy(q_wxyz: np.ndarray) -> tuple[float, float, float]:
    """Quaternion -> (roll, pitch, yaw) in radians, ZYX convention."""
    w, x, y, z = q_wxyz
    # roll (X)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    # pitch (Y)
    sinp = 2 * (w * y - z * x)
    sinp = float(np.clip(sinp, -1.0, 1.0))
    pitch = np.arcsin(sinp)
    # yaw (Z)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Roll-pitch-yaw (rad, ZYX) -> quaternion (w, x, y, z)."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


def rot_ned_to_enu(R_ned: np.ndarray) -> np.ndarray:
    """Rotate a body-attitude rotation matrix from NED-world to ENU-world.

    R_ned maps body vectors into the NED world. The transform we need for the
    GL scene is R_enu = P @ R_ned @ P^T with P = diag(swap_N_E, flip_D).
    """
    P = np.array([
        [0.0, 1.0, 0.0],   # E_enu = E_ned
        [1.0, 0.0, 0.0],   # N_enu = N_ned
        [0.0, 0.0, -1.0],  # U_enu = -D_ned
    ], dtype=np.float64)
    return P @ R_ned @ P.T
