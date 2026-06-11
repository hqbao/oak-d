"""SO(3) primitives -- skew, exp/log and the right Jacobian (pure NumPy).

Two ``so3_exp`` variants and two ``so3_log`` variants are kept on purpose:
across the old copies they were byte-identical for every typical-magnitude
rotation but differed in their near-singularity handling, and silently picking
one would change a project's numerics.

* :func:`so3_exp` vs :func:`so3_exp_unit` -- identical for any ``||phi|| >=
  1e-12``; only the (essentially never hit) ``< 1e-12`` branch differs:
  ``so3_exp`` returns the first-order ``I + skew(phi)`` (the bundle-adjustment
  convention), ``so3_exp_unit`` returns exactly ``I`` (the IMU convention).
* :func:`so3_log` vs :func:`so3_log_robust` -- identical away from ``theta ~=
  pi``; ``so3_log_robust`` adds a sign-robust axis recovery near ``pi`` (the
  pose-graph convention, where edge residuals can momentarily approach a
  half-turn). Plain :func:`so3_log` is the bundle/IMU convention.
"""
from __future__ import annotations

import numpy as np


def skew(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric (``hat``) matrix of a 3-vector: ``skew(w) @ v == w x v``."""
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map so3 -> SO3 (Rodrigues), bundle-adjustment convention.

    Near zero (``||phi|| < 1e-12``) returns the first-order ``I + skew(phi)``.
    For the IMU convention (exact identity at zero) use :func:`so3_exp_unit`.
    """
    theta = float(np.linalg.norm(phi))
    if theta < 1e-12:
        return np.eye(3) + skew(phi)
    k = phi / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_exp_unit(omega: np.ndarray) -> np.ndarray:
    """Exponential map so3 -> SO3 (Rodrigues), IMU convention.

    Identical to :func:`so3_exp` for ``||omega|| >= 1e-12``; near zero it
    returns exactly ``I`` (no first-order term), matching the IMU
    preintegration / odometry code's historical behaviour.
    """
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map SO3 -> so3 (inverse of :func:`so3_exp` / :func:`so3_exp_unit`).

    Standard small-residual form; valid everywhere except the ``theta ~= pi``
    singularity. For pose-graph edges that can approach a half-turn use
    :func:`so3_log_robust`.
    """
    cos_t = (np.trace(R) - 1.0) * 0.5
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    w = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        # near identity: R - R^T ~= 2*skew(phi)
        return 0.5 * w
    return (theta / (2.0 * np.sin(theta))) * w


def so3_log_robust(R: np.ndarray) -> np.ndarray:
    """Logarithm map SO3 -> so3 with a sign-robust ``theta ~= pi`` branch.

    Matches :func:`so3_log` away from the half-turn singularity. Near ``pi`` it
    recovers the axis from the symmetric part ``(R + I)/2`` (the pose-graph
    convention, where a momentary near-pi relative rotation must not blow up).
    """
    c = (np.trace(R) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    theta = float(np.arccos(c))
    if theta < 1e-9:
        # Near identity: vee of the skew-symmetric part (first-order).
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]])
    if np.pi - theta < 1e-6:
        # Near pi: recover axis from the symmetric part (sign-robust).
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        # fix signs from off-diagonals
        if axis[0] > 1e-6:
            axis[1] = np.copysign(axis[1], A[0, 1])
            axis[2] = np.copysign(axis[2], A[0, 2])
        elif axis[1] > 1e-6:
            axis[2] = np.copysign(axis[2], A[1, 2])
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return theta * axis
    w = theta / (2.0 * np.sin(theta))
    return w * np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]])


def so3_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3): ``Exp(phi + dphi) ~= Exp(phi) Exp(Jr(phi) dphi)``.

    Used by IMU preintegration to propagate the bias Jacobian of the
    preintegrated rotation. Falls back to the small-angle form near zero.
    """
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * K
    t2 = theta * theta
    return (np.eye(3)
            - (1.0 - np.cos(theta)) / t2 * K
            + (theta - np.sin(theta)) / (t2 * theta) * (K @ K))
