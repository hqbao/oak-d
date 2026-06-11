"""SE(3) primitives -- pack/inverse/adjoint and exp/log (pure NumPy).

Twist convention used throughout: ``xi = [rho(3); phi(3)]`` (translation part
first, rotation part second), with a left perturbation ``T <- Exp(xi) @ T`` for
the bundle/IMU code and a right perturbation for the pose graph -- the exp/log
maps below are convention-neutral (they are exact inverses of each other).

As with :mod:`sky.math.so3`, two variants are kept on purpose because the old
copies had genuine numerical drift at the singularities:

* :func:`se3_exp` vs :func:`se3_exp_unit` -- differ only through their rotation
  block (:func:`~sky.math.so3.so3_exp` vs :func:`~sky.math.so3.so3_exp_unit`),
  i.e. only for an essentially-never-hit ``||phi|| < 1e-12`` twist.
* :func:`se3_log` vs :func:`se3_log_robust` -- :func:`se3_log` recovers ``rho``
  with a linear ``solve`` against ``V`` (bundle convention); :func:`se3_log_robust`
  uses the closed-form ``V^{-1}`` series and the near-pi-robust
  :func:`~sky.math.so3.so3_log_robust` (pose-graph convention). These differ at
  the ~1e-12 level even for typical twists, so they are NOT interchangeable.
"""
from __future__ import annotations

import numpy as np

from .so3 import skew, so3_exp, so3_exp_unit, so3_log, so3_log_robust


def se3_from_Rp(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Pack a rotation + translation into a 4x4 homogeneous SE(3) matrix."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, np.float64)
    T[:3, 3] = np.asarray(p, np.float64)
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 SE(3): ``inv((R, t)) = (R^T, -R^T t)`` (no full solve)."""
    T = np.asarray(T, np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    """6x6 adjoint Ad(T) for the [rho; phi] (translation-first) twist order."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(t) @ R
    Ad[3:, 3:] = R
    return Ad


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map se3 -> SE3 (bundle convention). ``xi = [rho(3); phi(3)]`` -> 4x4."""
    rho = xi[:3]
    phi = xi[3:]
    theta = float(np.linalg.norm(phi))
    R = so3_exp(phi)
    if theta < 1e-12:
        V = np.eye(3) + 0.5 * skew(phi)
    else:
        K = skew(phi / theta)
        V = (np.eye(3)
             + (1.0 - np.cos(theta)) / theta * K
             + (theta - np.sin(theta)) / theta * (K @ K))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def se3_exp_unit(xi: np.ndarray) -> np.ndarray:
    """Exponential map se3 -> SE3 (IMU convention; rotation via :func:`so3_exp_unit`).

    Identical to :func:`se3_exp` except its rotation block returns exactly ``I``
    for a near-zero ``phi`` -- matching the IMU module's historical behaviour.
    """
    rho = xi[:3]
    phi = xi[3:]
    theta = float(np.linalg.norm(phi))
    R = so3_exp_unit(phi)
    if theta < 1e-12:
        V = np.eye(3) + 0.5 * skew(phi)
    else:
        K = skew(phi / theta)
        V = (np.eye(3)
             + (1.0 - np.cos(theta)) / theta * K
             + (theta - np.sin(theta)) / theta * (K @ K))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Logarithm map SE3 -> se3 (inverse of :func:`se3_exp`), bundle convention.

    Returns ``xi = [rho(3); phi(3)]``. ``rho`` is recovered by a linear solve
    against the left-Jacobian ``V``.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    phi = so3_log(R)
    theta = float(np.linalg.norm(phi))
    if theta < 1e-8:
        V = np.eye(3) + 0.5 * skew(phi)
    else:
        K = skew(phi / theta)
        V = (np.eye(3)
             + (1.0 - np.cos(theta)) / theta * K
             + (theta - np.sin(theta)) / theta * (K @ K))
    rho = np.linalg.solve(V, t)
    xi = np.empty(6)
    xi[:3] = rho
    xi[3:] = phi
    return xi


def se3_log_robust(T: np.ndarray) -> np.ndarray:
    """Logarithm map SE3 -> se3, pose-graph convention (closed-form ``V^{-1}``).

    Returns ``xi = [rho(3); phi(3)]`` (translation part first). Uses the
    closed-form inverse left-Jacobian series and the near-pi-robust
    :func:`~sky.math.so3.so3_log_robust`, so it stays well-conditioned for
    pose-graph edge residuals that momentarily approach a half-turn.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    phi = so3_log_robust(R)
    theta = float(np.linalg.norm(phi))
    if theta < 1e-9:
        Vinv = np.eye(3) - 0.5 * skew(phi)
    else:
        K = skew(phi)
        a = 1.0 / (theta * theta)
        b = (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        Vinv = np.eye(3) - 0.5 * K + (a - b) * (K @ K)
    rho = Vinv @ t
    return np.concatenate([rho, phi])
