"""Six-position accelerometer calibration (bias + scale + misalignment).

A raw MEMS accelerometer has three error sources that a single static "level the
gravity vector" step CANNOT separate:

* **bias**   -- a per-axis zero offset (the reading is not 0 at free-fall),
* **scale**  -- a per-axis gain error (1 g reads as 0.98 g / 1.02 g),
* **misalignment** -- the three sensitive axes are not perfectly orthogonal nor
  perfectly aligned with the case, so a pure +x acceleration leaks into y and z.

The classic way to observe all of them is the **six-position** (a.k.a. tumble)
test: hold the device still with each of its 6 faces up/down so gravity points
along +/-x, +/-y, +/-z in turn. At rest the *true* specific force has magnitude
exactly ``g`` in every pose, so a correct calibration must map every captured raw
vector onto the sphere of radius ``g``. This is enough to solve the full model.

Correction model (the same one used on the flight-controller)::

    a_cal = T @ (a_raw - b)

``b`` is the 3-vector bias (raw units, m/s^2) and ``T`` is a 3x3 matrix folding
the inverse scale and the misalignment together.

**Why direction, not just magnitude.** Constraining only the *magnitude*
(``|a_cal| = g``) at six poses is under-determined: six scalar equations cannot
pin the nine parameters -- magnitude alone only fixes the symmetric ``T^T T`` (an
ellipsoid), so the fit lands on the six captures yet does NOT generalise to other
orientations. The classic six-position test resolves this by also using the
*known direction* of gravity at each pose: when face ``+x`` is up the true
specific force is exactly ``g * [1,0,0]`` in the sensor frame, etc. That makes a
well-posed **linear** system::

    T @ a_k - c = g * dir_k          (c := T @ b)

stacked over all poses (3 equations each, 12 unknowns ``vec(T) + c``) and solved
by least squares, then ``b = T^{-1} c``. Using the direction recovers the FULL
(non-symmetric) misalignment and pins the gauge, so the calibration generalises
to any unseen orientation. Extra non-face poses only improve the fit.

The solver is pure NumPy (a single ``lstsq``), and is exercised by
``accel_calib_selftest`` with synthetically distorted data (known ``T``, ``b``
recovered to ~1e-12 and verified to generalise to unseen tilted poses) so the
maths is regression-locked offline before it ever touches hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Standard gravity (m/s^2). The calibration targets this magnitude; a site can
# override it with the local value if a survey-grade reference is needed.
G_STANDARD = 9.80665

# The six canonical face directions (gravity along +/- each sensor axis).
SIX_FACES = np.array([
    [+1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, +1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, +1.0], [0.0, 0.0, -1.0],
])


@dataclass(frozen=True)
class AccelCalibration:
    """Affine accelerometer correction ``a_cal = T @ (a_raw - b)``.

    ``T`` is 3x3 (lower-triangular as solved, but stored dense so a hand-edited
    or imported full matrix also works), ``b`` is the raw-unit bias (m/s^2).
    ``residual_g`` is the RMS of ``|a_cal| - g`` over the calibration captures
    (a quality figure: how far each corrected pose sits off the gravity sphere).
    """

    T: np.ndarray
    bias: np.ndarray
    residual_g: float = 0.0
    g: float = G_STANDARD

    @classmethod
    def identity(cls, g: float = G_STANDARD) -> "AccelCalibration":
        """A no-op calibration (raw passes through unchanged)."""
        return cls(np.eye(3), np.zeros(3), 0.0, g)

    def apply(self, a_raw: np.ndarray) -> np.ndarray:
        """Correct a raw accel vector (or an ``(N, 3)`` batch)."""
        a = np.asarray(a_raw, dtype=np.float64)
        return (a - self.bias) @ self.T.T

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "T": [[float(x) for x in row] for row in self.T],
            "bias": [float(x) for x in self.bias],
            "residual_g": float(self.residual_g),
            "g": float(self.g),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AccelCalibration":
        T = np.asarray(d["T"], dtype=np.float64).reshape(3, 3)
        b = np.asarray(d["bias"], dtype=np.float64).reshape(3)
        return cls(T, b, float(d.get("residual_g", 0.0)),
                   float(d.get("g", G_STANDARD)))


def infer_face_directions(captures, g: float = G_STANDARD) -> np.ndarray:
    """Snap each capture to its nearest +/- axis (the six-face gravity dir).

    For a guided six-face wizard the user is told which face to place, so the
    dominant axis of the mean raw vector identifies the gravity direction. This
    is only a convenience for the canonical face set / tests; the solver also
    accepts explicit directions for arbitrary tilted poses.
    """
    A = np.asarray(list(captures), dtype=np.float64).reshape(-1, 3)
    dirs = np.zeros_like(A)
    ax = np.argmax(np.abs(A), axis=1)
    for i, a in enumerate(ax):
        dirs[i, a] = np.sign(A[i, a]) or 1.0
    return dirs


def solve_accel_calibration(captures, directions=None, g: float = G_STANDARD
                            ) -> AccelCalibration:
    """Solve the affine accel calibration from >= 6 static captures.

    ``captures`` is an iterable of mean raw accel vectors (m/s^2), one per static
    pose. ``directions`` is the matching iterable of unit gravity directions in
    the sensor frame at each pose; if ``None`` they are inferred by snapping each
    capture to its nearest +/- axis (correct for the canonical six-face set).

    At least 6 poses spanning all +/- axes are required to observe every
    parameter; the canonical set is the six axis-up/down faces, and extra tilted
    poses only help. Returns the fitted :class:`AccelCalibration` with its RMS
    sphere residual. Raises ``ValueError`` on too few or degenerate captures.
    """
    A = np.asarray(list(captures), dtype=np.float64).reshape(-1, 3)
    N = A.shape[0]
    if N < 6:
        raise ValueError(f"need >= 6 static captures, got {N}")
    if directions is None:
        D = infer_face_directions(A, g)
    else:
        D = np.asarray(list(directions), dtype=np.float64).reshape(-1, 3)
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
    if D.shape[0] != N:
        raise ValueError("captures and directions length mismatch")

    # Linear system  T @ a_k - c = g * dir_k  (unknown x = [vec(T)(9), c(3)]).
    # Row block per pose: 3 equations (one per output component i).
    M = np.zeros((3 * N, 12))
    rhs = np.zeros(3 * N)
    for k in range(N):
        a = A[k]
        for i in range(3):
            row = 3 * k + i
            M[row, 3 * i:3 * i + 3] = a       # T_i. dotted with a_k
            M[row, 9 + i] = -1.0              # -c_i
            rhs[row] = g * D[k, i]
    x, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    T = x[:9].reshape(3, 3)
    c = x[9:12]
    if np.linalg.matrix_rank(T) < 3:
        raise ValueError("accel calibration degenerate (captures do not span "
                         "all three axes); place every face up and down")
    b = np.linalg.solve(T, c)
    if not (np.all(np.isfinite(T)) and np.all(np.isfinite(b))):
        raise ValueError("accel calibration did not converge (non-finite fit)")

    corrected = (A - b) @ T.T
    residual_g = float(np.sqrt(np.mean(
        (np.linalg.norm(corrected, axis=1) - g) ** 2)))
    return AccelCalibration(T, b, residual_g, g)
