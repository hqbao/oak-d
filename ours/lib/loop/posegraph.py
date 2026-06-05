"""SE(3) pose-graph optimisation — pure NumPy.

This is the backend optimiser for loop closure. Visual odometry (or windowed BA)
gives accurate *relative* motion between keyframes but accumulates *global*
drift, so after a long trajectory the start and end no longer line up even when
the camera physically returned to the same spot. A loop closure provides one
extra relative constraint ("keyframe 117 is right back at keyframe 3"), and
pose-graph optimisation (PGO) distributes the accumulated error over the whole
graph so the whole trajectory becomes globally consistent.

Formulation
-----------
- A node is a keyframe pose ``X_i = T_wc`` (camera->world, 4x4 SE3).
- An edge ``(i, j, Z_ij, Omega)`` carries a *measured* relative transform
  ``Z_ij = T_ci_cj`` (pose of cam j expressed in cam i, i.e. ``X_i^{-1} X_j``
  in the noise-free case) and a 6x6 information matrix ``Omega``.
- The error of an edge is ``e_ij = Log( Z_ij^{-1} · (X_i^{-1} X_j) )`` (a
  6-vector ``[rho; phi]``, translation part first to match :func:`se3_exp`).
- We minimise ``sum_ij e_ij^T Omega_ij e_ij`` by Gauss-Newton with a right
  perturbation ``X <- X · Exp(delta)``. The first node is pinned (gauge) with a
  strong prior so the global frame stays fixed.

Jacobians use the standard small-residual approximation ``J_r^{-1}(e) ~= I``
(exact at convergence; the relative measurements are accurate so every *edge*
residual stays small even when the *global* drift is large). With that,

    de/ddelta_i = -Ad( X_j^{-1} X_i )      de/ddelta_j = +I

which is the well-known Grisetti pose-graph linearisation. Levenberg-Marquardt
damping keeps it stable from a poor initial guess.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend.bundle import se3_exp, skew


# --------------------------------------------------------------------------- #
# SE(3) log + adjoint (se3_exp / so3_exp / skew live in bundle.py)
# --------------------------------------------------------------------------- #
def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) -> so3 (rotation matrix to rotation vector)."""
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


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) -> se3. Returns xi = [rho(3); phi(3)] (translation part first)."""
    R = T[:3, :3]
    t = T[:3, 3]
    phi = so3_log(R)
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


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    """6x6 adjoint Ad(T) for the [rho; phi] (translation-first) twist order."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(t) @ R
    Ad[3:, 3:] = R
    return Ad


def se3_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# --------------------------------------------------------------------------- #
# Batched SE(3) ops (vectorised over a stack of M transforms).
#
# These let :meth:`PoseGraph.optimize` assemble H and b for *all* edges in a
# handful of NumPy calls instead of a per-edge Python loop. That matters beyond
# raw speed: PGO runs on a background worker thread, and a scalar Python loop
# holds the GIL the whole time (measured: the read loop drops to ~14% of its
# solo rate during optimisation). Vectorised NumPy releases the GIL inside each
# C call, so the live read loop keeps running smoothly.
# --------------------------------------------------------------------------- #
def _inv_batch(T: np.ndarray) -> np.ndarray:
    """Inverse of a stack of SE(3) matrices, shape (M, 4, 4)."""
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    Rt = np.transpose(R, (0, 2, 1))
    Ti = np.zeros_like(T)
    Ti[:, 3, 3] = 1.0
    Ti[:, :3, :3] = Rt
    Ti[:, :3, 3] = -np.einsum("mij,mj->mi", Rt, t)
    return Ti


def _skew_batch(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrices for a stack of 3-vectors, shape (M, 3) -> (M,3,3)."""
    M = v.shape[0]
    K = np.zeros((M, 3, 3))
    K[:, 0, 1] = -v[:, 2]
    K[:, 0, 2] = v[:, 1]
    K[:, 1, 0] = v[:, 2]
    K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]
    K[:, 2, 1] = v[:, 0]
    return K


def _so3_log_batch(R: np.ndarray) -> np.ndarray:
    """Batched SO(3) log. Vectorised small/general regimes; the rare near-pi
    case (|theta - pi| < 1e-6) falls back to the scalar path per element."""
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(c)
    vee = np.stack([R[:, 2, 1] - R[:, 1, 2],
                    R[:, 0, 2] - R[:, 2, 0],
                    R[:, 1, 0] - R[:, 0, 1]], axis=1)
    phi = np.empty_like(vee)
    small = theta < 1e-9
    nearpi = (np.pi - theta) < 1e-6
    gen = ~small & ~nearpi
    # near identity: first-order vee of the skew part
    phi[small] = 0.5 * vee[small]
    # general
    if np.any(gen):
        w = theta[gen] / (2.0 * np.sin(theta[gen]))
        phi[gen] = w[:, None] * vee[gen]
    # near pi: rare, do it exactly element-wise
    for i in np.nonzero(nearpi)[0]:
        phi[i] = so3_log(R[i])
    return phi


def _se3_log_batch(T: np.ndarray) -> np.ndarray:
    """Batched SE(3) log -> twists [rho; phi], shape (M, 4, 4) -> (M, 6)."""
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    phi = _so3_log_batch(R)
    theta = np.linalg.norm(phi, axis=1)
    K = _skew_batch(phi)
    small = theta < 1e-9
    th = np.where(small, 1.0, theta)            # avoid div-by-zero in unused entries
    a = 1.0 / (th * th)
    b = (1.0 + np.cos(th)) / (2.0 * th * np.sin(th))
    coeff = np.where(small, 0.0, a - b)
    KK = K @ K
    Vinv = np.eye(3)[None] - 0.5 * K + coeff[:, None, None] * KK
    rho = np.einsum("mij,mj->mi", Vinv, t)
    return np.concatenate([rho, phi], axis=1)


def _se3_adjoint_batch(T: np.ndarray) -> np.ndarray:
    """Batched 6x6 adjoint for the [rho; phi] twist order, (M,4,4) -> (M,6,6)."""
    M = T.shape[0]
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    Ad = np.zeros((M, 6, 6))
    Ad[:, :3, :3] = R
    Ad[:, 3:, 3:] = R
    Ad[:, :3, 3:] = _skew_batch(t) @ R
    return Ad


def _se3_exp_batch(xi: np.ndarray) -> np.ndarray:
    """Batched SE(3) exp. xi = [rho; phi], shape (M, 6) -> (M, 4, 4).

    Matches :func:`bundle.se3_exp` element-wise (Rodrigues + left Jacobian V).
    """
    M = xi.shape[0]
    rho = xi[:, :3]
    phi = xi[:, 3:]
    theta = np.linalg.norm(phi, axis=1)
    small = theta < 1e-12
    th = np.where(small, 1.0, theta)            # safe divisor for unused entries
    Kf = _skew_batch(phi)                        # skew(phi)
    Ku = _skew_batch(phi / th[:, None])          # skew(unit axis)
    KuKu = Ku @ Ku
    I3 = np.eye(3)[None]
    sin, cos = np.sin(theta), np.cos(theta)
    # Rotation: small -> I + skew(phi); general -> Rodrigues with unit axis.
    R = np.where(small[:, None, None],
                 I3 + Kf,
                 I3 + sin[:, None, None] * Ku + (1.0 - cos)[:, None, None] * KuKu)
    # Left Jacobian V.
    cV = np.where(small, 0.5, (1.0 - cos) / th)
    cVV = np.where(small, 0.0, (th - sin) / th)
    V = I3 + cV[:, None, None] * np.where(small[:, None, None], Kf, Ku) \
        + cVV[:, None, None] * KuKu
    T = np.zeros((M, 4, 4))
    T[:, 3, 3] = 1.0
    T[:, :3, :3] = R
    T[:, :3, 3] = np.einsum("mij,mj->mi", V, rho)
    return T


# --------------------------------------------------------------------------- #
# Pose graph
# --------------------------------------------------------------------------- #
@dataclass
class Edge:
    i: int
    j: int
    Z: np.ndarray            # measured relative T_ci_cj (4x4)
    Omega: np.ndarray        # 6x6 information
    loop: bool = False


class PoseGraph:
    """Keyframe pose graph with SE(3) Gauss-Newton optimisation.

    Nodes are ``T_wc`` (camera->world). Add nodes in any order by integer id,
    add relative edges, then call :meth:`optimize`. Node 0 (or the lowest id) is
    pinned as the gauge anchor.
    """

    def __init__(self) -> None:
        self.nodes: dict[int, np.ndarray] = {}
        self.edges: list[Edge] = []

    def add_node(self, i: int, T_wc: np.ndarray) -> None:
        self.nodes[i] = np.asarray(T_wc, float).copy()

    def add_edge(self, i: int, j: int, Z: np.ndarray,
                 omega: np.ndarray | float = 1.0, loop: bool = False) -> None:
        if np.isscalar(omega):
            Om = np.eye(6) * float(omega)
        else:
            Om = np.asarray(omega, float)
        self.edges.append(Edge(i, j, np.asarray(Z, float).copy(), Om, loop))

    def _node_stack(self, ids: list[int]) -> np.ndarray:
        """Stack node poses into (N, 4, 4) in the given id order."""
        return np.stack([self.nodes[i] for i in ids], axis=0)

    def _edge_arrays(self, idx: dict[int, int]):
        """Static per-edge arrays (independent of current pose values)."""
        ei = np.array([idx[e.i] for e in self.edges], dtype=np.intp)
        ej = np.array([idx[e.j] for e in self.edges], dtype=np.intp)
        Zinv = _inv_batch(np.stack([e.Z for e in self.edges], axis=0))
        Om0 = np.stack([e.Omega for e in self.edges], axis=0)
        loop = np.array([e.loop for e in self.edges], dtype=bool)
        return ei, ej, Zinv, Om0, loop

    def total_error(self) -> float:
        if not self.edges:
            return 0.0
        ids = sorted(self.nodes.keys())
        idx = {nid: k for k, nid in enumerate(ids)}
        ei, ej, Zinv, Om0, _ = self._edge_arrays(idx)
        X = self._node_stack(ids)
        Xi, Xj = X[ei], X[ej]
        E = Zinv @ _inv_batch(Xi) @ Xj
        r = _se3_log_batch(E)
        return float(np.einsum("mi,mij,mj->m", r, Om0, r).sum())

    def optimize(self, iters: int = 30, anchor: int | None = None,
                 rel_tol: float = 1e-6, huber_delta: float = 0.5,
                 verbose: bool = False) -> dict:
        """Gauss-Newton (with LM damping). Mutates node poses in place.

        ``huber_delta`` applies a Huber robust kernel to **loop** edges only
        (odometry edges are trusted): a loop whose residual exceeds the threshold
        is down-weighted, so a few surviving false loop closures (perceptual
        aliasing) cannot drag the whole graph. Set to 0 to disable.

        The whole assembly is vectorised over edges (batched NumPy + scatter),
        so when this runs on the SLAM background thread it releases the GIL and
        does not freeze the live read loop.
        """
        ids = sorted(self.nodes.keys())
        idx = {nid: k for k, nid in enumerate(ids)}
        N = len(ids)
        if anchor is None:
            anchor = ids[0]
        a = idx[anchor]

        ei, ej, Zinv, Om0, loop = self._edge_arrays(idx)
        M = len(self.edges)
        ar6 = np.arange(6)
        # Block scatter indices into the 6N x 6N H (one set per block type) and b.
        ri = 6 * ei[:, None] + ar6[None, :]          # (M, 6) rows for node i
        rj = 6 * ej[:, None] + ar6[None, :]          # (M, 6) rows for node j
        # 2-D index grids for the four 6x6 blocks H[i,i], H[j,j], H[i,j], H[j,i].
        Rii, Cii = ri[:, :, None], ri[:, None, :]
        Rjj, Cjj = rj[:, :, None], rj[:, None, :]
        Rij, Cij = ri[:, :, None], rj[:, None, :]
        Rji, Cji = rj[:, :, None], ri[:, None, :]

        def assemble(X: np.ndarray):
            """Build (H, b) for the current node stack X (N,4,4)."""
            Xi, Xj = X[ei], X[ej]
            Xj_inv = _inv_batch(Xj)
            E = Zinv @ _inv_batch(Xi) @ Xj
            r = _se3_log_batch(E)                          # (M, 6)
            Ad = _se3_adjoint_batch(Xj_inv @ Xi)          # (M, 6, 6); Ji = -Ad
            # Huber down-weight on loop edges only.
            Om = Om0
            if huber_delta > 0.0 and np.any(loop):
                chi = np.sqrt(np.clip(np.einsum("mi,mij,mj->m", r, Om0, r), 0.0, None))
                w = np.where(chi <= huber_delta, 1.0, huber_delta / np.maximum(chi, 1e-12))
                w = np.where(loop, w, 1.0)
                Om = Om0 * w[:, None, None]
            AdT = np.transpose(Ad, (0, 2, 1))
            OmAd = Om @ Ad
            AdTOm = AdT @ Om
            Hii = AdT @ OmAd                # Ji^T Om Ji =  Ad^T Om Ad
            Hjj = Om                        # Jj^T Om Jj =  Om
            Hij = -AdTOm                    # Ji^T Om Jj = -Ad^T Om
            Hji = -OmAd                     # Jj^T Om Ji = -Om Ad
            bi = -np.einsum("mij,mj->mi", AdTOm, r)   # Ji^T Om r = -Ad^T Om r
            bj = np.einsum("mij,mj->mi", Om, r)       # Jj^T Om r =  Om r

            H = np.zeros((6 * N, 6 * N))
            b = np.zeros(6 * N)
            np.add.at(H, (Rii, Cii), Hii)
            np.add.at(H, (Rjj, Cjj), Hjj)
            np.add.at(H, (Rij, Cij), Hij)
            np.add.at(H, (Rji, Cji), Hji)
            np.add.at(b, ri, bi)
            np.add.at(b, rj, bj)
            return H, b

        lam = 1e-6
        cost_prev = self.total_error()
        cost0 = cost_prev
        X = self._node_stack(ids)
        it = 0
        for it in range(iters):
            H, b = assemble(X)

            # Pin the anchor node with a strong prior (gauge freedom removal).
            sa = slice(6 * a, 6 * a + 6)
            H[sa, sa] += np.eye(6) * 1e12

            # LM damping.
            H[np.diag_indices_from(H)] += lam * np.clip(np.diag(H), 1e-9, None)

            try:
                dx = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(H, -b, rcond=None)[0]

            # Trial update on a copy of the stack.
            steps = _se3_exp_batch(dx.reshape(N, 6))
            trial = X @ steps
            r_trial = _se3_log_batch(Zinv @ _inv_batch(trial[ei]) @ trial[ej])
            cost_new = float(np.einsum("mi,mij,mj->m", r_trial, Om0, r_trial).sum())
            if cost_new < cost_prev:
                X = trial
                lam = max(lam * 0.5, 1e-9)
                improved = (cost_prev - cost_new) / max(cost_prev, 1e-12)
                cost_prev = cost_new
                if verbose:
                    print(f"  pgo it{it:02d} cost={cost_new:.6g} lam={lam:.1e}")
                if improved < rel_tol:
                    break
            else:
                lam = min(lam * 8.0, 1e9)     # reject, keep X

        # Write the optimised stack back into the node dict.
        for nid in ids:
            self.nodes[nid] = X[idx[nid]]

        return {"iters": it + 1, "cost0": cost0, "cost1": cost_prev,
                "nodes": N, "edges": len(self.edges)}
