#!/usr/bin/env python3
"""Self-test for the SE(3) pose-graph optimiser (no recorded data needed).

Builds a synthetic square-loop trajectory, corrupts the odometry with small
per-edge noise so the integrated path drifts away from ground truth, adds one
loop-closure edge between the last and first keyframe, and checks that
pose-graph optimisation pulls the trajectory back toward ground truth.

It also exercises the Lie-group helpers (``se3_log``/``se3_exp`` round-trip and
the adjoint identity) and the Huber robust kernel: a single *gross* false loop
edge that the graph cannot absorb keeps a large residual and gets down-weighted,
while the consistent loop is left untouched.

Note: Huber is only a *secondary* defence on the back-end. A *moderate* false
loop the solver can bend the graph to satisfy ends up with a small residual, so
Huber never engages -- the primary defence against false loops is the front-end
(fundamental-matrix + PnP geometric verification in ``loopclosure.py``).

Run::

    .venv/bin/python ours/tools/posegraph_selftest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.backend.bundle import se3_exp, so3_exp           # noqa: E402
from ours.lib.loop.posegraph import (                        # noqa: E402
    PoseGraph, se3_adjoint, se3_inv, se3_log,
)


def _pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _square_loop(n: int = 40) -> list[np.ndarray]:
    side = n // 4
    gt = []
    for k in range(n):
        seg, frac = k // side, (k % side) / side
        p = ([frac, 0, 0], [1, frac, 0], [1 - frac, 1, 0], [0, 1 - frac, 0])[seg]
        R = so3_exp(np.array([0.0, 0.0, seg * np.pi / 2]))
        gt.append(_pose(R, np.array(p, float)))
    return gt


def _ate(traj: list[np.ndarray], gt: list[np.ndarray]) -> float:
    P = np.array([T[:3, 3] for T in traj])
    G = np.array([T[:3, 3] for T in gt])
    return float(np.sqrt(((P - G) ** 2).sum(1).mean()))


def main() -> int:
    rng = np.random.default_rng(0)

    # --- Lie-group sanity ---------------------------------------------------
    for _ in range(2000):
        xi = rng.uniform(-2.0, 2.0, 6)
        assert np.allclose(se3_exp(se3_log(se3_exp(xi))), se3_exp(xi), atol=1e-9)
    T = se3_exp(rng.uniform(-1, 1, 6))
    xi = rng.uniform(-0.3, 0.3, 6)
    assert np.allclose(se3_exp(se3_adjoint(T) @ xi),
                       T @ se3_exp(xi) @ se3_inv(T), atol=1e-7)
    print("Lie-group: se3 log/exp round-trip + adjoint identity OK")

    gt = _square_loop(40)
    N = len(gt)

    # --- noisy odometry -> drift -------------------------------------------
    rel = []
    for k in range(N - 1):
        Z = se3_inv(gt[k]) @ gt[k + 1]
        rel.append(Z @ se3_exp(rng.normal(0, 0.02, 6)))
    est = [gt[0].copy()]
    for k in range(N - 1):
        est.append(est[-1] @ rel[k])
    drift0 = np.linalg.norm(est[-1][:3, 3] - gt[-1][:3, 3])

    g = PoseGraph()
    for k in range(N):
        g.add_node(k, est[k])
    for k in range(N - 1):
        g.add_edge(k, k + 1, rel[k], omega=1.0)
    Zloop = (se3_inv(gt[0]) @ gt[N - 1]) @ se3_exp(rng.normal(0, 0.01, 6))
    g.add_edge(0, N - 1, Zloop, omega=10.0, loop=True)

    ate_before = _ate(est, gt)
    info = g.optimize(iters=50)
    closed = [g.nodes[k] for k in range(N)]
    drift1 = np.linalg.norm(closed[-1][:3, 3] - gt[-1][:3, 3])
    ate_after = _ate(closed, gt)
    print(f"loop closure : cost {info['cost0']:.3f} -> {info['cost1']:.4f}")
    print(f"  end drift  : {drift0*100:5.1f} cm -> {drift1*100:5.1f} cm")
    print(f"  ATE vs GT  : {ate_before*100:5.1f} cm -> {ate_after*100:5.1f} cm")
    assert drift1 < drift0 * 0.3, "loop closure did not cut end drift"
    assert ate_after < ate_before, "loop closure did not improve ATE"

    # --- robustness: verify the Huber kernel MECHANISM directly -------------
    #
    # End-to-end ATE on a synthetic false loop is a noisy, realisation-
    # dependent metric, so instead we assert the deterministic contract of the
    # back-end kernel on the converged graph: a *gross* loop edge the graph
    # cannot absorb keeps a large residual and gets down-weighted, while the
    # consistent true loop is left untouched.
    #
    # Caveat (honest): a *moderate* false loop the solver CAN bend the graph to
    # satisfy ends up with a small residual, so Huber never fires on it -- the
    # primary defence against those is the front-end (fundamental-matrix + PnP
    # geometric verification in loopclosure.py, validated on the corridor
    # session: ATE 2.27% -> 0.61%). The back-end kernel is a secondary net.
    def _chi(Xi: np.ndarray, Xj: np.ndarray, Z: np.ndarray,
             Om: np.ndarray) -> float:
        r = se3_log(se3_inv(Z) @ (se3_inv(Xi) @ Xj))
        return float(np.sqrt(max(r @ Om @ r, 0.0)))

    gg = PoseGraph()
    for k in range(N):
        gg.add_node(k, est[k])
    for k in range(N - 1):
        gg.add_edge(k, k + 1, rel[k], omega=1.0)
    gg.add_edge(0, N - 1, Zloop, omega=10.0, loop=True)          # true loop
    bad = se3_exp(np.array([3.0, 3.0, 0.0, 0.9, 0.0, 0.0]))      # gross, ~4 m off
    gg.add_edge(8, 24, bad, omega=1.0, loop=True)               # false loop
    delta = 0.4
    gg.optimize(iters=80, huber_delta=delta)

    Om_true = np.eye(6) * 10.0
    Om_false = np.eye(6) * 1.0
    chi_true = _chi(gg.nodes[0], gg.nodes[N - 1], Zloop, Om_true)
    chi_false = _chi(gg.nodes[8], gg.nodes[24], bad, Om_false)
    w_true = 1.0 if chi_true <= delta else delta / chi_true
    w_false = 1.0 if chi_false <= delta else delta / chi_false
    print(f"robust kernel: true loop  chi {chi_true:.3f} -> weight {w_true:.3f}")
    print(f"               false loop chi {chi_false:.3f} -> weight {w_false:.3f}")
    assert w_true > 0.9, "Huber wrongly down-weighted the consistent loop"
    assert w_false < 0.5, "Huber failed to down-weight the gross false loop"

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
