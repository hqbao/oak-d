"""SLAM map: persistent keyframes + loop closure + pose-graph correction.

This is the orchestrator that turns the drift-accumulating windowed-BA odometry
into a globally-consistent SLAM trajectory. It keeps **every** keyframe (unlike
the sliding-window BA, which drops old ones), so it can recognise a place seen
long ago and close the loop.

Responsibilities
----------------
- Store a persistent keyframe per insertion: its odometry pose ``T_wc`` plus a
  compact ORB appearance (:class:`KeyframeAppearance`) for place recognition.
- Maintain a pose graph: a chain of **odometry edges** (relative motion between
  consecutive keyframes, taken from the odometry poses) plus **loop edges**
  (precise relative transforms from geometric verification).
- When a loop is confirmed, :meth:`optimize` runs pose-graph optimisation and
  rewrites every keyframe pose so the whole trajectory becomes consistent.

The orchestrator is deliberately tracker-agnostic (it is handed ``T_wc`` +
image + depth at each keyframe), so the same map works offline in
``tools/vio_run.py`` and, later, on a background thread in the live source.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .loopclosure import LoopConfig, LoopDetector
from .posegraph import PoseGraph, se3_inv


@dataclass
class SlamConfig:
    loop: LoopConfig = field(default_factory=LoopConfig)
    odom_omega: float = 1.0       # information weight on odometry edges
    loop_omega: float = 5.0       # base weight on loop edges (scaled by inliers)
    max_loops_per_kf: int = 3     # cap verified loop edges added per query KF
    pgo_iters: int = 40
    # Spatial gate for loop candidates: only run the (expensive) ORB+PnP
    # verification against older keyframes whose current pose is within this
    # radius (metres) of the incoming keyframe. 0 = disabled (check ALL older
    # keyframes, the exact offline behaviour). A generous radius (> the expected
    # odometry drift) caps the otherwise-linear per-keyframe cost so the live
    # update rate can be raised without the worker falling behind. It can only
    # *miss* a loop if drift already exceeds the radius by revisit time.
    loop_search_radius_m: float = 0.0
    # --- keyframe budget (long-run memory / compute bound) ------------------
    # Motion-gated insertion: skip a new keyframe unless the camera has moved at
    # least ``kf_min_trans_m`` metres OR rotated ``kf_min_rot_deg`` degrees since
    # the last *inserted* keyframe. This makes the map grow with TRAJECTORY
    # length instead of run TIME -- a stationary or slowly-panning camera stops
    # piling up redundant keyframes, which is the main cause of unbounded memory
    # and the O(N^3) PGO cost on long sessions. 0/0 = disabled (insert on every
    # call, the original behaviour). The odometry edge is always taken between
    # consecutive *inserted* keyframes, so skipping frames keeps the chain exact.
    kf_min_trans_m: float = 0.0
    kf_min_rot_deg: float = 0.0
    # Hard ceiling on stored keyframes (0 = unlimited). When the count would
    # exceed it the OLDEST keyframe is dropped (its node + incident edges
    # removed, remaining ids renumbered). This bounds memory absolutely, but note
    # it *forgets* old places: a loop can only close against keyframes still in
    # the map, so set this well above the largest excursion you need to relocalise
    # against. Prefer the motion gate above as the primary bound; use this only as
    # a safety cap for runs that would otherwise grow without limit.
    max_keyframes: int = 0



class SlamMap:
    """Persistent keyframe graph with loop closure + pose-graph optimisation."""

    def __init__(self, K: np.ndarray, cfg: SlamConfig | None = None):
        self.K = np.asarray(K, dtype=np.float64)
        self.cfg = cfg or SlamConfig()
        self.detector = LoopDetector(self.K, self.cfg.loop)
        self.graph = PoseGraph()
        # parallel arrays, index = node id
        self.kf_orig: list[np.ndarray] = []     # odometry T_wc at insertion
        self.kf_pose: list[np.ndarray] = []      # current (corrected) T_wc
        self.kf_app: list = []                   # KeyframeAppearance
        self.kf_seq: list[int] = []
        self.loop_events: list[dict] = []        # {"cur","old","inliers","matches"}

    # ------------------------------------------------------------------ #
    def _needs_keyframe(self, T_wc: np.ndarray) -> bool:
        """Motion gate: True if the camera moved/rotated enough since the last
        inserted keyframe (or if gating is disabled / this is the first KF)."""
        tmin = self.cfg.kf_min_trans_m
        rmin = self.cfg.kf_min_rot_deg
        if (tmin <= 0.0 and rmin <= 0.0) or not self.kf_orig:
            return True
        dT = se3_inv(self.kf_orig[-1]) @ T_wc
        need = False
        if tmin > 0.0:
            need |= float(np.linalg.norm(dT[:3, 3])) >= tmin
        if rmin > 0.0:
            c = (np.trace(dT[:3, :3]) - 1.0) * 0.5
            ang = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
            need |= ang >= rmin
        return need

    def _drop_oldest(self) -> None:
        """Remove the oldest keyframe (id 0): renumber remaining ids down by one,
        drop its node + incident edges, and shift loop-event references."""
        self.kf_orig.pop(0)
        self.kf_pose.pop(0)
        self.kf_app.pop(0)
        self.kf_seq.pop(0)
        new = PoseGraph()
        for i, p in enumerate(self.kf_pose):
            new.add_node(i, p)                 # current corrected pose as init
        for e in self.graph.edges:
            if e.i == 0 or e.j == 0:
                continue                        # touched the removed node -> drop
            new.add_edge(e.i - 1, e.j - 1, e.Z, e.Omega, e.loop)
        self.graph = new
        shifted = []
        for ev in self.loop_events:
            if ev["cur"] == 0 or ev["old"] == 0:
                continue
            ev = dict(ev)
            ev["cur"] -= 1
            ev["old"] -= 1
            shifted.append(ev)
        self.loop_events = shifted

    # ------------------------------------------------------------------ #
    def add_keyframe(self, T_wc: np.ndarray, gray: np.ndarray,
                     depth_m: np.ndarray, seq: int = -1) -> list[dict]:
        """Insert a keyframe; returns the loop events confirmed at this KF.

        Skips insertion (returns ``[]``) when the motion gate
        (:meth:`_needs_keyframe`) says the camera has not moved enough, so the
        ORB appearance is never even computed for a redundant keyframe.
        """
        T_wc = np.asarray(T_wc, float).copy()
        if not self._needs_keyframe(T_wc):
            return []
        idx = len(self.kf_orig)
        app = self.detector.make_appearance(gray, depth_m)
        self.kf_orig.append(T_wc)
        self.kf_pose.append(T_wc.copy())
        self.kf_app.append(app)
        self.kf_seq.append(int(seq))
        self.graph.add_node(idx, T_wc)


        # Odometry edge from the previous keyframe (relative motion from the
        # odometry poses): Z = T_ci_cj = inv(X_i) X_j.
        if idx > 0:
            Z = se3_inv(self.kf_orig[idx - 1]) @ self.kf_orig[idx]
            self.graph.add_edge(idx - 1, idx, Z, omega=self.cfg.odom_omega)

        # Loop detection against older keyframes (skip the recent ones).
        events: list[dict] = []
        gap = self.cfg.loop.min_loop_gap
        if idx >= gap:
            radius = self.cfg.loop_search_radius_m
            cur_pos = T_wc[:3, 3]
            cands = []
            for old in range(0, idx - gap + 1):
                # Spatial gate (when enabled): skip far keyframes before paying
                # for ORB+PnP. Uses the current (corrected) pose of the old KF.
                if radius > 0.0:
                    if np.linalg.norm(self.kf_pose[old][:3, 3] - cur_pos) > radius:
                        continue
                res = self.detector.verify(app, self.kf_app[old])
                if res is not None:
                    T_cur_old, ninl, nmatch = res
                    cands.append((ninl, old, T_cur_old, nmatch))
            cands.sort(reverse=True)             # strongest (most inliers) first
            for ninl, old, T_cur_old, nmatch in cands[:self.cfg.max_loops_per_kf]:
                # Edge old(i) -> cur(j): Z = T_ci_cj = pose of cur in old frame =
                # inv(T_cur_old).
                Z = se3_inv(T_cur_old)
                w = self.cfg.loop_omega * (ninl / max(self.cfg.loop.min_inliers, 1))
                self.graph.add_edge(old, idx, Z, omega=w, loop=True)
                ev = {"cur": idx, "old": old, "inliers": ninl, "matches": nmatch,
                      "cur_seq": self.kf_seq[idx], "old_seq": self.kf_seq[old]}
                events.append(ev)
                self.loop_events.append(ev)

        # Hard ceiling: drop the oldest keyframe(s) once over budget. Done last
        # so the new keyframe and any loop edges it just formed are kept; the
        # cost is forgetting the very oldest place.
        cap = self.cfg.max_keyframes
        if cap > 0:
            while len(self.kf_orig) > cap:
                self._drop_oldest()
        return events

    # ------------------------------------------------------------------ #
    def has_loops(self) -> bool:
        return any(e.loop for e in self.graph.edges)

    def optimize(self, verbose: bool = False) -> dict:
        """Run pose-graph optimisation and rewrite all keyframe poses."""
        if not self.has_loops():
            return {"skipped": "no loop edges"}
        info = self.graph.optimize(iters=self.cfg.pgo_iters, verbose=verbose)
        for idx in range(len(self.kf_pose)):
            self.kf_pose[idx] = self.graph.nodes[idx].copy()
        return info

    # ------------------------------------------------------------------ #
    def correction(self, idx: int) -> np.ndarray:
        """World-frame correction ``T_corr @ inv(T_orig)`` for keyframe ``idx``.

        Apply to any pose anchored to that keyframe:
        ``T_new = correction(idx) @ T_old``.
        """
        return self.kf_pose[idx] @ se3_inv(self.kf_orig[idx])
