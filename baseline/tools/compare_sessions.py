#!/usr/bin/env python3
"""Compare two pose streams from recorded sessions.

Two modes:

1. Single-session VIO vs SLAM (default)::

       ./tools/compare_sessions.py sessions/loop1

   Compares ``basalt/vio_pose.jsonl`` against ``basalt/slam_pose.jsonl`` of
   the same session — useful to quantify how much drift the SLAM loop
   closure corrects on top of pure VIO.

2. Two-session comparison (skyslam vs basalt baseline)::

       ./tools/compare_sessions.py basalt_session skyslam_session \\
           --ref-stream slam --test-stream slam

Metrics
-------
- ATE  : Absolute Trajectory Error.  Aligns the test trajectory to the
         reference with SE(3) Umeyama (no scale), then reports RMSE,
         mean, median, max in metres.
- RPE  : Relative Pose Error.  For each pair of samples ``delta_s`` apart
         (default 1 s), compares the relative motion in the reference
         vs the test trajectory.  Reports translation (m) and rotation
         (deg) errors.

Both inputs are sampled at the reference timestamps via linear
interpolation in position and SLERP in quaternion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# Loaders (duplicated minimal from viz_session.py — keep tool standalone)
# ============================================================================

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_pose_stream(session: Path, stream: str) -> dict[str, np.ndarray]:
    """stream in {'vio', 'slam'}; returns ts_s (N), pos (N,3), quat (N,4 wxyz)."""
    fname = {"vio": "vio_pose.jsonl", "slam": "slam_pose.jsonl"}[stream]
    recs = _load_jsonl(session / "basalt" / fname)
    if not recs:
        raise SystemExit(f"empty or missing: {session}/basalt/{fname}")
    ts = np.array([r["ts_ns"] for r in recs], dtype=np.float64) / 1e9
    pos = np.array([r["pos"] for r in recs], dtype=np.float64)
    quat = np.array([r["quat_wxyz"] for r in recs], dtype=np.float64)
    # normalize quaternions defensively
    quat /= np.linalg.norm(quat, axis=1, keepdims=True).clip(min=1e-12)
    return {"ts_s": ts, "pos": pos, "quat": quat}


# ============================================================================
# Quaternion + SE3 math (minimal, no scipy dep)
# ============================================================================

def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(...,4) wxyz -> (...,3,3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two unit quaternions (wxyz)."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    th0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sth0 = np.sin(th0)
    s0 = np.sin((1 - t) * th0) / sth0
    s1 = np.sin(t * th0) / sth0
    return s0 * q0 + s1 * q1


def _resample(src_ts: np.ndarray, src_pos: np.ndarray, src_quat: np.ndarray,
              ref_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (pos, quat) at ref_ts. Returns (mask, pos, quat).

    Reference times outside src_ts range are masked out.
    """
    mask = (ref_ts >= src_ts[0]) & (ref_ts <= src_ts[-1])
    if not mask.any():
        return mask, np.zeros((0, 3)), np.zeros((0, 4))
    rt = ref_ts[mask]
    # position: linear interp per axis
    pos_i = np.column_stack([np.interp(rt, src_ts, src_pos[:, i]) for i in range(3)])
    # quaternion: SLERP between bracketing samples
    idx_hi = np.searchsorted(src_ts, rt)
    idx_hi = np.clip(idx_hi, 1, len(src_ts) - 1)
    idx_lo = idx_hi - 1
    t0 = src_ts[idx_lo]; t1 = src_ts[idx_hi]
    a = (rt - t0) / np.clip(t1 - t0, 1e-9, None)
    quat_i = np.empty((len(rt), 4))
    for k in range(len(rt)):
        quat_i[k] = _slerp(src_quat[idx_lo[k]], src_quat[idx_hi[k]], float(a[k]))
    return mask, pos_i, quat_i


def _umeyama_se3(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid SE(3) alignment src -> dst (no scale).

    Returns (R, t) such that dst ≈ R @ src.T + t[:, None].
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = src - mu_s
    D = dst - mu_d
    H = S.T @ D
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = mu_d - R @ mu_s
    return R, t


# ============================================================================
# Metrics
# ============================================================================

def ate(ref_pos: np.ndarray, test_pos: np.ndarray, *,
        align: bool = True) -> dict:
    """Absolute Trajectory Error after Umeyama SE(3) alignment."""
    if align and len(ref_pos) >= 3:
        R, t = _umeyama_se3(test_pos, ref_pos)
        aligned = (R @ test_pos.T).T + t
    else:
        aligned = test_pos
        R, t = np.eye(3), np.zeros(3)
    err = np.linalg.norm(aligned - ref_pos, axis=1)
    return {
        "n": int(len(err)),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))) if len(err) else 0.0,
        "mean_m": float(np.mean(err)) if len(err) else 0.0,
        "median_m": float(np.median(err)) if len(err) else 0.0,
        "max_m": float(np.max(err)) if len(err) else 0.0,
        "min_m": float(np.min(err)) if len(err) else 0.0,
        "align_R": R.tolist(),
        "align_t": t.tolist(),
        "err_per_sample": err,
        "aligned": aligned,
    }


def rpe(ref_ts: np.ndarray, ref_pos: np.ndarray, ref_quat: np.ndarray,
        test_pos: np.ndarray, test_quat: np.ndarray,
        delta_s: float = 1.0) -> dict:
    """Relative Pose Error over fixed time window.

    Both inputs must be on the same timestamps ``ref_ts``.
    """
    pairs = []
    j = 0
    n = len(ref_ts)
    for i in range(n):
        target = ref_ts[i] + delta_s
        while j < n and ref_ts[j] < target:
            j += 1
        if j >= n:
            break
        pairs.append((i, j))
    if not pairs:
        return {"n": 0, "trans_rmse_m": 0.0, "rot_rmse_deg": 0.0,
                "trans_err": np.zeros(0), "rot_err_deg": np.zeros(0)}

    R_ref = _quat_to_rot(ref_quat)
    R_test = _quat_to_rot(test_quat)
    trans_err = np.empty(len(pairs))
    rot_err = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        # relative motion in ref frame
        dR_ref = R_ref[i].T @ R_ref[j]
        dt_ref = R_ref[i].T @ (ref_pos[j] - ref_pos[i])
        dR_test = R_test[i].T @ R_test[j]
        dt_test = R_test[i].T @ (test_pos[j] - test_pos[i])

        # translation residual
        trans_err[k] = np.linalg.norm(dt_test - dt_ref)
        # rotation residual = angle of R_err
        R_err = dR_ref.T @ dR_test
        c = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
        rot_err[k] = np.degrees(np.arccos(c))

    return {
        "n": len(pairs),
        "delta_s": float(delta_s),
        "trans_rmse_m": float(np.sqrt(np.mean(trans_err ** 2))),
        "trans_mean_m": float(np.mean(trans_err)),
        "trans_max_m": float(np.max(trans_err)),
        "rot_rmse_deg": float(np.sqrt(np.mean(rot_err ** 2))),
        "rot_mean_deg": float(np.mean(rot_err)),
        "rot_max_deg": float(np.max(rot_err)),
        "trans_err": trans_err,
        "rot_err_deg": rot_err,
    }


# ============================================================================
# CLI
# ============================================================================

def _print_ate(label: str, m: dict) -> None:
    print(f"  ATE  ({label})")
    print(f"    n      : {m['n']}")
    print(f"    rmse   : {m['rmse_m']*100:8.2f} cm")
    print(f"    mean   : {m['mean_m']*100:8.2f} cm")
    print(f"    median : {m['median_m']*100:8.2f} cm")
    print(f"    max    : {m['max_m']*100:8.2f} cm")


def _print_rpe(label: str, m: dict) -> None:
    print(f"  RPE  ({label}, Δt={m['delta_s']}s)")
    print(f"    n           : {m['n']}")
    print(f"    trans rmse  : {m['trans_rmse_m']*100:8.2f} cm")
    print(f"    trans mean  : {m['trans_mean_m']*100:8.2f} cm")
    print(f"    trans max   : {m['trans_max_m']*100:8.2f} cm")
    print(f"    rot rmse    : {m['rot_rmse_deg']:8.2f} deg")
    print(f"    rot mean    : {m['rot_mean_deg']:8.2f} deg")
    print(f"    rot max     : {m['rot_max_deg']:8.2f} deg")


def _path_length(pos: np.ndarray) -> float:
    if len(pos) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))


def _summary_dict(label: str, ate_m: dict, rpe_m: dict, ref_len: float,
                  test_len: float, t_span: float) -> dict:
    return {
        "label": label,
        "ate": {k: ate_m[k] for k in
                ("n", "rmse_m", "mean_m", "median_m", "max_m", "min_m")},
        "rpe": {k: rpe_m[k] for k in
                ("n", "delta_s", "trans_rmse_m", "trans_mean_m", "trans_max_m",
                 "rot_rmse_deg", "rot_mean_deg", "rot_max_deg")},
        "ref_path_length_m": ref_len,
        "test_path_length_m": test_len,
        "time_span_s": t_span,
    }


def compare(ref: dict, test: dict, label: str, delta_s: float, no_align: bool,
            verbose: bool = True) -> dict:
    """Resample test to ref timestamps, compute ATE + RPE, print summary."""
    mask, test_pos, test_quat = _resample(test["ts_s"], test["pos"],
                                          test["quat"], ref["ts_s"])
    if not mask.any():
        print(f"  [{label}] no time overlap, skipped")
        return {}
    ref_ts = ref["ts_s"][mask]
    ref_pos = ref["pos"][mask]
    ref_quat = ref["quat"][mask]
    t_span = float(ref_ts[-1] - ref_ts[0])

    ate_m = ate(ref_pos, test_pos, align=not no_align)
    # use aligned test in pose space too, so RPE is in ref frame
    aligned_test_pos = ate_m["aligned"]
    # rotate test quaternions by alignment R as well
    R_align = np.array(ate_m["align_R"])
    R_test = _quat_to_rot(test_quat)
    R_test_aligned = R_align[None, :, :] @ R_test
    # rot -> quat
    aligned_test_quat = np.empty_like(test_quat)
    for i, R in enumerate(R_test_aligned):
        aligned_test_quat[i] = _rot_to_quat(R)

    rpe_m = rpe(ref_ts, ref_pos, ref_quat,
                aligned_test_pos, aligned_test_quat, delta_s=delta_s)

    if verbose:
        print(f"[{label}]")
        print(f"  time span  : {t_span:.2f}s  ({len(ref_ts)} ref samples)")
        print(f"  ref path   : {_path_length(ref_pos)*100:8.2f} cm")
        print(f"  test path  : {_path_length(aligned_test_pos)*100:8.2f} cm")
        _print_ate(label, ate_m)
        _print_rpe(label, rpe_m)
        print()

    return _summary_dict(label, ate_m, rpe_m,
                         _path_length(ref_pos),
                         _path_length(aligned_test_pos), t_span)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 -> wxyz quaternion."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_session", help="reference session folder")
    ap.add_argument("test_session", nargs="?", default=None,
                    help="test session folder (default: same as ref → "
                         "compare VIO vs SLAM within ref)")
    ap.add_argument("--ref-stream", default="slam", choices=("vio", "slam"),
                    help="reference pose stream (default: slam)")
    ap.add_argument("--test-stream", default="vio", choices=("vio", "slam"),
                    help="test pose stream (default: vio)")
    ap.add_argument("--delta-s", type=float, default=1.0,
                    help="RPE window in seconds (default 1.0)")
    ap.add_argument("--no-align", action="store_true",
                    help="skip SE3 Umeyama alignment before ATE/RPE")
    ap.add_argument("--out", default=None,
                    help="write summary JSON to this file")
    args = ap.parse_args()

    ref_dir = Path(args.ref_session).resolve()
    if not ref_dir.is_dir():
        print(f"not a directory: {ref_dir}", file=sys.stderr)
        return 2
    test_dir = Path(args.test_session).resolve() if args.test_session else ref_dir
    if not test_dir.is_dir():
        print(f"not a directory: {test_dir}", file=sys.stderr)
        return 2

    same = (ref_dir == test_dir)
    if same and args.ref_stream == args.test_stream:
        print("ref and test point to the same stream — nothing to compare.",
              file=sys.stderr)
        return 2

    print(f"REF   : {ref_dir.name} :: {args.ref_stream}")
    print(f"TEST  : {test_dir.name} :: {args.test_stream}")
    print(f"align : {'OFF' if args.no_align else 'SE(3) Umeyama'}")
    print()

    ref = _load_pose_stream(ref_dir, args.ref_stream)
    test = _load_pose_stream(test_dir, args.test_stream)

    summary = compare(ref, test,
                      label=f"{args.test_stream} vs {args.ref_stream}",
                      delta_s=args.delta_s,
                      no_align=args.no_align)

    if args.out:
        out = Path(args.out)
        # drop heavy arrays before writing
        summary.pop("err_per_sample", None)
        out.write_text(json.dumps(summary, indent=2))
        print(f"summary written: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
