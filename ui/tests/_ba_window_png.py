#!/usr/bin/env python3
"""End-to-end PNG proof for the "BA Window" visualiser.

Boots the SPLIT 2-process stack (imu_camera replay + vio with ``--ba-window``) on
a gold session, drives the REAL
:class:`~ui.modules.ipc_sources.IpcBaWindowSource` (it subscribes VIO's
``ba.window`` solve snapshots over IPC and buffers them in its slider deque),
renders the window's 2D top-down image with the REAL
:func:`~ui.viz.ba_render.render_ba_window`, and writes it to a PNG. No OpenGL is
involved (the BA view is pure 2D), so this runs headless.

Asserts the captured snapshots are real (a non-trivial keyframe window, shared
landmarks, observation rays, a finite reprojection error), that the source's
slider buffer accumulated them, and that the rendered PNG is non-blank with the
reprojection-error-coloured rays drawn.

Run::

    .venv/bin/python -m ui.tests._ba_window_png
    .venv/bin/python -m ui.tests._ba_window_png --out /tmp/ba_window.png
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.comms import IPCPubSub                                    # noqa: E402
from ui.modules import IpcBaWindowSource                          # noqa: E402
from ui.viz.ba_render import render_ba_window, _GOOD, _BAD       # noqa: E402


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def _await_calib(endpoint: str, timeout_s: float):
    got = threading.Event()
    box = [None]

    def on(wm):
        box[0] = wm
        got.set()
    c = IPCPubSub(endpoint, role="client", connect_timeout_s=timeout_s)
    c.subscribe("calib.bundle", on)
    c.start()
    try:
        if not got.wait(timeout=timeout_s):
            raise TimeoutError(f"no calib.bundle from {endpoint!r}")
    finally:
        c.stop()
    return box[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--out", default="/tmp/ba_window_lab_loop_30s.png")
    args = ap.parse_args()

    pid = os.getpid()
    cap_ep = f"oak.cap.bw{pid & 0xFFF:x}"
    vio_ep = f"oak.vio.bw{pid & 0xFFF:x}"
    py = sys.executable
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    lk = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}

    print("ba_window_png")
    print(f"  session={args.session} max-frames={args.max_frames}")

    vio_proc = subprocess.Popen(
        [py, "-m", "vio.main", "--capture-endpoint", cap_ep,
         "--endpoint", vio_ep, "--kf-every", str(args.kf_every),
         "--ba-window"], env=env, **lk)
    time.sleep(0.3)
    cap_proc = subprocess.Popen(
        [py, "-m", "imu_camera.main", "--endpoint", cap_ep,
         "--session", args.session, "--max-frames", str(args.max_frames)],
        env=env, **lk)
    procs = (cap_proc, vio_proc)

    snaps: list = []
    lock = threading.Lock()
    src = None
    try:
        bundle = _await_calib(vio_ep, timeout_s=25.0)
        W, H = int(bundle.width), int(bundle.height)
        print(f"  vio ready ({W}x{H})")

        # The REAL window source: ba.window solve snapshots, buffered for the slider.
        def on_snap(s) -> None:
            with lock:
                snaps.append(s)

        src = IpcBaWindowSource(vio_ep, connect_timeout_s=25.0)
        src.start(on_snap)

        cap_proc.wait(timeout=180.0)
        vio_proc.wait(timeout=180.0)
        time.sleep(1.0)                            # drain in-flight snapshots

        _check(src.error is None, f"source has no connect error ({src.error})")
        with lock:
            got = list(snaps)
        print(f"  captured {len(got)} BA-window snapshot(s)")
        _check(len(got) >= 1, "at least one ba.window snapshot reached the source")
        _check(src.snapshot_count() >= 1,
               f"the slider buffer accumulated snapshots "
               f"(count={src.snapshot_count()})")

        # Prefer a rich snapshot (most keyframes, then most observation rays).
        def score(m):
            return (int(m.n_kf), int(len(np.asarray(m.obs_kf))))
        m = max(got, key=score)
        n_obs = int(len(np.asarray(m.obs_kf)))
        ids = np.asarray(m.kf_ids)
        print(f"  chosen snapshot: seq {m.seq}  kf {m.n_kf}  lm {m.n_lm}  "
              f"obs {n_obs}  reproj {m.ba_reproj_px:.3f} px  "
              f"KF id {int(ids.min())}-{int(ids.max())}")
        _check(int(m.n_kf) >= 2, f"snapshot carries a real window (n_kf={m.n_kf})")
        _check(int(m.n_lm) >= 1 and int(m.n_lm) <= 100,
               f"snapshot carries shared landmarks within the cap (n_lm={m.n_lm})")
        _check(n_obs >= 1, f"snapshot carries observation rays (obs={n_obs})")
        obs_kf = np.asarray(m.obs_kf)
        obs_lm = np.asarray(m.obs_lm)
        _check(int(obs_kf.max()) < int(m.n_kf) and int(obs_lm.max()) < int(m.n_lm),
               "observation indices are in range for kf_ids / lm_ids")
        _check(np.isfinite(m.ba_reproj_px),
               f"window reprojection error is finite ({m.ba_reproj_px})")
        _check(m.kf_pos_pre.shape == m.kf_pos.shape
               and m.lm_xyz_pre.shape == m.lm_xyz.shape,
               "pre-solve poses/landmarks present for the before/after toggle")

        img = render_ba_window(m, 1100, 620, show_pre=False)
        _check(img.shape == (620, 1100, 3) and img.dtype == np.uint8,
               f"rendered (620,1100,3) uint8 (got {img.shape} {img.dtype})")
        # Non-blank: the geometry is actually drawn.
        nonbg = int((img.reshape(-1, 3) != np.array([13, 17, 23])).any(1).sum())
        _check(nonbg > 1000, f"rendered canvas is non-blank ({nonbg} px drawn)")
        # before/after toggle actually changes pixels on the real snapshot.
        img_pre = render_ba_window(m, 1100, 620, show_pre=True)
        d = int((img != img_pre).any(2).sum())
        _check(d > 100, f"before/after toggle changes pixels ({d} px)")

        # Persist (cv2 wants BGR; the canvas is RGB).
        import cv2
        out = Path(args.out)
        cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        flat = img.reshape(-1, 3).astype(int)
        green = int((np.abs(flat - np.array(_GOOD)).sum(1) < 60).sum())
        red = int((np.abs(flat - np.array(_BAD)).sum(1) < 60).sum())
        print(f"\n  wrote {out}  ({green} green / {red} red reprojection px)")
        print("BA-WINDOW PNG PASS")
        return 0
    finally:
        if src is not None:
            try:
                src.stop()
            except Exception:                                      # noqa: BLE001
                pass
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:                                  # noqa: BLE001
                    pass
        for p in procs:
            try:
                p.wait(timeout=5.0)
            except Exception:                                      # noqa: BLE001
                try:
                    p.kill()
                except Exception:                                  # noqa: BLE001
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
