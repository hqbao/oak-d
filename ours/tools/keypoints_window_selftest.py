#!/usr/bin/env python3
"""Headless self-test for the keypoint depth-tracker window + its overlay lib.

Two layers, both offline (no device):

1. **Library** (:mod:`ours.lib.viz.keypoint_overlay`): the per-id trail buffer,
   depth sampling, the depth->colour mapping (must be bit-identical to the depth
   panel's :func:`colorize_depth`), and the overlay drawing.
2. **Window** (:class:`ours.ui.keypoints_window.KeypointTrackWindow`): driven off
   a recorded session through :class:`ReplayKeypointWorker` under the offscreen Qt
   platform -- asserts it runs our REAL KLT frontend, renders the overlay pixmap,
   keeps track ids alive across frames, and prints the honest footer stats. The
   live OAK-D path is the only part left to the bench.

Run::

    QT_QPA_PLATFORM=offscreen python -m ours.tools.keypoints_window_selftest
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np                                                  # noqa: E402

from PyQt6.QtWidgets import QApplication                            # noqa: E402

from ours.lib.viz.depth_render import (                             # noqa: E402
    colorize_depth, turbo_bgr, turbo_bgr_array,
)
from ours.lib.viz.keypoint_overlay import (                         # noqa: E402
    TRAIL_LEN, TrackTrails, draw_overlay, marker_sizes, sample_depths,
)
from ours.ui.keypoints_window import (                              # noqa: E402
    KeypointTrackWindow, KeypointWorker, ReplayKeypointWorker,
)

_SESSION = "sessions/gold/lab_loop_30s"
_MAX_FRAMES = 24


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def _run_until(app, predicate, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline and not predicate():
        app.processEvents()
        time.sleep(0.005)


# --------------------------------------------------------------------------- #
# Library-level tests (pure, no Qt)
# --------------------------------------------------------------------------- #
def test_depth_color_matches_panel() -> None:
    print(" depth->colour is the SAME mapping as the depth panel")
    # Build a 1-row depth image of distinct depths; colorize_depth must agree
    # with the per-value turbo_bgr_array exactly (single source of truth).
    depths = np.array([0.3, 1.0, 2.5, 4.0, 8.0, 12.0], dtype=np.float32)
    img = depths.reshape(1, -1)
    panel = colorize_depth(img)[0]                  # (M, 3) BGR
    dots = turbo_bgr_array(depths)                  # (M, 3) BGR
    _check(np.array_equal(panel, dots),
           "turbo_bgr_array == colorize_depth pixel for pixel")
    _check(turbo_bgr(2.5) == tuple(int(x) for x in dots[2]),
           "scalar turbo_bgr matches the array form")


def test_sample_depths() -> None:
    print(" per-keypoint depth sampling (incl. invalid + out-of-bounds)")
    depth = np.zeros((10, 10), np.float32)
    depth[5, 5] = 2.0
    pts = np.array([[5, 5], [0, 0], [99, 99], [4, 5]], dtype=np.float32)
    z = sample_depths(depth, pts)
    _check(abs(z[0] - 2.0) < 1e-6, "valid keypoint reads its depth (2.0 m)")
    _check(z[1] == 0.0 and z[3] == 0.0, "no-stereo keypoints read 0 (invalid)")
    _check(z[2] == 0.0, "out-of-bounds keypoint clamps + reads 0 (no crash)")
    _check(sample_depths(depth, np.empty((0, 2))).shape == (0,),
           "empty keypoint set returns empty (no crash)")


def test_trails() -> None:
    print(" per-id trail buffer: continuity, cap=20, age, eviction")
    tr = TrackTrails(trail_len=TRAIL_LEN, max_stale=2)
    # Track id 7 moves right one px per frame for 25 frames.
    for f in range(25):
        ids = np.array([7, 8], dtype=np.int64)
        pts = np.array([[f, 0], [0, 0]], dtype=np.float32)
        tr.update(ids, pts)
    trail7 = tr.trail(7)
    _check(len(trail7) == TRAIL_LEN, f"trail capped at N={TRAIL_LEN} (got {len(trail7)})")
    _check(trail7[-1][0] == 24.0, "trail keeps the most recent position")
    _check(trail7[0][0] == 24.0 - (TRAIL_LEN - 1), "trail drops the oldest beyond N")
    _check(tr.age(7) == 24, f"age counts frames since first seen (got {tr.age(7)})")
    # Now stop reporting id 8 -> it must be evicted after max_stale frames.
    for f in range(25, 29):
        tr.update(np.array([7], np.int64), np.array([[f, 0]], np.float32))
    _check(tr.trail(8) == [], "stale track evicted after max_stale frames")
    _check(tr.new_count == 0, "no new tracks on a steady frame")
    tr.update(np.array([7, 99], np.int64), np.array([[29, 0], [1, 1]], np.float32))
    _check(tr.new_count == 1, "brand-new id counted in new_count")


def test_marker_scaling() -> None:
    print(" marker radii scale with frame size (small frame -> small dots)")
    big = marker_sizes((800, 1280))
    ref = marker_sizes((400, 640))
    small = marker_sizes((200, 320))
    _check(big[0] > ref[0] > small[0],
           f"dot radius scales with frame size ({small[0]}<{ref[0]}<{big[0]})")
    _check(ref[0] == 3, f"reference 640x400 keeps the tuned dot r=3 (got {ref[0]})")
    _check(all(v >= 1 for v in small),
           f"tiny frame still draws visible markers (>=1px) {small}")
    _check(big[1] > big[0] and big[2] > big[0],
           "halo + fresh-ring stay larger than the dot at any scale")


def test_draw_overlay() -> None:
    print(" overlay draws colour dots for valid + neutral for invalid depth")
    gray = np.full((40, 60), 100, np.uint8)
    depth = np.zeros((40, 60), np.float32)
    depth[20, 30] = 1.0                              # one valid keypoint
    ids = np.array([1, 2], dtype=np.int64)
    pts = np.array([[30, 20], [10, 10]], dtype=np.float32)   # 2nd has no depth
    tr = TrackTrails()
    tr.update(ids, pts)
    rgb = draw_overlay(gray, depth, ids, pts, tr)
    _check(rgb.shape == (40, 60, 3) and rgb.dtype == np.uint8,
           "overlay is an (H,W,3) uint8 RGB image")
    # The valid keypoint must paint a saturated (coloured) pixel near (30,20):
    patch = rgb[18:23, 28:33].reshape(-1, 3).astype(np.int32)
    sat = (patch.max(axis=1) - patch.min(axis=1)).max()
    _check(sat > 40, f"valid keypoint painted a coloured (non-grey) dot (sat={sat})")
    _check(rgb.max() > 0, "overlay is not blank")
    # Empty keypoint set must not crash and returns the (dimmed) background.
    blank = draw_overlay(gray, depth, np.empty((0,), np.int64),
                         np.empty((0, 2), np.float32), TrackTrails())
    _check(blank.shape == (40, 60, 3), "empty keypoint set returns a frame, no crash")


# --------------------------------------------------------------------------- #
# Window-level tests (offscreen Qt)
# --------------------------------------------------------------------------- #
def _replay_factory():
    return lambda: ReplayKeypointWorker(_SESSION, fps=120, max_frames=_MAX_FRAMES)


class _DeadWorker(KeypointWorker):
    """A worker that fails to open -- mimics a missing/busy OAK-D."""

    mode = "LIVE"

    def _frames(self):
        raise RuntimeError("X_LINK_DEVICE_NOT_FOUND")
        yield  # pragma: no cover  (makes this a generator)


def test_window_happy_path(app) -> None:
    print(" replay (happy path): real frontend + overlay render")
    win = KeypointTrackWindow(_replay_factory(), fps=120)
    win.resize(900, 700)
    win.show()                                       # showEvent -> start()
    _check(win._running, "window started the replay worker on show")
    _check(win._mode_pill.text() == "REPLAY", "mode pill shows REPLAY")

    # Drive the UI and read the rendered footer state after enough ticks (the
    # window's own timer drains the worker queue and renders the overlay). The
    # gold session's first frames are blown-out white (auto-exposure warmup) and
    # honestly show 0 keypoints, so we keep going until a populated frame renders.
    deadline = time.time() + 15.0
    last_seq = -1
    seen_seqs: list[int] = []
    max_trk = 0
    while time.time() < deadline and not win._ended:
        app.processEvents()
        time.sleep(0.005)
        txt = win._status.text()
        if txt.startswith("trk") and "SEQ" in txt:
            seq = int(txt.split("SEQ")[1].split()[0])
            max_trk = max(max_trk, int(txt.split("trk")[1].split()[0]))
            if seq != last_seq:
                seen_seqs.append(seq)
                last_seq = seq
        if len(seen_seqs) >= 8 and max_trk > 0:
            break

    pix = win._view.pixmap()
    _check(pix is not None and not pix.isNull(), "frame panel shows the overlay pixmap")
    _check(len(seen_seqs) >= 1, f"reported rendered frames (saw seq {seen_seqs[:5]})")
    _check(max_trk > 0, f"a populated frame rendered real keypoints (max trk={max_trk})")
    txt = win._status.text()
    for token in ("trk", "valid-z", "mean-age", "new", "SEQ"):
        _check(token in txt, f"footer reports honest stat '{token}' ({txt!r})")
    win.close()
    _check(not win._running, "window stopped the worker on close")


def test_window_frontend_continuity(app) -> None:
    print(" replay worker: ids persist + trails grow across frames")
    w = ReplayKeypointWorker(_SESSION, fps=240, max_frames=12)
    w.start()
    got: list = []
    deadline = time.time() + 12.0
    while time.time() < deadline:
        try:
            item = w.queue.get(timeout=0.2)
        except Exception:
            continue
        if item is None:
            break
        got.append(item)
    w.stop()
    _check(len(got) >= 5, f"worker produced samples (n={len(got)})")
    _check(all(s.rgb.ndim == 3 for s in got), "every sample carries an RGB overlay")
    _check(any(s.n_tracks > 0 for s in got), "the real KLT frontend found tracks")
    # Track ids from the first populated frame should reappear later (the SAME
    # keypoint is tracked across frames -- the whole point of the view).
    early = next((set(int(i) for i in s.ids) for s in got if s.n_tracks > 0), set())
    later = set()
    for s in got[3:]:
        later |= set(int(i) for i in s.ids)
    _check(len(early & later) > 0,
           f"persistent ids survive across frames (|shared|={len(early & later)})")
    _check(any(s.mean_age > 0.5 for s in got[3:]),
           "tracks accumulate age (trails span multiple frames)")
    _check(any(s.n_valid > 0 for s in got),
           "some keypoints carry a valid metric depth")


def test_window_device_absent(app) -> None:
    print(" device absent -> clean fail (no hang)")
    win = KeypointTrackWindow(lambda: _DeadWorker(), fps=120)
    win._startup_timeout_s = 6.0
    win.show()
    _run_until(app, lambda: win._failed, 8.0)
    _check(win._failed, "window surfaced the open failure")
    _check(not win._running, "failed window released the worker")
    win.close()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    _check(Path(_SESSION).exists(), f"gold session present: {_SESSION}")
    test_depth_color_matches_panel()
    test_sample_depths()
    test_trails()
    test_marker_scaling()
    test_draw_overlay()
    test_window_happy_path(app)
    test_window_frontend_continuity(app)
    test_window_device_absent(app)
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    print("keypoints_window_selftest")
    raise SystemExit(main())
