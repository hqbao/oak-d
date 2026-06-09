#!/usr/bin/env python3
"""Headless smoke test for the proc4 UI data path (ported to the ``ui`` project).

Spawns the 4-process stack using the SPLIT project entrypoints --
``imu_camera.main`` (replay) + ``vio.main`` + ``slam.main`` -- then drives a
:class:`~ui.main.IpcPoseSource` (the single-view's live marker) and a
:class:`~ui.main.SlamMapTracker` (the single model behind all 5 trajectory lines)
IN-PROCESS. Asserts that:

* the source emits one NED :class:`~ui.comms.lib.misc.pose.Pose` per replay frame,
  with finite positions and unit quaternions,
* the SLAM tracker shows CONTINUOUS keyframe dots from the ``slam.map`` stream
  even with ZERO loop closures (the bug fix: the old ``loop.correction``-driven
  tracker showed no dots along the path until the first loop fired),
* the tracker accumulates the VO (``pose.vo``) and VIO-BA (``pose.refined``)
  trails the 5-line view needs, and ``corrected_vio_snapshot`` returns the
  ``(positions, teleport_flags)`` contract the viewer's teleport-colouring reads,
* the Qt single-view MainWindow can be CONSTRUCTED with its 5 toggle buttons
  (exact labels) and a :class:`~ui.qt.viewer3d.Viewer3D` carrying its 5 trajectory
  line items (we don't enter the event loop because the test runs headless on CI /
  pyqtgraph + OpenGL might not be available -- but the construction proves the
  imports + wiring).

Menu coverage (the proc4 UI's restored View / Visualize / Calibration menus)
----------------------------------------------------------------------------
On top of the pose-path smoke test, :func:`test_menus` boots a real
``imu_camera(replay) + vio`` stack over IPC, builds the SAME menu actions
``run_ui`` builds (wired to the SAME IPC adapters in :mod:`ui.modules.ipc_sources`),
and exercises each one offscreen:

* **View** -- trigger a preset QAction and assert the viewer's camera moved.
* **Visualize -> triplet** -- open the :class:`~ui.qt.synced_window.SyncedViewWindow`
  and assert its worker's queue receives >= 1 ``TripletSample``.
* **Visualize -> keypoint** -- open the
  :class:`~ui.qt.keypoints_window.KeypointTrackWindow` and assert >= 1
  ``KeypointSample`` arrives, then close.
* **Visualize -> SLAM Map (3D room)** -- accumulate keyframes through an
  :class:`~ui.modules.ipc_sources.IpcSlamMapSource`, run its temporal occupancy
  fusion (``_build``), assert occupied voxel centres + green-by-height colours,
  and that :class:`~ui.qt.map_window.MapWindow` ingests them without raising.
* **Calibration -> Gyroscope Bias** -- driven at the adapter seam:
  :func:`test_imu_raw_source` publishes a known ``WireImuRaw`` batch on a test IPC
  server's ``imu.raw`` and asserts the :class:`~ui.modules.ipc_sources.IpcImuRawSource`
  callback fires once per sample with the right ``(3,)`` gyro/accel and ``t`` in
  SECONDS -- the exact contract the modal calib dialog's collector consumes.

Run::

    python -m ui.tests.ui_dataflow_selftest
    python -m ui.tests.ui_dataflow_selftest --no-menus   # pose-path only
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.comms.lib.misc.pose import Pose, PoseHistory                # noqa: E402
from ui.main import (                                               # noqa: E402
    IpcPoseSource, SlamMapTracker, _await_calib_bundle,
)


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--no-qt", action="store_true",
                    help="skip the Qt MainWindow construction test")
    ap.add_argument("--no-menus", action="store_true",
                    help="skip the View/Visualize/Calibration menu coverage")
    args = ap.parse_args()

    pid = os.getpid()
    cap_ep = f"oak.cap.u{pid & 0xFFF:x}"
    vio_ep = f"oak.vio.u{pid & 0xFFF:x}"
    slam_ep = f"oak.slm.u{pid & 0xFFF:x}"

    py = sys.executable
    base_env = dict(os.environ)
    log_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}

    print("ui_dataflow_selftest")
    print(f"  session={args.session} max-frames={args.max_frames}")

    vio_proc = subprocess.Popen(
        [py, "-m", "vio.main",
         "--capture-endpoint", cap_ep, "--endpoint", vio_ep,
         "--kf-every", str(args.kf_every)],
        env=base_env, **log_kwargs)
    slam_proc = subprocess.Popen(
        [py, "-m", "slam.main",
         "--vio-endpoint", vio_ep, "--endpoint", slam_ep],
        env=base_env, **log_kwargs)
    time.sleep(0.3)
    cap_proc = subprocess.Popen(
        [py, "-m", "imu_camera.main",
         "--endpoint", cap_ep, "--session", args.session,
         "--max-frames", str(args.max_frames)],
        env=base_env, **log_kwargs)

    procs = (cap_proc, vio_proc, slam_proc)

    try:
        # Wait for VIO + SLAM to be ready (their retained calib.bundle).
        _await_calib_bundle(vio_ep, timeout_s=20.0)
        print("  vio: ready")
        _await_calib_bundle(slam_ep, timeout_s=20.0)
        print("  slam: ready")

        # --- IpcPoseSource (live VIO marker) ---
        history = PoseHistory(capacity=10_000)
        source = IpcPoseSource(vio_ep, label="vio", connect_timeout_s=20.0)
        source.start(history.push)

        # --- SlamMapTracker (the 4 other lines) ---
        tracker = SlamMapTracker(slam_ep, vio_endpoint=vio_ep,
                                 connect_timeout_s=20.0)
        tracker.start()

        # Wait for the children to drain the whole replay (capture exits when
        # its replay ends; vio + slam exit when their END propagates).
        cap_proc.wait(timeout=60.0)
        vio_proc.wait(timeout=60.0)
        slam_proc.wait(timeout=60.0)

        # Give the source's recv thread a moment to drain any in-flight wire
        # messages that landed AFTER subprocess exit (capture's drain-on-close
        # buffers them but the local recv loop processes async).
        time.sleep(0.5)
        source.stop()
        tracker.stop()

        # ---------- Assertions ----------
        n_frames = args.max_frames
        pos, flags, latest = history.snapshot()
        snap_kf = tracker.refined_path_snapshot()
        snap_vo = tracker.vo_snapshot()
        snap_ba = tracker.ba_snapshot()
        corr_pos, corr_tp = tracker.corrected_vio_snapshot()
        _, _, _, n_loops = tracker.slam_overlay_snapshot()

        print(f"\n  received poses: {pos.shape[0]} (expected {n_frames})")
        print(f"  latest pose: {latest!r}")
        print(f"  slam kf snapshot: {snap_kf.shape} loops={n_loops}")
        print(f"  vo snapshot: {snap_vo.shape}  ba snapshot: {snap_ba.shape}")
        print(f"  corrected snapshot: pos={corr_pos.shape} "
              f"teleport={corr_tp.shape} ({int(corr_tp.sum())} flagged)")

        _check(cap_proc.returncode == 0,
               f"capture exited 0 (got {cap_proc.returncode})")
        _check(vio_proc.returncode == 0,
               f"vio exited 0 (got {vio_proc.returncode})")
        _check(slam_proc.returncode == 0,
               f"slam exited 0 (got {slam_proc.returncode})")
        _check(pos.shape[0] == n_frames,
               f"received one pose per frame (got {pos.shape[0]}/{n_frames})")
        _check(np.isfinite(pos).all(),
               "all positions finite (no NaN/inf from the NED conversion)")
        _check(isinstance(latest, Pose),
               "latest snapshot is a Pose")
        _check(latest is not None
               and np.isclose(float(np.linalg.norm(latest.quat_wxyz)), 1.0,
                              atol=1e-3),
               "latest quaternion is unit-norm")
        _check(snap_kf.dtype == np.float32 and snap_kf.shape[1] == 3,
               f"slam kf snapshot is (K, 3) float32 (got "
               f"{snap_kf.shape} {snap_kf.dtype})")
        # Regression guard for the reported bug: the SLAM line must show keyframe
        # dots ALONG THE PATH, not only after a loop closure. The continuous
        # slam.map stream accumulates a dot per keyframe, so with this replay
        # (which produces keyframes but need not close a loop) the snapshot is
        # NON-EMPTY. The old loop.correction-only tracker left this empty.
        _check(snap_kf.shape[0] > 0,
               f"slam kf dots present without a loop closure (got "
               f"{snap_kf.shape[0]} dots, loops={n_loops})")

        # The single tracker also feeds the VO + VIO-BA lines: pose.vo is
        # published per frame, so its trail tracks the odom trail; pose.refined
        # is per keyframe, so it is sparser but non-empty once keyframes exist.
        _check(snap_vo.dtype == np.float32 and snap_vo.ndim == 2
               and snap_vo.shape[1] == 3,
               f"vo snapshot is (N, 3) float32 (got {snap_vo.shape} "
               f"{snap_vo.dtype})")
        _check(snap_vo.shape[0] == n_frames,
               f"vo trail has one point per frame (got {snap_vo.shape[0]}/"
               f"{n_frames})")
        _check(np.isfinite(snap_vo).all(),
               "all vo positions finite (no NaN/inf from the NED conversion)")
        _check(snap_ba.dtype == np.float32 and snap_ba.ndim == 2
               and snap_ba.shape[1] == 3,
               f"ba snapshot is (N, 3) float32 (got {snap_ba.shape} "
               f"{snap_ba.dtype})")
        _check(snap_ba.shape[0] > 0,
               f"ba (pose.refined) trail accumulated keyframe poses (got "
               f"{snap_ba.shape[0]})")

        # corrected_vio_snapshot returns (positions, teleport_flags) -- the
        # exact contract the viewer's per-vertex teleport colouring consumes.
        _check(corr_pos.dtype == np.float32 and corr_pos.ndim == 2
               and corr_pos.shape[1] == 3,
               f"corrected positions are (M, 3) float32 (got {corr_pos.shape} "
               f"{corr_pos.dtype})")
        _check(corr_tp.dtype == bool and corr_tp.shape == (corr_pos.shape[0],),
               f"corrected teleport flags are (M,) bool aligned with positions "
               f"(got {corr_tp.shape} {corr_tp.dtype})")

        # ---------- Optional: build the Qt MainWindow ----------
        if not args.no_qt:
            try:
                _try_build_qt(vio_ep, slam_ep)
                _check(True, "Qt MainWindow constructs without error")
            except Exception as e:                                 # noqa: BLE001
                # Headless CI may not have a display / OpenGL; soft-fail.
                print(f"  [skip] Qt build skipped: {e}")

        # ---------- Menu coverage (View / Visualize / Calibration) ----------
        if not args.no_menus:
            test_menus(args)

        print("\nALL UI DATAFLOW SELFTESTS PASSED")
        return 0
    finally:
        _terminate_all(*procs)
        # Surface failure context if anything went wrong.
        for name, proc in (("capture", cap_proc), ("vio", vio_proc),
                           ("slam", slam_proc)):
            try:
                _out, err = proc.communicate(timeout=2.0)
            except Exception:                                      # noqa: BLE001
                err = b""
            if err.strip():
                print(f"\n  --- {name}.stderr ---\n"
                      f"{err.decode(errors='replace')}",
                      file=sys.stderr)


# --------------------------------------------------------------------------- #
# Menu coverage: the restored View / Visualize / Calibration menus
# --------------------------------------------------------------------------- #
def test_imu_raw_source() -> None:
    """Unit-test the calib seam: IpcImuRawSource splits a wire IMU batch.

    Drives the Calibration menu's data path WITHOUT a modal dialog: publish a
    known ``WireImuRaw`` batch (M=3) on a test server's ``imu.raw`` and assert
    :class:`~ui.modules.ipc_sources.IpcImuRawSource` fires the dialog callback
    once per sample, each carrying the right ``(3,)`` gyro/accel row and a
    timestamp in SECONDS (ns -> s) -- the exact ``(3,)``-at-a-time, float-seconds
    contract the gyro/accel collectors consume.
    """
    print("\n  [menus] IpcImuRawSource (Calibration seam)")
    from ui.comms import topics
    from ui.comms import IPCPubSub
    from ui.comms.wire import WireImuRaw
    from ui.modules import IpcImuRawSource

    ep = f"oak.imuraw.t{os.getpid() & 0xFFF:x}"
    # Three deterministic samples; ns timestamps so we can verify the /1e9.
    m = 3
    imu_ts = np.array([1_000_000_000, 1_005_000_000, 1_010_000_000], np.int64)
    gyro = np.array([[0.01, 0.02, 0.03],
                     [0.04, 0.05, 0.06],
                     [0.07, 0.08, 0.09]], np.float64)
    accel = np.array([[0.10, 0.20, 9.80],
                      [0.11, 0.21, 9.81],
                      [0.12, 0.22, 9.82]], np.float64)
    batch = WireImuRaw(seq=0, ts_ns=int(imu_ts[0]),
                       imu_ts=imu_ts, gyro=gyro, accel=accel)

    server = IPCPubSub(ep, role="server", retain_topics={topics.IMU_RAW},
                       blocking=True)
    server.start()
    got: list[tuple] = []
    done = threading.Event()

    def cb(g, a, t_s) -> None:
        got.append((np.asarray(g, float).copy(), np.asarray(a, float).copy(),
                    float(t_s)))
        if len(got) >= m:
            done.set()

    src = IpcImuRawSource(ep, device_id="SELFTEST-DEV", connect_timeout_s=10.0)
    src.start(cb)
    try:
        # retained imu.raw: a subscriber that connects after publish still gets
        # it, but publish after subscribe is the live path -- give the client a
        # moment to register, then publish.
        time.sleep(0.3)
        server.publish(topics.IMU_RAW, batch)
        ok = done.wait(timeout=10.0)
    finally:
        src.stop()
        server.close()

    _check(src.error is None, f"IpcImuRawSource has no connect error ({src.error})")
    _check(ok and len(got) == m,
           f"callback fired once per sample (got {len(got)}/{m})")
    for i in range(m):
        g, a, t_s = got[i]
        _check(g.shape == (3,) and a.shape == (3,),
               f"sample {i} rows are (3,) (got gyro {g.shape}, accel {a.shape})")
        _check(np.allclose(g, gyro[i]) and np.allclose(a, accel[i]),
               f"sample {i} gyro/accel match the published batch")
        _check(np.isclose(t_s, float(imu_ts[i]) * 1e-9),
               f"sample {i} t is in SECONDS (got {t_s}, "
               f"want {float(imu_ts[i]) * 1e-9})")
    print(f"    [ok] 3 (3,) samples, seconds timestamps, device "
          f"{src.device_id!r}")


def _spawn_cap_vio(session: str, max_frames: int, kf_every: int):
    """Boot imu_camera(replay) + vio over IPC; return (procs, cap_ep, vio_ep).

    Mirrors the boot order in :func:`main`: vio first (it retries the connect),
    then capture. The caller awaits each endpoint's retained ``calib.bundle``
    before driving the menus.
    """
    pid = os.getpid()
    cap_ep = f"oak.cap.m{pid & 0xFFF:x}"
    vio_ep = f"oak.vio.m{pid & 0xFFF:x}"
    py = sys.executable
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    lk = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    vio_proc = subprocess.Popen(
        [py, "-m", "vio.main", "--capture-endpoint", cap_ep,
         "--endpoint", vio_ep, "--kf-every", str(kf_every)], env=env, **lk)
    time.sleep(0.3)
    cap_proc = subprocess.Popen(
        [py, "-m", "imu_camera.main", "--endpoint", cap_ep,
         "--session", session, "--max-frames", str(max_frames)],
        env=env, **lk)
    return (cap_proc, vio_proc), cap_ep, vio_ep


def _drain_samples(work_queue: "queue.Queue", timeout_s: float,
                   want: int = 1) -> list:
    """Collect up to ``want`` non-None items off a worker queue.

    The Visualize windows' workers ship finished samples then a ``None`` END
    sentinel. The window normally drains this on its UI timer; here we read it
    directly so the assertion sees the samples regardless of redraw timing.
    Stops early on the END sentinel or once ``want`` samples are in hand.
    """
    out: list = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and len(out) < want:
        try:
            item = work_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:                              # END sentinel
            break
        out.append(item)
    return out


def attach_slam_map_source(vio_ep: str, slam_ep: str, bundle):
    """Attach an :class:`IpcSlamMapSource` to VIO's ``keyframe`` feed (no build).

    Returns the attached, subscribed occupancy-voxel source so the caller can let
    it accumulate keyframes in the BACKGROUND WHILE the replay/VIO is still alive,
    then fuse + build the voxel map + ``stop()`` it afterwards. VIO shuts down once
    the replay sends END, so the keyframe client must be up before that happens --
    attaching it up front is what makes the build see live kf rings. (Only VIO's
    ``keyframe`` client is started here; slam.map only re-colours the camera trail,
    which the voxel build does not need for this coverage.)
    """
    from ui.modules import IpcSlamMapSource

    W, H, K = int(bundle.width), int(bundle.height), bundle.K
    src = IpcSlamMapSource(vio_ep, slam_ep, K, width=W, height=H,
                           connect_timeout_s=20.0)
    _check(src._attach_or_fail(),
           f"slam-map source attached VIO kf rings ({src.error})")
    client = src._make_keyframe_client()
    client.start()
    return src


def build_slam_map_from(src):
    """Run the occupancy-voxel source's OFF-thread ``_build`` ONCE and assert it.

    Asserts the persistent occupancy fusion produced occupied voxels (voxel centres
    + green-by-height colours of matching length, one camera-path point per
    keyframe), then returns ``(points, colors, cams)`` for the caller to feed the
    window. Confirms the shared ``_KeyframeAccumulator`` base feeds the occupancy
    map with the kf depth/pose feed. The caller ``stop()``s the source.
    """
    # Give the (already-subscribed) source time to drain keyframes out of VIO's kf
    # rings before we build -- capture has drained by now in the caller; this just
    # waits for the keyframes to land in the accumulator.
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline and len(src._kf_depth) < 3:
        time.sleep(0.2)
    n_kf = len(src._kf_depth)
    _check(n_kf >= 1, f"slam-map source accumulated >=1 keyframe (got {n_kf})")
    # This smoke run just PROVES the fuse -> build -> emit path runs and the window
    # ingests its output. The log-odds fusion occupies a grid cell from a single
    # un-carved hit (L_OCC=0.85 > L_OCC_THRESH=0.5), so even a handful of keyframes
    # FUSES the grid (the INTERNAL occupied set is non-empty); we assert THAT here.
    # The RENDER, however, is gated HIGHER at L_DISPLAY (only HIGH-confidence,
    # many-times-re-observed surfaces show -- the behind-wall noise filter), so with
    # only a few keyframes the RENDERED count may legitimately be 0; the displayed
    # count + the wall/noise separation are proven on the FULL corridor in
    # ui.tests._map_display_sweep, and the carving (noise removal) in
    # ui.tests.occupancy_selftest. So here we check the build's OUTPUT shapes (always
    # valid) + that fusion actually happened (internal occupied set non-empty).
    points, colors, cams = src._build()
    _check(points.ndim == 2 and points.shape[1] == 3
           and points.dtype == np.float32,
           f"voxel centres are (N,3) float32 ({points.shape}, {points.dtype})")
    _check(colors.shape == points.shape and colors.dtype == np.float32,
           f"voxel colours match the centres ({colors.shape} vs {points.shape})")
    _check(cams.ndim == 2 and cams.shape[1] == 3 and len(cams) == n_kf,
           f"one camera-path point per keyframe ({len(cams)} vs {n_kf})")
    # The occupancy fusion must have folded the keyframes into the grid (the room
    # actually built, not an all-empty grid) -- assert the INTERNAL occupied set
    # (>= L_OCC_THRESH), which a single hit clears, rather than the higher RENDER
    # gate that a few keyframes need not yet reach.
    with src._lock:                                            # noqa: SLF001
        n_internal = sum(1 for lo in src._log.values()        # noqa: SLF001
                         if lo >= src.L_OCC_THRESH)
    _check(n_internal > 0,
           f"occupancy fusion produced internally-occupied voxels (got {n_internal})")
    return points, colors, cams


def test_menus(args) -> None:
    """Build the run_ui menu actions against a live capture+vio and fire them.

    Constructs the exact widgets + QActions ``run_ui`` builds (the SAME viewer,
    the SAME ``ipc_triplet_factory`` / ``ipc_keypoint_factory`` /
    ``IpcImuRawSource`` adapters), then triggers a View preset, the two Visualize
    windows, and the Calibration seam -- proving every restored menu action runs
    over IPC without throwing.
    """
    print("\n  [menus] booting capture+vio for menu coverage ...")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # (d) Calibration seam first -- it is self-contained (its own tiny server)
    # and proves the IMU adapter before we stand up the heavier capture+vio.
    test_imu_raw_source()

    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QApplication

    from ui.qt.viewer3d import Viewer3D, VIEW_PRESETS
    from ui.qt.synced_window import SyncedViewWindow, TripletSample
    from ui.qt.keypoints_window import KeypointTrackWindow, KeypointSample
    from ui.modules import (
        ipc_triplet_factory, ipc_keypoint_factory,
    )

    procs, cap_ep, vio_ep = _spawn_cap_vio(
        args.session, args.max_frames, args.kf_every)
    cap_proc, vio_proc = procs
    triplet_win = keypoint_win = None
    try:
        bundle = _await_calib_bundle(cap_ep, timeout_s=20.0)
        _await_calib_bundle(vio_ep, timeout_s=20.0)
        W, H = int(bundle.width), int(bundle.height)
        print(f"    capture {W}x{H}, device_id="
              f"{(bundle.device_id or 'default')!r}")

        app = QApplication.instance() or QApplication(sys.argv or ["test"])

        # Build the single viewer exactly as run_ui does (one view, no tabs).
        history = PoseHistory(capacity=10_000)
        viewer = Viewer3D(history, default_view="ISO")

        # ---------- (a) View preset action ----------
        preset = "TOP"
        before = (viewer.opts["azimuth"], viewer.opts["elevation"],
                  viewer.opts["distance"])
        act = QAction(preset.title(), viewer)
        act.triggered.connect(lambda _c=False, n=preset: viewer.set_view(n))
        act.trigger()
        app.processEvents()
        after = (viewer.opts["azimuth"], viewer.opts["elevation"],
                 viewer.opts["distance"])
        want = VIEW_PRESETS[preset]
        _check(before != after and (after[0], after[1], after[2]) == want,
               f"View '{preset}' moved the viewer "
               f"({before} -> {after}, want {want})")

        # ---------- (b) Visualize -> triplet ----------
        # Drain several samples (not just the first): the first frame legitimately
        # carries no IMU, so collecting a handful proves the imucam.sample + depth
        # streams actually join over IPC (imu_n > 0 on at least one frame).
        triplet_win = SyncedViewWindow(
            ipc_triplet_factory(cap_ep, W, H), fps=20)
        triplet_win.start()                          # boots the IPC worker
        app.processEvents()
        samples = _drain_samples(triplet_win._worker.queue, timeout_s=25.0,
                                 want=5)
        _check(triplet_win._worker.error is None,
               f"triplet worker has no error ({triplet_win._worker.error})")
        _check(len(samples) >= 1 and isinstance(samples[0], TripletSample),
               f"triplet worker queued >=1 TripletSample (got {len(samples)})")
        s0 = samples[0]
        _check(s0.gray_left.ndim == 2
               and s0.depth_m.shape == s0.gray_left.shape,
               "triplet sample carries (H,W) image + matching depth")
        max_imu = max(s.imu_n for s in samples)
        _check(max_imu > 0,
               f"triplet IMU rows joined over IPC (max imu_n={max_imu} "
               f"over {len(samples)} samples)")
        triplet_win.close()
        app.processEvents()
        _check(triplet_win._worker is None,
               "triplet window stopped its worker on close")
        print(f"    [ok] triplet: {len(samples)} samples, SEQ "
              f"{s0.seq}..{samples[-1].seq}, max imu_n {max_imu}")

        # ---------- (c) Visualize -> keypoint ----------
        # KLT seeds on frame 0 (no tracks yet) and the PnP solve lags a frame, so
        # again drain several to prove frame.tracks (from vio) + frame.depth (from
        # capture) really join: at least one frame must carry tracks.
        keypoint_win = KeypointTrackWindow(
            ipc_keypoint_factory(cap_ep, vio_ep, W, H), fps=20)
        keypoint_win.start()
        app.processEvents()
        ksamples = _drain_samples(keypoint_win._worker.queue, timeout_s=25.0,
                                  want=6)
        _check(keypoint_win._worker.error is None,
               f"keypoint worker has no error ({keypoint_win._worker.error})")
        _check(len(ksamples) >= 1 and isinstance(ksamples[0], KeypointSample),
               f"keypoint worker queued >=1 KeypointSample (got {len(ksamples)})")
        k0 = ksamples[0]
        _check(k0.rgb.ndim == 3 and k0.rgb.shape[2] == 3,
               "keypoint sample carries an (H,W,3) overlay")
        max_trk = max(k.n_tracks for k in ksamples)
        _check(max_trk > 0,
               f"keypoint tracks joined over IPC (max n_tracks={max_trk} "
               f"over {len(ksamples)} samples)")
        keypoint_win.close()
        app.processEvents()
        _check(keypoint_win._worker is None,
               "keypoint window stopped its worker on close")
        print(f"    [ok] keypoint: {len(ksamples)} samples, SEQ "
              f"{k0.seq}..{ksamples[-1].seq}, max trk {max_trk}")

        # Attach the SLAM-Map (occupancy-voxel) source's keyframe client NOW, so it
        # accumulates keyframes in the background while VIO is still alive -- VIO
        # shuts down once the replay ENDs, so its kf rings must be attached before
        # then. We fuse + build + assert it in (e).
        slam_src = attach_slam_map_source(vio_ep, vio_ep, bundle)

        # ---------- (e) Visualize -> SLAM Map (3D room) ----------
        # Build the occupancy-voxel map from the keyframes the source accumulated
        # above (temporal occupancy fusion -> occupied voxel centres + green-by-
        # height colours), and assert MapWindow ingests it without raising. The
        # GLViewWidget can't actually render offscreen here, but update() touching
        # its GL items + the source's fusion math are what we exercise.
        from ui.qt.map_window import MapWindow
        try:
            mpts, mcols, mcams = build_slam_map_from(slam_src)
        finally:
            slam_src.stop()
        mwin = MapWindow(title="SLAM Map (3D room)")
        # update() touches the GL scatter items; it must not raise on the GUI
        # thread (GL paint is deferred; offscreen it just won't draw).
        mwin.update(mpts, mcols, mcams)
        # Empty inputs exercise the early-return / guard paths (no crash on a blank
        # map or a colour/point length mismatch).
        mwin.update(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                    np.zeros((0, 3), np.float32))
        mwin.update(None, None, None)
        app.processEvents()
        mwin.close()
        app.processEvents()
        print(f"    [ok] slam-map: {len(mpts)} occupied voxels, "
              f"{len(mcams)} cams on the path")
    finally:
        for w in (triplet_win, keypoint_win):
            if w is not None:
                try:
                    w.close()
                except Exception:                                  # noqa: BLE001
                    pass
        _terminate_all(*procs)
        for name, proc in (("capture", cap_proc), ("vio", vio_proc)):
            try:
                _out, err = proc.communicate(timeout=2.0)
            except Exception:                                      # noqa: BLE001
                err = b""
            if err.strip():
                print(f"\n  --- menus {name}.stderr ---\n"
                      f"{err.decode(errors='replace')}", file=sys.stderr)


def _try_build_qt(vio_ep: str, slam_ep: str) -> None:
    """Build the single-view QMainWindow once; assert its wiring; destroy.

    Proves the single-view widget tree is wireable: ONE Viewer3D carrying the 5
    trajectory line items, plus the 5 per-line toggle buttons (exact labels) on
    the Controls toolbar. Does NOT call ``app.exec()`` so the test stays
    headless. Imports Qt inside this function so the rest of the file imports
    cleanly even without PyQt6 installed.
    """
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QPushButton, QToolBar, QWidget, QVBoxLayout,
    )
    import pyqtgraph.opengl as gl

    from ui.qt import theme
    from ui.qt.viewer3d import Viewer3D

    app = QApplication.instance() or QApplication(sys.argv or ["test"])
    win = tb = central = viewer = None
    try:
        history = PoseHistory(capacity=128)
        viewer = Viewer3D(history, default_view="ISO")
        # The 5 lines the single view renders, each its own GLLinePlotItem.
        line_items = [viewer._vo, viewer._traj, viewer._ba,
                      viewer._corrected, viewer._refined]
        _check(all(isinstance(it, gl.GLLinePlotItem) for it in line_items),
               "Viewer3D carries 5 trajectory GLLinePlotItem lines "
               "(_vo/_traj/_ba/_corrected/_refined)")
        _check(all(it in viewer.items for it in line_items),
               "all 5 trajectory lines are added to the Viewer3D scene")

        central = QWidget()
        QVBoxLayout(central).addWidget(viewer)
        win = QMainWindow()
        win.setStyleSheet(theme.QSS)
        win.setCentralWidget(central)

        # Build the 5 toggles EXACTLY as run_ui does and assert labels + wiring.
        tb = QToolBar("Controls")
        win.addToolBar(tb)
        toggles = []
        for label, setter in (("VO", viewer.set_vo_visible),
                              ("VIO", viewer.set_vio_visible),
                              ("VIO-BA", viewer.set_ba_visible),
                              ("SLAM-corrected VIO", viewer.set_corrected_visible),
                              ("SLAM", viewer.set_slam_visible)):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.toggled.connect(setter)
            tb.addWidget(btn)
            toggles.append(btn)
        labels = [b.text() for b in toggles]
        want = ["VO", "VIO", "VIO-BA", "SLAM-corrected VIO", "SLAM"]
        _check(labels == want,
               f"5 toggle buttons with exact labels (got {labels}, want {want})")
        _check(all(b.isCheckable() and b.isChecked() for b in toggles),
               "all 5 toggles are checkable and start CHECKED (visible)")

        # Toggling a button hides its line; toggling back shows it -- proves the
        # toggled(bool) -> visibility-setter wiring is live.
        toggles[0].setChecked(False)             # VO off
        app.processEvents()
        _check(not viewer._vo.visible(),
               "unchecking 'VO' hides the VO line via the wired setter")
        toggles[0].setChecked(True)              # VO back on
        app.processEvents()
        _check(viewer._vo.visible(),
               "re-checking 'VO' shows the VO line again")
        # Don't show -- headless construction proof only.
    finally:
        # PyQt5/6 cleans up via Python GC, but explicitly drop refs so the
        # `QApplication.instance()` re-use on a subsequent test call doesn't
        # see a stale widget.
        del win, tb, central, viewer
        # Don't quit -- the QApplication singleton may be re-used.


def _terminate_all(*procs: subprocess.Popen) -> None:
    for p in procs:
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:                                      # noqa: BLE001
                pass
    for p in procs:
        try:
            p.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:                                      # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
