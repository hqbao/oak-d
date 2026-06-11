#!/usr/bin/env python3
"""Headless selftest for :class:`~ui.modules.ipc_sources.IpcStereoRawSource`.

Drives the stereo RAW-pair source THROUGH THE REAL CAPTURE PROCESS (no device):
boots ``imu_camera.main`` in replay on a gold session that HAS right frames, then
attaches an in-process :class:`IpcStereoRawSource` to capture's ``imucam.sample``
topic and asserts it delivers genuine UNRECTIFIED left+right pairs -- the stream
the upcoming stereo camera-calibration wizard consumes.

Asserts:

* The source connects with no error (capture rings attach + IPC connect).
* It delivers ``>= K`` records where BOTH ``gray_left`` and ``gray_right`` are
  present, the same shape, plausible (uint8, 2D), and the seq is monotonic.
* The session genuinely carries right frames (a precondition we verify up front
  via :class:`~imu_camera.io.reader.SessionReader`, so a green run can't come from
  a mono session silently latching the source's mono guard).
* The mono guard fires when the right frame is absent (a focused unit check
  publishing a known ``WireImuCamPacket`` with ``gray_right_ref=None``).
* Clean start/stop -- no crash / hang -- under ``QT_QPA_PLATFORM=offscreen``.

Run::

    QT_QPA_PLATFORM=offscreen python -m ui.tests.stereo_source_selftest
    python -m ui.tests.stereo_source_selftest --session sessions/gold/lab_loop_30s
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

from imu_camera.io.reader import SessionReader                       # noqa: E402
from ui.comms import topics                                         # noqa: E402
from ui.comms import IPCPubSub, RingRegistry                        # noqa: E402
from ui.comms.ring_registry import default_capture_specs           # noqa: E402
from ui.comms.wire import SharedArrayRef, WireImuCamPacket         # noqa: E402
from ui.main import _await_calib_bundle                             # noqa: E402
from ui.modules import IpcStereoRawSource                          # noqa: E402

#: How many BOTH-present pairs prove the stream is live (well under a 30-frame
#: replay so the assertion is robust to a few startup frames being missed while
#: the client registers its subscription).
WANT_PAIRS = 8


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def _session_has_right(session: str) -> tuple[int, int]:
    """Confirm the session's frame 0 has a right frame; return (H, W).

    This is the precondition the whole test rests on: if the session were mono,
    the source's mono guard would latch and (correctly) deliver zero pairs, which
    would look like a different failure. We verify the session is stereo FIRST so
    a delivery failure can only mean a real bug.
    """
    reader = SessionReader(session)
    f = reader.load_frame(0, load_right=True)
    if f.gray_right is None:
        raise SystemExit(
            f"session {session!r} has no right frame at frame 0 -- pick a stereo "
            f"gold session (this test needs both frames).")
    h, w = f.gray_left.shape
    return int(h), int(w)


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            proc.terminate()
        except Exception:                                          # noqa: BLE001
            pass
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:                                          # noqa: BLE001
            pass


def test_mono_guard() -> None:
    """The mono guard latches an error + emits no half-pair when right is None.

    Publishes a known ``WireImuCamPacket`` whose ``gray_right_ref`` is None on a
    tiny test server's ``imucam.sample``; asserts the source NEVER fires the
    callback (no half-pair) and sets the documented mono-guard :attr:`error`.
    """
    print("\n  [unit] IpcStereoRawSource mono guard")
    ep = f"oak.stereomono.t{os.getpid() & 0xFFF:x}"
    h, w = 16, 24
    # The source attaches capture rings (the converter read_copies the LEFT gray
    # out of shared memory), so the test server must create matching rings and
    # write a left frame into the slot the wire ref points at.
    specs = default_capture_specs(endpoint=ep, width=w, height=h)
    rings = RingRegistry().create_all(specs)
    left = (np.arange(h * w, dtype=np.uint8).reshape(h, w))
    slot = 0
    left_ref: SharedArrayRef = rings.get(f"{ep}.gray_left").write(slot, left)
    pkt = WireImuCamPacket(
        seq=0, ts_ns=1_000, gray_left_ref=left_ref, gray_right_ref=None,
        imu_ts=np.zeros(0, np.int64),
        gyro=np.zeros((0, 3), np.float64), accel=np.zeros((0, 3), np.float64))

    server = IPCPubSub(ep, role="server",
                       retain_topics={topics.IMUCAM_SAMPLE}, blocking=True)
    server.start()
    got: list[tuple] = []

    def cb(seq, ts_ns, gl, gr) -> None:
        got.append((seq, ts_ns, gl, gr))

    src = IpcStereoRawSource(ep, w, h, connect_timeout_s=10.0)
    src.start(cb)
    try:
        _check(src.error is None,
               f"mono-guard source connected without error ({src.error})")
        time.sleep(0.3)                                # let the client register
        server.publish(topics.IMUCAM_SAMPLE, pkt)
        time.sleep(0.5)                                # let the recv thread run
    finally:
        src.stop()
        server.close()
        rings.close()

    _check(len(got) == 0,
           f"mono packet emitted NO half-pair (got {len(got)} callbacks)")
    _check(src.error is not None and "right frame" in src.error,
           f"mono guard latched a clear error (got {src.error!r})")
    print(f"    [ok] mono guard: 0 callbacks, error={src.error!r}")


def test_no_frame_watchdog() -> None:
    """Watchdog latches a clear error if connected but ZERO frames arrive.

    This is the regression for the silent-hang the operator hit: the source
    connects fine (capture's socket + rings are up) but no ``imucam.sample`` ever
    reaches it, so the wizard would sit on the placeholder forever. We bring up a
    server + rings that publish NOTHING, attach the source with a short
    ``frame_timeout_s``, and assert it (a) connects with no error, then (b) latches
    the documented no-frame error AFTER the timeout (never before).
    """
    print("\n  [unit] IpcStereoRawSource no-frame watchdog")
    ep = f"oak.stereowd.t{os.getpid() & 0xFFF:x}"
    h, w = 16, 24
    # The source attaches capture's rings on start, so they must exist even though
    # we publish nothing (a real capture always has its rings up before frames).
    specs = default_capture_specs(endpoint=ep, width=w, height=h)
    rings = RingRegistry().create_all(specs)
    server = IPCPubSub(ep, role="server",
                       retain_topics={topics.IMUCAM_SAMPLE}, blocking=False)
    server.start()
    got: list[tuple] = []

    def cb(seq, ts_ns, gl, gr) -> None:
        got.append((seq, ts_ns, gl, gr))

    # Short watchdog so the test is fast; well above the ~0 connect latency here.
    src = IpcStereoRawSource(ep, w, h, connect_timeout_s=10.0,
                             frame_timeout_s=1.0)
    src.start(cb)
    try:
        _check(src.error is None,
               f"watchdog source connected without error ({src.error})")
        # BEFORE the timeout the error must NOT be set (no false positive while a
        # real first frame is still in flight).
        time.sleep(0.3)
        _check(src.error is None,
               f"no error before the watchdog window ({src.error!r})")
        # AFTER the timeout (no frame ever published) the watchdog must fire.
        time.sleep(1.2)
        _check(src.error is not None,
               "watchdog latched an error after the no-frame window")
        _check("no stereo frames" in (src.error or ""),
               f"watchdog error is the documented no-frame message "
               f"({src.error!r})")
    finally:
        src.stop()
        server.close()
        rings.close()

    _check(len(got) == 0, f"watchdog path delivered no frames (got {len(got)})")
    print(f"    [ok] watchdog: 0 frames -> error={src.error!r}")


def test_full_launcher_late_subscriber(session: str) -> None:
    """Late-joining stereo source against the FULL ``launcher.main`` replay graph.

    THE critical gap the original Phase-2 test never covered: the wizard opens
    AFTER capture is already streaming ``imucam.sample`` to VIO (a second, late
    subscriber to the same shared-memory-ring topic) under the real multi-process
    launcher, with the ``--auto-suffix`` endpoint the launcher actually uses. We
    boot ``launcher.main --auto-suffix --no-ui`` (capture + vio + slam), resolve
    the SUFFIXED capture endpoint exactly as the launcher derives it, wait for the
    whole pipeline to be up (VIO's retained ``calib.bundle``), then attach the
    source the same way :func:`ui.main.run_ui._open_camera_calib` does and assert
    real stereo pairs arrive at the late subscriber.
    """
    print("\n  [integration] late subscriber via FULL launcher.main replay")
    py = sys.executable
    env = dict(os.environ)
    proc = subprocess.Popen(
        [py, "-m", "launcher.main", "--auto-suffix", "--no-ui",
         "--session", session, "--max-frames", "120"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # The launcher derives its suffix from ITS OWN pid: `.l<pid & 0xFFF:x>`
    # (see launcher/main.py). The capture endpoint is then `oak.cap<suffix>`.
    suffix = f".l{proc.pid & 0xFFF:x}"
    cap_ep = f"oak.cap{suffix}"
    vio_ep = f"oak.vio{suffix}"
    print(f"    launcher pid={proc.pid} cap_ep={cap_ep!r} vio_ep={vio_ep!r}")

    pairs: list[tuple] = []
    enough = threading.Event()
    lock = threading.Lock()

    def cb(seq, ts_ns, gl, gr) -> None:
        with lock:
            pairs.append((int(seq), gl is not None and gr is not None))
            if len(pairs) >= WANT_PAIRS:
                enough.set()

    src = None
    try:
        # Wait for VIO's calib bundle: proves the WHOLE pipeline is up and VIO has
        # ALREADY been subscribed to capture's imucam.sample (so the source below
        # is genuinely a LATE second subscriber to that ring topic).
        vb = _await_calib_bundle(vio_ep, timeout_s=40.0)
        w, h = int(vb.width), int(vb.height)
        print(f"    vio ready {w}x{h}; capture already streaming to VIO")
        time.sleep(1.5)                                # ensure we join LATE
        src = IpcStereoRawSource(cap_ep, w, h, device_id="LAUNCHER-LATE",
                                 connect_timeout_s=20.0)
        src.start(cb)
        _check(src.error is None,
               f"late source connected without error ({src.error})")
        got_enough = enough.wait(timeout=20.0)
    finally:
        if src is not None:
            src.stop()
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10.0)
            except Exception:                                      # noqa: BLE001
                try:
                    proc.kill()
                except Exception:                                  # noqa: BLE001
                    pass

    with lock:
        n = len(pairs)
        both_ok = all(b for _s, b in pairs)
    print(f"    late subscriber received {n} pairs (wanted >= {WANT_PAIRS})")
    _check(src is not None and src.error is None,
           f"late source ran without error ({None if src is None else src.error})")
    _check(got_enough and n >= WANT_PAIRS,
           f"late subscriber got >= {WANT_PAIRS} stereo pairs (got {n})")
    _check(both_ok, "every late-delivered pair carries BOTH frames")
    print(f"    [ok] late subscriber via full launcher: {n} BOTH-present pairs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--no-launcher", action="store_true",
                    help="skip the full launcher.main late-subscriber integration "
                         "test (it boots 3 child processes; ~20 s)")
    args = ap.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    print("stereo_source_selftest")
    print(f"  session={args.session} max-frames={args.max_frames}")

    h, w = _session_has_right(args.session)
    print(f"  session has stereo frames: {w}x{h} (right frame present)")

    cap_ep = f"oak.cap.s{os.getpid() & 0xFFF:x}"
    py = sys.executable
    env = dict(os.environ)
    cap_proc = subprocess.Popen(
        [py, "-m", "imu_camera.main",
         "--endpoint", cap_ep, "--session", args.session,
         "--max-frames", str(args.max_frames)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Collected pairs (thread-safe handoff off the source's recv thread).
    pairs: list[tuple] = []
    enough = threading.Event()
    lock = threading.Lock()

    def cb(seq, ts_ns, gl, gr) -> None:
        with lock:
            pairs.append((int(seq), int(ts_ns), gl, gr))
            if len(pairs) >= WANT_PAIRS:
                enough.set()

    src = None
    try:
        # Wait for capture to be ready (its retained calib.bundle) before we
        # attach -- capture creates its shared-memory rings during start-up.
        _await_calib_bundle(cap_ep, timeout_s=20.0)
        print("  capture: ready")

        src = IpcStereoRawSource(cap_ep, w, h, device_id="SELFTEST-DEV",
                                 connect_timeout_s=20.0)
        src.start(cb)
        _check(src.error is None,
               f"IpcStereoRawSource connected without error ({src.error})")

        # Drain until we have enough pairs or the replay finishes.
        got_enough = enough.wait(timeout=30.0)
        # Let the replay finish + the recv thread drain any in-flight packets.
        cap_proc.wait(timeout=40.0)
        time.sleep(0.5)
    finally:
        if src is not None:
            src.stop()                                 # must be clean (no hang)
        _terminate(cap_proc)
        try:
            _out, err = cap_proc.communicate(timeout=2.0)
        except Exception:                                          # noqa: BLE001
            err = b""
        if err.strip():
            print(f"\n  --- capture.stderr ---\n"
                  f"{err.decode(errors='replace')}", file=sys.stderr)

    with lock:
        n = len(pairs)
    print(f"\n  received pairs: {n} (wanted >= {WANT_PAIRS})")
    _check(src is not None and src.error is None,
           f"source ran without error ({None if src is None else src.error})")
    _check(got_enough and n >= WANT_PAIRS,
           f"delivered >= {WANT_PAIRS} stereo pairs (got {n})")

    # Every delivered record must be a genuine, plausible BOTH-present pair.
    seqs = []
    for seq, ts_ns, gl, gr in pairs:
        seqs.append(seq)
        _check(gl is not None and gr is not None,
               f"pair seq={seq} carries BOTH frames")
        _check(gl.dtype == np.uint8 and gr.dtype == np.uint8,
               f"pair seq={seq} frames are uint8 "
               f"(got {gl.dtype}/{gr.dtype})")
        _check(gl.ndim == 2 and gr.ndim == 2,
               f"pair seq={seq} frames are 2D (got {gl.ndim}D/{gr.ndim}D)")
        _check(gl.shape == gr.shape == (h, w),
               f"pair seq={seq} both frames are {(h, w)} "
               f"(got {gl.shape}/{gr.shape})")
        # The grays must be independent objects (read_copied out of the ring),
        # not two views of the same buffer.
        _check(gl is not gr and gl.base is None and gr.base is None,
               f"pair seq={seq} frames are independent owned arrays")

    _check(seqs == sorted(seqs),
           f"delivered seqs are monotonic (got {seqs[:6]}...)")
    # Sanity: left != right for at least one pair (a true stereo pair differs;
    # this would catch a wiring bug that delivered the left frame twice).
    distinct = any(not np.array_equal(gl, gr) for _s, _t, gl, gr in pairs)
    _check(distinct,
           "at least one pair has left != right (genuine stereo, not duplicated)")
    print(f"    [ok] {n} BOTH-present pairs, all {(h, w)} uint8 2D, "
          f"seq {seqs[0]}..{seqs[-1]}, device {src.device_id!r}")

    # Focused unit check of the mono guard (no replay needed).
    test_mono_guard()
    # The no-frame watchdog (the silent-hang regression) -- fast, no replay.
    test_no_frame_watchdog()
    # The critical gap: a LATE subscriber through the REAL multi-process launcher.
    if args.no_launcher:
        print("\n  [integration] full-launcher late-subscriber test SKIPPED "
              "(--no-launcher)")
    else:
        test_full_launcher_late_subscriber(args.session)

    print("\nALL STEREO SOURCE SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
