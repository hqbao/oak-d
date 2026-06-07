"""4-process launcher: boot capture + vio + slam in background, run UI in foreground.

The launcher's only job is process lifecycle management:

1. Spawn ``capture`` in background (it owns the OAK-D, or replays a session).
2. Spawn ``vio`` and ``slam`` in background (they connect to capture's retained
   ``calib.bundle`` over IPC, then start their own IPC endpoints).
3. Run the ``ui`` process in the FOREGROUND so the Qt event loop has the GUI
   focus and Ctrl-C / window-close cleanly tears everything down.
4. On UI exit (clean or crash), send SIGTERM to capture / vio / slam, wait for
   them to drain (each has a SIGTERM handler that runs the same finally block
   the replay-end path uses), then SIGKILL stragglers.

Endpoint naming
---------------
By default the launcher uses the canonical endpoint names ``oak.capture``,
``oak.vio``, ``oak.slam`` so external tools (the Phase 10 calibration / visualize
tools that subscribe via IPC) work without configuration.  ``--endpoint-suffix
SUFFIX`` (or ``--auto-suffix``) uniquifies them per launcher PID so two
launchers can co-exist (e.g. dev vs CI on the same machine).

Run::

    python -m ours.proc.launcher                                 # live, default
    python -m ours.proc.launcher --session sessions/gold/lab_loop_30s   # replay
    python -m ours.proc.launcher --width 1280 --height 800 --fps 15
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from multiprocessing import shared_memory
from pathlib import Path

LOG = logging.getLogger("ours.proc.launcher")


# Endpoint roles + the ring names each role's process owns. Mirrors
# `default_capture_specs()` (capture) and `default_vio_specs()` (vio); slam
# attaches but owns no rings. Used by `_cleanup_orphans` to brute-force unlink
# every stale POSIX shm segment from prior crashed runs so a fresh launch
# doesn't trip macOS's per-process fd / shm caps with EMFILE.
_RING_NAMES_BY_ROLE = {
    "cap": ("gray_left", "gray_right", "depth_m"),
    "vio": ("kf_gray", "kf_depth"),
    "slm": (),
}
_RING_SLOTS = 64


def _cleanup_orphans() -> None:
    """Best-effort: unlink every stale `oak.*` SHM segment + IPC socket file.

    macOS has no public listing API for POSIX shared memory, so we discover
    candidate endpoints by listing the launcher's IPC socket directory (each
    launch leaves `oak.{cap,vio,slm}.<suffix>.sock` there). For each endpoint
    we then try to unlink every (ring, slot) name owned by that role. Missing
    segments are silently skipped -- this is a guard against accumulation, not
    a correctness operation.
    """
    sock_dir = Path(tempfile.gettempdir()) / "ours_ipc"
    if not sock_dir.is_dir():
        return
    endpoints: set[str] = set()
    for p in glob.glob(str(sock_dir / "oak.*.sock")):
        endpoints.add(Path(p).name[:-len(".sock")])
    if not endpoints:
        return
    unlinked = 0
    for ep in sorted(endpoints):
        parts = ep.split(".")
        if len(parts) != 3:
            continue
        role = parts[1]
        for ring in _RING_NAMES_BY_ROLE.get(role, ()):
            for i in range(_RING_SLOTS):
                shm_name = f"{ep}.{ring}.{i}"
                try:
                    shm = shared_memory.SharedMemory(name=shm_name,
                                                     create=False)
                    shm.close()
                    shm.unlink()
                    unlinked += 1
                except FileNotFoundError:
                    pass
                except Exception:                                  # noqa: BLE001
                    pass
    sock_removed = 0
    for p in glob.glob(str(sock_dir / "oak.*.sock")):
        try:
            os.unlink(p)
            sock_removed += 1
        except FileNotFoundError:
            pass
    if unlinked or sock_removed:
        LOG.info("launcher: cleanup_orphans freed %d stale SHM segments + "
                 "%d socket files from %d prior endpoints",
                 unlinked, sock_removed, len(endpoints))


# --------------------------------------------------------------------------- #
def _spawn(py: str, mod: str, args: list[str], *, env: dict[str, str],
           name: str) -> subprocess.Popen:
    """Spawn a child python process; stdout / stderr inherited from launcher."""
    cmd = [py, "-m", mod, *args]
    p = subprocess.Popen(cmd, env=env)
    LOG.info("launcher: spawned %s pid=%d -> %s", name, p.pid, " ".join(cmd))
    return p


def _terminate(procs: list[subprocess.Popen], *, deadline_s: float = 10.0,
               step_s: float = 0.2) -> None:
    """SIGTERM all procs, wait for clean exit, SIGKILL any straggler."""
    for p in procs:
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:                                      # noqa: BLE001
                pass
    deadline = time.monotonic() + float(deadline_s)
    while time.monotonic() < deadline and any(p.poll() is None for p in procs):
        time.sleep(step_s)
    for p in procs:
        if p.poll() is None:
            LOG.warning("launcher: SIGKILL on pid %d (clean shutdown timeout)",
                        p.pid)
            try:
                p.kill()
            except Exception:                                      # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=None,
                    help="replay this session directory instead of opening the OAK-D")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap replay frames (0 = all)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--kf-every", type=int, default=5)
    ap.add_argument("--no-gyro", action="store_true",
                    help="live: disable IMU gyro use (pure-vision)")
    ap.add_argument("--recalibrate-bias", action="store_true",
                    help="live: re-measure gyro bias instead of using the cached one")
    ap.add_argument("--worker", action="store_true",
                    help="run BA / SLAM solves in their own child processes")
    ap.add_argument("--endpoint-suffix", default="",
                    help="append SUFFIX to canonical endpoint names so two "
                         "launchers can co-exist (e.g. 'dev', 'ci')")
    ap.add_argument("--auto-suffix", action="store_true",
                    help="derive endpoint suffix from this launcher's PID")
    ap.add_argument("--no-ui", action="store_true",
                    help="don't open the UI -- useful for capture-only headless runs")
    args = ap.parse_args()

    # ---- Best-effort cleanup of stale SHM + sockets from prior crashed runs.
    # macOS POSIX shm persists past process death (SIGKILL skips unlink); after
    # enough crashed launches the kernel namespace fills up and the next
    # capture's shm_open() fails with EMFILE. We run this BEFORE spawning so
    # the children start in a clean namespace.
    _cleanup_orphans()

    # ---- Endpoint names --------------------------------------------------
    if args.auto_suffix:
        # `oak.cap.l<pidhex>` -- short enough that the per-slot suffix still
        # fits inside macOS's 30-char POSIX shm name limit. See
        # `SharedArrayRing.create` for the assertion.
        suffix = f".l{os.getpid() & 0xFFF:x}"
    elif args.endpoint_suffix:
        suffix = "." + args.endpoint_suffix
    else:
        suffix = ""
    cap_ep = f"oak.cap{suffix}" if suffix else "oak.capture"
    vio_ep = f"oak.vio{suffix}" if suffix else "oak.vio"
    slam_ep = f"oak.slm{suffix}" if suffix else "oak.slam"
    LOG.info("launcher: endpoints cap=%r vio=%r slam=%r",
             cap_ep, vio_ep, slam_ep)

    py = sys.executable
    env = dict(os.environ)

    # ---- Build per-proc argv ---------------------------------------------
    capture_args: list[str] = ["--endpoint", cap_ep,
                               "--width", str(args.width),
                               "--height", str(args.height),
                               "--fps", str(args.fps)]
    if args.session:
        capture_args += ["--session", args.session]
        if args.max_frames > 0:
            capture_args += ["--max-frames", str(args.max_frames)]
    else:
        capture_args += ["--live"]
        if args.no_gyro:
            capture_args += ["--no-gyro"]
        if args.recalibrate_bias:
            capture_args += ["--recalibrate-bias"]

    vio_args = ["--capture-endpoint", cap_ep, "--endpoint", vio_ep,
                "--kf-every", str(args.kf_every)]
    if args.no_gyro:
        vio_args += ["--no-gyro"]
    if args.worker:
        vio_args += ["--worker"]

    slam_args = ["--capture-endpoint", cap_ep,
                 "--vio-endpoint", vio_ep,
                 "--endpoint", slam_ep]
    if args.worker:
        slam_args += ["--worker"]

    ui_args = ["--vio-endpoint", vio_ep, "--slam-endpoint", slam_ep]

    # ---- Boot order ------------------------------------------------------
    # capture FIRST so the retained `calib.bundle` is published as soon as it
    # builds the frontend. vio + slam connect with retried `IpcClientBus.start`
    # so booting them after capture is fine; this just minimises the connect
    # retry noise in the log.
    procs: list[subprocess.Popen] = []
    try:
        cap_proc = _spawn(py, "ours.proc.capture", capture_args, env=env,
                          name="capture")
        procs.append(cap_proc)
        # tiny sleep so capture's IPC server is listening before vio / slam
        # try their first connect (vio retries so this is cosmetic, but it
        # gives a clean first-attempt success in the log).
        time.sleep(0.2)
        vio_proc = _spawn(py, "ours.proc.vio", vio_args, env=env, name="vio")
        procs.append(vio_proc)
        slam_proc = _spawn(py, "ours.proc.slam", slam_args, env=env, name="slam")
        procs.append(slam_proc)

        # SIGTERM handler so a `kill <launcher_pid>` from outside cleans up
        # the whole tree (not just the launcher process itself).
        #
        # CRITICAL: do NOT call `_terminate(procs)` here -- `_terminate` polls
        # each `Popen.poll()`, which calls `os.waitpid(pid, WNOHANG)` on the
        # same pid the main thread is blocked in `ui_proc.wait()` on. The two
        # waitpid callers race for the single reap event, leaving Popen's
        # `returncode` stuck at None on the loser, so the handler's
        # `_terminate` loop spins the full 10 s deadline and SIGKILLs the UI
        # even though it already exited cleanly. Instead just forward SIGTERM
        # to each child (they likely already got it from the process-group
        # signal anyway) and `os._exit` immediately; children either finish
        # their own shutdown or get reaped by init when launcher dies.
        def _on_sigterm(_signo, _frame):
            LOG.info("launcher: SIGTERM -> forwarding to children + exiting")
            for p in procs:
                try:
                    p.terminate()
                except Exception:                                  # noqa: BLE001
                    pass
            os._exit(143)                                         # 128 + SIGTERM
        signal.signal(signal.SIGTERM, _on_sigterm)

        if args.no_ui:
            LOG.info("launcher: --no-ui set; waiting for capture to exit "
                     "(Ctrl-C to stop)")
            try:
                cap_proc.wait()
            except KeyboardInterrupt:
                LOG.info("launcher: SIGINT -> stopping")
            rc = cap_proc.returncode if cap_proc.returncode is not None else 0
            # After capture exits, vio + slam see END on their inputs (capture's
            # publisher bridge converts each Flow's `_emit_end` to a wire END
            # then drains them onto the socket before close). Give them a
            # natural-exit window BEFORE `_terminate` SIGKILLs them: each one
            # has its own 120 s drain ceiling so a busy back-end won't lose data.
            LOG.info("launcher: waiting for vio + slam to drain naturally ...")
            for child in (vio_proc, slam_proc):
                try:
                    child.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    LOG.warning("launcher: %s pid=%d still running after 30 s; "
                                "_terminate will SIGTERM/SIGKILL",
                                child.args, child.pid)
        else:
            # UI in foreground -- inherits stdout / stderr / stdin so the user
            # sees Qt warnings and can Ctrl-C cleanly.
            ui_proc = subprocess.Popen([py, "-m", "ours.proc.ui", *ui_args],
                                       env=env)
            procs.append(ui_proc)
            try:
                rc = ui_proc.wait()
            except KeyboardInterrupt:
                LOG.info("launcher: SIGINT -> stopping UI")
                try:
                    ui_proc.terminate()
                except Exception:                                  # noqa: BLE001
                    pass
                rc = ui_proc.wait(timeout=5.0)
    finally:
        LOG.info("launcher: shutting down background procs ...")
        _terminate(procs)
        LOG.info("launcher: bye")

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
