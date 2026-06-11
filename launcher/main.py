"""4-process launcher: boot imu_camera + vio + slam in background, run ui foreground.

The launcher's only job is process lifecycle management:

1. Spawn ``imu_camera`` (capture) in background (it owns the OAK-D, or replays a
   session).
2. Spawn ``vio`` and ``slam`` in background (they connect to capture's retained
   ``calib.bundle`` over IPC, then start their own IPC endpoints).
3. Run the ``ui`` process in the FOREGROUND so the Qt event loop has the GUI
   focus and Ctrl-C / window-close cleanly tears everything down.
4. On UI exit (clean or crash), send SIGTERM to capture / vio / slam, wait for
   them to drain (each has a SIGTERM handler that runs the same finally block
   the replay-end path uses), then SIGKILL stragglers.

This is a behaviour-for-behaviour port of the pre-split ``ours.proc.launcher``,
retargeted onto the four split projects' ``<project>.main`` entrypoints
(``imu_camera.main`` / ``vio.main`` / ``slam.main`` / ``ui.main``). The only
wire-level change is that the new ``imu_camera.main`` DEFAULTS to replay and
takes an explicit ``--live`` flag for hardware (the old ``ours.proc.capture``
defaulted to live), so the live branch passes ``--live`` and the replay branch
passes ``--session``.

Endpoint naming
---------------
By default the launcher uses the canonical endpoint names ``oak.capture``,
``oak.vio``, ``oak.slam`` so external tools (calibration / visualize tools that
subscribe via IPC) work without configuration.  ``--endpoint-suffix SUFFIX`` (or
``--auto-suffix``) uniquifies them per launcher PID so two launchers can co-exist
(e.g. dev vs CI on the same machine).

Run::

    python -m launcher.main                                       # live, default
    python -m launcher.main --session sessions/gold/lab_loop_30s  # replay
    python -m launcher.main --width 1280 --height 800 --fps 15
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

# Import-safe: ui.main imports PyQt6 LAZILY inside run_ui, so pulling in this one
# constant does NOT drag Qt into the launcher (verified by the QT-FREE check).
from ui.main import RESTART_EXIT_CODE

LOG = logging.getLogger("launcher.main")


# Endpoint roles + the ring names each role's process owns. Mirrors
# `imu_camera.comms.ring_registry.default_capture_specs()` (capture) and
# `default_vio_specs()` (vio); slam attaches but owns no rings. Used by
# `_cleanup_orphans` to unlink every stale POSIX shm segment from prior crashed
# runs so a fresh launch doesn't trip macOS's per-process fd / shm caps with
# EMFILE.
#
# Each live ring is now ONE block named `{ep}.{ring}` (slots are byte offsets --
# see SharedArrayRing). `_RING_SLOTS` is retained only for the legacy cleanup
# pass that reclaims the OLD per-slot `{ep}.{ring}.{i}` names left by crashed
# PRE-upgrade runs during the transition.
_RING_NAMES_BY_ROLE = {
    "cap": ("gray_left", "gray_right", "depth_m"),
    "vio": ("kf_gray", "kf_depth"),
    "slm": (),
}
_RING_SLOTS = 64


def _endpoints_history_path() -> Path:
    """File where every launcher persists its endpoint trio. cleanup reads it
    so a prior run whose sock files were already deleted (manual cleanup,
    /tmp reaper, etc.) is still recoverable next launch."""
    return Path(tempfile.gettempdir()) / "ours_ipc" / ".endpoints_seen"


def _record_endpoints(eps: list[str]) -> None:
    p = _endpoints_history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        seen: set[str] = set()
        if p.is_file():
            seen = {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}
        seen.update(eps)
        # Cap so the file doesn't grow forever; recent entries first.
        capped = sorted(seen)[:1024]
        p.write_text("\n".join(capped) + "\n")
    except OSError:
        pass


def _cleanup_orphans() -> None:
    """Best-effort: unlink every stale `oak.*` SHM segment + IPC socket file.

    macOS has no public listing API for POSIX shared memory, so we union
    endpoint candidates from THREE sources, in order of recency:
      1. The launcher's IPC socket directory (`oak.*.sock`).
      2. A persistent endpoints-seen file (every launcher records itself,
         so prior endpoints whose sock files were already deleted are still
         covered).
      3. (fallback) Brute-force the 4096 `l<hex>` PID-suffix space when both
         of the above turn up nothing -- runs on the EMFILE recovery path
         only, since it adds ~2.6 s on macOS.
    Missing segments are silently skipped -- this is a guard against
    accumulation, not a correctness operation.
    """
    sock_dir = Path(tempfile.gettempdir()) / "ours_ipc"
    endpoints: set[str] = set()
    if sock_dir.is_dir():
        for p in glob.glob(str(sock_dir / "oak.*.sock")):
            endpoints.add(Path(p).name[:-len(".sock")])
    hist = _endpoints_history_path()
    if hist.is_file():
        try:
            for ln in hist.read_text().splitlines():
                ln = ln.strip()
                if ln.startswith("oak."):
                    endpoints.add(ln)
        except OSError:
            pass
    if not endpoints:
        return
    unlinked = 0
    for ep in sorted(endpoints):
        parts = ep.split(".")
        if len(parts) != 3:
            continue
        role = parts[1]
        for ring in _RING_NAMES_BY_ROLE.get(role, ()):
            # Current layout: each ring is ONE block named exactly `{ep}.{ring}`
            # (slots are byte offsets inside it -- see SharedArrayRing). Plus a
            # legacy pass over the OLD per-slot names `{ep}.{ring}.{i}` so
            # segments leaked by a PRE-upgrade crashed run are still reclaimed
            # during the transition. Both are best-effort; missing names skipped.
            candidates = [f"{ep}.{ring}"]
            candidates += [f"{ep}.{ring}.{i}" for i in range(_RING_SLOTS)]
            for shm_name in candidates:
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
    if sock_dir.is_dir():
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
def build_capture_args(args, cap_ep: str) -> list[str]:
    """Build the ``imu_camera.main`` argv from the parsed launcher ``args``.

    Pure (no I/O, no spawning) so the flag-forwarding contract is unit-testable
    without launching subprocesses. ``imu_camera.main`` defaults to REPLAY and takes
    an explicit ``--live`` for hardware (inverse of the old ours.proc.capture, which
    defaulted to live):

    * replay -> ``--session PATH [--max-frames N]``
    * live   -> ``--live [--no-gyro] [--recalibrate-bias] [--use-camera-calib]``

    ``--use-camera-calib`` is forwarded ONLY in the live branch and ONLY when set --
    it opts the capture process into the operator's saved stereo calib (default OFF =
    factory). Only capture needs it; vio/slam get whatever calib capture publishes on
    the retained ``calib.bundle``. ``--vl53l9cx`` applies to both modes, so it is
    appended after the mode branch.
    """
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
        if args.use_camera_calib:
            capture_args += ["--use-camera-calib"]
    if args.vl53l9cx:
        capture_args += ["--vl53l9cx"]
    return capture_args


def build_vio_args(args, cap_ep: str, vio_ep: str, slam_ep: str,
                   use_worker: bool) -> list[str]:
    """Build the ``vio.main`` argv from the parsed launcher ``args``.

    Pure (no I/O, no spawning) so the flag-forwarding contract is unit-testable
    without launching subprocesses -- the same discipline as ``build_capture_args``.

    ``--tight`` selects the tight-coupled backend; only on that path do we wire the
    ``--slam-endpoint`` (closed-loop SLAM->VIO feedback) and forward
    ``--stabilize-velocity`` (Phase-4 velocity regularisation -- CV prior + gated
    ZUPT). Both are OPT-IN and tight-only: ``--stabilize-velocity`` is appended ONLY
    when ``args.tight AND args.stabilize_velocity``, so the loose path and the
    tight-without-flag path are unchanged (the offline oracle stays byte-identical).
    A ``--stabilize-velocity`` without ``--tight`` is dropped here (the caller warns).
    """
    vio_args: list[str] = ["--capture-endpoint", cap_ep, "--endpoint", vio_ep,
                           "--kf-every", str(args.kf_every)]
    if args.no_gyro:
        vio_args += ["--no-gyro"]
    if use_worker:
        vio_args += ["--worker"]
    if args.tight:
        vio_args += ["--tight"]
        # CLOSED-LOOP feedback (slam -> vio): give VIO the slam endpoint so its
        # --tight live pose subscribes loop.correction and the SLAM pose-graph
        # correction is fed back into the live pose (drift bounded on revisits).
        # Only on the --tight path; the loose pipeline never wires it.
        vio_args += ["--slam-endpoint", slam_ep]
        # Phase-4 velocity regularisation: tight-only, opt-in. Forwarded ONLY when
        # BOTH --tight AND --stabilize-velocity are set, so the default end-to-end
        # path (and the oracle) never see it.
        if args.stabilize_velocity:
            vio_args += ["--stabilize-velocity"]
    return vio_args


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
    ap.add_argument("--live", action="store_true",
                    help="open the OAK-D (the default when no --session is given; "
                         "accepted explicitly for convenience). Ignored if --session is set.")
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
    ap.add_argument("--use-camera-calib", action="store_true",
                    help="live: apply the operator's SAVED per-device stereo calib "
                         "(from the wizard) instead of the FACTORY calib. Default "
                         "OFF -- factory is the trusted reference. Forwarded to the "
                         "capture subprocess.")
    ap.add_argument("--worker", action="store_true",
                    help="run the heavy BA/SLAM solves in worker subprocesses "
                         "(GIL-free). Off by default -- SLAM already stays "
                         "responsive via its latest-only in-process inbox.")
    ap.add_argument("--endpoint-suffix", default="",
                    help="append SUFFIX to canonical endpoint names so two "
                         "launchers can co-exist (e.g. 'dev', 'ci')")
    ap.add_argument("--auto-suffix", action="store_true",
                    help="derive endpoint suffix from this launcher's PID")
    ap.add_argument("--no-ui", action="store_true",
                    help="don't open the UI -- useful for capture-only headless runs")
    ap.add_argument("--vl53l9cx", action="store_true",
                    help="simulate a VL53L9CX-class ToF camera in the capture "
                         "process: compute depth at the source resolution then "
                         "downsample gray + depth to 54x42 (works live + replay)")
    ap.add_argument("--tight", action="store_true",
                    help="run the VIO process with its TIGHT-coupled backend "
                         "(joint visual + IMU window optimiser) instead of the "
                         "default loose windowed-BA backend. Forwarded to "
                         "vio.main --tight; loose stays the default.")
    ap.add_argument("--stabilize-velocity", action="store_true",
                    help="tight only: enable Phase-4 velocity regularisation "
                         "(CV prior + gated ZUPT) to curb 54x42/shake velocity "
                         "divergence. Forwarded to vio.main --stabilize-velocity "
                         "only with --tight; ignored (warned) on the loose path.")
    args = ap.parse_args()

    # SLAM keeps its live map current via a LATEST-ONLY in-process inbox (set in
    # slam.main) -- it drops a backlog instead of lagging, with NO worker
    # subprocess (so no resource_tracker semaphore noise on every shutdown /
    # Restart). `--worker` is an opt-in for running the heavy solves GIL-free in
    # child processes; off by default.
    use_worker = bool(args.worker)

    # ---- Endpoint names (computed ONCE, identical across restarts) --------
    # The auto-suffix is derived from THIS launcher's PID, so re-spawning the
    # pipeline (Restart button) re-creates the same-named endpoints + rings.
    # _cleanup_orphans + SharedArrayRing.create's cleanup_stale reclaim any
    # leftovers from the prior generation each iteration.
    if args.auto_suffix:
        # `oak.cap.l<pidhex>` -- the ring shm name is `{ep}.{ring}` and must fit
        # inside macOS's 30-char POSIX shm name limit. See `SharedArrayRing.create`
        # for the gate.
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
    # Persist our endpoints so the NEXT launcher's `_cleanup_orphans` can
    # recover them even if our sock files are deleted between runs.
    _record_endpoints([cap_ep, vio_ep, slam_ep])

    py = sys.executable
    env = dict(os.environ)

    # ---- Build per-proc argv ---------------------------------------------
    # Capture argv (mode branch + flag forwarding) lives in build_capture_args so
    # the contract is unit-testable without spawning subprocesses.
    capture_args = build_capture_args(args, cap_ep)

    # VIO argv (flag forwarding, incl. --tight / --slam-endpoint /
    # --stabilize-velocity) lives in build_vio_args so the contract is
    # unit-testable without spawning subprocesses.
    if args.stabilize_velocity and not args.tight:
        # --stabilize-velocity only affects the tight backend's velocity state;
        # the loose path has no velocity to regularise, so warn + drop it (the
        # builder already gates it behind --tight, this just tells the operator).
        LOG.warning("launcher: --stabilize-velocity has no effect without "
                    "--tight (loose path has no velocity state); ignoring it")
    vio_args = build_vio_args(args, cap_ep, vio_ep, slam_ep, use_worker)

    # NB: the new `slam.main` is a PURE consumer of VIO's output and -- unlike the
    # pre-split `ours.proc.slam` -- intentionally DROPPED `--capture-endpoint`
    # (its docstring: "We deliberately don't subscribe to capture at all"). So we
    # wire only `--vio-endpoint` / `--endpoint` here; passing the old
    # `--capture-endpoint` would make slam's argparse abort on startup.
    slam_args = ["--vio-endpoint", vio_ep,
                 "--endpoint", slam_ep]
    if use_worker:
        slam_args += ["--worker"]

    # The UI's calib + visualise windows subscribe capture directly (IMU /
    # imucam.sample / frame.depth), so it must know the suffixed live endpoint
    # under an --auto-suffix run.
    ui_args = ["--capture-endpoint", cap_ep,
               "--vio-endpoint", vio_ep, "--slam-endpoint", slam_ep]

    # ---- SIGTERM handler (registered ONCE) -------------------------------
    # `kill <launcher_pid>` from outside must clean up the whole tree, not just
    # the launcher itself. The handler reads a STABLE mutable `procs` holder
    # that the restart loop clear()s + repopulates each generation, so it always
    # signals the CURRENT generation's children regardless of how many restarts
    # have happened.
    #
    # CRITICAL: do NOT call `_terminate(procs)` here -- `_terminate` polls each
    # `Popen.poll()`, which calls `os.waitpid(pid, WNOHANG)` on the same pid the
    # main thread is blocked in `ui_proc.wait()` on. The two waitpid callers race
    # for the single reap event, leaving Popen's `returncode` stuck at None on
    # the loser, so the handler's `_terminate` loop spins the full 10 s deadline
    # and SIGKILLs the UI even though it already exited cleanly. Instead just
    # forward SIGTERM to each child (they likely already got it from the
    # process-group signal anyway) and `os._exit` immediately; children either
    # finish their own shutdown or get reaped by init when launcher dies.
    procs: list[subprocess.Popen] = []

    def _on_sigterm(_signo, _frame):
        LOG.info("launcher: SIGTERM -> forwarding to children + exiting")
        for p in procs:
            try:
                p.terminate()
            except Exception:                                      # noqa: BLE001
                pass
        os._exit(143)                                             # 128 + SIGTERM
    signal.signal(signal.SIGTERM, _on_sigterm)

    # ---- Boot order ------------------------------------------------------
    # capture FIRST so the retained `calib.bundle` is published as soon as it
    # builds the frontend. vio + slam connect with retried `IpcClientBus.start`
    # so booting them after capture is fine; this just minimises the connect
    # retry noise in the log.
    def _spawn_pipeline() -> None:
        """Clear `procs` in place and spawn a fresh capture+vio+slam generation.

        Mutates the SHARED `procs` holder (clear + append) so the once-registered
        SIGTERM handler always sees the live generation. Best-effort SHM cleanup
        runs FIRST so the prior generation's segments are reclaimed before the
        same-named rings are re-created (macOS POSIX shm persists past SIGKILL;
        a stale namespace eventually trips capture's shm_open() with EMFILE).
        """
        _cleanup_orphans()
        procs.clear()
        cap_proc = _spawn(py, "imu_camera.main", capture_args, env=env,
                          name="imu_camera")
        procs.append(cap_proc)
        # tiny sleep so capture's IPC server is listening before vio / slam
        # try their first connect (vio retries so this is cosmetic, but it
        # gives a clean first-attempt success in the log).
        time.sleep(0.2)
        procs.append(_spawn(py, "vio.main", vio_args, env=env, name="vio"))
        procs.append(_spawn(py, "slam.main", slam_args, env=env,
                            name="slam"))

    if args.no_ui:
        # No restart button without a UI -- the --no-ui path runs exactly ONCE,
        # independent of the restart loop below.
        try:
            _spawn_pipeline()
            cap_proc = procs[0]
            vio_proc, slam_proc = procs[1], procs[2]
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
        finally:
            LOG.info("launcher: shutting down background procs ...")
            _terminate(procs)
            LOG.info("launcher: bye")
        return int(rc)

    # ---- Restart loop ----------------------------------------------------
    # Each iteration spawns a FRESH capture+vio+slam+ui generation, blocks on the
    # UI, then tears that generation down. The IPC bus is one-way (server->client)
    # so the UI cannot reset vio/slam in place; the robust "chay lai tu dau" is a
    # full respawn, which the UI requests via the RESTART_EXIT_CODE return code.
    rc = 0
    try:
        while True:
            _spawn_pipeline()
            # UI in foreground -- inherits stdout / stderr / stdin so the user
            # sees Qt warnings and can Ctrl-C cleanly.
            ui_proc = subprocess.Popen([py, "-m", "ui.main", *ui_args],
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
            # Tear down THIS generation on the main thread (no waitpid race: the
            # UI is already reaped by `ui_proc.wait()` above).
            LOG.info("launcher: shutting down background procs ...")
            _terminate(procs)
            if rc == RESTART_EXIT_CODE:
                LOG.info("launcher: restart requested -> respawning pipeline")
                continue
            break
    finally:
        LOG.info("launcher: bye")

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
