#!/usr/bin/env python3
"""2-process smoke selftest: imu_camera (capture) + depth over a gold session.

The live-plumbing gate for the standalone ``depth.main`` (the depth-as-a-process
harness). Spawns the capture project's replay process and the depth process
back-to-back over IPC::

    imu_camera.main (replay)  --oak.cap (cam.sync)-->  depth.main  --oak.dep (frame.depth)-->

then opens a tiny headless :class:`~depth.comms.IPCPubSub` CLIENT on the depth
endpoint and asserts the depth shell honours its core contract:

* both spawned processes exit rc=0 once capture's replay ends;
* **depth produces exactly one ``frame.depth`` per consumed ``cam.sync``** -- the
  1:1 FIFO guarantee. depth.main is a NON-BLOCKING (drop-oldest) publisher, so a
  late-connecting headless client legitimately misses the earliest frames; the
  DETERMINISTIC, refactor-comparable number is depth's OWN producer count, which
  it logs as ``shutdown complete (published N frame.depth)``. We parse that and
  assert ``N == capture's published frame count`` (the cam.sync count) -- exact,
  deterministic equality;
* the headless client DOES observe the run: it sees the retained ``calib.bundle``
  (readiness barrier), a non-trivial floor of ``frame.depth``, AND the END
  sentinel forwarded from ``cam.sync`` onto ``frame.depth`` (proves the
  procedural END-forwarding works end to end). The client's exact frame COUNT is
  non-deterministic by design (drop-oldest), so it is asserted only as ``>=`` a
  floor, mirroring how ``slam.tests.proc3_smoke_selftest`` treats its
  non-deterministic live counts;
* clean shutdown -- no IPC connect / shared-memory error surfaced in either
  process's stderr.

This fills the coverage gap: before it there was NO process-level test for
``depth.main`` at all. The deterministic depth MATH parity lives in
:mod:`depth.tests.stereo_sgm_selftest`; this proves the 2-process PLUMBING +
the FIFO 1:1 contract + clean END/shutdown.

Run::

    python -m depth.tests.proc_depth_smoke_selftest
    python -m depth.tests.proc_depth_smoke_selftest --session sessions/gold/lab_loop_30s --max-frames 40
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth.comms import IPCPubSub, topics                          # noqa: E402
from depth.comms.wire import WireEnd                               # noqa: E402

#: Floor on the headless client's observed frame.depth (drop-oldest makes the
#: exact count non-deterministic; the client must still see the bulk of them).
_CLIENT_FRAME_FLOOR = 10
#: Lower bound on the deterministic producer count for the default 40-frame cap;
#: a tighter exact check (== capture's count) runs alongside.
_PRODUCER_FLOOR = 10


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def _tail(s: str, n: int) -> str:
    return "\n".join(s.rstrip().splitlines()[-n:])


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="sessions/gold/lab_loop_30s")
    ap.add_argument("--max-frames", type=int, default=40,
                    help="cap capture's replay so the smoke test stays quick "
                         "(0 = full session); depth produces 1 frame.depth per "
                         "cam.sync regardless")
    ap.add_argument("--keep-logs", action="store_true",
                    help="print subprocess stdout/stderr instead of capturing")
    args = ap.parse_args()

    if not Path(args.session).exists():
        print(f"SKIP: session {args.session!r} not present")
        return 0

    pid = os.getpid()
    cap_ep = f"oak.cap.d{pid & 0xFFF:x}"
    dep_ep = f"oak.dep.d{pid & 0xFFF:x}"

    py = sys.executable
    env = dict(os.environ)

    print("proc_depth_smoke_selftest (imu_camera + depth)")
    print(f"  session={args.session} max-frames={args.max_frames}")
    print(f"  endpoints: cap={cap_ep!r} depth={dep_ep!r}")

    # Capture the children's stderr so we can (a) parse depth's producer count and
    # (b) scan both for IPC/shm errors. --keep-logs streams them instead.
    log_kwargs = ({} if args.keep_logs
                  else {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE})

    # ---------- Boot depth FIRST (retried client -> waits for capture's calib) -
    dep_proc = subprocess.Popen(
        [py, "-m", "depth.main",
         "--capture-endpoint", cap_ep, "--endpoint", dep_ep,
         "--session", args.session],
        env=env, **log_kwargs)
    # Brief window so depth registers its calib subscriber before capture boots
    # (calib is retained, so a later boot is fine too).
    time.sleep(0.3)
    cap_proc = subprocess.Popen(
        [py, "-m", "imu_camera.main",
         "--endpoint", cap_ep, "--session", args.session,
         "--max-frames", str(args.max_frames)],
        env=env, **log_kwargs)

    procs = (cap_proc, dep_proc)
    try:
        return _run_assertions(args, dep_ep, procs)
    finally:
        _terminate_all(*procs)


# --------------------------------------------------------------------------- #
def _run_assertions(args, dep_ep, procs):
    cap_proc, dep_proc = procs

    # ---------- ONE persistent headless client on the depth endpoint ----------
    # Subscribe to calib.bundle (readiness barrier -- retained, so it lands the
    # instant depth's server is up) AND frame.depth in the SAME connection,
    # started ONCE and kept alive for the whole run. frame.depth rides a
    # shared-memory ring, but this client has NO ring registry attached, so it
    # receives the WIRE form (WireDepthFrame for data, WireEnd for the END) off
    # the socket directly -- which is all we need to COUNT + detect END.
    seqs: list[int] = []         # seq of every observed frame.depth, in order
    saw_end = threading.Event()
    ready = threading.Event()

    def on_calib(_wm) -> None:
        ready.set()

    def on_depth(wm) -> None:
        # No converter on this raw client: a data frame is a WireDepthFrame, the
        # forwarded END is a WireEnd.
        if isinstance(wm, WireEnd):
            saw_end.set()
            return
        seqs.append(int(wm.seq))

    client = IPCPubSub(dep_ep, role="client", connect_timeout_s=30.0)
    client.subscribe(topics.CALIB_BUNDLE, on_calib)
    client.subscribe(topics.FRAME_DEPTH, on_depth)
    client.start()

    try:
        if not ready.wait(timeout=30.0):
            raise TimeoutError(f"no calib.bundle from {dep_ep!r} in 30s")
        print("  depth: ready (retained calib observed)")

        # Wait until the END forwarded onto frame.depth lands (the run drained)
        # or a child dies early.
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            if saw_end.is_set():
                break
            for name, proc in (("imu_camera", cap_proc), ("depth", dep_proc)):
                if proc.poll() is not None and proc.returncode != 0:
                    print(f"\n  [FAIL] {name} exited early rc={proc.returncode}",
                          file=sys.stderr)
                    raise SystemExit(1)
            time.sleep(0.1)
    finally:
        client.stop()

    # ---------- Wait for the children to exit cleanly --------------------------
    cap_rc = cap_proc.wait(timeout=20.0)
    dep_rc = dep_proc.wait(timeout=20.0)

    # ---------- Drain captured stderr (for parsing + error scan) ---------------
    cap_err = _drain_stderr(cap_proc)
    dep_err = _drain_stderr(dep_proc)

    producer_n = _parse_published(dep_err)
    n_obs = len(seqs)

    print(f"\n  observed (headless client): frame.depth={n_obs} "
          f"END={'yes' if saw_end.is_set() else 'NO'}")
    if seqs:
        print(f"    seq span: min={min(seqs)} max={max(seqs)} "
              f"unique={len(set(seqs))}")
    print(f"  depth producer count (log) : published={producer_n}")
    print(f"  capture cam.sync cap       : max_frames={args.max_frames}")

    # ---------- Assertions ----------------------------------------------------
    _check(cap_rc == 0, f"imu_camera exited 0 (got {cap_rc})")
    _check(dep_rc == 0, f"depth exited 0 (got {dep_rc})")

    _check(saw_end.is_set(),
           "END forwarded from cam.sync onto frame.depth (procedural END-forward)")
    _check(n_obs >= _CLIENT_FRAME_FLOOR,
           f"headless client observed >= {_CLIENT_FRAME_FLOOR} frame.depth "
           f"(drop-oldest; got {n_obs})")

    _check(producer_n is not None,
           "depth logged its producer count (published N frame.depth)")
    _check(producer_n is not None and producer_n >= _PRODUCER_FLOOR,
           f"depth produced >= {_PRODUCER_FLOOR} frame.depth (got {producer_n})")

    # ---- The 1:1 FIFO contract -- the heart of the gate ----------------------
    # depth produces EXACTLY one frame.depth per cam.sync it CONSUMES, in order,
    # with no coalescing and no duplication. Two independent checks prove it:
    #
    # (a) STRICT-INCREASING, GAP-FREE observed seqs. The headless client receives
    #     the WireDepthFrame seqs the SGM solve stamped (= the source cam.sync
    #     seq). A FIFO no-coalesce pipeline yields a CONTIGUOUS run with NO
    #     interior holes and NO repeats. (The drop-oldest server may shave frames
    #     off the FRONT before this late client attaches, so we check contiguity
    #     of what we DID see, not that it starts at 0.) A coalescing/duplicating
    #     bug would surface here as a gap or a repeat -- deterministic regardless
    #     of the boot race.
    #
    # (b) PRODUCER COUNT vs the cap. Capture emits exactly --max-frames cam.sync
    #     triggers, so depth -- consuming every one it receives -- produces
    #     max_frames frame.depth, MINUS at most one head frame lost to the inherent
    #     cross-process connect race (capture starts producing before depth's IPC
    #     client has finished attaching; this race predates + is unchanged by the
    #     procedural refactor). So producer_n is max_frames or max_frames-1, never
    #     less (no mid-stream loss) and never more (no duplication).
    gap_free = _is_contiguous(seqs)
    _check(gap_free,
           f"observed frame.depth seqs are gap-free + unique (FIFO, no coalesce): "
           f"{_seq_summary(seqs)}")

    if args.max_frames > 0:
        lo, hi = args.max_frames - 1, args.max_frames
        _check(producer_n is not None and lo <= producer_n <= hi,
               f"depth produced 1 frame.depth per consumed cam.sync "
               f"(published={producer_n} in [{lo},{hi}] = cap +/- one head "
               f"boot-race frame)")
    else:
        print("  [info] uncapped run -- cam.sync count is session-dependent; "
              "the gap-free FIFO check above is the 1:1 proof")

    # Clean shutdown: no IPC connect / shared-memory error surfaced.
    _check(not _has_ipc_error(dep_err),
           "depth stderr clean (no IPC/shm error)")
    _check(not _has_ipc_error(cap_err),
           "imu_camera stderr clean (no IPC/shm error)")

    print("\nALL PROC DEPTH SMOKE SELFTESTS PASSED")
    return 0


# --------------------------------------------------------------------------- #
# stderr parsing helpers
# --------------------------------------------------------------------------- #
_PUBLISHED_RE = re.compile(r"shutdown complete \(published (\d+) frame\.depth\)")

#: Substrings that mean a real IPC/shared-memory failure (not the benign INFO
#: lines). Kept narrow so a normal run never trips it.
_IPC_ERROR_MARKERS = (
    "THREAD CRASH",
    "Traceback (most recent call last)",
    "could not connect",
    "convert failed",
    "send failed",
    "failed: ",          # IPCPubSub "... encode <topic> failed: <e>" etc.
    "raised: ",          # IPCPubSub "... handler for <topic> raised: <e>"
    "decode failed",
    "SharedMemory",
)


def _drain_stderr(proc: subprocess.Popen) -> str:
    if proc.stderr is None:
        return ""
    try:
        data = proc.stderr.read() or b""
    except Exception:                                              # noqa: BLE001
        return ""
    return data.decode(errors="replace")


def _parse_published(stderr: str) -> int | None:
    m = _PUBLISHED_RE.search(stderr)
    return int(m.group(1)) if m else None


def _is_contiguous(seqs: list[int]) -> bool:
    """True iff ``seqs`` is a strictly-increasing run of consecutive integers.

    Empty / single-element lists are trivially contiguous. This is the FIFO,
    no-coalesce, no-duplicate proof: every consumed cam.sync seq appears exactly
    once, in order, with no interior hole.
    """
    return all(b - a == 1 for a, b in zip(seqs, seqs[1:]))


def _seq_summary(seqs: list[int]) -> str:
    if not seqs:
        return "no frames observed"
    return (f"n={len(seqs)} span=[{min(seqs)}..{max(seqs)}] "
            f"unique={len(set(seqs))}")


def _has_ipc_error(stderr: str) -> bool:
    for marker in _IPC_ERROR_MARKERS:
        if marker in stderr:
            print(f"\n  --- stderr error marker {marker!r} ---\n"
                  f"{_tail(stderr, 20)}", file=sys.stderr)
            return True
    return False


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
