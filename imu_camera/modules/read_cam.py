"""read_cam producer: pull stereo on a schedule, trigger the IMU pack.

One half of the acquisition front-end (``imu_cam`` is the other). It owns the
*schedule*: one stereo pair per scheduler tick (``fps`` Hz). For each pair it
publishes a single :class:`~imu_camera.comms.messages.CamSync` (the frames + their
device timestamp) on ``cam.sync`` -- the trigger the
:class:`~imu_camera.modules.pipeline.ImuCamWorker` reacts to.

This module is exactly ONE producer: the :class:`ReadCamModule` (a plain
``threading.Thread``) plus the pull-based frame sources (replay offline / live
OAK-D). depthai is only touched by :class:`LiveCamSource`, imported lazily, so the
offline path never pulls the device library.

A source is pull-based -- :meth:`CamSource.read` returns the next
``(seq, ts_ns, gray_left, gray_right)`` or ``None`` when exhausted -- because the
camera producer, unlike the free-running IMU, decides *when* to grab a frame.

Procedural shape: :class:`ReadCamModule` was a reactive ``SourceModule`` whose
single step published ``cam.sync``. It is now a plain ``threading.Thread`` that
publishes ``cam.sync`` inline and forwards END on the same topic when the source
exhausts -- the framework's ``SourceModule.run`` / step-chain machinery was just
glue for a one-step source, so it is written out explicitly here. The public
surface (constructor, ``start`` / ``stop`` / ``join`` / ``is_alive`` / ``done`` /
``error``) is unchanged so ``imu_camera.main`` + the selftest drive it as before.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from imu_camera.comms import LocalPubSub, topics
from imu_camera.comms.messages import END, CamSync
from imu_camera.io.reader import SessionReader


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #
class CamSource:
    """Pull-based stereo source."""

    def open(self) -> None:
        """Acquire the source (open files / device). Optional."""

    def read(self):
        """Return the next ``(seq, ts_ns, gray_left, gray_right)`` or ``None``."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the source. Optional."""


class ReplayCamSource(CamSource):
    """Yields a recorded session's stereo frames in order (offline, deterministic)."""

    def __init__(self, reader: SessionReader, *, load_right: bool = True,
                 max_frames: int = 0) -> None:
        self._reader = reader
        self._load_right = bool(load_right)
        n = len(reader)
        self._n = n if max_frames <= 0 else min(max_frames, n)
        self._i = 0

    def read(self):
        if self._i >= self._n:
            return None
        f = self._reader.load_frame(self._i, load_right=self._load_right)
        self._i += 1
        return (int(f.seq), int(f.ts_ns), f.gray_left,
                f.gray_right if self._load_right else None)


class LiveCamSource(CamSource):
    """Grabs synced stereo pairs from a shared OAK-D (raw left + raw right).

    Reads the mono pair off a
    :class:`~imu_camera.device.oak_live.SharedLiveDevice` (the OAK-D is
    single-client, so the camera and IMU readers must share ONE device/pipeline).
    It pairs left/right by sequence number -- the cameras are hardware-synced, so a
    shared ``seq`` is a true same-instant pair -- and tags the pair with the left
    frame's device timestamp, the clock the IMU module drains against. depthai is
    pulled lazily by the shared device; hardware-only.
    """

    def __init__(self, device) -> None:
        self.device = device
        self._pend_l: dict[int, object] = {}
        self._pend_r: dict[int, object] = {}

    def open(self) -> None:
        self.device.acquire()

    @staticmethod
    def _seq(msg) -> int:
        try:
            return int(msg.getSequenceNum())
        except Exception:
            return -1

    @staticmethod
    def _gray(frame) -> np.ndarray:
        g = frame.getCvFrame()
        if g.ndim == 3:                                  # BGR -> luminance (601)
            g = (g[..., 0] * 0.114 + g[..., 1] * 0.587
                 + g[..., 2] * 0.299).astype(np.uint8)
        return g

    def read(self):
        dev = self.device
        while dev.is_running():
            ld = dev.poll("left")
            while True:
                nxt = dev.poll("left")
                if nxt is None:
                    break
                ld = nxt
            if ld is not None:
                self._pend_l[self._seq(ld)] = ld
            while True:
                nxt = dev.poll("right")
                if nxt is None:
                    break
                self._pend_r[self._seq(nxt)] = nxt
            common = self._pend_l.keys() & self._pend_r.keys()
            if not common:
                for buf in (self._pend_l, self._pend_r):
                    if len(buf) > 8:
                        for k in sorted(buf)[:-8]:
                            buf.pop(k, None)
                time.sleep(0.002)
                continue
            seq = max(common)
            ld = self._pend_l.pop(seq)
            rd = self._pend_r.pop(seq)
            for k in [k for k in self._pend_l if k < seq]:
                self._pend_l.pop(k, None)
            for k in [k for k in self._pend_r if k < seq]:
                self._pend_r.pop(k, None)
            try:
                ts_ns = int(ld.getTimestampDevice().total_seconds() * 1e9)
            except Exception:
                ts_ns = time.monotonic_ns()
            return seq, ts_ns, self._gray(ld), self._gray(rd)
        return None

    def close(self) -> None:
        self.device.release()


# --------------------------------------------------------------------------- #
# Producer thread
# --------------------------------------------------------------------------- #
class ReadCamModule(threading.Thread):
    """Producer thread: emit one :class:`~imu_camera.comms.messages.CamSync` per frame.

    A plain procedural replacement for the old reactive ``SourceModule`` (which
    pushed each frame through a one-step publish chain). It owns the schedule and
    publishes ``cam.sync`` directly, then forwards END on ``cam.sync`` when the
    source exhausts (or ``stop`` is requested) so the downstream
    :class:`~imu_camera.modules.pipeline.ImuCamWorker` drains and shuts down.

    ``fps`` sets the schedule; ``realtime`` paces ticks to it (live-like) versus
    running free (deterministic offline replay). ``source`` supplies the frames
    (``ReplayCamSource`` offline, ``LiveCamSource`` on the bench).

    The name + ``start`` / ``stop`` / ``join`` / ``is_alive`` / ``done`` / ``error``
    surface matches the old ``SourceModule`` so ``imu_camera.main`` (which polls
    ``is_alive`` and calls ``stop`` / ``done.wait``) and the selftest are unchanged.
    """

    def __init__(self, bus: LocalPubSub, source: CamSource, *, fps: int = 20,
                 realtime: bool = False) -> None:
        super().__init__(name="cam", daemon=True)
        self.bus = bus
        self.source = source
        self.fps = max(1, int(fps))
        self.realtime = bool(realtime)
        self.error: str | None = None
        self._stop = threading.Event()
        self.done = threading.Event()   #: set after the producer loop emits END

    def stop(self) -> None:
        """Request the produce loop to break at the next item boundary."""
        self._stop.set()

    def run(self) -> None:
        # Produce frames, publishing cam.sync inline; on exit emit END once on
        # cam.sync and signal done. Mirrors the old SourceModule.run (produce ->
        # _run_chain(publish) -> _emit_end -> done.set), specialised to the single
        # cam.sync publish.
        try:
            self._produce()
        finally:
            self.bus.publish(topics.CAM_SYNC, END)
            self.done.set()

    def _produce(self) -> None:
        try:
            self.source.open()
        except Exception as e:                                    # noqa: BLE001
            # e.g. the OAK-D is absent (X_LINK_DEVICE_NOT_FOUND). Record the
            # reason and return cleanly so the producer still emits END -- the
            # graph drains and the UI can surface the failure instead of hanging.
            self.error = f"camera open failed: {e}"
            return
        period = 1.0 / self.fps
        try:
            next_tick = time.monotonic()
            while not self._stop.is_set():
                if self.realtime:
                    now = time.monotonic()
                    if now < next_tick:
                        time.sleep(next_tick - now)
                    next_tick += period
                try:
                    item = self.source.read()
                except Exception as e:                            # noqa: BLE001
                    self.error = f"camera read failed: {e}"
                    break
                if item is None:
                    break
                seq, ts_ns, gray_left, gray_right = item
                self.bus.publish(topics.CAM_SYNC, CamSync(
                    seq=seq, ts_ns=ts_ns,
                    gray_left=gray_left, gray_right=gray_right))
        finally:
            self.source.close()
