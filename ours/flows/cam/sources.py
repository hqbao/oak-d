"""Stereo-frame sources for :class:`~ours.flows.cam.CamFlow`.

The cam flow drives the *schedule* (it pulls one stereo pair per scheduler tick)
but the *origin* of the frames is injected as a ``CamSource`` so the same flow
runs offline (replay of a recorded session) and on the bench (the OAK-D cameras).

A source is pull-based -- :meth:`CamSource.read` returns the next
``(seq, ts_ns, gray_left, gray_right)`` or ``None`` when exhausted -- because the
camera flow, unlike the free-running IMU, decides *when* to grab a frame.

Only :class:`LiveCamSource` touches depthai, imported lazily inside :meth:`open`.
"""
from __future__ import annotations

import time

import numpy as np

from ...lib.io.reader import SessionReader


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

    Reads the mono pair off a :class:`~ours.lib.oak_live.SharedLiveDevice` (the
    OAK-D is single-client, so the camera and IMU readers must share ONE
    device/pipeline). It pairs left/right by sequence number -- the cameras are
    hardware-synced, so a shared ``seq`` is a true same-instant pair -- and tags
    the pair with the left frame's device timestamp, the clock the IMU flow
    drains against. depthai is pulled lazily by the shared device; hardware-only.
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
