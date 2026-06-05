"""Time-synchronised (image, depth, IMU) sample bundles -- the first VIO block.

This is the transparent input layer of the from-scratch VIO: it turns the three
raw recorded streams -- the feature image, its aligned depth map, and the IMU --
into a sequence of bundles where all three are aligned on the SAME device clock.

Why this is even possible (the honest part)
-------------------------------------------
The recorder writes every stream with a device-clock ``ts_ns`` timestamp:

* ``frames.jsonl`` -- one timestamp per stereo frame. The stored left image is
  the **rectified left** (the image you extract features from) and the depth map
  is on that SAME rectified-left grid, so image + depth are already aligned in
  BOTH space and time: one capture instant, one pixel grid. (Live, the matcher's
  ``dense_depth_rectified_left`` returns exactly this same pair -- rectified left
  + its depth -- so an offline bundle is structurally identical to a live one.)
* ``imu.jsonl`` -- one timestamp per IMU sample (gyro rad/s + accel m/s^2, IMU
  frame), on the same clock.

So "synchronise the three" reduces to one well-defined operation: for each frame
at time ``t_cur``, take the IMU samples that fall in the interval since the
previous frame, ``(t_prev, t_cur]``. That block of IMU is exactly the motion that
happened DURING the step from the previous image to this one -- the segment a VIO
preintegrates (:func:`oakd.vio.imu.preintegrate_imu` /
:meth:`oakd.vio.imu.GyroPreintegrator.delta_rotation`). Nothing here invents
data; it only groups real samples by time.

Conventions
-----------
* ``gray`` is uint8 ``(H, W)`` -- the rectified-left feature image.
* ``depth_m`` is float32 ``(H, W)`` metres on the same grid; ``0.0`` == invalid.
* ``imu`` spans ``(t_prev_ns, t_ns]``; the first frame has an EMPTY segment
  (there is no previous frame, hence no inter-frame motion) -- that is correct,
  not a gap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .reader import SessionReader


@dataclass
class ImuSegment:
    """The IMU samples spanning one inter-frame interval (IMU frame).

    Selection only -- the timestamps are the samples' real device-clock times,
    never clamped to the interval endpoints. Clamping/interpolation to the exact
    endpoints is the integrator's job, so this stays a pure, inspectable slice.
    """

    ts_ns: np.ndarray  # (K,) int64 device-clock nanoseconds, increasing
    gyro: np.ndarray   # (K, 3) rad/s, IMU frame
    accel: np.ndarray  # (K, 3) m/s^2 specific force, IMU frame

    def __len__(self) -> int:
        return int(self.ts_ns.shape[0])

    @property
    def span_s(self) -> float:
        """Seconds between the first and last sample (0 if fewer than 2)."""
        if len(self) < 2:
            return 0.0
        return float(int(self.ts_ns[-1]) - int(self.ts_ns[0])) * 1e-9


@dataclass
class SyncedSample:
    """One time-aligned bundle: feature image + aligned depth + IMU segment.

    ``gray`` and ``depth_m`` are the same capture instant on the same pixel grid
    (rectified left); ``imu`` is the IMU measured over ``(t_prev_ns, t_ns]`` --
    the motion during the step from the previous frame to this one.
    """

    seq: int
    t_prev_ns: int          # interval start (== t_ns for the first frame)
    t_ns: int               # this frame's capture time (interval end)
    gray: np.ndarray        # (H, W) uint8, rectified-left feature image
    depth_m: np.ndarray     # (H, W) float32 metres, aligned to gray; 0 == invalid
    imu: ImuSegment         # IMU samples spanning (t_prev_ns, t_ns]
    K: np.ndarray           # 3x3 intrinsics for gray / depth

    @property
    def dt_s(self) -> float:
        """Inter-frame interval in seconds (0 for the first frame)."""
        return (self.t_ns - self.t_prev_ns) * 1e-9


def slice_imu(ts_ns: np.ndarray, gyro: np.ndarray, accel: np.ndarray,
              t0_ns: int, t1_ns: int, *, bracket: bool = True) -> ImuSegment:
    """Select the IMU samples covering the interval ``(t0_ns, t1_ns]``.

    Returns every sample with ``t0 < ts <= t1``. With ``bracket=True`` (default)
    one extra sample on each side is included when available -- the last sample
    ``<= t0`` and the first sample ``> t1`` -- so an integrator can interpolate
    angular velocity / acceleration right up to both endpoints (integrating over
    ``[t0, t1]`` needs the samples that bracket each end). The bracketing samples
    keep their true timestamps; clamping to the endpoints is the integrator's job.

    With ``bracket=False`` the segments of consecutive frames tile the timeline
    exactly (each IMU sample lands in exactly one segment), which is the easy
    invariant to test.

    Empty interval (``t1 <= t0``, e.g. the first frame) -> empty segment.
    """
    ts = np.asarray(ts_ns, dtype=np.int64)
    g = np.asarray(gyro, dtype=np.float64)
    a = np.asarray(accel, dtype=np.float64)
    if t1_ns <= t0_ns or ts.size == 0:
        return ImuSegment(ts_ns=ts[:0].copy(),
                          gyro=g[:0].copy(), accel=a[:0].copy())

    # Samples strictly after t0 and up to (incl.) t1: indices [lo, hi).
    lo = int(np.searchsorted(ts, t0_ns, side="right"))
    hi = int(np.searchsorted(ts, t1_ns, side="right"))
    if bracket:
        lo = max(lo - 1, 0)          # one sample at/just before t0
        hi = min(hi + 1, ts.size)    # one sample just after t1
    return ImuSegment(ts_ns=ts[lo:hi].copy(),
                      gyro=g[lo:hi].copy(), accel=a[lo:hi].copy())


def iter_synced(reader: SessionReader, *, load_right: bool = False,
                bracket: bool = True) -> Iterator[SyncedSample]:
    """Yield :class:`SyncedSample` bundles from a recorded session, in order.

    For each stereo frame this loads the rectified-left feature image + its
    aligned depth, then attaches the IMU samples measured since the previous
    frame (:func:`slice_imu`). The whole VIO can then be developed against these
    bundles -- one transparent, offline-testable input -- with no device attached.

    ``load_right=True`` also decodes the (raw, unrectified) right image, e.g. to
    recompute depth from scratch; the default skips it for speed.
    """
    imu = reader.load_imu()
    ts_i, gyro, accel = imu["ts_ns"], imu["gyro"], imu["accel"]

    t_prev: int | None = None
    for i in range(len(reader)):
        fr = reader.load_frame(i, load_right=load_right)
        t0 = fr.ts_ns if t_prev is None else t_prev
        seg = slice_imu(ts_i, gyro, accel, t0, fr.ts_ns, bracket=bracket)
        yield SyncedSample(
            seq=fr.seq,
            t_prev_ns=t0,
            t_ns=fr.ts_ns,
            gray=fr.gray_left,
            depth_m=fr.depth_m,
            imu=seg,
            K=fr.K,
        )
        t_prev = fr.ts_ns
