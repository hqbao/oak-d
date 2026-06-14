"""``tof_downsample`` step: simulate a VL53L9CX-class ToF sensor from the OAK-D.

The OAK-D is the stand-in for a real ToF camera we do not have. The user's
validated idea is **compute high, downsample**: run SGM at the SOURCE resolution
(``--width``/``--height``, where stereo actually works), then downsample the
rectified-left gray + the metric depth to the fixed ToF grid ``TOF_W x TOF_H``.

Why downsample beats a direct low-res stereo solve: a 640x400 SGM depth has dense,
sub-pixel-accurate disparities; collapsing each output cell to the MEDIAN of its
valid (>0) source-depth pixels fills the small holes a low-res solve would leave
and averages the noise -- the clean, dense per-pixel depth a real ToF returns
(empirically ~3x the valid coverage of direct 54x42 stereo).

This single step REPLACES the normal ``publish_imucam`` / ``compute_depth`` /
``publish_depth`` trio in the ToF chain (see :mod:`imu_camera.modules.pipeline`):
depth is computed here at source res, both the gray and the depth are reduced to
``TOF_W x TOF_H``, and the step publishes the 54x42 ``imucam.sample`` (gray +
calibrated IMU) AND the 54x42 ``frame.depth``. The capture process sizes its
shared-memory rings + the broadcast ``calib.bundle`` to the same 54x42 grid (with
an anisotropically scaled K), so the whole pipeline stays self-consistent.

Downsampling rules (deliberately different per channel):

* GRAY -> area-average downsample (:func:`_area_resize_gray`) -- area averaging
  is the correct anti-aliasing reduction for an intensity image. This is a pure
  NumPy reimplementation of ``cv2.resize(..., INTER_AREA)`` (each output pixel is
  the fractional-AREA-weighted average of the source pixels it overlaps), so the
  flight runtime needs no OpenCV. Verified bit-exact against ``cv2.INTER_AREA``
  (max abs diff 0 on gold frames).
* DEPTH (metres) -> BLOCK-MEDIAN of the VALID (>0) source pixels per output cell,
  0 where a cell has no valid source depth. We must NOT use cv2/linear on depth:
  linear interpolation would blend across depth discontinuities (object edges)
  and across the 0-valued holes, inventing physically wrong "ramp" depths. A
  block reduce that ignores invalid pixels is the honest ToF behaviour. Block
  sizes are non-integer (640/54, 400/42), so we partition with ``linspace`` bin
  edges that tile the WHOLE frame (no dropped border rows/cols).
"""
from __future__ import annotations

import numpy as np

from imu_camera.comms import LocalPubSub, topics
from imu_camera.comms.messages import DepthFrame, ImuCamPacket
from imu_camera.comms.runtime import NUMBA_PARALLEL_LOCK
from sky.depth.stereo import SGMStereoMatcher


def _block_median_valid(depth: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reduce a metric ``depth`` map to ``(out_h, out_w)`` by per-cell median.

    Each output cell aggregates the VALID (>0) source-depth pixels inside its
    block; cells with no valid source pixel stay 0 (a genuine ToF "no return").
    Block edges come from ``linspace`` so they tile the entire source frame even
    when ``src / out`` is non-integer (640/54, 400/42) -- no border is dropped.

    The median is robust to the SGM speckle a mean would smear, and it never
    blends a real surface with a 0-hole or an across-edge pixel (those are
    excluded as invalid before the median).
    """
    src_h, src_w = depth.shape
    # Bin edges along each axis: out_h+1 / out_w+1 monotone integer boundaries
    # that start at 0 and end at the full extent, so every source pixel lands in
    # exactly one cell and every cell is covered.
    row_edges = np.linspace(0, src_h, out_h + 1).astype(np.int64)
    col_edges = np.linspace(0, src_w, out_w + 1).astype(np.int64)

    out = np.zeros((out_h, out_w), dtype=np.float32)
    for r in range(out_h):
        r0, r1 = row_edges[r], row_edges[r + 1]
        for c in range(out_w):
            c0, c1 = col_edges[c], col_edges[c + 1]
            block = depth[r0:r1, c0:c1]
            valid = block[block > 0.0]
            if valid.size:
                out[r, c] = np.median(valid)
    return out


def _area_weights(n_in: int, n_out: int) -> np.ndarray:
    """1-D fractional-AREA averaging matrix ``(n_out, n_in)`` for a downscale.

    Row ``o`` holds the weights with which the ``n_in`` source samples contribute
    to output sample ``o``: output cell ``o`` covers the source interval
    ``[o*s, (o+1)*s)`` with ``s = n_in / n_out``, and each source pixel ``i``
    contributes the LENGTH of its overlap with that interval (its fractional
    area). The row is normalised to sum to 1, so the result is the area-weighted
    AVERAGE. This is exactly the 1-D kernel ``cv2.resize(..., INTER_AREA)``
    applies (verified bit-exact for the 640->54 / 400->42 ToF reduction).

    Downscale only (``n_out <= n_in``); the ToF grid is always smaller than the
    source, so the upscale branch INTER_AREA falls back to (bilinear) is never
    needed here.
    """
    scale = n_in / n_out
    w = np.zeros((n_out, n_in), dtype=np.float64)
    for o in range(n_out):
        start = o * scale
        end = (o + 1) * scale
        i = int(np.floor(start))
        # Accumulate each source pixel's overlap length with [start, end).
        while i < end and i < n_in:
            lo = max(start, float(i))
            hi = min(end, float(i + 1))
            w[o, i] = hi - lo
            i += 1
    # Normalise each output row to a weighted average (guards the tiny
    # floating-point drift in the boundary cells so rows sum to exactly 1).
    w /= w.sum(axis=1, keepdims=True)
    return w


def _area_resize_gray(gray: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pure-NumPy ``cv2.resize(gray, (out_w, out_h), INTER_AREA)`` (downscale).

    Separable area averaging: average along rows then columns via the 1-D
    fractional-area matrices from :func:`_area_weights`. Rounding matches OpenCV's
    ``saturate_cast<uchar>`` (round-half-AWAY-from-zero, then clamp to [0, 255]),
    so the uint8 output is bit-identical to ``cv2.INTER_AREA`` on these inputs.
    """
    src = np.asarray(gray, dtype=np.float64)
    src_h, src_w = src.shape
    # (out_h, src_h) @ (src_h, src_w) @ (src_w, out_w) -> (out_h, out_w).
    wr = _area_weights(src_h, out_h)
    wc = _area_weights(src_w, out_w)
    avg = wr @ src @ wc.T
    # OpenCV rounds half away from zero (not numpy's round-half-to-even) before
    # the uint8 saturating cast; all values here are >= 0, so floor(x + 0.5).
    rounded = np.floor(avg + 0.5)
    return np.clip(rounded, 0.0, 255.0).astype(np.uint8)


def tof_downsample(matcher: SGMStereoMatcher, bus: LocalPubSub,
                   out_w: int, out_h: int, msg: ImuCamPacket) -> None:
    """Simulate a VL53L9CX ToF frame: SGM at source res, then reduce to 54x42.

    Was ``ToFDownsampleStep(Step)``; the matcher (was ``ctx.state["matcher"]``),
    the bus and the output grid are now explicit args. Logic byte-identical.

    Consumes the calibrated :class:`ImuCamPacket` (source-res stereo + the synced,
    calibrated IMU), runs the dense matcher at source res, downsamples gray +
    depth to ``(out_h, out_w)``, and publishes both ``imucam.sample`` (the 54x42
    gray bundled with the SAME IMU) and ``frame.depth`` (the 54x42 metric depth).
    Returns ``None`` -- this function IS the ToF publisher (chain ends here).
    """
    with NUMBA_PARALLEL_LOCK:        # SGM uses numba parallel=True
        # Depth + the tracking-grid left at SOURCE resolution (exactly the
        # normal compute_depth): rectify_left=True on the live matcher
        # returns float32, so cast the gray back to uint8 (storage contract;
        # the bilinear-interp precision is already spent).
        gray_track, depth = matcher.dense_depth_rectified_left(
            msg.gray_left, msg.gray_right)
    if gray_track.dtype != np.uint8:
        gray_track = np.clip(gray_track, 0.0, 255.0).astype(np.uint8)

    # GRAY -> ToF grid by area averaging (pure NumPy, bit-exact vs INTER_AREA).
    gray_tof = _area_resize_gray(gray_track, out_h, out_w)

    # DEPTH (metres) -> ToF grid by block-median of valid pixels (NOT linear).
    depth_tof = _block_median_valid(depth, out_h, out_w)

    # imucam.sample: the 54x42 gray bundled with the SAME calibrated IMU
    # (gray_right is dropped -- a ToF returns intensity + depth, no stereo
    # pair, and nothing downstream of the ToF path consumes the right frame).
    packet_tof = ImuCamPacket(
        seq=msg.seq, ts_ns=msg.ts_ns,
        gray_left=gray_tof, gray_right=None,
        imu_ts=msg.imu_ts, gyro=msg.gyro, accel=msg.accel)
    bus.publish(topics.IMUCAM_SAMPLE, packet_tof)

    # frame.depth: the 54x42 metric depth aligned to the 54x42 gray.
    bus.publish(topics.FRAME_DEPTH,
                DepthFrame(msg.seq, msg.ts_ns, gray_tof, depth_tof))
