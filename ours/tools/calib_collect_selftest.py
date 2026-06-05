"""Self-test for the IMU calibration capture state machines (offline).

Simulates a 200 Hz IMU stream through the planted sensor distortion and drives
the collectors exactly as the live wizard would, checking:

* :class:`StaticCollector` -- ignores motion, only becomes ``ready`` after a
  clean still window; motion resets the streak.
* :class:`SixFaceCollector` -- captures one mean per distinct face, requires
  motion between captures (no double-count), rejects ambiguous (tilted)
  orientations, and the calibration it solves recovers the planted distortion on
  unseen poses.

No hardware: the IMU samples are generated from a known model so the result is
deterministic and regression-locked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.imu.accel_calib import G_STANDARD, SIX_FACES  # noqa: E402
from ours.lib.imu.calib_collect import (  # noqa: E402
    SixFaceCollector,
    StaticCollector,
    StaticCollectorConfig,
)

_M = np.array([[1.018, 0.012, -0.009],
               [-0.007, 0.982, 0.015],
               [0.010, -0.013, 1.005]])
_B = np.array([0.21, -0.15, 0.08])
_RATE = 200.0
_DT = 1.0 / _RATE


def _raw(true_force, rng, noise=0.01):
    return _M @ true_force + _B + rng.normal(0.0, noise, size=3)


def _feed_still(obj, true_force, dur_s, t0, rng, gyro_noise=0.005):
    """Feed `dur_s` of still samples at a face.

    Returns ``(t, last_status, captured)`` where ``captured`` is the face index
    captured during this window (or None). The capture fires on the first ready
    sample, after which later statuses carry ``just_captured=None``, so we scan
    every feed rather than only the last one.
    """
    t = t0
    last = None
    captured = None
    n = int(dur_s * _RATE)
    for _ in range(n):
        g = rng.normal(0.0, gyro_noise, size=3)
        a = _raw(true_force, rng)
        last = obj.feed(g, a, t)
        if getattr(last, "just_captured", None) is not None:
            captured = last.just_captured
        t += _DT
    return t, last, captured


def _feed_motion(obj, dur_s, t0, rng):
    """Feed `dur_s` of clearly-moving samples (large gyro)."""
    t = t0
    n = int(dur_s * _RATE)
    for _ in range(n):
        g = np.array([1.0, -0.8, 0.6]) + rng.normal(0, 0.1, size=3)
        a = _raw(np.array([0, 0, G_STANDARD]), rng, noise=2.0)
        obj.feed(g, a, t)
        t += _DT
    return t


def main() -> int:
    rng = np.random.default_rng(0)
    ok = True

    # --- StaticCollector: motion resets, stillness completes --------------
    sc = StaticCollector(StaticCollectorConfig(window_s=0.6, min_samples=30))
    t = 0.0
    # 0.3 s still (not enough), then motion, then 0.8 s still (enough)
    t, _, _ = _feed_still(sc, SIX_FACES[4] * G_STANDARD, 0.3, t, rng)
    ok_reset = not sc.ready
    t = _feed_motion(sc, 0.2, t, rng)
    ok_reset &= (sc.n == 0)                     # motion cleared the streak
    t, _, _ = _feed_still(sc, SIX_FACES[4] * G_STANDARD, 0.8, t, rng)
    ok_ready = sc.ready and sc.n >= 30
    am = sc.accel_mean
    ok_ready &= int(np.argmax(np.abs(am))) == 2 and am[2] > 0   # +Z up
    print(f"StaticCollector reset-on-motion: {'OK' if ok_reset else 'FAIL'}")
    print(f"StaticCollector ready+mean: {'OK' if ok_ready else 'FAIL'}")
    ok &= ok_reset and ok_ready

    # --- SixFaceCollector: full six-face run ------------------------------
    six = SixFaceCollector()
    t = 0.0
    order = [4, 5, 0, 1, 2, 3]      # +Z,-Z,+X,-X,+Y,-Y
    captured_order = []
    for fi in order:
        t = _feed_motion(six, 0.3, t, rng)         # move to the face
        t, st, cap = _feed_still(six, SIX_FACES[fi] * G_STANDARD, 1.0, t, rng)
        if cap is not None:
            captured_order.append(cap)
    ok_six = six.complete and set(captured_order) == set(range(6))
    print(f"SixFaceCollector captured all 6: {'OK' if ok_six else 'FAIL'}")
    ok &= ok_six

    # No double-count: feeding the same face again must not add a 7th.
    t, st, _ = _feed_still(six, SIX_FACES[2] * G_STANDARD, 1.0, t, rng)
    ok_nodup = len(six.captured_faces) == 6 and st.complete
    print(f"SixFaceCollector no double-count: {'OK' if ok_nodup else 'FAIL'}")
    ok &= ok_nodup

    # Solved calibration generalises to unseen tilted poses.
    cal = st.calibration
    dirs = rng.normal(size=(30, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    test_raw = np.array([_raw(d * G_STANDARD, rng, noise=0.0) for d in dirs])
    err = float(np.max(np.abs(
        np.linalg.norm(cal.apply(test_raw), axis=1) - G_STANDARD)))
    print(f"solved cal generalises (max |a|-g) = {err:.3e} m/s^2")
    ok_gen = err < 0.05
    ok &= ok_gen

    # --- ambiguous orientation rejected -----------------------------------
    six2 = SixFaceCollector()
    t = 0.0
    tilt = np.array([1.0, 1.0, 0.0]) / np.sqrt(2) * G_STANDARD   # 45 deg, no face
    t = _feed_motion(six2, 0.3, t, rng)
    t, st2, _ = _feed_still(six2, tilt, 1.0, t, rng)
    ok_amb = len(six2.captured_faces) == 0
    print(f"ambiguous tilt rejected: {'OK' if ok_amb else 'FAIL'}")
    ok &= ok_amb

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
