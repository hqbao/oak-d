#!/usr/bin/env python3
"""Regression guard for BUG C: ``imu_moving`` propagates IMU -> PnP freeze gate.

Background
----------
``OdometryConfig.min_inliers_for_translation`` freezes translation when PnP
returns few inliers (textureless wall). BUG C: a motion-blurred shake ALSO
starves PnP of inliers, but there the camera IS moving -- freezing pins the
marker through real motion (the "move + shake -> ours-ba/slam freezes in place"
symptom).

The fix introduced :attr:`ours.lib.flow.messages.ImuPrior.imu_moving`: a loose
"definitely moving" gate (gyro > 0.3 rad/s OR |accel| - g > 0.5 m/s^2)
computed in :class:`~ours.flows.odometry.preintegrate_prior.PreintegratePrior`
and threaded all the way down to
:meth:`~ours.lib.odometry.odometry.RGBDVisualOdometry.estimate` via
:class:`~ours.flows.odometry.estimate_motion.EstimateMotion`. When
``imu_moving=True`` the freeze is vetoed (vision keeps its translation guess);
when ``imu_moving=False`` the freeze stays as before.

This test guards both halves of the pipeline:

1. **PreintegratePrior unit**: feed two synthetic
   :class:`~ours.lib.flow.messages.ImuCamPacket` (one with gyro = 1 rad/s, one
   with gyro ~ 0 + |accel| ~ g) into the actual task, and assert the resulting
   :class:`ImuPrior.imu_moving` flags are ``True`` and ``False`` respectively.
   Boundary samples (just under / just over the 0.3 rad/s threshold) round out
   the gate proof.

2. **End-to-end propagation**: build the real
   :class:`~ours.flows.odometry.OdometryFlow` on a local Bus, spy on
   :meth:`RGBDVisualOdometry.estimate` to record its ``imu_moving`` kwarg, then
   publish ``imucam.sample`` + ``frame.depth`` pairs and assert the spy saw the
   right value (``True`` for the shake packet, ``False`` for the still packet).

Run::

    python -m ours.tools.imu_moving_propagation_selftest
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.flows.odometry.odometry_flow import OdometryFlow           # noqa: E402
from ours.flows.odometry.preintegrate_prior import (                 # noqa: E402
    PreintegratePrior, _MOVING_GYRO, _MOVING_ACCEL_DEV, _GRAVITY,
)
from ours.lib.flow import Bus, topics                                # noqa: E402
from ours.lib.flow.flow import FlowContext                           # noqa: E402
from ours.lib.flow.messages import DepthFrame, ImuCamPacket, ImuPrior, END  # noqa: E402
from ours.lib.odometry import odometry as odo_mod                    # noqa: E402

H, W = 64, 96
FX = FY = 80.0
CX, CY = W / 2.0, H / 2.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
R_IMU_CAM = np.eye(3, dtype=np.float64)


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
def _make_packet(seq: int, *, gyro_mag: float,
                 accel_mag: float = _GRAVITY) -> ImuCamPacket:
    """Build a synthetic ImuCamPacket whose IMU samples have the requested
    gyro magnitude and accelerometer magnitude (= |accel|).

    gyro is pure +X rotation; accel is pure +Z (camera-frame gravity).
    """
    n = 5
    imu_ts = np.linspace(0, 1e8, n, dtype=np.int64)         # 0..100 ms in ns
    gyro = np.zeros((n, 3), dtype=np.float64)
    gyro[:, 0] = gyro_mag
    accel = np.zeros((n, 3), dtype=np.float64)
    accel[:, 2] = accel_mag
    return ImuCamPacket(
        seq=seq, ts_ns=int(imu_ts[-1]),
        gray_left=np.zeros((H, W), dtype=np.uint8),
        gray_right=None,
        imu_ts=imu_ts, gyro=gyro, accel=accel,
    )


def _make_depth(seq: int, *, ts_ns: int) -> DepthFrame:
    return DepthFrame(
        seq=seq, ts_ns=ts_ns,
        gray_left=np.zeros((H, W), dtype=np.uint8),
        depth_m=np.full((H, W), 2.0, dtype=np.float32),
    )


# --------------------------------------------------------------------------- #
# Part 1. PreintegratePrior unit-level gate
# --------------------------------------------------------------------------- #
def _run_preint(packet: ImuCamPacket) -> ImuPrior:
    """Run the real PreintegratePrior task on a packet, return its ImuPrior."""
    bus = Bus()
    ctx = FlowContext(bus, "preint-test")
    ctx.state["R_imu_cam"] = R_IMU_CAM
    ctx.state["use_gyro"] = True
    ctx.state["priors"] = {}
    PreintegratePrior().run(ctx, packet)
    return ctx.state["priors"][packet.seq]


def test_preintegrate_gate() -> None:
    print(" preintegrate_prior gate (synthetic IMU)")

    # Above threshold -> imu_moving=True (clear shake).
    p = _run_preint(_make_packet(seq=1, gyro_mag=1.0))
    _check(p.imu_moving is True,
           f"gyro=1.0 rad/s (>> {_MOVING_GYRO}) -> imu_moving=True "
           f"(got: {p.imu_moving})")

    # At rest -> imu_moving=False (still + ~1 g).
    p = _run_preint(_make_packet(seq=2, gyro_mag=0.0, accel_mag=_GRAVITY))
    _check(p.imu_moving is False,
           f"gyro=0, |accel|=g -> imu_moving=False (got: {p.imu_moving})")

    # Just BELOW the gyro gate -> still not "definitely moving".
    p = _run_preint(_make_packet(seq=3, gyro_mag=_MOVING_GYRO * 0.5))
    _check(p.imu_moving is False,
           f"gyro=0.5*{_MOVING_GYRO} -> imu_moving=False (got: {p.imu_moving})")

    # Just ABOVE the gyro gate -> moving.
    p = _run_preint(_make_packet(seq=4, gyro_mag=_MOVING_GYRO * 1.5))
    _check(p.imu_moving is True,
           f"gyro=1.5*{_MOVING_GYRO} -> imu_moving=True (got: {p.imu_moving})")

    # Big linear accel (no rotation) crosses the accel gate.
    p = _run_preint(_make_packet(seq=5, gyro_mag=0.0,
                                 accel_mag=_GRAVITY + _MOVING_ACCEL_DEV * 2))
    _check(p.imu_moving is True,
           f"|accel|-g = {2 * _MOVING_ACCEL_DEV} > {_MOVING_ACCEL_DEV} -> "
           f"imu_moving=True (got: {p.imu_moving})")


# --------------------------------------------------------------------------- #
# Part 2. End-to-end: imu_moving flows from PreintegratePrior into vo.estimate
# --------------------------------------------------------------------------- #
def test_end_to_end_propagation() -> None:
    print(" end-to-end imu_moving -> vo.estimate kwarg (real OdometryFlow)")

    captured: list[dict] = []
    captured_lock = threading.Lock()

    real_estimate = odo_mod.RGBDVisualOdometry.estimate

    def spy_estimate(self, cur_obs, depth_m, R_prior=None, dt_s=None,
                     imu_moving=False):
        with captured_lock:
            captured.append({"seq_obs_n": len(cur_obs),
                             "R_prior_is_none": R_prior is None,
                             "imu_moving": bool(imu_moving)})
        # Don't actually solve -- we just want to verify the kwarg arrival.
        # Returning the current pose preserves the flow's downstream contract
        # (PublishPose / EmitKeyframe still receive a valid pose).
        return self.pose

    odo_mod.RGBDVisualOdometry.estimate = spy_estimate
    try:
        bus = Bus()
        flow = OdometryFlow(bus, K=K, R_imu_cam=R_IMU_CAM,
                            kf_every=10, use_gyro=True, latest_only=False)
        flow.expected_ends = 2

        # Track outputs so we can assert at least one frame went through.
        odoms: list[int] = []
        bus.subscribe(topics.POSE_ODOM,
                      lambda m: odoms.append(m.seq) if m is not END else None)

        flow.start()
        try:
            # Frame 0: IMU at rest -> expect imu_moving=False at vo.estimate.
            p0 = _make_packet(seq=0, gyro_mag=0.0, accel_mag=_GRAVITY)
            bus.publish(topics.IMUCAM_SAMPLE, p0)
            bus.publish(topics.FRAME_DEPTH, _make_depth(0, ts_ns=int(p0.ts_ns)))

            # Frame 1: clear shake -> expect imu_moving=True at vo.estimate.
            p1 = _make_packet(seq=1, gyro_mag=1.0)
            bus.publish(topics.IMUCAM_SAMPLE, p1)
            bus.publish(topics.FRAME_DEPTH, _make_depth(1, ts_ns=int(p1.ts_ns)))

            # Frame 2: at rest again -> expect imu_moving=False.
            p2 = _make_packet(seq=2, gyro_mag=0.05, accel_mag=_GRAVITY)
            bus.publish(topics.IMUCAM_SAMPLE, p2)
            bus.publish(topics.FRAME_DEPTH, _make_depth(2, ts_ns=int(p2.ts_ns)))

            # END so the flow drains.
            bus.publish(topics.IMUCAM_SAMPLE, END)
            bus.publish(topics.FRAME_DEPTH, END)

            ok = flow.done.wait(timeout=10.0)
            _check(ok, f"OdometryFlow drained END within 10 s (done={ok})")
        finally:
            flow.stop()

        # The spy must have seen the same number of estimate() calls as frames.
        _check(len(captured) == 3,
               f"vo.estimate called once per frame (got: {len(captured)} calls)")

        # Per-frame propagation. Note: the FIRST frame has no preintegrated
        # prior to pop (PullPrior may set prior=None on frame 0 depending on
        # ordering), so the EstimateMotion task may pass imu_moving=False
        # there. We focus on the discriminating frame (frame 1, shake).
        seen = [c["imu_moving"] for c in captured]
        print(f"  [info] imu_moving sequence at vo.estimate: {seen}")

        _check(any(c["imu_moving"] is True for c in captured),
               "at least ONE frame propagated imu_moving=True end-to-end "
               f"(seen: {seen}) -- the shake packet (frame 1, gyro=1.0 rad/s) "
               "must veto the freeze")

        _check(any(c["imu_moving"] is False for c in captured),
               "at least ONE frame propagated imu_moving=False end-to-end "
               f"(seen: {seen}) -- the still packets must NOT veto the freeze")

        # And specifically: the shake frame's vo.estimate call must be True.
        # The chain is FIFO so the seq order is the same as the call order
        # (modulo possibly missing frame 0 if its prior was not joined).
        # We assert the SHAKE frame's call carried True.
        # Find the call whose index matches the shake (frame index 1 of 3).
        if len(captured) >= 2:
            _check(captured[1]["imu_moving"] is True,
                   f"the 2nd vo.estimate call (= shake frame 1) had "
                   f"imu_moving=True (got: {captured[1]['imu_moving']})")
        if len(captured) >= 3:
            _check(captured[2]["imu_moving"] is False,
                   f"the 3rd vo.estimate call (= still frame 2) had "
                   f"imu_moving=False (got: {captured[2]['imu_moving']})")

        _check(len(odoms) == 3,
               f"flow emitted pose.odom for each frame (got: {len(odoms)}/3)")
    finally:
        odo_mod.RGBDVisualOdometry.estimate = real_estimate


# --------------------------------------------------------------------------- #
def main() -> int:
    print("imu_moving_propagation_selftest")
    test_preintegrate_gate()
    test_end_to_end_propagation()
    print("\nALL IMU_MOVING PROPAGATION SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
