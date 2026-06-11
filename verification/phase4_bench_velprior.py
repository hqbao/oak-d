#!/usr/bin/env python3
"""Phase-4 A/B benchmark: loose-vs-tight ATE with the velocity-stabilisation
priors flipped ON, WITHOUT editing the frozen ``loose_vs_tight_bench`` harness.

It reuses ``loose_vs_tight_bench.run_session`` verbatim (same per-frame loop,
same scoring, same 54x42 reduction) and only flips the new VioConfig flags by
monkeypatching the ``WindowedVIOConfig`` *class default* for ``stabilize_velocity``
for the duration of a run -- so ``run_ba`` flips ``vel_cv_prior``/``vel_zupt`` on
via ``dataclasses.replace`` exactly as the live ``--tight`` knob would. The
harness behaviour (and the byte-parity oracle, which never sets the flag) is
untouched.

Modes per session/res: OFF (baseline), CV (cv prior only), CVZ (cv + zupt).

Run::

    .venv/bin/python verification/phase4_bench_velprior.py \
        --only push_shake_20s push_straight_fast_15s push_straight_15s lab_loop_30s
"""
from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vio.mathlib.backend.vio_window as vw  # noqa: E402
from verification.loose_vs_tight_bench import (  # noqa: E402
    GOLD_DIR,
    run_session,
)


@contextmanager
def _vio_flags(*, cv: bool, zupt: bool, sigma_vel_cv: float,
               sigma_vel_zupt: float):
    """Temporarily make ``run_ba``'s nested VioConfig carry the velocity priors.

    ``run_session`` builds ``WindowedVIOConfig()`` then ``replace(.vio,
    imu_info_weight=...)``. We patch ``run_ba`` to additionally ``replace`` the
    flags onto ``vio_cfg`` before the solve -- equivalent to the live
    ``stabilize_velocity`` knob but with per-mode sigmas. Restored on exit.
    """
    orig_run_ba = vw.WindowedVIOMap.run_ba

    def patched(self):
        # Flip the flags on the instance's nested VioConfig for this map only.
        self.cfg.vio = replace(
            self.cfg.vio, vel_cv_prior=cv, vel_zupt=zupt,
            sigma_vel_cv=sigma_vel_cv, sigma_vel_zupt=sigma_vel_zupt)
        return orig_run_ba(self)

    vw.WindowedVIOMap.run_ba = patched
    try:
        yield
    finally:
        vw.WindowedVIOMap.run_ba = orig_run_ba


def _run(session_dir, res, mbv, *, cv, zupt, scv, szu):
    with _vio_flags(cv=cv, zupt=zupt, sigma_vel_cv=scv, sigma_vel_zupt=szu):
        return run_session(session_dir, backend="vio", resolution=res,
                           imu_info_weight=True, min_ba_views=mbv)


def _fmt(m):
    if m is None:
        return f"{'--':>8s} {'--':>7s} {'--':>8s} {'--':>8s}"
    return (f"{m['ate_cm']:>8.2f} {m['scale']:>7.3f} "
            f"{m['max_step_cm']:>8.1f} {m['phantom_cm']:>8.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[
        "push_shake_20s", "push_straight_fast_15s", "push_straight_15s",
        "lab_loop_30s"])
    ap.add_argument("--tof-min-ba-views", type=int, default=1)
    ap.add_argument("--sigma-vel-cv", type=float, default=0.15)
    ap.add_argument("--sigma-vel-zupt", type=float, default=0.5)
    args = ap.parse_args()

    print("Phase-4 velocity-prior A/B (TIGHT, imu_info_weight=True)")
    print(f"  sigma_vel_cv={args.sigma_vel_cv}  "
          f"sigma_vel_zupt={args.sigma_vel_zupt}")
    print("  columns per mode: ATE(cm) scale maxstep(cm) phantom(cm)")
    hdr = (f"{'session':24s} {'res':6s} {'mode':5s} "
           f"{'ATE':>8s} {'scale':>7s} {'maxstep':>8s} {'phantom':>8s}")
    print("=" * len(hdr))

    for name in args.only:
        sd = GOLD_DIR / name
        if not sd.exists():
            print(f"!! missing {name}")
            continue
        print(hdr)
        print("-" * len(hdr))
        for res in ("full", "tof54"):
            mbv = args.tof_min_ba_views if res == "tof54" else None
            off = _run(sd, res, mbv, cv=False, zupt=False,
                       scv=args.sigma_vel_cv, szu=args.sigma_vel_zupt)
            cvm = _run(sd, res, mbv, cv=True, zupt=False,
                       scv=args.sigma_vel_cv, szu=args.sigma_vel_zupt)
            cvz = _run(sd, res, mbv, cv=True, zupt=True,
                       scv=args.sigma_vel_cv, szu=args.sigma_vel_zupt)
            for mode, m in (("OFF", off), ("CV", cvm), ("CVZ", cvz)):
                print(f"{name:24s} {res:6s} {mode:5s} {_fmt(m)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
