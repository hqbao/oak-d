#!/usr/bin/env python3
"""Offscreen Qt selftest for the tracking-lost master-warning badge.

Exercises the debounced tracking-lost overlay added to
:class:`~ui.qt.viewer3d.Viewer3D` WITHOUT entering the Qt event loop (so it runs
headless on CI / wherever OpenGL can't actually present a frame). It drives the
viewer's own ``_refresh`` tick directly -- the same method the 60 Hz timer calls
-- after pushing crafted poses into a real :class:`~ui.comms.lib.misc.pose.PoseHistory`,
and asserts the operator-facing behaviour:

* badge stays HIDDEN while tracking is OK,
* a SINGLE dropped (lost) pose does NOT flash the badge (debounce),
* the badge latches VISIBLE only after ``LOST_DEBOUNCE_POSES`` consecutive
  distinct lost poses,
* the badge HIDES immediately on the first recovery (OK) pose,
* the drone origin marker is recoloured to ``theme.BAD`` while LOST and back to
  ``theme.GOOD`` on recovery,
* the AMBER tier: a latched-lost pose with ``inertial_dr=True`` (the --tight IMU
  still dead-reckoning) shows the amber ``⚠ VISION LOST · INERTIAL DR`` badge +
  amber marker, vs the red ``⚠ TRACKING LOST`` badge + red marker when
  ``inertial_dr=False`` (no inertial fallback),
* the amber<->red switch happens LIVE while the lost state stays latched (DR
  stops -> badge goes red) without re-arming the debounce.

Run::

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m ui.tests.tracking_lost_badge_selftest
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Force the headless Qt platform BEFORE any Qt import so the test never needs a
# display / GL surface (mirrors the offscreen pattern in ui_dataflow_selftest).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.comms.lib.misc.pose import Pose, PoseHistory          # noqa: E402
from ui.qt import theme                                       # noqa: E402
from ui.qt.viewer3d import LOST_DEBOUNCE_POSES, Viewer3D       # noqa: E402


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def _push_pose(history: PoseHistory, viewer: Viewer3D, *, ok: bool,
               dr: bool = False) -> None:
    """Append one fresh pose then run a single viewer refresh tick.

    A NEW ``Pose`` object each call is what makes the debounce advance: it counts
    DISTINCT lost poses (keyed on object identity), so re-ticking the same held
    pose would not double-count -- exactly the source-idle case. ``dr`` sets the
    ``inertial_dr`` flag (TIGHT IMU still dead-reckoning) that drives the amber
    tier of the badge.
    """
    history.push(Pose(t=0.0,
                      pos_ned=np.zeros(3),
                      vel_ned=np.zeros(3),
                      quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                      tracking_ok=ok,
                      inertial_dr=dr))
    viewer._refresh()


def _badge_shown(viewer: Viewer3D) -> bool:
    """True when the badge is in its SHOWN state, regardless of ancestry.

    ``QWidget.isVisible()`` is False whenever an ANCESTOR is not visible, and
    this headless test never ``show()``-s the parent ``Viewer3D``. ``isVisibleTo``
    reports the badge's OWN show/hide flag relative to its parent -- the state
    ``_set_lost_latched`` actually toggles -- so it is the correct probe here.
    """
    return viewer._lost_badge.isVisibleTo(viewer)


def _dot_rgb(viewer: Viewer3D) -> tuple[float, float, float]:
    """The current RGB (0-1) of the drone origin marker's filled dot."""
    c = viewer._drone.dot.color
    return (float(c[0]), float(c[1]), float(c[2]))


def _expected_rgb(hexstr: str) -> tuple[float, float, float]:
    from PyQt6.QtGui import QColor
    c = QColor(hexstr)
    return (c.redF(), c.greenF(), c.blueF())


def _rgb_close(a, b, tol: float = 1e-3) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv or ["test"])
    viewer = None
    try:
        history = PoseHistory(capacity=256)
        viewer = Viewer3D(history, default_view="ISO")
        # Give the widget a concrete size so badge positioning is exercised.
        viewer.resize(800, 600)
        app.processEvents()

        badge = viewer._lost_badge

        print(f"\n  debounce threshold: LOST_DEBOUNCE_POSES = "
              f"{LOST_DEBOUNCE_POSES}")
        _check(LOST_DEBOUNCE_POSES >= 2,
               "debounce threshold > 1 so a single dropped frame cannot flash")

        # ---- (1) OK stream: badge hidden, marker green --------------------
        for _ in range(4):
            _push_pose(history, viewer, ok=True)
        app.processEvents()
        _check(not _badge_shown(viewer),
               "badge is HIDDEN while tracking is OK")
        _check(not viewer._lost_latched,
               "debounced LOST state is False while OK")
        _check(_rgb_close(_dot_rgb(viewer), _expected_rgb(theme.GOOD)),
               "drone marker is GREEN (theme.GOOD) while OK")

        # ---- (2) single dropped frame: NO flash ---------------------------
        _push_pose(history, viewer, ok=False)         # 1 lost pose only ...
        _push_pose(history, viewer, ok=True)          # ... then recovered
        app.processEvents()
        _check(not _badge_shown(viewer),
               f"a SINGLE lost pose does NOT show the badge "
               f"(needs {LOST_DEBOUNCE_POSES} consecutive)")
        _check(_rgb_close(_dot_rgb(viewer), _expected_rgb(theme.GOOD)),
               "marker stays GREEN through a single dropped frame")

        # ---- (3) sustained loss: badge latches at the threshold -----------
        # Feed exactly THRESHOLD-1 lost poses: still hidden (off-by-one guard).
        for _ in range(LOST_DEBOUNCE_POSES - 1):
            _push_pose(history, viewer, ok=False)
        app.processEvents()
        _check(not _badge_shown(viewer),
               f"badge still HIDDEN after {LOST_DEBOUNCE_POSES - 1} lost poses "
               f"(one short of threshold)")

        # The Nth consecutive lost pose latches the badge ON.
        _push_pose(history, viewer, ok=False)
        app.processEvents()
        _check(_badge_shown(viewer),
               f"badge becomes VISIBLE after {LOST_DEBOUNCE_POSES} "
               f"consecutive lost poses")
        _check(viewer._lost_latched, "debounced LOST state latched True")
        _check(badge.text() == "⚠ TRACKING LOST",
               "badge text reads '⚠ TRACKING LOST'")
        _check(_rgb_close(_dot_rgb(viewer), _expected_rgb(theme.BAD)),
               "drone marker recoloured RED (theme.BAD) while LOST")
        # Badge sits near the top-centre of the 800-wide view.
        cx = badge.x() + badge.width() / 2.0
        _check(abs(cx - 400.0) <= 2.0 and badge.y() < 40,
               f"badge pinned top-centre (cx={cx:.0f}, y={badge.y()})")

        # Holding the SAME lost pose (no new pose) must not churn -- re-tick.
        viewer._refresh()
        app.processEvents()
        _check(_badge_shown(viewer) and viewer._lost_latched,
               "badge stays latched on a held (unchanged) lost pose")

        # ---- (4) recovery: badge hides immediately ------------------------
        _push_pose(history, viewer, ok=True)
        app.processEvents()
        _check(not _badge_shown(viewer),
               "badge HIDES immediately on the first OK (recovery) pose")
        _check(not viewer._lost_latched,
               "debounced LOST state cleared on recovery")
        _check(_rgb_close(_dot_rgb(viewer), _expected_rgb(theme.GOOD)),
               "drone marker restored to GREEN on recovery")

        # ---- (5) AMBER tier: vision lost but IMU still dead-reckoning ------
        # Latch lost with inertial_dr=True every frame -> the badge must show
        # the AMBER inertial-DR tier (not the red hard-lost one), marker amber.
        for _ in range(LOST_DEBOUNCE_POSES):
            _push_pose(history, viewer, ok=False, dr=True)
        app.processEvents()
        _check(_badge_shown(viewer),
               "badge VISIBLE after sustained vision-lost+DR poses")
        _check(viewer._lost_latched, "debounced LOST state latched True (DR)")
        _check("INERTIAL DR" in badge.text(),
               f"badge text shows inertial DR tier (text={badge.text()!r})")
        _check(badge.text() == "⚠ VISION LOST · INERTIAL DR",
               "amber badge text reads '⚠ VISION LOST · INERTIAL DR'")
        _check(_rgb_close(_dot_rgb(viewer), _expected_rgb(theme.WARN)),
               "drone marker is AMBER (theme.WARN) while inertial DR")

        # ---- (6) live DR -> hard-lost flip (still latched, no debounce reset)
        # The SAME latched-lost state, now with inertial_dr=False (DR stopped):
        # the badge must flip to the RED hard-lost tier IMMEDIATELY -- a
        # presentation change, not a debounce re-arm.
        _push_pose(history, viewer, ok=False, dr=False)
        app.processEvents()
        _check(_badge_shown(viewer) and viewer._lost_latched,
               "badge stays latched through the DR->hard-lost flip")
        _check(badge.text() == "⚠ TRACKING LOST",
               "badge flips to red '⚠ TRACKING LOST' when DR stops")
        _check(_rgb_close(_dot_rgb(viewer), _expected_rgb(theme.BAD)),
               "drone marker flips to RED (theme.BAD) when DR stops")

        # ---- (7) hard-lost first (no DR ever): RED from the start ----------
        _push_pose(history, viewer, ok=True)          # recover -> reset latch
        app.processEvents()
        _check(not _badge_shown(viewer), "badge hidden after recovery (pre-7)")
        for _ in range(LOST_DEBOUNCE_POSES):
            _push_pose(history, viewer, ok=False, dr=False)
        app.processEvents()
        _check(_badge_shown(viewer), "badge VISIBLE after sustained hard-lost")
        _check(badge.text() == "⚠ TRACKING LOST",
               "hard-lost (no DR) shows the red '⚠ TRACKING LOST' badge")
        _check(_rgb_close(_dot_rgb(viewer), _expected_rgb(theme.BAD)),
               "drone marker RED for hard-lost (no inertial fallback)")

        # ---- (8) red -> amber flip while latched (DR resumes) --------------
        _push_pose(history, viewer, ok=False, dr=True)
        app.processEvents()
        _check(_badge_shown(viewer) and viewer._lost_latched,
               "badge stays latched through the hard-lost->DR flip")
        _check(badge.text() == "⚠ VISION LOST · INERTIAL DR",
               "badge flips back to amber when DR resumes")
        _check(_rgb_close(_dot_rgb(viewer), _expected_rgb(theme.WARN)),
               "drone marker flips back to AMBER when DR resumes")

        # ---- (9) recovery from amber clears everything --------------------
        _push_pose(history, viewer, ok=True)
        app.processEvents()
        _check(not _badge_shown(viewer),
               "badge HIDES on recovery from the amber tier")
        _check(_rgb_close(_dot_rgb(viewer), _expected_rgb(theme.GOOD)),
               "drone marker restored to GREEN on recovery from DR")

        print("\n  ALL TRACKING-LOST BADGE CHECKS PASSED")
        return 0
    finally:
        del viewer
        # Don't quit -- the QApplication singleton may be re-used.


if __name__ == "__main__":
    raise SystemExit(main())
