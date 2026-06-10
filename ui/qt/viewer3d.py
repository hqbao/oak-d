"""3D scene built with pyqtgraph's OpenGL viewport.

Renders, in ENU display coordinates (positions converted from internal NED),
the FIVE trajectory streams of the proc4 single-view UI plus the live marker:
  * ground grid + axis triad at world origin
  * live drone position as a triad (forward=red, right=green, down=cyan reversed to up)
  * VO polyline (dim grey): the pure-vision ``pose.vo`` trail (drifts most)
  * VIO trajectory polyline (NVG green): the live VIO ``pose.odom`` trail + marker
  * VIO-BA polyline (violet-blue): the windowed-BA ``pose.refined`` keyframe path
  * SLAM-corrected-VIO polyline (warm orange, with teleport vertices flashed in
    red): the dense VIO trail rubber-sheeted by SLAM's loop corrections
  * SLAM keyframe polyline (HUD cyan) + amber keyframe dots: the SLAM map
Each line has a visibility setter so the single-view's 5 toggle buttons can
show/hide it independently.

On top of the trajectory streams the view carries an operator-grade
TRACKING-LOST master-warning overlay driven solely by the abstract
``pose.tracking_ok`` / ``pose.inertial_dr`` flags (kept generic / multi-chip --
no chip specifics): a debounced badge pinned top-centre (see
``LOST_DEBOUNCE_POSES``) plus a recolour of the live drone origin marker, so a
tracking loss is impossible to miss on the big 3D view, not only in the side
TelemetryPanel's small OK/DR/LOST readout. The badge is two-tier: while latched
lost it shows AMBER ``⚠ VISION LOST · INERTIAL DR`` when the (``--tight``) IMU is
still dead-reckoning a valid pose (``pose.inertial_dr``), and RED ``⚠ TRACKING
LOST`` when there is no inertial fallback (loose path frozen). The amber/red
choice is a per-frame PRESENTATION of the single latched-lost state, so it can
switch live (e.g. DR stops -> goes red) without re-arming the debounce.
"""
from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6 import QtCore, QtGui
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QLabel

from ui.comms.lib.misc import frames
from ui.comms.lib.misc.pose import Pose, PoseHistory
from . import theme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qcolor(hexstr: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    c = QColor(hexstr)
    return (c.redF(), c.greenF(), c.blueF(), alpha)


def _make_grid(size_m: float = 20.0, step_m: float = 1.0,
               color=_qcolor(theme.GRID, 0.55)) -> gl.GLGridItem:
    g = gl.GLGridItem()
    g.setSize(size_m, size_m, 0)
    g.setSpacing(step_m, step_m, 0)
    g.setColor(QColor(theme.GRID))
    return g


def _make_world_axes(length: float = 1.5) -> list[gl.GLLinePlotItem]:
    """Origin axes shown in ENU: E (red), N (green), U (cyan)."""
    items: list[gl.GLLinePlotItem] = []
    specs = [
        ((1, 0, 0), theme.AXIS_E),   # East (X in scene)
        ((0, 1, 0), theme.AXIS_N),   # North (Y in scene)
        ((0, 0, 1), theme.AXIS_U),   # Up (Z in scene)
    ]
    for direction, hexc in specs:
        pts = np.array([(0, 0, 0), tuple(length * d for d in direction)],
                       dtype=np.float32)
        line = gl.GLLinePlotItem(
            pos=pts, color=_qcolor(hexc, 1.0), width=2.5, antialias=True,
        )
        items.append(line)
    return items


# ---------------------------------------------------------------------------
# Drone marker — body axes triad
# ---------------------------------------------------------------------------

class _DroneTriad:
    """Three colored line segments that follow the drone's pose.

    Body axes (FRD) are drawn in scene coordinates after NED->ENU rotation:
      Forward (red), Right (green), Down (cyan, but rendered as -Z for clarity).
    """

    def __init__(self, length: float = 0.6) -> None:
        self.length = float(length)
        # initialise at origin pointing along world axes
        zero = np.zeros((2, 3), dtype=np.float32)
        self.fwd = gl.GLLinePlotItem(pos=zero, color=_qcolor(theme.AXIS_N), width=3.0,
                                     antialias=True)
        self.right = gl.GLLinePlotItem(pos=zero, color=_qcolor(theme.AXIS_E), width=3.0,
                                       antialias=True)
        self.down = gl.GLLinePlotItem(pos=zero, color=_qcolor(theme.AXIS_U), width=3.0,
                                      antialias=True)
        # small filled marker at the body origin
        self.dot = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=_qcolor(theme.GOOD, 1.0),
            size=12.0,
            pxMode=True,
        )
        # Tracking-status recolour state. Cached so the dot colour is pushed
        # only on an actual status transition, never on every per-frame pose
        # update -- ``set_status`` is the single owner of the dot's colour.
        self._status = "ok"

    def items(self) -> list:
        return [self.fwd, self.right, self.down, self.dot]

    # Tracking status -> origin-marker dot colour. Mirrors the badge tiers:
    # OK = green, inertial dead-reckoning = amber, hard-lost = red.
    _STATUS_COLOR = {"ok": theme.GOOD, "dr": theme.WARN, "lost": theme.BAD}

    def set_status(self, status: str) -> None:
        """Recolour the origin marker for the tracking status (3-state).

        ``status`` is one of ``"ok"`` (green ``theme.GOOD``), ``"dr"`` (amber
        ``theme.WARN`` -- vision lost, IMU dead-reckoning) or ``"lost"`` (red
        ``theme.BAD`` -- no inertial fallback). Only touches ``setData`` on an
        actual status transition so a held state costs nothing (no per-frame
        ``setData`` churn). The origin dot is the operator's primary eye-target,
        so recolouring it makes the loss catch the eye on the marker itself,
        mirroring the overlay badge.
        """
        if status == self._status:
            return
        self._status = status
        self.dot.setData(color=_qcolor(self._STATUS_COLOR[status], 1.0))

    def update(self, pose: Pose) -> None:
        # body origin in scene coords (ENU)
        p_enu = frames.ned_to_enu(pose.pos_ned).astype(np.float32)
        # rotate body axes into ENU
        R_enu = frames.rot_ned_to_enu(pose.R).astype(np.float32)
        x_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        z_b = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        fwd_dir = R_enu @ x_b
        right_dir = R_enu @ y_b
        down_dir = R_enu @ z_b

        L = self.length
        self.fwd.setData(pos=np.stack([p_enu, p_enu + L * fwd_dir]))
        self.right.setData(pos=np.stack([p_enu, p_enu + L * right_dir]))
        # down_dir already points downward in the scene (-Z when level)
        self.down.setData(pos=np.stack([p_enu, p_enu + L * down_dir]))
        self.dot.setData(pos=p_enu.reshape(1, 3))


# ---------------------------------------------------------------------------
# Main 3D viewport widget
# ---------------------------------------------------------------------------

# Tracking-lost debounce: number of CONSECUTIVE distinct lost poses before the
# master-warning badge latches on. VIO emits poses at the source rate (replay /
# live ~15-30 Hz), so ~5 lost poses is roughly 0.2-0.35 s of continuous loss --
# inside the 0.25-0.4 s target. We count distinct lost poses (not raw 60 Hz
# refresh ticks) because ``history.snapshot()`` returns the SAME ``latest`` pose
# between source updates; counting refreshes would falsely accumulate while the
# source is merely idle. A single dropped frame (1 lost pose, then OK) never
# reaches the threshold, so it cannot flash. Recovery clears instantly on the
# first OK pose -- no lingering nuisance.
LOST_DEBOUNCE_POSES = 5

# (azimuth_deg, elevation_deg, distance_m)  -- pyqtgraph GLViewWidget conventions
VIEW_PRESETS: dict[str, tuple[float, float, float]] = {
    "ISO":    (45.0,  28.0, 14.0),
    "TOP":    (-90.0,  89.9, 14.0),
    "FRONT":  (-90.0,   0.0, 14.0),
    "BACK":   ( 90.0,   0.0, 14.0),
    "LEFT":   (180.0,   0.0, 14.0),
    "RIGHT":  (  0.0,   0.0, 14.0),
}


class Viewer3D(gl.GLViewWidget):
    """OpenGL viewport with grid, axes, trajectory line and drone triad."""

    def __init__(self, history: PoseHistory, parent=None,
                 default_view: str = "ISO") -> None:
        super().__init__(parent)
        self.history = history
        self.setBackgroundColor(QColor(theme.BG))

        # ---- static scene -------------------------------------------------
        self.addItem(_make_grid(size_m=20.0, step_m=1.0))
        for ax in _make_world_axes(length=1.5):
            self.addItem(ax)

        # ---- VO polyline (optional) --------------------------------------
        # The PURE-VISION ``pose.vo`` trail (dim desaturated grey), drawn FIRST so
        # it sits BEHIND every brighter line — it is the rawest, most-drifting
        # stream, so it reads as the noisy baseline. Populated only when the
        # source exposes a VO snapshot (set via ``set_vo_path_source``); a
        # harmless empty line otherwise.
        self._vo_getter: Callable[[], np.ndarray] | None = None
        self._vo = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=_qcolor(theme.VO_PATH, 0.80),
            width=2.0,
            antialias=True,
            mode="line_strip",
        )
        self.addItem(self._vo)

        # ---- trajectory polyline -----------------------------------------
        self._traj = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=_qcolor(theme.TRACE_PATH, 0.95),
            width=2.0,
            antialias=True,
            mode="line_strip",
        )
        self.addItem(self._traj)

        # ---- VIO-BA polyline (optional) ----------------------------------
        # The windowed-BA ``pose.refined`` keyframe trail (violet-blue): the
        # sparse BA-optimised poses VIO emits per keyframe, distinct from the
        # dense green f2f trail. Populated only when the source exposes a BA
        # snapshot (set via ``set_ba_path_source``); a harmless empty line
        # otherwise.
        self._ba_getter: Callable[[], np.ndarray] | None = None
        self._ba = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=_qcolor(theme.BA_PATH, 0.85),
            width=2.0,
            antialias=True,
            mode="line_strip",
        )
        self.addItem(self._ba)

        # ---- refined-map polyline (optional) -----------------------------
        # The BA/SLAM-refined keyframe trajectory, drawn in HUD cyan BEHIND the
        # green live path so ours-ba/ours-slam visibly show the corrected map the
        # heavy optimiser produced while the marker stays the responsive f2f tip.
        # Populated only when the source exposes ``refined_path_snapshot`` (set via
        # ``set_refined_path_source``); a harmless empty line otherwise. Fed REAL
        # refined poses (BA window kf positions / SLAM corrected kf poses).
        self._refined_getter: Callable[[], np.ndarray] | None = None
        self._refined = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=_qcolor(theme.REFINED_PATH, 0.85),
            width=2.0,
            antialias=True,
            mode="line_strip",
        )
        self.addItem(self._refined)

        # ---- SLAM-corrected-VIO polyline (optional) ----------------------
        # The DENSE VIO trail rubber-sheeted by SLAM's loop corrections (warm
        # orange), drawn BEHIND the marker like the refined line. Populated only
        # when the source exposes the corrected snapshot (set via
        # ``set_corrected_path_source``); a harmless empty line otherwise. This
        # is the per-frame VIO path DEFORMED so its keyframe anchors land on the
        # loop-corrected SLAM keyframe positions -- not the sparse keyframe line.
        # The getter returns ``(positions, teleport_flags)``; vertices SLAM
        # pulled far (|delta| > TELEPORT_M) are recoloured red per-vertex,
        # MIRRORING the green ``_traj`` teleport recolour (see ``_refresh``).
        self._corrected_getter: \
            Callable[[], tuple[np.ndarray, np.ndarray]] | None = None
        self._corrected = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=_qcolor(theme.CORRECTED_PATH, 0.85),
            width=2.0,
            antialias=True,
            mode="line_strip",
        )
        self.addItem(self._corrected)

        # ---- drone triad --------------------------------------------------
        self._drone = _DroneTriad(length=0.6)
        for it in self._drone.items():
            self.addItem(it)

        # ---- SLAM keyframe overlay (optional) ----------------------------
        # Populated only when the source exposes a SLAM overlay (set via
        # ``set_overlay_source``). Three layers, all fed REAL SlamMap outputs:
        #   * amber dots  -> every keyframe position
        #   * red dots    -> matched (revisited) keyframes, drawn bigger on top
        #   * magenta line-> loop-closure (teleport) links cur<->old, in a
        #                    distinct colour so they read as map corrections,
        #                    NOT part of the VIO trajectory.
        # ---- SLAM keyframe overlay (optional) ----------------------------
        # Populated only when the source exposes a SLAM overlay (set via
        # ``set_overlay_source``). Two layers, both fed REAL SlamMap outputs:
        #   * amber dots -> every keyframe position (the persistent map)
        #   * red dot    -> the keyframe just revisited; blink-fades on each new
        #                   loop closure to show WHERE the pose snapped back to.
        # The teleport MOTION itself is drawn by recolouring the trajectory
        # segment magenta (see _refresh), so no separate link line is needed.
        self._overlay_getter: Callable[[], tuple] | None = None
        self._slam_kf = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=_qcolor(theme.WARN, 0.85), size=8.0, pxMode=True,
        )
        self._slam_kf_match = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=_qcolor(theme.BAD, 1.0), size=16.0, pxMode=True,
        )
        self.addItem(self._slam_kf)
        self.addItem(self._slam_kf_match)
        # Flash state: the matched dot blink-fades when a NEW loop closes
        # (flash_id changes), then clears — so the highlight tracks the latest
        # revisit instead of piling up the whole loop history.
        self._flash_id = 0
        self._flash_t0 = 0.0
        self._flash_match = np.zeros((0, 3), dtype=np.float32)

        # ---- per-line "head" dots ----------------------------------------
        # One GLScatterPlotItem per line marking that line's NEWEST vertex, in
        # the SAME colour as the line, so it's obvious which lines are live and
        # where each one's leading data is. Added LAST so they draw ON TOP of
        # every line. Each head follows its line's visibility toggle (see the
        # set_*_visible setters) and is set to the last vertex of its line on
        # each refresh (empty -> nothing drawn). The cyan SLAM head marks the
        # LEADING keyframe -- distinct from the amber all-keyframe dots above.
        _HEAD_SIZE = 11.0
        self._vo_head = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=_qcolor(theme.VO_PATH, 0.95), size=_HEAD_SIZE, pxMode=True)
        self._vio_head = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=_qcolor(theme.TRACE_PATH, 0.95), size=_HEAD_SIZE, pxMode=True)
        self._ba_head = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=_qcolor(theme.BA_PATH, 0.95), size=_HEAD_SIZE, pxMode=True)
        self._corrected_head = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=_qcolor(theme.CORRECTED_PATH, 0.95), size=_HEAD_SIZE,
            pxMode=True)
        self._slam_head = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=_qcolor(theme.REFINED_PATH, 0.95), size=_HEAD_SIZE,
            pxMode=True)
        for _head in (self._vo_head, self._vio_head, self._ba_head,
                      self._corrected_head, self._slam_head):
            self.addItem(_head)

        # ---- follow-cam state --------------------------------------------
        self._follow = False
        self.set_view(default_view if default_view in VIEW_PRESETS else "ISO")

        # ---- tracking-lost master-warning badge --------------------------
        # A child QLabel parented to THIS widget (not a GL scene item): the
        # GL viewport is a QOpenGLWidget, and compositing a plain QWidget over
        # it -- then ``raise_()``-ing it above the GL surface -- is the simplest
        # robust overlay (no GL text rendering, legible over any scene). It is
        # repositioned in ``resizeEvent`` and hidden whenever tracking is OK.
        #
        # Placement: TOP-CENTRE -- the conventional master-caution location in a
        # cockpit/MFD layout, where the operator's eye returns between glances.
        # ``theme.BAD`` (#ff3b30) red text on a semi-transparent dark backing so
        # it stays legible over bright or dark scene geometry; bold + large so a
        # tracking loss is impossible to miss.
        self._lost_badge = QLabel("⚠ TRACKING LOST", self)
        self._lost_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._lost_badge.setStyleSheet(self._badge_qss(theme.BAD))
        self._lost_badge.adjustSize()
        self._lost_badge.hide()

        # Debounced tracking-lost state (see LOST_DEBOUNCE_POSES). ``_lost_count``
        # counts CONSECUTIVE distinct lost poses; ``_last_pose_id`` is the id of
        # the pose object last counted, so a held (unchanged) ``latest`` between
        # source updates is not double-counted. ``_lost_latched`` is the public
        # debounced state the badge + marker follow. ``_shown_sev`` is the badge's
        # currently-DISPLAYED severity (None = hidden / "dr" = amber / "lost" =
        # red), tracked so ``setText`` / ``setStyleSheet`` / the marker recolour
        # fire only on an actual severity change -- the amber<->red switch while
        # latched is a presentation flip, not a debounce re-arm.
        self._lost_count = 0
        self._last_pose_id: int | None = None
        self._lost_latched = False
        self._shown_sev: str | None = None

        # ---- refresh timer (UI 60 Hz, decoupled from source rate) --------
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ---- public API ------------------------------------------------------

    def set_view(self, name: str) -> None:
        if name not in VIEW_PRESETS:
            return
        az, el, dist = VIEW_PRESETS[name]
        self.setCameraPosition(azimuth=az, elevation=el, distance=dist)
        self.opts["center"] = QtGui.QVector3D(0, 0, 0)
        self.update()

    def set_follow(self, on: bool) -> None:
        self._follow = bool(on)

    def set_overlay_source(self, getter: Callable[[], tuple]) -> None:
        """Register a callable returning the live SLAM overlay.

        ``getter()`` must return ``(kf_ned, match_ned, loop_segs)`` (positions
        in NED): keyframe dots, matched keyframes, and ``[cur, old]`` loop
        segments. Polled each refresh; pass ``None`` to disable.
        """
        self._overlay_getter = getter

    def set_vo_path_source(self, getter: Callable[[], np.ndarray]) -> None:
        """Register a callable returning the pure-vision VO trajectory.

        ``getter()`` returns an ``(N, 3)`` array of ``pose.vo`` positions in NED
        (the rawest, most-drifting stream). Polled each refresh and drawn as the
        dim-grey VO line behind every other line; pass ``None`` to disable.
        """
        self._vo_getter = getter

    def set_ba_path_source(self, getter: Callable[[], np.ndarray]) -> None:
        """Register a callable returning the windowed-BA (``pose.refined``) path.

        ``getter()`` returns an ``(N, 3)`` array of BA-keyframe positions in NED.
        Polled each refresh and drawn as the violet-blue VIO-BA line; pass
        ``None`` to disable.
        """
        self._ba_getter = getter

    def set_refined_path_source(self, getter: Callable[[], np.ndarray]) -> None:
        """Register a callable returning the SLAM keyframe trajectory.

        ``getter()`` returns an ``(N, 3)`` array of SLAM keyframe positions in
        NED (the corrected keyframe path). Polled each refresh and drawn as the
        cyan SLAM-map line behind the live green path; pass ``None`` to disable.
        """
        self._refined_getter = getter

    def set_corrected_path_source(
            self, getter: Callable[[], tuple[np.ndarray, np.ndarray]]) -> None:
        """Register a callable returning the corrected (deformed) VIO trajectory.

        ``getter()`` returns ``(positions, teleport_flags)``: an ``(M, 3)`` array
        of dense VIO positions in NED AFTER SLAM's loop corrections have been
        rubber-sheeted onto them, plus an ``(M,)`` bool array flagging the
        vertices SLAM pulled far (``|correction_delta| > TELEPORT_M``). Polled
        each refresh and drawn as the warm-orange corrected-VIO line (teleport
        vertices recoloured red) behind the live green path; pass ``None`` to
        disable.
        """
        self._corrected_getter = getter

    # ---- per-line visibility (the single-view's 5 toggle buttons) --------

    def set_vo_visible(self, on: bool) -> None:
        """Show/hide the dim-grey VO (pure-vision ``pose.vo``) line + head dot."""
        on = bool(on)
        self._vo.setVisible(on)
        self._vo_head.setVisible(on)

    def set_vio_visible(self, on: bool) -> None:
        """Show/hide the live green VIO (``pose.odom``) line + head dot."""
        on = bool(on)
        self._traj.setVisible(on)
        self._vio_head.setVisible(on)

    def set_ba_visible(self, on: bool) -> None:
        """Show/hide the violet-blue VIO-BA (``pose.refined``) line + head dot."""
        on = bool(on)
        self._ba.setVisible(on)
        self._ba_head.setVisible(on)

    def set_corrected_visible(self, on: bool) -> None:
        """Show/hide the warm-orange SLAM-corrected-VIO line + head dot."""
        on = bool(on)
        self._corrected.setVisible(on)
        self._corrected_head.setVisible(on)

    def set_slam_visible(self, on: bool) -> None:
        """Show/hide the cyan SLAM keyframe line, head dot AND its keyframe dots.

        The SLAM toggle owns the whole SLAM overlay -- the corrected keyframe
        trajectory line, its cyan leading-keyframe head dot, and the amber/red
        keyframe + revisit dots -- so hiding it clears the entire SLAM map from
        the view, not just the line.
        """
        on = bool(on)
        self._refined.setVisible(on)
        self._slam_head.setVisible(on)
        self._slam_kf.setVisible(on)
        self._slam_kf_match.setVisible(on)

    # ---- internal --------------------------------------------------------

    def _refresh(self) -> None:
        traj, flags, latest = self.history.snapshot()
        if traj.shape[0] >= 2:
            # convert the whole trajectory NED -> ENU in one shot
            traj_enu = frames.ned_to_enu(traj.astype(np.float64)).astype(np.float32)
            # Per-vertex colour: normal odometry green, loop-closure teleport
            # segments magenta so the "swoosh back to memory" reads as a map
            # correction rather than real camera motion. A line-strip segment
            # is coloured teleport if EITHER endpoint is a teleport sample.
            col = np.tile(np.array(_qcolor(theme.TRACE_PATH, 0.95),
                                   dtype=np.float32), (traj_enu.shape[0], 1))
            tp = np.asarray(flags, dtype=bool)
            seg = tp.copy()
            seg[:-1] |= tp[1:]   # also colour the vertex leading INTO a teleport
            col[seg] = np.array((1.0, 0.2, 1.0, 0.95), dtype=np.float32)
            self._traj.setData(pos=traj_enu, color=col)
        if latest is not None:
            self._drone.update(latest)
            self._update_tracking_lost(latest)
            # Green VIO head dot at the newest VIO vertex (in ADDITION to the
            # orientation triad). `latest.pos_ned` is the leading pose.
            self._set_head(self._vio_head,
                           np.asarray(latest.pos_ned, np.float64).reshape(1, 3))
            if self._follow:
                p_enu = frames.ned_to_enu(latest.pos_ned)
                self.opts["center"] = QtGui.QVector3D(*p_enu.astype(float))
                self.update()
        else:
            self._set_head(self._vio_head, None)
        self._refresh_vo()
        self._refresh_ba()
        self._refresh_refined()
        self._refresh_corrected()
        self._refresh_overlay()

    def _update_tracking_lost(self, latest: Pose) -> None:
        """Advance the debounced tracking-lost state from the newest pose.

        Counts CONSECUTIVE distinct lost poses (keyed on object identity so a
        held ``latest`` between source updates is not recounted). The badge +
        marker latch to LOST only after ``LOST_DEBOUNCE_POSES`` such poses, so a
        single dropped frame cannot flash; recovery clears on the FIRST OK pose.

        The latch is on ``tracking_ok`` ALONE (the debounce never re-arms on the
        amber<->red switch). Once latched, the SEVERITY shown -- amber inertial
        DR vs red hard-lost -- is chosen per-frame from ``latest.inertial_dr``,
        so the badge can flip colour live (e.g. DR stops -> goes red) while it
        stays latched.
        """
        pid = id(latest)
        is_new = pid != self._last_pose_id
        self._last_pose_id = pid

        if latest.tracking_ok:
            # Recovery is immediate -- no debounce on the way back.
            self._lost_count = 0
            self._lost_latched = False
            self._apply_badge(None)             # hidden / marker green
            return

        if is_new:
            self._lost_count += 1
        if self._lost_count >= LOST_DEBOUNCE_POSES:
            self._lost_latched = True
        if self._lost_latched:
            # Amber while the IMU is still dead-reckoning a valid pose (tight),
            # red when there is no inertial fallback (loose path frozen).
            self._apply_badge("dr" if latest.inertial_dr else "lost")

    @staticmethod
    def _badge_qss(color: str) -> str:
        """Stylesheet for the master-warning badge in the given accent ``color``.

        ``color`` is ``theme.BAD`` (red, hard tracking loss) or ``theme.WARN``
        (amber, inertial dead-reckoning); both sit on the same semi-transparent
        dark backing so the badge stays legible over any scene geometry.
        """
        return (
            f"color: {color};"
            " background-color: rgba(13, 17, 23, 200);"   # theme.BG @ ~78% alpha
            f" border: 1px solid {color};"
            " border-radius: 4px;"
            " font-weight: bold;"
            " font-size: 20px;"
            " letter-spacing: 2px;"
            " padding: 6px 18px;")

    # Latched-lost severity -> (badge text, badge accent colour, marker status).
    _BADGE_SPEC = {
        "dr":   ("⚠ VISION LOST · INERTIAL DR", theme.WARN, "dr"),
        "lost": ("⚠ TRACKING LOST", theme.BAD, "lost"),
    }

    def _apply_badge(self, severity: str | None) -> None:
        """Apply the badge + drone-marker presentation for ``severity`` once.

        ``severity`` is ``None`` (not lost -> badge hidden, marker green), ``"dr"``
        (amber inertial-DR badge) or ``"lost"`` (red tracking-lost badge).
        Idempotent: a no-op when the displayed severity is unchanged, so neither
        the QLabel show/hide/restyle nor the marker recolour churns on a held
        state -- but an amber<->red flip (both latched) DOES restyle, since the
        severity actually changed.
        """
        if severity == self._shown_sev:
            return
        self._shown_sev = severity
        if severity is None:
            self._drone.set_status("ok")
            self._lost_badge.hide()
            return
        text, color, marker = self._BADGE_SPEC[severity]
        self._drone.set_status(marker)
        self._lost_badge.setText(text)
        self._lost_badge.setStyleSheet(self._badge_qss(color))
        self._position_badge()                  # re-pin (text width changed)
        self._lost_badge.show()
        self._lost_badge.raise_()               # composite above the GL surface

    def _position_badge(self) -> None:
        """Centre the badge horizontally near the top edge of the viewport."""
        b = self._lost_badge
        b.adjustSize()
        x = (self.width() - b.width()) // 2
        b.move(max(0, x), 12)                   # 12 px below the top edge

    def resizeEvent(self, ev) -> None:
        """Keep the master-warning badge pinned to the top-centre of the view."""
        super().resizeEvent(ev)
        # The base ``GLViewWidget.__init__`` can fire a resize BEFORE this
        # subclass builds the badge; ignore those early events.
        if getattr(self, "_lost_badge", None) is not None:
            self._position_badge()

    def _set_head(self, head, pos_ned) -> None:
        """Point a head scatter item at one NED vertex (or hide it).

        ``pos_ned`` is a ``(1, 3)`` NED row (the line's newest vertex) which we
        convert NED->ENU exactly like the lines; ``None`` -> an empty ``(0, 3)``
        so nothing draws (used when the line has no points yet).
        """
        if pos_ned is None:
            head.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return
        enu = frames.ned_to_enu(np.asarray(pos_ned, np.float64)).astype(
            np.float32).reshape(1, 3)
        head.setData(pos=enu)

    def _refresh_line(self, getter, item, head) -> None:
        """Poll a plain ``(N, 3)`` NED getter and push it to a line + head item.

        Shared by the VO / VIO-BA / SLAM lines (all single-colour line strips):
        an empty/short result collapses the line to a harmless single point and
        hides the head dot; otherwise the head dot is placed on the LAST (newest)
        vertex in the same NED->ENU frame as the line.
        """
        if getter is None:
            return
        pts = getter()
        if pts is None or len(pts) < 2:
            item.setData(pos=np.zeros((1, 3), dtype=np.float32))
            self._set_head(head, None)
            return
        arr = np.asarray(pts, np.float64)
        enu = frames.ned_to_enu(arr).astype(np.float32)
        item.setData(pos=enu)
        self._set_head(head, arr[-1:])           # head on the newest vertex

    def _refresh_vo(self) -> None:
        self._refresh_line(self._vo_getter, self._vo, self._vo_head)

    def _refresh_ba(self) -> None:
        self._refresh_line(self._ba_getter, self._ba, self._ba_head)

    def _refresh_refined(self) -> None:
        self._refresh_line(self._refined_getter, self._refined, self._slam_head)

    def _refresh_corrected(self) -> None:
        if self._corrected_getter is None:
            return
        result = self._corrected_getter()
        # The corrected getter returns (positions, teleport_flags). An empty /
        # short path collapses to a single point (same as the other lines) and
        # hides the head dot.
        pts, tp_flags = result
        if pts is None or len(pts) < 2:
            self._corrected.setData(pos=np.zeros((1, 3), dtype=np.float32))
            self._set_head(self._corrected_head, None)
            return
        arr = np.asarray(pts, np.float64)
        enu = frames.ned_to_enu(arr).astype(np.float32)
        # Per-vertex colour: normal corrected orange, with teleport vertices
        # recoloured red so a segment SLAM pulled far on a loop closure reads as
        # a flagged "swoosh". MIRRORS the green ``_traj`` recolour exactly: a
        # line-strip segment is teleport if EITHER endpoint is a teleport sample
        # (so the vertex leading INTO a teleport is recoloured too).
        col = np.tile(np.array(_qcolor(theme.CORRECTED_PATH, 0.85),
                               dtype=np.float32), (enu.shape[0], 1))
        tp = np.asarray(tp_flags, dtype=bool)
        seg = tp.copy()
        seg[:-1] |= tp[1:]
        col[seg] = np.array(_qcolor(theme.TELEPORT, 0.95), dtype=np.float32)
        self._corrected.setData(pos=enu, color=col)
        self._set_head(self._corrected_head, arr[-1:])  # head on newest vertex

    def _refresh_overlay(self) -> None:
        if self._overlay_getter is None:
            return
        kf_ned, match_ned, _loop_segs, flash_id = self._overlay_getter()
        empty = np.zeros((0, 3), dtype=np.float32)
        # Amber dots: every keyframe, always on.
        self._slam_kf.setData(
            pos=(frames.ned_to_enu(kf_ned.astype(np.float64)).astype(np.float32)
                 if len(kf_ned) else empty))

        now = time.monotonic()
        # A new loop just closed -> start a fresh flash on the revisited dot.
        if flash_id != self._flash_id:
            self._flash_id = flash_id
            self._flash_t0 = now
            self._flash_match = (
                frames.ned_to_enu(match_ned.astype(np.float64)).astype(
                    np.float32) if len(match_ned) else empty)

        # Blink (3 Hz) under a linear fade over the flash duration, then clear.
        dur = 1.8
        t = now - self._flash_t0
        if t > dur or not len(self._flash_match):
            self._slam_kf_match.setData(pos=empty)
            return
        fade = 1.0 - t / dur
        blink = 0.5 + 0.5 * float(np.cos(2.0 * np.pi * 3.0 * t))
        a = float(np.clip(fade * blink, 0.0, 1.0))
        self._slam_kf_match.setData(
            pos=self._flash_match, color=_qcolor(theme.BAD, a), size=16.0)
