"""ui process: PyQt6 viewer with separate VIO + SLAM tabs (IPC subscriber).

Connects (over IPC) to the ``vio`` and ``slam`` endpoints, mirrors each onto a
**dedicated tab** of a single :class:`QMainWindow`. Per user spec
(``docs/PROC4_ARCHITECTURE.md`` §6):

* **VIO tab** -- live marker = ``pose.odom`` from VIO; map overlay (behind the
  marker) = ``pose.refined`` from VIO's windowed BA. This is the responsive
  frame-to-frame view; the marker never lags because it never waits on the
  back-end.
* **SLAM tab** -- live marker = the SAME ``pose.odom`` from VIO; map overlay =
  loop-corrected keyframe positions from SLAM's ``loop.correction``. So the
  SLAM tab shows the "best-known map" the loop closer has converged on, with
  the same live cursor.

The two tabs are independent QWidget views inside one Qt event loop -- no
duplicate Qt apps, no per-tab subprocess. The IPC subscribers live on
background threads (one :class:`IpcClientBus` per source) and marshal pose
samples onto the GUI thread via the existing :class:`PoseSource` callback
contract (the viewer already handles the cross-thread hop via signal/slot).

Calibration handshake
---------------------
Like VIO and SLAM, the UI waits for the retained ``calib.bundle`` on EACH
endpoint it cares about before building views. The bundle's
``width`` / ``height`` come from there so the future calibration / visualise
dialogs (Phase 10) know the capture resolution without needing a CLI flag.

Run::

    python -m ours.proc.ui                                  # default endpoints
    python -m ours.proc.ui --vio-endpoint oak.vio.test --slam-endpoint oak.slam.test
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.lib.flow.messages import END                             # noqa: E402
from ours.lib.ipc import IpcClientBus                              # noqa: E402
from ours.lib.ipc.messages import (                                # noqa: E402
    WireCalibBundle, WireLoopCorrection, WirePoseMsg,
)
from ours.lib.misc.frames import rot_to_quat                       # noqa: E402
from ours.lib.misc.pose import Pose, PoseHistory                   # noqa: E402
from ours.ui.source import PoseSource                              # noqa: E402

LOG = logging.getLogger("ours.proc.ui")

DEFAULT_VIO_ENDPOINT = "oak.vio"
DEFAULT_SLAM_ENDPOINT = "oak.slam"


# Camera optical (x right, y down, z forward) -> NED, matching
# `ours.ui.live_source.FlowPoseSource` so both code paths display in the same
# convention. _P_OPT_TO_FRD reorders attitude columns from optical to FRD body.
_M_OPT_TO_NED = np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
_P_OPT_TO_FRD = np.array([[0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0]])


def _wire_pose_to_ned(wm: WirePoseMsg, prev_pos: np.ndarray | None,
                      prev_t: float | None) -> tuple[Pose, np.ndarray, float]:
    """Convert a wire pose (T_world_cam, optical) into a viewer NED Pose.

    Returns ``(pose, new_prev_pos, new_prev_t)`` so the caller can carry the
    velocity finite-difference state forward.
    """
    T = wm.T_world_cam
    pos_opt = T[:3, 3]
    R_opt = T[:3, :3]
    pos_ned = _M_OPT_TO_NED @ pos_opt
    R_ned = _M_OPT_TO_NED @ R_opt @ _P_OPT_TO_FRD
    q_ned = rot_to_quat(R_ned)
    now = time.monotonic()
    if prev_t is None:
        vel_ned = np.zeros(3)
    else:
        dt = max(now - prev_t, 1e-6)
        vel_ned = (pos_ned - prev_pos) / dt
    ok = bool(wm.info.get("ok", True))
    pose = Pose(t=now, pos_ned=pos_ned, vel_ned=vel_ned,
                quat_wxyz=q_ned, tracking_ok=ok)
    return pose, pos_ned, now


# --------------------------------------------------------------------------- #
# IPC-driven pose source: subscribe to one endpoint's pose.odom topic + emit
# `Pose` samples to the standard `PoseSource` callback the viewer consumes.
# --------------------------------------------------------------------------- #
class IpcPoseSource(PoseSource):
    """A :class:`PoseSource` that drains ``pose.odom`` off an IPC endpoint.

    Unlike :class:`ours.ui.live_source.FlowPoseSource`, this does NOT open the
    camera or run any flow graph: VIO does that in its own process. The UI
    just subscribes to VIO's ``pose.odom`` (the realtime frame-to-frame
    marker) and pushes each pose to the viewer via the standard callback.
    """

    def __init__(self, endpoint: str, *, label: str = "ipc",
                 connect_timeout_s: float = 30.0) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.label = label
        self._connect_timeout_s = float(connect_timeout_s)
        self._client: IpcClientBus | None = None
        self._prev_pos: np.ndarray | None = None
        self._prev_t: float | None = None
        self._frames = 0
        self._last_fps_t = 0.0

    # ------------------------------------------------------------------ #
    def _on_pose(self, wm: WirePoseMsg | object) -> None:
        if wm is END:
            return                                # source closed; viewer stays
        pose, prev_pos, prev_t = _wire_pose_to_ned(
            wm, self._prev_pos, self._prev_t)     # type: ignore[arg-type]
        self._prev_pos = prev_pos
        self._prev_t = prev_t
        self._emit(pose)
        # Cheap FPS estimate: rolling 0.5 s window.
        self._frames += 1
        now = time.monotonic()
        if self._last_fps_t == 0.0:
            self._last_fps_t = now
        if now - self._last_fps_t >= 0.5:
            self.fps = self._frames / (now - self._last_fps_t)
            self._frames = 0
            self._last_fps_t = now

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        """Block until stop: the IPC client thread does all the real work."""
        client = IpcClientBus(self.endpoint,
                              connect_timeout_s=self._connect_timeout_s)
        client.subscribe("pose.odom", self._on_pose)
        try:
            client.start()
        except (TimeoutError, ConnectionError) as e:
            self._fail(f"connect to {self.endpoint}: {e}")
            return
        self._client = client
        # Wait for stop -- the IPC client's recv thread fires `_on_pose` for
        # every incoming wire pose; the work is done on that thread, not here.
        try:
            while not self._stop.is_set():
                time.sleep(0.1)
        finally:
            try:
                client.stop()
            except Exception:                                      # noqa: BLE001
                pass
            self._client = None


# --------------------------------------------------------------------------- #
# Loop-correction subscriber for the SLAM tab: collects the latest corrected
# keyframe positions so the viewer can draw the SLAM map behind the marker.
# --------------------------------------------------------------------------- #
class SlamMapTracker(threading.Thread):
    """Background subscriber: latch the latest ``loop.correction`` payload.

    The viewer reads :meth:`refined_path_snapshot` to render the SLAM map.
    Cheap (single lock + copy on snapshot) so it can be polled at GUI rate
    (10-30 Hz) without contention.
    """

    def __init__(self, endpoint: str, *,
                 connect_timeout_s: float = 30.0) -> None:
        super().__init__(name=f"slam-map-{endpoint}", daemon=True)
        self.endpoint = endpoint
        self._connect_timeout_s = float(connect_timeout_s)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._client: IpcClientBus | None = None
        self._kf_ned: np.ndarray = np.zeros((0, 3), np.float32)
        self._n_loops = 0

    # ------------------------------------------------------------------ #
    def _on_loop(self, wm: WireLoopCorrection | object) -> None:
        if wm is END:
            return
        kf_poses = dict(wm.kf_poses)             # type: ignore[attr-defined]
        if not kf_poses:
            return
        # `loop.correction` carries refined CAMERA-OPTICAL keyframe poses keyed
        # by kf_id. Convert each to NED position for the viewer overlay.
        ids = sorted(kf_poses)
        pos_opt = np.asarray([kf_poses[i][:3, 3] for i in ids],
                             dtype=np.float64)
        kf_ned = (pos_opt @ _M_OPT_TO_NED.T).astype(np.float32)
        with self._lock:
            self._kf_ned = kf_ned
            self._n_loops = int(wm.n_loops)      # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    def refined_path_snapshot(self) -> np.ndarray:
        with self._lock:
            return self._kf_ned.copy()

    def slam_overlay_snapshot(self):
        with self._lock:
            return (self._kf_ned.copy(),
                    np.zeros((0, 3), np.float32),
                    [], self._n_loops)

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        client = IpcClientBus(self.endpoint,
                              connect_timeout_s=self._connect_timeout_s)
        client.subscribe("loop.correction", self._on_loop)
        try:
            client.start()
        except (TimeoutError, ConnectionError) as e:
            LOG.error("slam-map-tracker: connect to %s failed: %s",
                      self.endpoint, e)
            return
        self._client = client
        try:
            self._stop.wait()
        finally:
            try:
                client.stop()
            except Exception:                                      # noqa: BLE001
                pass
            self._client = None

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------- #
# Calibration handshake -- mirrors the one used in ours.proc.vio / ours.proc.slam
# --------------------------------------------------------------------------- #
def _await_calib_bundle(endpoint: str, timeout_s: float) -> WireCalibBundle:
    bundle: list[WireCalibBundle | None] = [None]
    got = threading.Event()

    def on_calib(wm: WireCalibBundle) -> None:
        bundle[0] = wm
        got.set()

    client = IpcClientBus(endpoint, connect_timeout_s=timeout_s)
    client.subscribe("calib.bundle", on_calib)
    client.start()
    try:
        if not got.wait(timeout=timeout_s):
            raise TimeoutError(
                f"ui: no calib.bundle from {endpoint!r} in {timeout_s}s")
    finally:
        client.stop()
    assert bundle[0] is not None
    return bundle[0]


# --------------------------------------------------------------------------- #
def run_ui(*, vio_endpoint: str = DEFAULT_VIO_ENDPOINT,
           slam_endpoint: str = DEFAULT_SLAM_ENDPOINT,
           calib_timeout_s: float = 60.0,
           default_view: str = "ISO") -> int:
    """Open the 2-tab Qt UI and block on the Qt event loop."""
    # Import Qt lazily so a headless smoke / CI run that doesn't need the GUI
    # can import this module without pulling PyQt6.
    from PyQt6.QtWidgets import (
        QApplication, QHBoxLayout, QLabel, QMainWindow, QTabWidget, QVBoxLayout,
        QWidget,
    )

    from ours.ui import theme
    from ours.ui.viewer3d import Viewer3D
    from ours.ui.panels import TelemetryPanel

    # 1. Wait for VIO + SLAM to be ready (and learn the capture resolution).
    LOG.info("ui: waiting for calib.bundle on %s ...", vio_endpoint)
    vio_bundle = _await_calib_bundle(vio_endpoint, calib_timeout_s)
    LOG.info("ui: vio ready (%dx%d)", vio_bundle.width, vio_bundle.height)
    LOG.info("ui: waiting for calib.bundle on %s ...", slam_endpoint)
    slam_bundle = _await_calib_bundle(slam_endpoint, calib_timeout_s)
    LOG.info("ui: slam ready (%dx%d)", slam_bundle.width, slam_bundle.height)

    # 2. Qt app.
    app = QApplication(sys.argv if sys.argv else ["ours-ui"])
    app.setStyleSheet(theme.QSS)

    # 3. Build the two tabs. Each owns its own PoseHistory + Viewer3D +
    #    IpcPoseSource. The VIO tab additionally polls its refined trajectory
    #    line (a future enhancement -- not wired yet, see Phase 8 limitations
    #    in docs/PROC4_ARCHITECTURE.md). The SLAM tab polls the loop-corrected
    #    keyframe positions from SLAM's loop.correction stream.
    def _build_tab(name: str, history: PoseHistory, viewer: Viewer3D,
                   source: PoseSource,
                   slam_tracker: SlamMapTracker | None) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        # Header label so the operator can tell at a glance which tab is which
        # (the QTabWidget tab text already says it, but a sub-header keeps the
        # contract with the 1-window MainWindow look).
        hdr = QLabel(f"{name} · live marker = pose.odom · "
                     f"map = {'loop-corrected keyframes' if slam_tracker is not None else 'odom only'}")
        hdr.setObjectName("HeaderSub")
        root.addWidget(hdr)
        body = QWidget()
        bh = QHBoxLayout(body)
        bh.setContentsMargins(0, 0, 0, 0)
        bh.setSpacing(6)
        bh.addWidget(viewer, 1)
        panel = TelemetryPanel(history, source_fps_getter=lambda: source.fps)
        bh.addWidget(panel, 0)
        root.addWidget(body, 1)
        if slam_tracker is not None:
            viewer.set_overlay_source(slam_tracker.slam_overlay_snapshot)
            viewer.set_refined_path_source(slam_tracker.refined_path_snapshot)
        return w

    vio_history = PoseHistory(capacity=200_000)
    slam_history = PoseHistory(capacity=200_000)

    vio_viewer = Viewer3D(vio_history, default_view=default_view)
    slam_viewer = Viewer3D(slam_history, default_view=default_view)

    vio_source = IpcPoseSource(vio_endpoint, label="vio",
                               connect_timeout_s=calib_timeout_s)
    slam_pose_source = IpcPoseSource(vio_endpoint, label="slam-pose",
                                     connect_timeout_s=calib_timeout_s)
    slam_tracker = SlamMapTracker(slam_endpoint,
                                  connect_timeout_s=calib_timeout_s)
    slam_tracker.start()

    # 4. Push poses from the sources -> PoseHistory. The viewer redraws on a
    #    QTimer (~30 Hz) inside Viewer3D and reads from PoseHistory under a
    #    lock, so it's safe to append from the IPC recv thread directly.
    vio_source.start(vio_history.push)
    slam_pose_source.start(slam_history.push)

    tabs = QTabWidget()
    tabs.addTab(_build_tab("VIO", vio_history, vio_viewer, vio_source, None),
                "VIO")
    tabs.addTab(_build_tab("SLAM", slam_history, slam_viewer,
                           slam_pose_source, slam_tracker),
                "SLAM")

    win = QMainWindow()
    win.setWindowTitle("OAK-D · 4-proc · VIO | SLAM")
    win.resize(1400, 860)
    win.setStyleSheet(theme.QSS)
    win.setCentralWidget(tabs)
    win.show()

    # SIGTERM from the launcher closes the window so the Qt loop exits cleanly
    # (signal arrives on the main thread; QApplication.quit is thread-safe).
    def _on_sigterm(_signo, _frame):
        LOG.info("ui: SIGTERM -> quitting")
        app.quit()
    signal.signal(signal.SIGTERM, _on_sigterm)

    LOG.info("ui: entering Qt event loop")
    try:
        rc = app.exec()
    finally:
        vio_source.stop()
        slam_pose_source.stop()
        slam_tracker.stop()
    return int(rc)


# --------------------------------------------------------------------------- #
def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vio-endpoint", default=DEFAULT_VIO_ENDPOINT)
    ap.add_argument("--slam-endpoint", default=DEFAULT_SLAM_ENDPOINT)
    ap.add_argument("--calib-timeout", type=float, default=60.0,
                    help="seconds to wait for each retained calib.bundle")
    ap.add_argument("--default-view", default="ISO",
                    choices=("ISO", "TOP", "FRONT", "SIDE"),
                    help="initial 3D viewer preset")
    args = ap.parse_args()
    return run_ui(
        vio_endpoint=args.vio_endpoint,
        slam_endpoint=args.slam_endpoint,
        calib_timeout_s=args.calib_timeout,
        default_view=args.default_view,
    )


if __name__ == "__main__":
    raise SystemExit(main())
