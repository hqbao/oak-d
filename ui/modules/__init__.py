"""``ui.modules`` -- the NO-Qt reactive bus sinks + the IPC source adapters.

Two layers share the name "ui"; they are different things, don't confuse them:

* ``ui.modules`` (HERE) -- reactive bus *sinks* (one thread each) that terminate
  the local bus by consuming topics for display, plus the IPC *adapters* that
  republish the cross-process capture / VIO topics onto that local bus. The sinks
  hold **NO Qt** (``import ui.modules.collector`` pulls zero PyQt); the adapters
  pull PyQt6 transitively because their worker base classes live in the Qt
  windows. NEITHER imports depthai -- the UI is device-agnostic by contract.
* ``ui.qt`` -- the actual **Qt GUI** (windows, viewer3d, panels). It builds a
  local bus and plugs these sinks in. The dependency is one-way: ``ui.qt``
  imports ``ui.modules`` sinks, never the reverse; ``ui.modules.ipc_sources``
  imports ``ui.qt`` worker base classes (the only crossing edge).

Ported verbatim from ``ours.flows.ui`` + ``ours.proc.ui_ipc_sources``
(Flow -> Module, Task -> Step, Bus -> LocalPubSub, IpcSubscriberFlow ->
IPCSubscriber, IpcClientBus -> IPCPubSub(role="client")). Topic strings unchanged.

The NO-Qt sinks (eagerly exported; importing them pulls zero PyQt):

* :class:`~ui.modules.collector.UiCollectorModule` -- records ``pose.odom`` /
  ``pose.refined`` / ``loop.correction`` for offline scoring. ``expected_ends = 3``.
* :class:`~ui.modules.render.UiRenderModule` -- bridges ``pose.odom`` to a viewer
  callback (the live 3D marker).
* :class:`~ui.modules.tracks.UiTracksModule` -- ``frame.tracks`` + ``frame.inliers``
  + ``frame.depth`` (joined by seq) for the keypoint-depth window.
* :class:`~ui.modules.triplet.UiTripletModule` -- ``frame.depth`` + ``imucam.sample``
  (joined by seq) for the image|depth|IMU window.

The IPC source adapters (feed the UNCHANGED ``ui.qt`` windows over IPC). These
are exposed via a LAZY module ``__getattr__`` so that importing the NO-Qt sinks
stays Qt-free: the adapters' worker base classes live in ``ui.qt``, so eagerly
importing :mod:`ui.modules.ipc_sources` here would pull PyQt6 into every
``import ui.modules.*``. ``from ui.modules import IpcImuRawSource`` (etc.) still
works -- it triggers the lazy import on first access:

* :class:`~ui.modules.ipc_sources.IpcImuRawSource` -- capture ``imu.raw`` for the
  gyro / accel calib dialogs.
* :class:`~ui.modules.ipc_sources.IpcTripletWorker` -- capture
  ``imucam.sample`` + ``frame.depth`` for the triplet window.
* :class:`~ui.modules.ipc_sources.IpcKeypointWorker` -- capture ``frame.depth`` +
  vio ``frame.tracks`` / ``frame.inliers`` for the keypoint window.
* :class:`~ui.modules.ipc_sources.IpcSlamMapSource` -- vio ``keyframe`` (+ its kf
  rings) + slam ``slam.map`` (corrected poses) fused into the room cloud for the
  SLAM 3D-map window.
* :class:`~ui.modules.ipc_sources.IpcFloorPlanSource` -- vio ``keyframe`` (+ its
  kf rings) back-projected + binned onto the ground plane into a 2D top-down
  OCCUPANCY raster for the Floor Plan (top-down) window -- a LIGHT, no-GL
  alternative to the 3D map. Shares the ``_KeyframeAccumulator`` base with the
  SLAM-map source (NO copy-paste of the SHM/recv wiring).
"""
from .collector import UiCollectorModule
from .render import UiRenderModule
from .tracks import UiTracksModule, TracksWithFrame
from .triplet import UiTripletModule

# The IPC adapters are pulled lazily (PEP 562 module __getattr__) so the NO-Qt
# bus sinks above stay importable WITHOUT PyQt6 (the adapters' worker base
# classes live in ui.qt). Accessing any of these names triggers the import.
_IPC_ADAPTERS = frozenset({
    "IpcImuRawSource", "IpcGyroFuseSource", "IpcTripletWorker",
    "IpcKeypointWorker", "IpcSlamMapSource", "IpcFloorPlanSource",
    "ipc_triplet_factory", "ipc_keypoint_factory", "ipc_slam_map_factory",
    "ipc_floor_plan_factory",
})

__all__ = [
    # NO-Qt bus sinks
    "UiCollectorModule",
    "UiRenderModule",
    "UiTracksModule",
    "TracksWithFrame",
    "UiTripletModule",
    # IPC source adapters (lazy)
    "IpcImuRawSource",
    "IpcGyroFuseSource",
    "IpcTripletWorker",
    "IpcKeypointWorker",
    "IpcSlamMapSource",
    "IpcFloorPlanSource",
    "ipc_triplet_factory",
    "ipc_keypoint_factory",
    "ipc_slam_map_factory",
    "ipc_floor_plan_factory",
]


def __getattr__(name: str):
    """Lazily resolve the Qt-pulling IPC adapters on first access (PEP 562)."""
    if name in _IPC_ADAPTERS:
        from . import ipc_sources
        return getattr(ipc_sources, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
