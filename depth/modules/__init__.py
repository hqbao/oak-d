"""``depth.modules`` -- the two depth steps (the depth half of the pipeline).

depth runs INLINE on the capture project's ``imu_cam`` thread today, so these two
steps are BYTE-IDENTICAL (modulo the project import prefix) to
``imu_camera/modules/{compute_depth,publish_depth}.py``:

* :class:`~depth.modules.compute_depth.ComputeDepthStep` -- run the SGM matcher
  (``sky.depth.stereo``) on a raw stereo pair and emit a
  :class:`~depth.comms.messages.DepthFrame` (rectified-left + metric depth).
* :class:`~depth.modules.publish_depth.PublishDepthStep` -- publish that frame on
  ``frame.depth``.

The standalone process (:mod:`depth.main`) composes them on a
:class:`~depth.comms.LocalPubSub` behind an
:class:`~depth.comms.IPCSubscriber` bridge that feeds the raw ``cam.sync``
stereo in over IPC, mirroring how the capture project composes them inline (see
``imu_camera.modules.pipeline.ImuCamModule``).
"""

from .compute_depth import ComputeDepthStep
from .publish_depth import PublishDepthStep

__all__ = ["ComputeDepthStep", "PublishDepthStep"]
