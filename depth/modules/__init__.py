"""``depth.modules`` -- the two depth stages (the depth half of the pipeline).

depth runs INLINE on the capture project's ``imu_cam`` thread today via the
capture project's OWN ``imu_camera.modules.{compute_depth,publish_depth}`` Steps;
this standalone process runs the SAME math, but as plain PROCEDURAL functions so a
reader sees the data flow as straight-line calls:

* :func:`~depth.modules.compute_depth.compute_depth` -- run the SGM matcher
  (``sky.depth.stereo``) on a raw stereo pair and return a
  :class:`~depth.comms.messages.DepthFrame` (rectified-left + metric depth).
* :func:`~depth.modules.publish_depth.publish_depth` -- publish that frame on
  ``frame.depth``.

The standalone process (:mod:`depth.main`) calls them BACK-TO-BACK from the
:class:`~depth.comms.IPCSubscriber` callback for each raw ``cam.sync`` -- no
reactive ``Module`` / ``Step`` orchestration layer in between.
"""

from .compute_depth import compute_depth
from .publish_depth import publish_depth

__all__ = ["compute_depth", "publish_depth"]
