"""``ours.flows`` -- the live-pipeline orchestration layer.

Each subpackage is one *flow*: a single thread that runs a short list of *tasks*
sequentially (one task per file). A flow uses the libraries in ``ours.lib`` --
both the algorithm libraries (stereo, odometry, ...) and the flow-framework
library ``ours.lib.flow`` (``Flow`` / ``SourceFlow`` / ``Task`` / ``Bus`` /
``topics`` / ``messages``). The flows hold no maths of their own.

    capture   grabs stereo frames + IMU            -> frame.raw, imu.sample
    depth     rectify + SGM dense depth            -> frame.depth
    odometry  KLT + RGB-D PnP (+ gyro prior)       -> pose.odom, keyframe
    backend   sliding-window bundle adjustment     -> pose.refined
    slam      ORB loop closure + pose graph        -> loop.correction
    ui        collects poses for display / scoring

================================ HARD RULE ================================
Flows NEVER call each other directly. The ONLY way one flow influences another
is by publishing a message on a :class:`~ours.lib.flow.pubsub.Bus` topic that the
other flow subscribes to. No flow imports, holds a reference to, or invokes a
method on another flow. Concretely:

  * A flow publishes its outputs with ``ctx.bus.publish(topic, msg)`` (usually
    in a small dedicated "publish_*" task at the end of a chain).
  * A flow declares the topics it consumes with ``self.on(topic, [tasks...])``
    and the topics it produces with ``self.forwards_to(topic, ...)``.
  * The topic names live in ``ours.lib.flow.topics`` and the message types in
    ``ours.lib.flow.messages`` -- together they ARE the inter-flow contract.

Why: this keeps every flow independently testable and swappable (the long-term
goal is to replace black-box modules one at a time), and guarantees no hidden
coupling or cross-thread method calls. The pipeline graph is therefore fully
described by which topics each flow reads and writes -- nothing else.
==========================================================================

Offline tools (``ours.tools``) call the ``ours.lib`` algorithm libraries directly
and do NOT use this package; the flows exist for the live, threaded runtime. Wire
and run them with ``ours.app`` (or its ``--session`` replay mode for offline
validation).
"""
