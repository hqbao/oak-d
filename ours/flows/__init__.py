"""``ours.flows`` -- the live-pipeline orchestration layer.

Each subpackage is one *flow*: a single thread that runs a short list of *tasks*
sequentially and talks to the other flows only over the pub/sub
:class:`~ours.lib.flow.pubsub.Bus`. The flows wrap the pure algorithms in
``ours.lib``; they hold no maths of their own.

    capture   grabs stereo frames + IMU            -> frame.raw, imu.sample
    depth     rectify + SGM dense depth            -> frame.depth
    odometry  KLT + RGB-D PnP (+ gyro prior)       -> pose.odom, keyframe
    backend   sliding-window bundle adjustment     -> pose.refined
    slam      ORB loop closure + pose graph        -> loop.correction
    ui        collects poses for display / scoring

Offline tools (``ours.tools``) call ``ours.lib`` directly and do NOT use this
package; the flows exist for the live, threaded runtime. Wire and run them with
``ours.app`` (or its ``--session`` replay mode for offline validation).
"""
