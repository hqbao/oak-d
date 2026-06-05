"""Our from-scratch RGB-D visual-inertial pipeline (library-free).

This root package holds everything we implement ourselves while replacing the
DepthAI BasaltVIO + RTABMap black boxes one module at a time:

  * ``ours.lib``                 — the algorithm library + runtime infrastructure.
                                   Pure logic grouped into subpackages (frontend,
                                   stereo, imu, odometry, backend, loop, io,
                                   config) plus the core ``pose`` / ``frames`` /
                                   ``pngio`` helpers and the flow/task/pubsub
                                   building blocks for the live pipeline.
  * ``ours.flows``               — live-pipeline orchestration: one directory per
                                   flow (capture, depth, odometry, backend, slam,
                                   ui), each a thread of sequential tasks that
                                   talk over the pub/sub bus.
  * ``ours.depthai_ours_vio``    — the live OAK-D source driving ``ours.lib``
  * ``ours.tools``               — offline scoring, self-tests and inspectors
                                   (these call ``ours.lib`` directly, not the flows)
  * ``ours.sources`` / ``ours.ui`` — our own pose-source base + Qt 3D viewer

This package is fully self-contained: it imports nothing from ``oakd`` (the
baseline's core) so the two pipelines share no code. We accept the small
duplication (pose/frames/pngio/sources/ui) in exchange for a clean split. The
library baseline we are replacing lives in ``baseline`` (with ``oakd`` as its
core).
"""
