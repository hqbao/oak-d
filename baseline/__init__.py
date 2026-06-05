"""DepthAI library baseline (BasaltVIO + RTABMapSLAM) — the black box we replace.

This root package holds the pose sources that run DepthAI's on-device blobs:

  * ``baseline.depthai_vio``   — BasaltVIO odometry source
  * ``baseline.depthai_slam``  — BasaltVIO + RTABMapSLAM loop-closing source
  * ``baseline.tools``         — the main 3D viewer entry point plus session
                                 recording / replay / comparison utilities

Both sources depend only on the neutral ``oakd`` core package. Our from-scratch
replacement lives in ``ours``.
"""
