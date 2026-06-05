"""Our from-scratch RGB-D visual-inertial pipeline (library-free).

This root package holds everything we implement ourselves while replacing the
DepthAI BasaltVIO + RTABMap black boxes one module at a time:

  * ``ours.vio``                 — the algorithm library (KLT, corners, PnP, IMU
                                   preintegration, stereo SGM, windowed BA, pose
                                   graph + loop closure, the synced-input bundle)
  * ``ours.depthai_ours_vio``    — the live OAK-D source driving ``ours.vio``
  * ``ours.tools``               — offline scoring, self-tests and inspectors

Shared infrastructure (pose types, frames math, the Qt viewer, the session
recorder, the PNG codec) lives in the neutral ``oakd`` core package, which this
package depends on. The library baseline we are replacing lives in ``baseline``.
"""
