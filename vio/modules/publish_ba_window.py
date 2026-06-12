"""``publish_ba_window`` step: emit the windowed-BA solve snapshot on ``ba.window``.

OPT-IN (``--ba-window``). The default BackendModule does NOT register this step
(``capture_window`` defaults False), so the deterministic / oracle path never
builds the capture engine and this step never runs -- the byte-parity oracle is
UNAFFECTED. When enabled, the backend builds the capture-aware engine (see
:func:`vio.mathlib.engine.make_ba_engine` with ``capture_window=True``) whose
``ba_step_capture`` stashes a :class:`~vio.mathlib.engine.steps.BaWindowSnap` on
the overlay channel every keyframe the solve actually ran.

This step is chained AFTER :class:`~vio.modules.run_ba.RunBA` (so the engine has
already submitted this keyframe and the overlay is fresh) and BEFORE
:class:`~vio.modules.publish_refined.PublishRefined`. It polls
``engine.poll_overlay()``; when that is a ``BaWindowSnap`` it publishes one
:class:`~vio.comms.messages.BaWindow` on ``ba.window`` (the 1:1 columns the UI's
"BA Window" visualiser renders), then forwards the refined pose UNCHANGED so the
``pose.refined`` publish is identical to the no-capture path.

There are NO images on the wire (mirrors ``slam.loop`` / ``publish_loops``); the
snapshot is pure POD (window keyframe poses + landmarks + observation rays).
"""
from __future__ import annotations

from vio.comms import topics
from vio.comms.messages import BaWindow, PoseMsg
from vio.comms import Step
from vio.mathlib.engine import Engine
from vio.mathlib.engine.steps import BaWindowSnap


class PublishBaWindow(Step):
    name = "publish_ba_window"

    def run(self, ctx, msg: PoseMsg):
        engine: Engine = ctx.state["engine"]
        snap = engine.poll_overlay()
        # The overlay is a BaWindowSnap ONLY on the capture engine + a keyframe
        # whose solve ran; anything else (None on warmup) is simply skipped.
        if isinstance(snap, BaWindowSnap):
            ctx.bus.publish(topics.BA_WINDOW, BaWindow(
                seq=int(snap.seq), ts_ns=int(snap.ts_ns),
                kf_ids=snap.kf_ids, kf_quat=snap.kf_quat, kf_pos=snap.kf_pos,
                lm_ids=snap.lm_ids, lm_xyz=snap.lm_xyz,
                obs_kf=snap.obs_kf, obs_lm=snap.obs_lm, obs_uv=snap.obs_uv,
                obs_reproj_px=snap.obs_reproj_px,
                ba_reproj_px=float(snap.ba_reproj_px),
                kf_quat_pre=snap.kf_quat_pre, kf_pos_pre=snap.kf_pos_pre,
                lm_xyz_pre=snap.lm_xyz_pre,
                n_kf=int(snap.n_kf), n_lm=int(snap.n_lm)))
        # Forward the refined pose UNCHANGED so PublishRefined emits it identically.
        return msg
