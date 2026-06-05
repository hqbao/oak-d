"""ui flow: consume poses for display / scoring.

Subscribes ``pose.odom`` (the real-time trajectory), ``pose.refined`` (back-end
corrections) and ``loop.correction`` (SLAM corrections). In the live app a
renderer task would push these to the Qt 3D viewer; here the default collector
just records them so an offline run can be scored against Basalt.

It is a *sink*: it has no downstream topics, so it waits for END on all three
subscribed topics (``expected_ends = 3``) before declaring itself done -- which
guarantees every upstream flow has fully drained.
"""
from .collector import UiCollectorFlow
from .render import UiRenderFlow
from .tracks import UiTracksFlow
from .triplet import UiTripletFlow

__all__ = ["UiCollectorFlow", "UiRenderFlow", "UiTracksFlow", "UiTripletFlow"]
