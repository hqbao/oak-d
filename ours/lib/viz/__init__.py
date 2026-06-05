"""Visualisation helpers shared by the dev tools and the Qt UI.

These modules draw *only* what a message actually carries (honest visualisation):
no parallel pipeline is run to synthesise data for display. cv2 is used purely as
a drawing backend here and is imported by these modules only -- the base VIO
path and the always-on UI never pull it.
"""
