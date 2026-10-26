"""`jet` as one more palette, without touching the repo. /tmp only.

The repo has gray, ironbow, rainbow, lava and arctic (irsim/isp/palette.py); `rainbow` is an HSV
hue sweep, not jet. This is the classic piecewise-linear jet (dark blue -> blue -> cyan -> yellow
-> red -> dark red), written as control points in the format `_CONTROL_POINTS` already uses and
built with the repo's own `_interpolate`, so the later repo change is this table plus the name in
two Literals -- see repo_patch_jet.diff. The default stays what the sensor YAML says (gray,
white_hot).
"""
from __future__ import annotations

import numpy as np

from irsim.isp.palette import PALETTES, _interpolate

JET_CONTROL_POINTS = [
    (0, (0, 0, 128)),
    (32, (0, 0, 255)),
    (96, (0, 255, 255)),
    (160, (255, 255, 0)),
    (224, (255, 0, 0)),
    (255, (128, 0, 0)),
]


def jet_table() -> np.ndarray:
    return _interpolate(JET_CONTROL_POINTS)


def table_for(name: str) -> np.ndarray:
    """256x3 uint8 for a repo palette name or 'jet'."""
    if name == "jet":
        return jet_table()
    return PALETTES[name]
