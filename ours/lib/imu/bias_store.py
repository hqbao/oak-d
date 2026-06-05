"""Persisted gyro-bias cache, keyed by device id.

The gyro bias is a **sensor** property -- roughly fixed for a given physical
OAK-D -- so it should be calibrated once and reused on later runs, NOT measured
every START. This is the opposite of the gravity-align level, which is the
direction of gravity in the camera frame at START and therefore depends on how
the camera is held/mounted at that instant -- that one is inherently per-run and
is never cached here.

The cache is a tiny JSON file under the (gitignored) repo ``.cache`` dir, keyed
by the device id so several cameras never clobber each other::

    {"<device_id>": {"bias": [bx, by, bz], "n": 137, "ts": 1718000000.0}}

``bias`` is in the RAW gyroscope sensor frame (rad/s), exactly the quantity the
live capture subtracts before integrating the rotation prior.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# Repo-root/.cache/imu_bias.json (.cache is gitignored). parents: capture-> ...
# this file is ours/lib/imu/bias_store.py, so parents[3] is the repo root.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / ".cache" / "imu_bias.json"


def default_path() -> Path:
    """Where the bias cache lives (repo ``.cache/imu_bias.json``)."""
    return _DEFAULT_PATH


def _load_all(path: Path) -> dict:
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def load_gyro_bias(device_id: str,
                   path: Path | None = None) -> np.ndarray | None:
    """Return the cached gyro bias (rad/s, sensor frame) or ``None`` if absent."""
    entry = _load_all(path or _DEFAULT_PATH).get(str(device_id))
    if not isinstance(entry, dict) or "bias" not in entry:
        return None
    b = np.asarray(entry["bias"], dtype=np.float64)
    if b.shape != (3,) or not np.all(np.isfinite(b)):
        return None
    return b


def save_gyro_bias(device_id: str, bias: np.ndarray, n_samples: int,
                   path: Path | None = None) -> Path:
    """Persist the gyro bias for ``device_id`` (merges into the existing file)."""
    p = path or _DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _load_all(p)
    data[str(device_id)] = {
        "bias": [float(x) for x in np.asarray(bias, dtype=np.float64)],
        "n": int(n_samples),
        "ts": time.time(),
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(p)        # atomic on POSIX -> never leaves a half-written cache
    return p
