#!/usr/bin/env python3
"""Gravity Sphere -- "see" accelerometer bias / scale / misalignment (ALGORITHMS §1.3).

WHAT THIS TEACHES
-----------------
A raw MEMS accelerometer has three error sources that a single "level the gravity
vector" step CANNOT separate (``sky.sensors.accel_calib``):

* **bias**        -- a per-axis zero offset (the reading is not 0 at free-fall),
* **scale**       -- a per-axis gain error (1 g reads 0.98 / 1.02 g),
* **misalignment**-- the three sensitive axes are not orthogonal / not aligned
  with the case, so a pure +x acceleration leaks into y and z.

The six-position (tumble) test makes all three observable: hold each face up/down
so true gravity points along +/- x, +/- y, +/- z. At rest the *true* specific
force has magnitude exactly ``g`` in every pose, so a correct calibration maps
every captured raw vector onto the sphere of radius ``g``. The affine correction
is ``a_cal = T (a_raw - b)`` -- see ``ALGORITHMS.md`` §1.3.

THE PICTURE (the most illuminating view for this stage, per §1.3)
----------------------------------------------------------------
One 3D scatter:

* the translucent **g-sphere** (radius ``g`` = 9.80665 m/s^2) -- where every
  resting accel vector MUST land if the sensor were perfect;
* the **RAW** six face means (RED dots) -- BEFORE calibration they form a tilted,
  off-centre **ellipsoid**: the centre offset IS the bias, the per-axis half-axis
  lengths ARE the scale, the tilt IS the misalignment;
* the **CALIBRATED** vectors ``T (a_raw - b)`` (GREEN dots) -- they SNAP onto the
  g-sphere. The leftover scatter off the sphere is exactly ``residual_g``.

bias / per-axis scale / residual_g are annotated on the figure, so the three
error sources are literally visible and the "snap" is quantified.

WHAT THE DEVICE ACTUALLY PROVIDES -- BE HONEST ABOUT THE DATA
-------------------------------------------------------------
Investigated: the calib store (``sky.sensors.calib_store``) persists only the
**solved** ``AccelCalibration`` (``T``, ``bias``, ``residual_g``, ``g``, n_poses)
per device id in ``.cache/imu_calib.json`` -- the six *raw* face mean vectors that
``SixFaceCollector`` captured are NOT kept (it solves and discards them), and no
recorded accel stream / six-face raw capture is stored on disk. So we cannot show
the literal captured dots. Two honest sources, in priority order:

1. ``--device <id>`` / auto (default): load the REAL stored ``AccelCalibration``
   and RECONSTRUCT the six raw face means by inverting the stored model exactly --
   ``a_raw_k = b + T^{-1} (g * dir_k)`` -- the inverse of ``a_cal = T(a_raw-b)``.
   These reconstructed dots are the vectors the device's OWN stored calibration
   maps onto the sphere: applying ``T(a-b)`` to them lands on ``g * dir_k`` by
   construction (so the GREEN dots sit exactly on the sphere and ``residual_g`` is
   the stored fit residual, faithfully reproduced as a tiny perturbation). The
   figure is clearly labelled "reconstructed from STORED calib (T,b)".

2. ``--demo``: a self-contained DEMONSTRATION -- synthesize six raw faces from a
   *known* injected (bias, scale, misalignment), re-solve with the real
   ``solve_accel_calibration``, and show the solver recovering the model + the
   snap. No device / disk needed. Labelled "SYNTHETIC demo".

The tool prints and the figure title states which source was used.

WHY OFFLINE / STANDALONE
------------------------
This tool only IMPORTS ``sky.sensors.accel_calib`` / ``calib_store`` (read-only)
and reads the calib JSON. It touches NOTHING in the live path / comms / oracle, so
gap=0 is trivially unaffected. matplotlib runs on the headless ``Agg`` backend
(heavy deps are fine -- this is an offline learning tool, per the spec).

SHARED RENDER CORE (reused LIVE in the calib wizard)
----------------------------------------------------
The figure-drawing is factored into :func:`render_gravity_sphere`
``(raw_vecs, calib_or_None, ...) -> RGB uint8 image`` so the SAME routine draws
both this offline payoff AND the LIVE accel six-face wizard
(``ui.qt.calib_dialogs.AccelCalibDialog``): as the operator tumbles the device the
captured raw vectors land on the sphere (RED), and after the 6th face + solve they
snap onto the g-sphere (GREEN) -- exactly this figure, in real time. The wizard
imports ONLY ``render_gravity_sphere`` (lazily, when the dialog opens) and feeds it
the six raw face means + the solved ``AccelCalibration`` it already holds locally,
so the live view needs NO comms / IPC change. :func:`render_sphere_png` is now a
thin disk-writing wrapper over that core (DRY: one figure, two callers).

USAGE
-----
Headless render of the REAL stored calib (auto-picks the only stored device)::

    .venv/bin/python -m imu_camera.tools.gravity_sphere --render /tmp/gravity_sphere.png

Pick a specific device, or fall back to the synthetic demo::

    .venv/bin/python -m imu_camera.tools.gravity_sphere --device 14442C10C164D5D200 \\
        --render /tmp/gravity_sphere.png
    .venv/bin/python -m imu_camera.tools.gravity_sphere --demo --render /tmp/gravity_sphere.png

Dependencies: numpy + matplotlib (Agg). ``sky.*`` is imported read-only, never
modified.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Run as a module (-m) or as a script: make the repo root importable either way.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# READ-ONLY imports of the shared sky library (the model + solver + store). This
# tool never mutates sky.* -- it only consumes the persisted calibration.
from sky.sensors import calib_store                               # noqa: E402
from sky.sensors.accel_calib import (                             # noqa: E402
    G_STANDARD,
    SIX_FACES,
    AccelCalibration,
    solve_accel_calibration,
)


# --------------------------------------------------------------------------- #
# Compute core -- the six raw faces + the calibration, from a chosen source.
# --------------------------------------------------------------------------- #
@dataclass
class SphereData:
    """Everything the figure needs, plus provenance for honest labelling.

    ``raw_faces`` are the six raw mean accel vectors (m/s^2), ``cal_faces`` are
    ``T (raw - b)`` (should sit on the g-sphere), ``directions`` are the unit
    gravity directions per pose. ``calib`` is the (stored or re-solved) model.
    ``source`` is a one-line human description of where the data came from.
    """

    raw_faces: np.ndarray        # (6, 3)
    cal_faces: np.ndarray        # (6, 3)
    directions: np.ndarray       # (6, 3) unit
    calib: AccelCalibration
    source: str
    synthetic: bool              # True for --demo (must be labelled as such)


def _axis_scales(T: np.ndarray) -> np.ndarray:
    """Per-axis sensor scale (gain) recovered from the correction matrix ``T``.

    The forward model is ``a_cal = T (a_raw - b)``, so the sensor's own forward
    distortion is ``a_raw - b = T^{-1} a_cal``: a unit calibrated input along axis
    i produces a raw vector ``T^{-1} e_i`` whose LENGTH is that axis' gain. The
    per-axis scale is therefore the column norm of ``T^{-1}`` (a clean readout of
    "1 g reads as X g" even when misalignment couples the axes).
    """
    Tinv = np.linalg.inv(T)
    return np.linalg.norm(Tinv, axis=0)


def _reconstruct_raw_faces(cal: AccelCalibration) -> np.ndarray:
    """Invert the stored model to recover the six raw face means it calibrates.

    The forward correction is ``a_cal = T (a_raw - b)``. A perfect resting pose
    with face k up reads ``a_cal = g * dir_k`` on the sphere, so the raw vector
    the stored model maps there is the exact inverse::

        a_raw_k = b + T^{-1} (g * dir_k)

    These are the genuine "before" dots for THIS device's stored calibration:
    feeding them back through ``cal.apply`` lands on ``g * dir_k`` by construction,
    so the BEFORE ellipsoid and the AFTER snap are both faithful to the persisted
    (T, b) -- we are showing the device's own stored model, not inventing data.
    """
    Tinv = np.linalg.inv(cal.T)
    target = cal.g * SIX_FACES                     # (6,3): g*dir per face
    return cal.bias[None, :] + target @ Tinv.T     # (6,3) raw vectors


def from_stored_calib(device_id: str | None,
                      path: Path | None = None) -> SphereData:
    """Source 1: REAL stored ``AccelCalibration`` -> reconstructed raw faces.

    If ``device_id`` is None, auto-pick the single device that has an accel calib
    (the common case -- one camera). Raises ``LookupError`` if no stored accel
    calibration is found, so the caller can fall back to ``--demo``.
    """
    p = path or calib_store.default_path()
    chosen = device_id
    if chosen is None:
        chosen = _auto_device(p)
        if chosen is None:
            raise LookupError(
                f"no stored accel calibration in {p}; run with --demo")
    cal = calib_store.load_accel_calib(chosen, p)
    if cal is None:
        raise LookupError(
            f"device {chosen!r} has no stored accel calibration in {p}; "
            "run with --demo or pick another --device")

    raw = _reconstruct_raw_faces(cal)
    cal_faces = cal.apply(raw)                      # lands on g*dir by construction
    src = (f"REAL stored calib (device {chosen}); six raw faces RECONSTRUCTED "
           f"from stored (T,b) -- residual_g={cal.residual_g:.4f} m/s^2")
    return SphereData(raw, cal_faces, SIX_FACES.copy(), cal, src, synthetic=False)


def _auto_device(path: Path) -> str | None:
    """Return the lone device id that has an accel calib, or None.

    Picks the only device with a stored accel block; if several exist we still
    return the first deterministically (sorted) so the render is reproducible, but
    the common deployment has exactly one camera.
    """
    # _load_all is the store's own loader (handles legacy migration); using it
    # keeps us honest about exactly what the store would read.
    data = calib_store._load_all(path)              # noqa: SLF001 (read-only introspection)
    ids = sorted(k for k, v in data.items()
                 if isinstance(v, dict) and isinstance(v.get("accel"), dict))
    return ids[0] if ids else None


# Known injected distortion for the synthetic demo -- a plausible MEMS: ~1-2%
# per-axis scale error, a few-degrees misalignment, and a real bias. Kept here as
# named constants so the demo is reproducible and the figure can show "injected
# vs recovered". These are NOT real device data -- the figure says SYNTHETIC.
_DEMO_BIAS = np.array([0.18, -0.27, 0.11])         # m/s^2 raw zero offset
_DEMO_SENSOR = np.array([                           # raw = SENSOR @ true + bias
    [1.018, 0.020, -0.013],                         # +x leaks a little into y/z
    [-0.015, 0.987, 0.022],                         # off-diagonals = misalignment
    [0.011, -0.018, 1.009],                         # diagonals = per-axis scale
])


def from_demo(g: float = G_STANDARD,
              noise_std: float = 0.03,
              seed: int = 7) -> SphereData:
    """Source 2: synthesize six raw faces from a KNOWN model, then re-solve.

    Builds raw captures ``a_raw_k = SENSOR @ (g*dir_k) + bias + noise`` from the
    injected ``_DEMO_SENSOR`` / ``_DEMO_BIAS``, runs the REAL
    ``solve_accel_calibration`` to recover ``(T, b)``, and returns both clouds. A
    small per-capture noise makes the ``residual_g`` non-zero and the "leftover
    scatter" visible (so the demo teaches the residual too). Clearly synthetic.
    """
    rng = np.random.default_rng(seed)
    dirs = SIX_FACES.copy()
    raw = np.empty((6, 3))
    for k in range(6):
        true_sf = g * dirs[k]                        # perfect specific force
        raw[k] = _DEMO_SENSOR @ true_sf + _DEMO_BIAS
        raw[k] += rng.normal(0.0, noise_std, size=3)
    # Recover with the production solver (using the known face directions, exactly
    # as SixFaceCollector does on the 6th capture).
    cal = solve_accel_calibration(raw, directions=dirs, g=g)
    cal_faces = cal.apply(raw)
    src = ("SYNTHETIC demo: six raw faces built from a KNOWN injected "
           f"(bias,scale,misalign)+noise, recovered by solve_accel_calibration "
           f"-> residual_g={cal.residual_g:.4f} m/s^2")
    return SphereData(raw, cal_faces, dirs, cal, src, synthetic=True)


# --------------------------------------------------------------------------- #
# Render -- the 3D gravity-sphere figure (matplotlib, headless Agg).
# --------------------------------------------------------------------------- #
def _g_sphere(g: float, n: int = 24):
    """Wireframe lon/lat grid of the radius-``g`` sphere (for ``plot_wireframe``)."""
    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0, np.pi, n)
    x = g * np.outer(np.cos(u), np.sin(v))
    y = g * np.outer(np.sin(u), np.sin(v))
    z = g * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def render_gravity_sphere(
    raw_vecs: np.ndarray,
    calib: AccelCalibration | None,
    *,
    directions: np.ndarray | None = None,
    g: float = G_STANDARD,
    source: str = "",
    synthetic: bool = False,
    figsize: tuple[float, float] = (11, 9),
    dpi: int = 120,
) -> np.ndarray:
    """Draw the gravity-sphere figure and return it as an RGB ``uint8`` image.

    This is the SHARED render core used by BOTH the offline ``--render`` tool
    (via :func:`render_sphere_png`, which just writes the returned image to disk)
    AND the LIVE accel-calibration wizard (``ui.qt.calib_dialogs.AccelCalibDialog``,
    which shows the returned array as a ``QPixmap``). Keeping ONE drawing routine
    means the live "snap" the operator sees during capture is pixel-for-pixel the
    same figure the offline payoff shows -- no second copy to drift.

    It is deliberately TOLERANT of partial input so it can animate the wizard's
    1->6 fill:

    * ``raw_vecs``  -- ``(k, 3)`` raw mean accel vectors already captured
      (``0 <= k <= 6``). They are ALWAYS plotted as the RED "before" dots, so the
      off-centre/tilted ellipsoid grows as the operator tumbles the device.
    * ``calib``     -- the solved :class:`AccelCalibration`, or ``None`` while the
      six faces are still being collected. When present, the GREEN "after" dots
      ``T(raw - b)``, the snap connectors, the bias marker, and the quantitative
      annotation box are drawn -- the post-solve payoff. When ``None`` only the
      sphere + accumulating raw dots are shown (with a "capturing..." note).
    * ``directions``-- optional ``(k, 3)`` unit gravity directions per captured
      face. Unused for drawing (the dots speak for themselves) but accepted so the
      caller can pass the collector's per-face dirs without reshaping; kept for a
      symmetric, self-documenting signature.

    The render runs on the headless ``Agg`` backend (no display / Qt needed), so
    it is safe to call from a worker or a Qt UI thread alike. matplotlib is
    imported LAZILY here so merely importing this module stays light.
    """
    import matplotlib
    matplotlib.use("Agg")                            # headless: no display / Qt
    import matplotlib.pyplot as plt
    # The "3d" projection is registered automatically by matplotlib when
    # add_subplot(projection="3d") is called (no explicit mpl_toolkits import
    # needed on matplotlib >= 3.x), so we don't import it here.

    raw = np.asarray(raw_vecs, dtype=np.float64).reshape(-1, 3)
    # The calibration's own g (if solved) is authoritative; otherwise the caller's.
    g = float(calib.g) if calib is not None else float(g)
    del directions                                   # accepted but not drawn (see docstring)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # 1) translucent g-sphere wireframe -- the target every resting vector must
    #    land on. Drawn faint so the dots read clearly on top.
    sx, sy, sz = _g_sphere(g)
    ax.plot_wireframe(sx, sy, sz, color="#7aa6c2", linewidth=0.4, alpha=0.35)

    # 2) RAW captured faces (RED) -- the off-centre, tilted ellipsoid that grows
    #    1->6 as the operator tumbles (bias=centre offset, scale=half-axes, tilt).
    if raw.shape[0] > 0:
        ax.scatter(raw[:, 0], raw[:, 1], raw[:, 2], c="#d62728", s=70,
                   depthshade=False, edgecolors="k", linewidths=0.5,
                   label="RAW face means (before)")

    if calib is not None:
        # --- post-solve payoff: the SNAP onto the g-sphere. ------------------- #
        calf = calib.apply(raw)
        # 3) CALIBRATED faces (GREEN) -- snapped onto the g-sphere.
        ax.scatter(calf[:, 0], calf[:, 1], calf[:, 2], c="#2ca02c", s=70,
                   depthshade=False, edgecolors="k", linewidths=0.5,
                   label="T(a-b) calibrated (after)")
        # Thin grey connector from each raw dot to its calibrated landing point:
        # the "snap" made literal (how far / which way calibration moved each).
        for k in range(raw.shape[0]):
            ax.plot([raw[k, 0], calf[k, 0]],
                    [raw[k, 1], calf[k, 1]],
                    [raw[k, 2], calf[k, 2]],
                    color="0.55", linewidth=0.8, alpha=0.8)

        # 4) bias marker: the RAW ellipsoid centre IS the bias offset from origin.
        b = calib.bias
        ax.scatter([b[0]], [b[1]], [b[2]], c="#ff7f0e", marker="X", s=90,
                   depthshade=False, label="bias (raw ellipsoid centre)")
        ax.plot([0.0, b[0]], [0.0, b[1]], [0.0, b[2]],
                color="#ff7f0e", linewidth=1.2, linestyle="--", alpha=0.9)
    else:
        calf = None
        b = None

    # World axes through the origin (so the off-centre of RAW is obvious).
    lim = g * 1.35
    for vec, c in (((lim, 0, 0), "k"), ((0, lim, 0), "k"), ((0, 0, lim), "k")):
        ax.plot([-vec[0], vec[0]], [-vec[1], vec[1]], [-vec[2], vec[2]],
                color=c, linewidth=0.4, alpha=0.3)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))                     # equal aspect: a sphere, not an egg
    ax.set_xlabel("ax  (m/s^2)")
    ax.set_ylabel("ay  (m/s^2)")
    ax.set_zlabel("az  (m/s^2)")
    ax.view_init(elev=22, azim=-58)

    # --- Title + quantitative annotation box. --------------------------------- #
    if synthetic:
        tag = "SYNTHETIC DEMO (not real device data)"
    elif calib is None:
        tag = f"CAPTURING ({raw.shape[0]}/6 faces)"
    else:
        tag = "calibrated"
    title = ("Gravity Sphere -- accelerometer bias / scale / misalignment "
             f"made visible\n[{tag}]  g = {g:.5f} m/s^2")
    ax.set_title(title, fontsize=11)

    if calib is not None:
        # Full quantitative payoff once solved.
        scales = _axis_scales(calib.T)
        raw_norms = np.linalg.norm(raw, axis=1)
        cal_norms = np.linalg.norm(calf, axis=1)
        raw_rms = float(np.sqrt(np.mean((raw_norms - g) ** 2)))
        cal_rms = float(np.sqrt(np.mean((cal_norms - g) ** 2)))
        info = (
            f"bias  b = [{b[0]:+.4f}, {b[1]:+.4f}, {b[2]:+.4f}]  m/s^2\n"
            f"per-axis scale = [{scales[0]:.4f}, {scales[1]:.4f}, {scales[2]:.4f}]"
            "  (1 g reads as ...)\n"
            f"misalign (off-diag of T) max = {_max_offdiag(calib.T):.4f}\n"
            f"residual_g (fit) = {calib.residual_g:.4f}  m/s^2\n"
            f"|a|-g  RMS:  RAW = {raw_rms:.3f}  ->  CAL = {cal_rms:.4f}  m/s^2  "
            "(the snap)"
        )
    else:
        # Mid-capture: show how full the sphere is + which faces remain.
        info = (
            f"capturing the six faces: {raw.shape[0]}/6 landed (RED).\n"
            "tumble the device through every face up & down --\n"
            "after the 6th, T(a-b) snaps them onto the g-sphere (GREEN)."
        )
    # Off-axes text box so it never overlaps the 3D cloud.
    fig.text(0.015, 0.015, info, fontsize=9, family="monospace",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round", facecolor="#f4f4f4", edgecolor="0.7"))
    if source:
        fig.text(0.985, 0.015, source, fontsize=7.5, color="0.35",
                 va="bottom", ha="right", wrap=True)

    # Legend only once there is something labelled to show.
    if raw.shape[0] > 0:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # Rasterize the figure to an RGB uint8 array via the Agg canvas (no temp file).
    fig.set_dpi(dpi)
    fig.canvas.draw()
    # buffer_rgba() is the stable Agg accessor across matplotlib 3.x; drop alpha.
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = np.ascontiguousarray(rgba[:, :, :3])
    plt.close(fig)
    return img


def render_sphere_png(data: SphereData, out_path: str) -> str:
    """Write the gravity-sphere 3D figure (RAW ellipsoid + calibrated snap).

    Thin disk-writing wrapper over the shared :func:`render_gravity_sphere` core
    (DRY: the offline tool and the live wizard draw the SAME figure). Encodes the
    returned RGB array as a PNG with matplotlib's image writer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    img = render_gravity_sphere(
        data.raw_faces, data.calib,
        directions=data.directions, g=data.calib.g,
        source=data.source, synthetic=data.synthetic)
    out_path = str(Path(out_path).resolve())
    mpimg.imsave(out_path, img)
    return out_path


def _max_offdiag(T: np.ndarray) -> float:
    """Largest off-diagonal magnitude of ``T`` (a coarse misalignment figure)."""
    off = T - np.diag(np.diag(T))
    return float(np.max(np.abs(off)))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build(args) -> SphereData:
    """Resolve the data source per the CLI flags (with honest fallback)."""
    if args.demo:
        return from_demo()
    try:
        return from_stored_calib(args.device,
                                 Path(args.calib) if args.calib else None)
    except LookupError as exc:
        if args.device is not None or args.calib is not None:
            # The user explicitly asked for a stored source -- do NOT silently
            # swap to synthetic; surface the miss so the report is honest.
            raise
        print(f"note: {exc}", file=sys.stderr)
        print("note: falling back to --demo (synthetic)", file=sys.stderr)
        return from_demo()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Gravity Sphere: visualize accel bias/scale/misalignment as a "
                    "RAW ellipsoid snapping onto the g-sphere after T(a-b) "
                    "(ALGORITHMS.md §1.3). Offline; imports sky.* read-only.")
    ap.add_argument("--device", default=None,
                    help="device id whose STORED accel calib to show "
                         "(default: auto-pick the only stored device)")
    ap.add_argument("--calib", default=None,
                    help="path to the imu_calib.json (default: repo .cache)")
    ap.add_argument("--demo", action="store_true",
                    help="ignore stored data: synthesize from a KNOWN model and "
                         "show the solver recovering it (clearly labelled)")
    ap.add_argument("--render", metavar="PNG", default=None,
                    help="headless: write the gravity-sphere PNG and exit "
                         "(no display / Qt needed)")
    args = ap.parse_args(argv)

    data = _build(args)
    cal = data.calib
    scales = _axis_scales(cal.T)
    raw_norms = np.linalg.norm(data.raw_faces, axis=1)
    cal_norms = np.linalg.norm(data.cal_faces, axis=1)
    g = cal.g

    print(f"source: {data.source}")
    print(f"bias        = {np.array2string(cal.bias, precision=4)}  m/s^2")
    print(f"per-axis scale = {np.array2string(scales, precision=4)}")
    print(f"residual_g  = {cal.residual_g:.4f}  m/s^2")
    print(f"|a|-g RMS:  RAW = {np.sqrt(np.mean((raw_norms - g) ** 2)):.4f}  ->  "
          f"CAL = {np.sqrt(np.mean((cal_norms - g) ** 2)):.4f}  m/s^2")

    if args.render:
        out = render_sphere_png(data, args.render)
        print(f"wrote {out}")
        return 0

    print("note: this tool is headless-only; pass --render <PNG> to write the figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
