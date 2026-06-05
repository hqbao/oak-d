#!/usr/bin/env python3
"""Visualise our from-scratch SGM depth -- side by side with the chip depth.

Two ways to see *how our depth map looks*:

  * **Recorded replay** (default, no hardware): replays a gold session and shows
    ``[ rectified-left | OUR SGM depth | chip depth ]`` so you can eyeball our
    depth against the OAK-D's reference frame by frame, in real time.

  * **Live** (``--live``, needs an OAK-D plugged in): taps the two RAW cameras
    (``CAM_B``/``CAM_C``) and rectifies BOTH frames + runs our SGM ourselves --
    exactly the VPU-free path the live VIO source uses, so the depth you preview
    here is byte-identical to what the VIO consumes. NO chip ``StereoDepth`` node
    is in the loop (so there is no "width multiple of 16" constraint, and nothing
    reads the VPU depth). Shows ``[ rectified-left | OUR SGM depth ]``.

Our depth is :class:`oakd.vio.SGMStereoMatcher` (own rectification + dense
semi-global matching, library-free). The chip depth shown in replay is only the
reference oracle, exactly like ``stereo_selftest.py`` -- it is NOT used to
produce our map. cv2 here is a dev-tool display dependency (windowing +
colormap), same as the other ``tools/*`` viewers; it is not in any production
path.

Usage::

    # replay the default gold session, full-accuracy 8-path SGM
    python tools/stereo_view.py

    # a specific session, fast (live) preset
    python tools/stereo_view.py --session sessions/gold/corridor_60s --fast

    # live from the camera (fast preset recommended)
    python tools/stereo_view.py --live --fast

Keys: SPACE pause/resume, ``n`` step one frame (when paused), ``q`` / ESC quit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.vio import (  # noqa: E402
    SessionReader, SGMConfig, SGMStereoMatcher, StereoCalib,
)
from ours.vio.stereo import HAVE_NUMBA  # noqa: E402
from ours.vio.resolution import ResolutionProfile  # noqa: E402

# Fixed depth range (metres) for the colormap, so colours are stable across
# frames (a per-frame autoscale makes the scene "breathe" and hides drift).
_D_MIN = 0.3
_D_MAX = 8.0


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """Metric depth (m, 0 == invalid) -> BGR turbo image (near = red)."""
    valid = depth_m > 1e-6
    norm = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid.any():
        z = np.clip(depth_m, _D_MIN, _D_MAX)
        # Invert so near is hot (red) and far is cool (blue), like the chip view.
        t = 1.0 - (z - _D_MIN) / (_D_MAX - _D_MIN)
        norm[valid] = (t[valid] * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _text_style(width: int, ref: float = 0.6, ref_thick: int = 2):
    """Font scale + thickness + top-left margin scaled to the frame width.

    All overlay text was sized for the 640-wide baseline; on a small frame
    (e.g. 54 px) that fixed size covers the whole image. Scale everything by
    ``width / 640`` so the labels keep the same RELATIVE size at any resolution,
    with a small floor so they stay renderable on tiny frames.
    """
    s = max(0.15, ref * width / 640.0)
    thick = max(1, round(ref_thick * width / 640.0))
    mx = max(2, round(8 * width / 640.0))
    my = max(round(s * 18) + 2, 6)  # baseline y for top-anchored text
    return s, thick, mx, my


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    s, thick, mx, my = _text_style(img.shape[1])
    cv2.putText(out, text, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, s,
                (255, 255, 255), thick, cv2.LINE_AA)
    return out


def _gray_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class TemporalHold:
    """Display-only stabiliser: hold a pixel's last valid depth for a few frames.

    Our SGM validity decision (LR-check + uniqueness) is noisy at the margin:
    ~35% of pixels flip valid/invalid every frame even when the depth VALUE is
    steady, so the black holes "dance" and the view flickers. Holding each
    pixel's last valid value for up to ``hold`` frames removes that shimmer
    (verified on a real dump: mask-flip 35% -> ~5%). This is exactly what the
    OAK-D / RealSense temporal filter does.

    NOTE: this is for the *viewer* only. It must NOT feed VIO -- holding stale
    depth would lag the geometry. The matcher itself is untouched.
    """

    def __init__(self, hold: int = 3) -> None:
        self.hold = hold
        self._held: np.ndarray | None = None
        self._age: np.ndarray | None = None

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        v = depth > 0
        if self._held is None:
            self._held = depth.copy()
            self._age = np.where(v, 0, 999).astype(np.int32)
            return self._held
        self._age = np.where(v, 0, self._age + 1)
        self._held = np.where(v, depth, self._held)
        self._held[self._age > self.hold] = 0.0
        return self._held


def _check_engine() -> None:
    """Warn loudly when numba is missing -- pure-numpy SGM looks like a freeze."""
    if not HAVE_NUMBA:
        print("=" * 70)
        print("WARNING: numba NOT available -> pure-NumPy SGM (~seconds/frame).")
        print("The UI will look frozen. Run with the project venv instead:")
        print("    .venv/bin/python tools/stereo_view.py ...")
        print("=" * 70)


def _warmup(matcher: SGMStereoMatcher, h: int, w: int) -> None:
    """Trigger the one-time numba JIT compile BEFORE opening the GUI.

    The first ``dense_depth`` call compiles every parallel kernel, which can take
    tens of seconds. Doing it here (with a console message) keeps the window from
    appearing hung on the first real frame.
    """
    if not HAVE_NUMBA:
        return
    print("compiling SGM kernels (one-time JIT, ~10-40 s)...", flush=True)
    t0 = time.perf_counter()
    dummy = np.zeros((h, w), dtype=np.uint8)
    matcher.dense_depth(dummy, dummy)
    print(f"ready ({time.perf_counter() - t0:.1f} s)", flush=True)



def run_replay(session_dir: Path, cfg: SGMConfig, fps: float,
               hold: int = 3) -> int:
    reader = SessionReader(session_dir)
    if len(reader) == 0:
        print(f"no frames in {session_dir}")
        return 1
    matcher = SGMStereoMatcher.from_calib(reader.calib, cfg)
    fx, B = reader.K[0, 0], reader.calib.baseline_m
    print(f"session {reader.dir.name}: {len(reader)} frames, "
          f"baseline {B*100:.1f} cm, engine "
          f"{'numba' if HAVE_NUMBA else 'pure-numpy'}")
    _check_engine()
    _warmup(matcher, reader.calib.left.height, reader.calib.left.width)
    print("keys: SPACE pause | n step | q quit")

    win = "stereo_view  [ left | OURS (SGM) | chip ]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    i = 0
    paused = False
    period = 1.0 / max(fps, 1e-3)
    stab = TemporalHold(hold) if hold > 0 else None
    while 0 <= i < len(reader):
        t0 = time.perf_counter()
        f = reader.load_frame(i, load_right=True)
        ours = matcher.dense_depth(f.gray_left, f.gray_right)
        ms = (time.perf_counter() - t0) * 1e3

        # Compare metric depth where both are valid (chip 0.1-12 m).
        chip = f.depth_m
        both = (ours > 0) & (chip > 0.1) & (chip < 12.0)
        rel = (np.abs(ours[both] - chip[both]) / chip[both]
               if both.any() else np.array([0.0]))
        shown = stab(ours) if stab is not None else ours
        panel = np.hstack([
            _label(_gray_bgr(f.gray_left), "left"),
            _label(colorize_depth(shown),
                   f"OURS {ms:.0f}ms med{100*np.median(rel):.0f}%"),
            _label(colorize_depth(chip), "chip"),
        ])
        cv2.imshow(win, panel)

        wait = max(1, int((period - (time.perf_counter() - t0)) * 1000)) \
            if not paused else 0
        key = cv2.waitKey(0 if paused else wait) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if paused and key == ord("n"):
            i += 1
            continue
        if not paused:
            i += 1
    cv2.destroyAllWindows()
    return 0


def run_live(cfg: SGMConfig, width: int, height: int, fps: int,
             dump_dir: Path | None = None, dump_n: int = 40,
             hold: int = 3, mains_hz: int = 50,
             exposure_us: int = 10000, iso: int = 100,
             exit_after_dump: bool = False) -> int:
    import depthai as dai  # lazy: replay mode works without depthai

    _check_engine()
    left_socket = dai.CameraBoardSocket.CAM_B
    right_socket = dai.CameraBoardSocket.CAM_C
    win = "stereo_view LIVE  [ left | OURS (SGM) ]"

    with dai.Pipeline() as p:
        left = p.create(dai.node.Camera).build(left_socket, sensorFps=fps)
        right = p.create(dai.node.Camera).build(right_socket, sensorFps=fps)
        # Mains-frequency lamps (LED/fluorescent) flicker at 2x the line
        # frequency. With free-running auto-exposure the integration window
        # beats against that flicker -> the bright lamp pulses frame to frame.
        # Anti-banding pins the exposure time to a multiple of the flicker
        # period so it cancels; a fixed manual exposure removes it entirely.
        ab = {50: dai.CameraControl.AntiBandingMode.MAINS_50_HZ,
              60: dai.CameraControl.AntiBandingMode.MAINS_60_HZ}.get(
                  mains_hz, dai.CameraControl.AntiBandingMode.OFF)
        for cam in (left, right):
            cam.initialControl.setAntiBandingMode(ab)
            if exposure_us > 0:
                cam.initialControl.setManualExposure(int(exposure_us), int(iso))
        if exposure_us > 0:
            print(f"camera: manual exposure {exposure_us}us iso{iso}")
        else:
            print(f"camera: auto-exposure, anti-banding {mains_hz}Hz")

        # VPU-FREE preview (matches the live VIO source exactly): tap the two
        # RAW cameras and rectify BOTH ourselves + run our SGM. No chip
        # StereoDepth in the loop -> no "width must be multiple of 16" constraint
        # and the depth shown here is byte-identical to what the VIO consumes.
        left_out = left.requestOutput((width, height))
        right_out = right.requestOutput((width, height))
        q_left = left_out.createOutputQueue(maxSize=4, blocking=False)
        q_right = right_out.createOutputQueue(maxSize=4, blocking=False)

        # Chip depth is an OPTIONAL offline oracle, read ONLY when dumping. The
        # chip StereoDepth node requires the input width to be a multiple of 16,
        # so the dump path (and only it) inherits that constraint.
        q_depth = None
        if dump_dir is not None:
            if width % 16 != 0:
                print(f"ERROR: --dump needs the chip StereoDepth oracle, whose "
                      f"input width must be a multiple of 16 (got {width}). "
                      f"Re-run --dump with e.g. --width {((width + 15)//16)*16}.")
                return 2
            stereo = p.create(dai.node.StereoDepth)
            stereo.setExtendedDisparity(False)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(False)
            stereo.setDepthAlign(left_socket)
            left_out.link(stereo.left)
            right_out.link(stereo.right)
            q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
        p.start()

        ch = p.getDefaultDevice().readCalibration()

        # Build our matcher from the live calibration exactly like the VIO source
        # (oakd.sources.depthai_ours_vio): assemble the same JSON shape so
        # StereoCalib.from_json applies the identical cm->m extrinsic convention,
        # then ``rectify_left=True`` makes the matcher rectify BOTH raw frames.
        def _intr(sock):
            Ki = np.array(ch.getCameraIntrinsics(sock, width, height),
                          dtype=np.float64)
            dist = list(ch.getDistortionCoefficients(sock))
            return {"fx": float(Ki[0, 0]), "fy": float(Ki[1, 1]),
                    "cx": float(Ki[0, 2]), "cy": float(Ki[1, 2]),
                    "dist": [float(x) for x in dist],
                    "width": int(width), "height": int(height)}

        T_lr = np.array(ch.getCameraExtrinsics(left_socket, right_socket),
                        dtype=np.float64).reshape(4, 4)
        calib = StereoCalib.from_json({
            "intrinsics_left": _intr(left_socket),
            "intrinsics_right": _intr(right_socket),
            "T_left_right": T_lr.tolist(),
        })
        matcher = SGMStereoMatcher.from_calib(calib, cfg, rectify_left=True)
        print(f"live {width}x{height}@{fps}  engine "
              f"{'numba' if HAVE_NUMBA else 'pure-numpy'}  (VPU-free, own rectify)")
        print(f"[stereo_view] SGM cfg: ndisp={cfg.num_disparities} "
              f"downscale={cfg.downscale} -> compute width "
              f"{width // max(1, cfg.downscale)}px "
              f"(internal ndisp {cfg.num_disparities // max(1, cfg.downscale)})")
        _warmup(matcher, height, width)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.startWindowThread()  # macOS/Linux: keep the HighGUI event loop alive
        print("keys: q quit")

        dump_f = None
        dumped = 0
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_f = open(dump_dir / "stats.jsonl", "w")
            print(f"DUMP: saving first {dump_n} matched pairs to {dump_dir}")

        def _as_gray(msg):
            g = msg.getCvFrame()
            if g.ndim == 3:
                g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
            return g

        # The two RAW cameras arrive on independent queues, so they are rarely
        # both ready on the same poll AND a left frame may be paired with a
        # stale right (or vice-versa). Feeding such a temporally MISMATCHED pair
        # to the matcher makes the disparity jump every frame -> violent flicker.
        # Fix: buffer by sequence number and only compute on a MATCHED pair (same
        # seq); CAM_B/CAM_C are hardware frame-synced so equal seq = same capture.
        # Separately: pump the HighGUI loop (cv2.waitKey) EVERY iteration so the
        # macOS window never goes "Not Responding" while waiting for frames.
        pend_l: dict[int, np.ndarray] = {}
        pend_r: dict[int, np.ndarray] = {}
        pend_d: dict[int, np.ndarray] = {}
        shown_a_frame = False
        stab = TemporalHold(hold) if hold > 0 else None
        while p.isRunning():
            got = False
            while True:
                m = q_left.tryGet()
                if m is None:
                    break
                pend_l[m.getSequenceNum()] = _as_gray(m)
                got = True
            while True:
                m = q_right.tryGet()
                if m is None:
                    break
                pend_r[m.getSequenceNum()] = _as_gray(m)
                got = True
            if q_depth is not None:
                while True:
                    m = q_depth.tryGet()
                    if m is None:
                        break
                    pend_d[m.getSequenceNum()] = m.getFrame()  # uint16 mm

            common = pend_l.keys() & pend_r.keys()
            # When dumping we also want the chip depth for the SAME capture as an
            # oracle. Chip depth lags rectifiedLeft by ~1 frame, so require it in
            # the match too -- otherwise we'd pick a seq, miss its (not-yet-
            # arrived) depth, then prune it and never save any chip frame.
            if q_depth is not None:
                common = common & pend_d.keys()
            if common:
                seq = max(common)
                gl = pend_l[seq]
                gr = pend_r[seq]
                chip = pend_d.get(seq)  # grab BEFORE pruning below
                # drop this and any older buffered frames (we only show newest)
                pend_l = {k: v for k, v in pend_l.items() if k > seq}
                pend_r = {k: v for k, v in pend_r.items() if k > seq}
                pend_d = {k: v for k, v in pend_d.items() if k > seq}
                t0 = time.perf_counter()
                # Rectify BOTH raw frames ourselves and get depth on the
                # rectified-left grid (exactly what the VIO frontend tracks on).
                rect_left, ours = matcher.dense_depth_rectified_left(gl, gr)
                ms = (time.perf_counter() - t0) * 1e3
                shown = stab(ours) if stab is not None else ours
                # Display the RECTIFIED left (uint8), so the image and the depth
                # share the same grid the VIO uses. Bilinear remap can land just
                # outside [0,255], so clip before the cast.
                disp_left = np.clip(rect_left, 0, 255).astype(np.uint8)
                # Diagnostics: exposure (mean/std of left) tells us if the camera
                # is auto-exposure hunting (alternating bright/dark -> flicker);
                # valid% tells us how much depth survived this frame.
                lmean = float(disp_left.mean()); lstd = float(disp_left.std())
                vr = float((ours > 0).mean()) * 100.0
                left_bgr = _label(_gray_bgr(disp_left), "left (rect)")
                ds, dthick, dmx, _ = _text_style(width, ref=0.5, ref_thick=1)
                cv2.putText(left_bgr,
                            f"seq{seq} exp{lmean:.0f}/{lstd:.0f} val{vr:.0f}%",
                            (dmx, height - max(3, round(12 * width / 640.0))),
                            cv2.FONT_HERSHEY_SIMPLEX, ds,
                            (0, 255, 0), dthick, cv2.LINE_AA)
                panel = np.hstack([
                    left_bgr,
                    _label(colorize_depth(shown), f"OURS {ms:.0f}ms"),
                ])
                cv2.imshow(win, panel)
                shown_a_frame = True
                if dump_f is not None and dumped < dump_n:
                    # L = rectified left (gold convention), R = raw right.
                    cv2.imwrite(str(dump_dir / f"{dumped:03d}_L.png"), disp_left)
                    cv2.imwrite(str(dump_dir / f"{dumped:03d}_R.png"), gr)
                    np.save(dump_dir / f"{dumped:03d}_depth.npy",
                            ours.astype(np.float32))
                    if chip is not None:
                        np.save(dump_dir / f"{dumped:03d}_chip.npy",
                                (chip.astype(np.float32) * 1e-3))  # mm -> m
                    dump_f.write(
                        f'{{"i":{dumped},"seq":{seq},"lmean":{lmean:.2f},'
                        f'"lstd":{lstd:.2f},"valid_pct":{vr:.2f},'
                        f'"ms":{ms:.1f}}}\n')
                    dump_f.flush()
                    dumped += 1
                    if dumped >= dump_n:
                        print(f"DUMP complete: {dumped} frames in {dump_dir}")
                        if exit_after_dump:
                            break
            elif not shown_a_frame:
                # Paint the placeholder ONLY before the first real frame. After a
                # frame has been shown, NEVER repaint black -- otherwise every
                # poll without a fresh pair overwrites the live image with the
                # "waiting" screen, producing a real/black/real/black flicker.
                # Leaving the window untouched keeps the last frame on screen.
                placeholder = np.zeros((height, width * 2, 3), dtype=np.uint8)
                cv2.imshow(win, _label(placeholder, "waiting for camera..."))

            # Always pump the GUI + read keys, every iteration.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if not got:
                time.sleep(0.002)
        if dump_f is not None:
            dump_f.close()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="sessions/gold/lab_straight_20s",
                    help="gold session to replay (ignored with --live)")
    ap.add_argument("--live", action="store_true",
                    help="pull from a connected OAK-D instead of replaying")
    ap.add_argument("--fast", action="store_true",
                    help="use the live SGM preset (half-res, 4-path) -- faster")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="replay/live frame rate cap [15]")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--num-disparities", type=int, default=None,
                    dest="num_disparities",
                    help="(live) SGM disparity search range px; default None = "
                         "auto-scale from the 640x400 baseline (96*width/640). "
                         "Must stay below the internal compute width "
                         "(width/downscale) or near depth is lost")
    ap.add_argument("--dump", default=None,
                    help="(live) save raw L/R + depth + stats to this dir for "
                         "offline analysis of the flicker")
    ap.add_argument("--dump-n", type=int, default=40,
                    help="number of matched pairs to dump [40]")
    ap.add_argument("--exit-after-dump", action="store_true",
                    help="(live) quit automatically once the dump is complete")
    ap.add_argument("--hold", type=int, default=3,
                    help="temporal-hold frames to de-flicker the depth view "
                         "(display only, not used for VIO) [3]")
    ap.add_argument("--no-stabilize", action="store_true",
                    help="disable temporal hold (show raw per-frame depth)")
    ap.add_argument("--mains-hz", type=int, default=50, choices=[0, 50, 60],
                    help="(live) anti-banding for mains-lamp flicker; 0=off, "
                         "50 for VN/EU, 60 for US [50]")
    ap.add_argument("--exposure", type=int, default=10000,
                    help="(live) lock manual exposure in microseconds; use a "
                         "multiple of 10000 (=1 mains cycle @50Hz) to cancel "
                         "lamp flicker; 0 = auto-exposure [10000]")
    ap.add_argument("--iso", type=int, default=100,
                    help="(live) ISO for manual exposure [100]")
    args = ap.parse_args()

    hold = 0 if args.no_stabilize else args.hold
    if args.live:
        # Match the EXACT SGM config the live VIO source would use at this
        # resolution: ResolutionProfile scales num_disparities (and the live
        # preset's census/paths) from the 640x400 baseline, so the depth you
        # preview here is what the VIO actually consumes. (Replay keeps the flat
        # preset since the recorded frames are always 640x400.)
        prof = ResolutionProfile.for_resolution(args.width, args.height,
                                                num_disparities=args.num_disparities)
        cfg = prof.sgm(fast=args.fast)
        print(f"[stereo_view] resolution profile: {prof.describe()}")
        dump_dir = Path(args.dump) if args.dump else None
        return run_live(cfg, args.width, args.height, int(args.fps),
                        dump_dir, args.dump_n, hold, args.mains_hz,
                        args.exposure, args.iso, args.exit_after_dump)
    cfg = SGMConfig.live() if args.fast else SGMConfig()
    return run_replay(Path(args.session), cfg, args.fps, hold)


if __name__ == "__main__":
    raise SystemExit(main())
