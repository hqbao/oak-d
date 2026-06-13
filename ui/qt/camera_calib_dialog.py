"""Qt stereo camera-calibration WIZARD (Phase 4 -- the operator-driven flow).

Ties Phases 1-3 together into ONE operator tool: show a checkerboard (Phase 1),
capture diverse stereo views off an injected RAW stereo stream (Phase 2) while a
tested collector gates the dataset (Phase 3 ``collector``), solve the intrinsics +
extrinsic OFF the UI thread (Phase 3 ``solve``), grade the result with the shipped
``calib_check`` suite, and persist a reader-compatible ``calib.json`` (Phase 3
``writer``). The HEAVY logic lives in :mod:`ui.calib` + the worker; this
dialog is thin glue + rendering, so the calibration math stays unit-tested there.

Stream contract (device-free) -- modelled on :mod:`ui.qt.calib_dialogs`
----------------------------------------------------------------------
The wizard drives ANY object exposing ``start(callback)`` / ``stop()`` / ``.error``
/ ``.device_id``. In proc4 the caller injects
:class:`ui.modules.ipc_sources.IpcStereoRawSource` (capture's RAW ``imucam.sample``
left+right over IPC) -- the UI never opens a device. The stream calls
:meth:`_feed_pair` on its BACKGROUND thread; that method only appends to a
thread-safe queue. A UI-thread ``QTimer`` (:meth:`_drain`) then renders the latest
frame and refreshes the bars, so ALL Qt access stays on one thread (no locks, no
cross-thread Qt). This split is also what makes the wizard offline-unit-testable: a
test calls :meth:`_feed_pair` with synthetic frames and ticks :meth:`_drain`
directly, no device required (see ``ui/tests/camera_calib_dialog_selftest.py``).

Threading -- why the UI thread NEVER runs cv2 detection
-------------------------------------------------------
``collector.feed`` -> ``detect_corners`` -> ``cv2.findChessboardCorners`` is SLOW
(~100-300 ms) on a board-LESS frame, which is the normal case while the operator is
still aiming. Running that on the UI thread (and looping it over the whole queue
backlog) saturates the GUI thread: the preview never repaints, the status freezes,
and even the watchdog ``.error`` is never re-read. So detection runs OFF the UI
thread on a dedicated :class:`_DetectWorker` (mirroring :class:`_SolveWorker`):

  * :meth:`_drain` (UI, ~30 Hz) **drains the queue to the NEWEST pair** (discarding
    stale frames -- a live preview only needs the most recent), paints that gray
    cheaply EVERY tick, re-syncs the bars from the LAST detection result, polls the
    watchdog, and returns promptly -- it never runs cv2.
  * at most one detection is in flight; :meth:`_drain` hands the worker the latest
    pair only when none is running, so a slow board-less detect can never stall the
    UI and detection is naturally throttled to "as fast as it can keep up".
  * the worker emits the :class:`FrameStatus` + the left corners back to the UI
    thread via a queued signal; :meth:`_on_detected` feeds the collector (the only
    place ``collector`` is mutated) and arms the next detection.

cv2 POLICY -- the module import must NOT load OpenCV
---------------------------------------------------
The flight runtime is cv2-free and stays so. cv2 is pulled in ONLY transitively, at
CAPTURE time (``collector.feed`` -> ``detect_corners``) and SOLVE time
(``solve_stereo``), both of which lazy-import cv2 themselves. This module therefore
imports NEITHER cv2 NOR :mod:`ui.calib` at top level -- the calib helpers
are imported lazily inside the methods that first need them, so merely importing the
dialog module (e.g. when ``ui.main`` builds its menu) loads no OpenCV.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QSpinBox, QDoubleSpinBox, QVBoxLayout, QWidget,
)

from . import theme

# Marker colours drawn over the live preview (RGB tuples). Found corners are NVG
# green; a "not found this frame" preview gets no markers (the status says why).
_CORNER_RGB = (124, 255, 92)        # theme.GOOD as RGB -- detected corners
_PREVIEW_W = 480                    # preview label target width (keeps it light)

# B1 -- progress-bar chunk stylesheets: a gate reads GREEN (theme.GOOD) only once
# it is genuinely satisfied, AMBER (theme.WARN) while still needed. Applied to the
# ::chunk subcontrol so only the fill colour changes (the bar frame is unaffected).
_BAR_GOOD = f"QProgressBar::chunk {{ background-color: {theme.GOOD}; }}"
_BAR_WARN = f"QProgressBar::chunk {{ background-color: {theme.WARN}; }}"
# Tilt bar visible fill is capped below full until skew_ok, so an incomplete tilt
# gate can never read as a finished ~100% bar (the "hang" the operator misreads).
_TILT_CAP_PCT = 80
# B2 -- after a frame is accepted, hold a green "view banked" confirmation on the
# preview border for this many UI ticks (~33 ms each) so it survives long enough to
# be seen (a single-frame ~33 ms green flash at 30 Hz is unmissable-by-design).
_ACCEPT_HOLD_TICKS = 12             # ~12 * 33 ms ~= 400 ms
# Preview border stylesheets for the banked-confirmation flash (B2): green on the
# accept edge, the neutral panel edge on the revert edge.
_PREVIEW_QSS_BANKED = (
    f"QLabel#Preview {{ border: 2px solid {theme.GOOD}; border-radius: 4px; }}")
_PREVIEW_QSS_IDLE = (
    f"QLabel#Preview {{ border: 1px solid {theme.PANEL_EDGE}; border-radius: 4px; }}")

# H3 -- sane bounds for severity-colouring the two key results metrics. These are
# soft hints (the calib_check verdict line owns the real PASS/WARN/FAIL call): a
# good stereo solve lands well under ~0.5 px RMS, and an OAK-D baseline is ~75 mm,
# so a measured baseline outside a generous 50-100 mm stereo band is flagged.
_RMS_GOOD_PX = 0.5
_BASELINE_GOOD_MM = (50.0, 100.0)   # in-band -> GOOD; outside -> BAD
_BASELINE_EXPECT_MM = 75.0          # echoed "expect ~75 for OAK-D" hint

# Liveness telemetry: detections/sec is averaged over a short trailing window so the
# in-dialog readout is steady (a single slow board-less detect would otherwise make
# an instantaneous rate jump around). 2 s balances responsiveness vs. jitter.
_LIVENESS_WINDOW_S = 2.0


class _DetectWorker(QThread):
    """Run the SLOW per-frame board detection (``collector.feed``) OFF the UI thread.

    ``collector.feed`` -> ``detect_corners`` -> ``cv2.findChessboardCorners`` can take
    100-300 ms on a board-less frame (the normal case while aiming), x2 (L+R). Doing
    that on the GUI thread freezes the wizard, so -- exactly like :class:`_SolveWorker`
    runs the solve off-thread -- this worker runs ONE detection on a background thread
    and emits the resulting :class:`~sky.calib.collector.FrameStatus` plus the
    detected LEFT corners (for the overlay) back to a UI-thread slot via a queued
    signal. The UI thread keeps exactly one detection in flight (see
    :meth:`CameraCalibWizard._maybe_detect`), so detection is self-throttling and can
    never stall the preview.

    The worker does NOT touch Qt or the collector's mutation path beyond calling
    ``feed`` (which only appends to the collector's accepted list): the dialog runs at
    most one worker at a time and never reads the collector concurrently, so there is
    no shared-state race -- the collector is only ever fed from one worker, serialised
    by the "one in flight" rule on the UI thread.
    """

    #: Emitted with ``(status, left_corners | None)`` for the pair that was detected.
    done = pyqtSignal(object, object)

    def __init__(self, collector, gray_left: np.ndarray, gray_right: np.ndarray,
                 parent=None) -> None:
        super().__init__(parent)
        self._coll = collector
        # Snapshot the grays so the worker never aliases the live queue's arrays.
        self._left = gray_left
        self._right = gray_right

    def run(self) -> None:                                            # noqa: D102
        # Lazy import inside the worker thread: keeps cv2 off the dialog's import
        # path AND off the GUI thread (detection + solve are the only cv2 users).
        from sky.calib.detect import detect_corners
        try:
            status = self._coll.feed(self._left, self._right)
            # Re-detect the LEFT corners for the overlay ONLY when this frame found
            # them, so the preview can draw the green lock markers. This second
            # detect is cheap on a board-FULL frame (the slow path is board-less,
            # which returns no corners and skips this) and stays off the UI thread.
            corners = None
            if status.found_left:
                corners = detect_corners(self._left, self._coll.pattern_cols,
                                         self._coll.pattern_rows)
        except Exception:                                            # noqa: BLE001
            # A detection error must not kill the worker thread silently or wedge the
            # pump: report an empty result so the UI re-arms the next detection.
            self.done.emit(None, None)
            return
        self.done.emit(status, corners)


class _SolveWorker(QThread):
    """Run the (blocking ~1 s) stereo solve OFF the UI thread.

    ``cv2.calibrateCamera`` / ``stereoCalibrate`` can block for ~1 s; running them
    on the GUI thread would freeze the wizard. This worker takes the collected
    views + board geometry, runs :func:`sky.calib.solve.solve_stereo`
    (which lazy-imports cv2), and emits exactly one of :attr:`done` /
    :attr:`failed`. The dialog connects both to UI-thread slots, so the result is
    marshalled back safely by Qt's queued-connection across the thread boundary.
    """

    #: Emitted with the solved :class:`StereoCalibResult` on success.
    done = pyqtSignal(object)
    #: Emitted with a human message on any solve exception.
    failed = pyqtSignal(str)

    def __init__(self, views, cols: int, rows: int, square_size_m: float,
                 image_size: tuple[int, int], dump_path=None, parent=None) -> None:
        super().__init__(parent)
        # Snapshot the inputs so the worker never touches live collector state.
        self._views = views
        self._cols = int(cols)
        self._rows = int(rows)
        self._square_m = float(square_size_m)
        self._image_size = (int(image_size[0]), int(image_size[1]))
        # Always-on debug dump of the REAL captured corners (a calibration is
        # infrequent): lets the operator send us one file to reproduce a failure.
        self._dump_path = dump_path

    def run(self) -> None:                                            # noqa: D102
        # Lazy import inside the worker thread: keeps cv2 off the dialog's import
        # path AND off the GUI thread (the solve is the only cv2 user here).
        from sky.calib.solve import solve_stereo
        try:
            result = solve_stereo(self._views, self._cols, self._rows,
                                  self._square_m, self._image_size,
                                  dump_path=self._dump_path)
        except Exception as exc:                                     # noqa: BLE001
            # Any solve failure (degenerate views, cv2 error) is surfaced to the
            # operator rather than crashing the worker silently.
            self.failed.emit(f"solve failed: {exc}")
            return
        self.done.emit(result)


class _BoardWindow(QLabel):
    """Resizable checkerboard target window (no nested event loop).

    The operator points the OAK-D at this (or drags it to a second screen). It is
    shown NON-modally from the already-running wizard -- the main Qt event loop
    drives it, so there is NO ``app.exec()``. The wizard keeps a STRONG reference
    (``_board_win``) so the window is not garbage-collected. The previous
    ``checkerboard._show_fullscreen`` ran a NESTED ``app.exec()`` AND let its local
    QLabel be collected, so inside the running app the board flashed on-screen and
    vanished before the operator could see it (the first reported bug).

    It is a NORMAL, RESIZABLE, movable, minimisable window -- NOT forced
    fullscreen (the second reported bug: the operator could not shrink it). Resize
    it, drag it to another monitor, or use the OS controls to maximise/fullscreen.
    The board RESCALES to the window PRESERVING the square aspect ratio (a stretched
    board would bias the recovered focal lengths). Esc / double-click closes it.
    """

    def __init__(self, src_pixmap: QPixmap) -> None:
        super().__init__(None)                       # top-level resizable window
        self._src = src_pixmap                        # full-res source; rescaled on resize
        self.setWindowTitle(
            "Calibration checkerboard -- resize / drag freely · Esc to close")
        self.setStyleSheet("background-color: #000000;")
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self._rescale()
        QShortcut(QKeySequence(QtCore.Qt.Key.Key_Escape), self,
                  activated=self.close)

    def _rescale(self) -> None:
        """Fit the board into the current window size WITHOUT stretching it."""
        sz = self.size()
        if sz.width() > 1 and sz.height() > 1 and not self._src.isNull():
            self.setPixmap(self._src.scaled(
                sz, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:             # noqa: N802, D102
        self._rescale()
        super().resizeEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:   # noqa: N802, D102
        self.close()


class CameraCalibWizard(QDialog):
    """Operator wizard: show a board, capture diverse stereo views, solve, save.

    Phases composed
    ---------------
    * board inputs (``cols`` / ``rows`` inner corners + the REAL ``square_mm``)
      and a "Show checkerboard" button (Phase 1 generator + fullscreen ``--show``),
    * live capture off the injected RAW stereo stream -> a
      :class:`~sky.calib.collector.StereoCheckerboardCollector` with a live
      left-frame preview + corner overlay + per-axis coverage + operator guidance,
    * an OFF-thread solve (:class:`_SolveWorker`) reporting per-camera + stereo
      reprojection RMS, the baseline (mm), a K summary, and the shipped
      ``calib_check`` PASS / WARN / FAIL verdict,
    * a "Save calib.json…" button (:func:`sky.calib.writer.write_calib_json`)
      that WARNS the operator if ``calib_check`` did not PASS.
    """

    def __init__(self, parent=None, *, device_id: str | None = None,
                 width: int = 640, height: int = 400,
                 stream: object | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(theme.QSS)
        self.setWindowTitle("Calibrate Camera · Stereo")
        self.setMinimumWidth(620)

        # The wizard drives ANY object with start()/stop()/.error/.device_id. In
        # proc4 the caller ALWAYS injects IpcStereoRawSource -- there is no
        # in-process default to fall back to, so a None stream is a programming
        # error; surface it clearly rather than silently opening nothing.
        if stream is None:
            raise ValueError(
                "camera calib wizard needs an injected `stream` in the "
                "device-free proc4 UI (e.g. IpcStereoRawSource).")
        self._stream = stream
        self._device_id = device_id
        # (width, height) -- cv2 convention; the calib JSON resolution + collector
        # image_size both come from here (the capture resolution in calib.bundle).
        self._image_size = (int(width), int(height))

        # Thread-safe hand-off: the stream's recv thread only appends here; the
        # UI-thread _drain pops and feeds the collector (mirrors the IMU dialogs).
        self._queue: deque = deque(maxlen=256)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)               # ~30 Hz UI drain
        self._timer.timeout.connect(self._drain)
        self._running = False

        # Built lazily on START (needs the board geometry the operator enters).
        self._coll = None
        self._last_status = None
        self._last_left = None                    # last left gray for the preview
        self._last_corners = None                 # last detected LEFT corners (overlay)
        self._preview_buf = None                  # keep QImage backing alive
        self._result = None                       # solved StereoCalibResult
        self._verdict = None                      # ("PASS"|"WARN"|"FAIL", text)
        self._dump_path: str | None = None        # last solve's debug-dump .npz path
        self._solver: _SolveWorker | None = None
        # OFF-thread detection: at most ONE _DetectWorker runs at a time. _drain only
        # arms a new one when this is None (or finished), which both throttles
        # detection and guarantees the collector is fed from a single worker at a
        # time (no concurrent feed). Cleared in _on_detected and torn down on close.
        self._detector: _DetectWorker | None = None

        # Liveness telemetry surfaced IN the dialog (the UI process's logs do not
        # reach the launcher terminal): total frames received off the stream, plus a
        # trailing window of detection-completion timestamps for a detections/sec
        # readout. So if the stream is dead or detection is stalling, the OPERATOR
        # sees it in the status line rather than staring at a frozen preview.
        self._frames_received = 0
        self._detect_times: deque = deque()       # monotonic ts of recent detections

        # B2 -- "view banked" confirmation latch: counts down UI ticks after an
        # accept edge so the green preview border survives ~400 ms. 0 = idle (border
        # neutral). Styles are only touched on the accept edge and the revert edge,
        # never every frame, so this adds no per-tick stylesheet churn.
        self._accept_hold = 0

        # Retained reference to the fullscreen checkerboard window (None = closed).
        # MUST be kept on the instance: a local-only window is GC'd and flashes shut.
        self._board_win: _BoardWindow | None = None

        self._build_ui()
        self._set_phase_idle()

    # -- UI construction --------------------------------------------------- #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        title = QLabel("STEREO CAMERA CALIBRATION")
        title.setObjectName("PanelTitle")
        root.addWidget(title)
        hint = QLabel(
            "Enter the board geometry and the REAL printed/on-screen square size, "
            "show the board, then press START and sweep it across the view -- "
            "near/far, corner to corner, and TILTED (not flat-on).")
        hint.setObjectName("DialogHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # --- Board inputs: cols/rows inner corners + the real square size (mm). ---
        form = QGridLayout()
        form.addWidget(QLabel("Inner corners (cols):"), 0, 0)
        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(2, 30)
        self._cols_spin.setValue(9)
        form.addWidget(self._cols_spin, 0, 1)
        form.addWidget(QLabel("Inner corners (rows):"), 0, 2)
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(2, 30)
        self._rows_spin.setValue(6)
        form.addWidget(self._rows_spin, 0, 3)
        form.addWidget(QLabel("Square size (mm):"), 1, 0)
        self._square_spin = QDoubleSpinBox()
        self._square_spin.setRange(1.0, 500.0)
        self._square_spin.setDecimals(2)
        self._square_spin.setValue(25.0)
        self._square_spin.setSingleStep(0.5)
        # Make the unit explicit IN the field (B3): the #1 operator error is entering
        # the wrong physical square size, which silently scales the whole calib.
        self._square_spin.setSuffix(" mm")
        form.addWidget(self._square_spin, 1, 1)
        self._show_btn = QPushButton("Show checkerboard")
        self._show_btn.clicked.connect(self._on_show_board)
        # H1: warn that fullscreen-board covers the WHOLE screen on a single monitor.
        self._show_btn.setToolTip(
            "Opens the board in a RESIZABLE window — shrink it, drag it to a second "
            "monitor, or print the board instead. Press Esc to close it.")
        form.addWidget(self._show_btn, 1, 2, 1, 2)
        form_wrap = QWidget()
        form_wrap.setLayout(form)
        root.addWidget(form_wrap)

        # PERMANENT board-params caption (B3 + H1): always visible, unlike the
        # transient status line. Stresses that square_mm is the REAL measured size
        # and that "Show checkerboard" needs a second monitor / printed board.
        board_caption = QLabel(
            "Square size = the REAL size: printed at 100%, or MEASURED on-screen "
            "with a ruler.  ·  “Show checkerboard” opens a RESIZABLE window — "
            "shrink/drag it to a second monitor or print it; Esc closes it.")
        board_caption.setObjectName("DialogHint")
        board_caption.setWordWrap(True)
        root.addWidget(board_caption)

        # --- Live preview (left frame + corner overlay). ---
        self._preview = QLabel()
        self._preview.setObjectName("Preview")
        self._preview.setMinimumSize(_PREVIEW_W, int(_PREVIEW_W * 3 / 4))
        self._preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._preview.setText("(live preview appears here once you press START)")
        # Neutral idle border; flips green for ~400 ms after each accept (B2).
        self._preview.setStyleSheet(_PREVIEW_QSS_IDLE)
        root.addWidget(self._preview)

        # --- Capture status: count bar + tilt-coverage bar + guidance line. ---
        # Count bar shows RAW counts ("7/15"), not a bare percentage (B1): the
        # range is rebound to 0..n_target on START (the collector owns n_target).
        self._count_bar = QProgressBar()
        self._count_bar.setRange(0, 1)            # placeholder until START rebinds
        self._count_bar.setFormat("views %v/%m")
        root.addWidget(self._count_bar)
        # Tilt bar reads as INCOMPLETE until skew_ok is genuinely True (B1): the
        # fill is capped below full and labelled "need more" while still required,
        # so one mildly-tilted view can't read as a finished (near-100%) hang.
        self._tilt_bar = QProgressBar()
        self._tilt_bar.setRange(0, 100)
        self._tilt_bar.setFormat("tilt: need more")
        root.addWidget(self._tilt_bar)
        self._status = QLabel("Idle.")
        self._status.setObjectName("DialogHint")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        # Always-visible stream-liveness readout (connection + frames received +
        # detections/sec). Lets the operator SEE a dead stream or a stalled detect
        # from inside the dialog -- the UI process's stdout never reaches them.
        self._liveness_lbl = QLabel("stream: idle")
        self._liveness_lbl.setObjectName("DialogMono")
        self._liveness_lbl.setWordWrap(True)
        root.addWidget(self._liveness_lbl)

        # --- Result line (RMS / baseline / K) + calib_check verdict. ---
        self._result_lbl = QLabel("result = —")
        self._result_lbl.setObjectName("DialogMono")
        self._result_lbl.setWordWrap(True)
        root.addWidget(self._result_lbl)
        self._verdict_lbl = QLabel("")
        self._verdict_lbl.setObjectName("DialogMono")
        self._verdict_lbl.setWordWrap(True)
        root.addWidget(self._verdict_lbl)

        # --- Buttons: START / SAVE / CLOSE. ---
        btns = QHBoxLayout()
        self._start_btn = QPushButton("START")
        self._start_btn.clicked.connect(self._on_start)
        self._save_btn = QPushButton("Save calib.json…")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        close_btn = QPushButton("CLOSE")
        close_btn.clicked.connect(self.close)
        btns.addWidget(self._start_btn)
        btns.addWidget(self._save_btn)
        btns.addStretch(1)
        btns.addWidget(close_btn)
        root.addLayout(btns)

    # -- phase helpers ----------------------------------------------------- #
    def _set_phase_idle(self) -> None:
        """Inputs enabled, capture/solve idle."""
        for w in (self._cols_spin, self._rows_spin, self._square_spin):
            w.setEnabled(True)

    # -- board show (Phase 1) ---------------------------------------------- #
    def _on_show_board(self) -> None:
        """Show the checkerboard fullscreen for the shine-on-screen workflow.

        Builds the Phase-1 board, scales it to the wizard's screen PRESERVING the
        square aspect ratio (a stretched board would bias the recovered focal
        lengths), and shows it in a retained, NON-modal :class:`_BoardWindow`. This
        replaces the old ``checkerboard._show_fullscreen`` path, whose nested
        ``app.exec()`` + dropped widget reference made the board flash and close
        instantly inside the running wizard. Dismiss with Esc / double-click.
        cv2-free: the generator pulls no OpenCV.
        """
        from ui.calib.checkerboard import make_checkerboard

        cols = int(self._cols_spin.value())
        rows = int(self._rows_spin.value())
        img = np.ascontiguousarray(
            make_checkerboard(cols, rows, square_px=100, margin_squares=1.0))
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)        # copies pixels -> `img` lifetime is moot
        self._close_board_window()           # replace any prior board cleanly
        # A RESIZABLE window (the _BoardWindow rescales the board to fit, aspect
        # preserved) -- NOT forced fullscreen, so the operator can shrink/move it.
        self._board_win = _BoardWindow(pix)  # STRONG ref -> no GC flash (the bug)
        screen = self.screen()
        if screen is not None:
            g = screen.availableGeometry()
            self._board_win.resize(int(g.width() * 0.7), int(g.height() * 0.7))
            self._board_win.move(
                g.center() - self._board_win.rect().center())
        self._board_win.show()
        self._status.setText(
            "Board shown in a resizable window -- shrink/drag it (or maximise it on "
            "a second screen), point the OAK-D at it, then sweep + tilt. Press Esc "
            "or double-click the board to close it. MEASURE one on-screen square "
            "with a ruler and enter that as Square size (mm).")

    def _close_board_window(self) -> None:
        """Close + release the fullscreen board window (idempotent)."""
        win = self._board_win
        self._board_win = None
        if win is not None:
            try:
                win.close()
            except Exception:                                    # noqa: BLE001
                pass

    # -- capture lifecycle ------------------------------------------------- #
    def _on_start(self) -> None:
        """(Re)build the collector from the entered geometry and start capturing."""
        # Lazy import: collector pulls detect -> cv2 only when first FED, but we
        # import the class here (cheap, cv2-free) to build it.
        from sky.calib.collector import StereoCheckerboardCollector

        cols = int(self._cols_spin.value())
        rows = int(self._rows_spin.value())
        self._coll = StereoCheckerboardCollector(cols, rows, self._image_size)
        self._last_status = None
        self._last_left = None
        self._last_corners = None
        self._result = None
        self._verdict = None
        # Reset liveness telemetry for the new run.
        self._frames_received = 0
        self._detect_times.clear()
        self._save_btn.setEnabled(False)
        self._result_lbl.setText("result = —")
        self._verdict_lbl.setText("")
        self._start_btn.setText("RESTART")
        # B1: rebind the count bar to the real target so it reads "0/15", not "0%".
        # (n_target lives on the collector's config; tests may override it post-start,
        # but _refresh re-syncs the max from the live status every tick.)
        self._count_bar.setRange(0, max(1, self._coll.cfg.n_target))
        self._count_bar.setValue(0)
        self._count_bar.setStyleSheet(_BAR_WARN)
        self._tilt_bar.setValue(0)
        self._tilt_bar.setFormat("tilt: need more")
        self._tilt_bar.setStyleSheet(_BAR_WARN)
        # Clear any lingering banked-confirmation flash from a previous run.
        self._accept_hold = 0
        self._preview.setStyleSheet(_PREVIEW_QSS_IDLE)
        # Lock the geometry inputs while a dataset is being collected against them.
        for w in (self._cols_spin, self._rows_spin, self._square_spin):
            w.setEnabled(False)
        self._status.setText("Capturing… point the board at BOTH cameras.")
        self._start_stream()

    def _start_stream(self) -> None:
        if self._running:
            return
        self._queue.clear()
        self._stream.start(self._feed_pair)
        self._timer.start()
        self._running = True

    def _stop_stream(self) -> None:
        self._timer.stop()
        # The wizard does NOT own the injected stream (the caller's `finally`
        # stops it), mirroring the IMU dialogs' `_owns_stream=False` contract.
        self._running = False
        # Wait out any in-flight detection so no worker thread outlives the capture
        # (it touches no UI; this is a brief, bounded wait -- mirrors the solver
        # teardown). Its queued `done` may still fire harmlessly into _on_detected.
        detector = self._detector
        if detector is not None and detector.isRunning():
            detector.wait(2000)

    def _feed_pair(self, seq, ts_ns, gray_left, gray_right) -> None:
        """Stream recv thread: queue ONLY -- no collector / Qt access here.

        Matches :class:`ui.modules.ipc_sources.IpcStereoRawSource`'s callback
        signature ``(seq, ts_ns, gray_left, gray_right)``. We copy nothing: the
        source already ``read_copy``-ed both grays out of shared memory, so the
        record owns its arrays.
        """
        # Count for the in-dialog liveness readout. Plain int increment is atomic
        # enough here (single writer thread, UI thread only reads), and the queue's
        # ``maxlen`` already bounds memory if the UI falls behind (stale frames drop).
        self._frames_received += 1
        self._queue.append((np.asarray(gray_left), np.asarray(gray_right)))

    def _drain(self) -> None:
        """UI thread (~30 Hz): poll the watchdog, paint the latest gray, arm detect.

        This MUST return promptly -- it never runs cv2. It (1) surfaces any stream
        ``.error`` (watchdog / mono-guard), (2) drains the queue to the NEWEST pair
        and discards the stale backlog (a live preview only needs the most recent
        frame -- this is what stops the old unbounded ``while`` loop from saturating
        the UI thread), (3) paints that gray + the last detection overlay cheaply,
        (4) re-syncs the bars from the last detection status, and (5) hands the latest
        pair to the OFF-thread :class:`_DetectWorker` IFF none is already running (so
        at most one detect is in flight -- self-throttling, never blocking).
        """
        # Mono-guard / connect failures land on the stream's .error; surface and
        # stop, exactly like the IMU dialogs poll their stream each tick. Because
        # _drain now returns promptly (no cv2 loop), the watchdog .error is actually
        # re-read every tick and surfaced -- the freeze hid this path entirely.
        if self._stream.error:
            self._on_error(self._stream.error)
            return
        if self._coll is None:
            return

        # (2) Drain-to-latest: keep ONLY the newest pair, drop the stale backlog.
        latest = None
        while self._queue:
            latest = self._queue.popleft()
        if latest is not None:
            self._last_left = latest[0]

        # (3)+(4) Cheap, unconditional repaint + bar sync -- every tick, regardless
        # of whether a detection is running or succeeding. This is the smooth live
        # view the operator aims with; it is NEVER gated on detection.
        self._refresh()

        # (5) Arm the off-thread detection on the latest pair if the worker is free.
        if latest is not None:
            self._maybe_detect(latest[0], latest[1])

    def _maybe_detect(self, gray_left: np.ndarray, gray_right: np.ndarray) -> None:
        """Start ONE off-thread detection if none is in flight (self-throttling).

        Keeping at most one :class:`_DetectWorker` alive means a slow board-less
        detect can never queue up behind another, and the collector is only ever fed
        from a single worker at a time -- so there is no concurrent ``feed`` and no
        lock is needed. While a detect runs, newer frames are simply previewed (and
        the freshest one is picked up the next time the worker frees up).
        """
        if self._coll is None or self._coll.complete:
            return
        if self._detector is not None and self._detector.isRunning():
            return
        self._detector = _DetectWorker(self._coll, gray_left, gray_right,
                                       parent=self)
        self._detector.done.connect(self._on_detected)
        self._detector.start()

    def _on_detected(self, status, corners) -> None:
        """UI-thread slot: the off-thread detection of one pair finished.

        Records the status (so the bars + guidance update), stamps the liveness
        detections/sec window, caches the overlay corners, and -- if the dataset is
        now complete -- stops capture and kicks off the OFF-thread solve. A ``None``
        status means the worker hit an error mid-detect; we just drop it and let the
        next ``_drain`` tick re-arm, so a transient detection failure never wedges
        the pump.
        """
        # The worker has finished; let it be GC'd and free the slot for the next one.
        self._detector = None
        if status is None or self._coll is None:
            return
        self._last_status = status
        self._last_corners = corners
        # Liveness: log this detection's completion time and trim the trailing window.
        now = time.monotonic()
        self._detect_times.append(now)
        while self._detect_times and now - self._detect_times[0] > _LIVENESS_WINDOW_S:
            self._detect_times.popleft()
        # Repaint immediately so the overlay + bars reflect this detection at once
        # (rather than waiting up to one timer interval for the next _drain).
        self._refresh()
        if self._coll.complete:
            # Stop capturing the instant the dataset is good, then solve.
            self._stop_stream()
            self._begin_solve()

    # -- live rendering ---------------------------------------------------- #
    def _refresh(self) -> None:
        """Repaint the preview + status bars from the latest frame / status.

        Cheap by construction: the LEFT gray blit is a numpy->QImage copy, and the
        corner overlay reuses the LAST off-thread detection's corners
        (``self._last_corners``) -- this method NEVER runs cv2, so it can be called
        unconditionally every UI tick without risk of stalling the GUI thread.
        """
        st = self._last_status
        # Always refresh the in-dialog liveness readout (connection + frames + det/s).
        self._refresh_liveness()
        if self._last_left is not None:
            # Overlay the LEFT corners from the LAST detection when that detection
            # found them, so the operator sees the lock visually (not just a count).
            # The corners were computed OFF the UI thread (no cv2 here).
            corners = (self._last_corners if st is not None and st.found_left
                       else None)
            self._show_left(self._last_left, corners)

        # B2: tick the "view banked" confirmation BEFORE the early-out below, so the
        # green preview border decays on every tick (not just on status frames).
        self._tick_accept_confirmation(st)

        if st is None:
            return
        # --- B1 count bar: RAW counts, GREEN once count_ok, AMBER while short. ---
        # Re-sync the max each tick (tests/callers may change n_target post-start);
        # setValue is clamped to the range so an over-target count still pins full.
        target = max(1, st.n_target)
        if self._count_bar.maximum() != target:
            self._count_bar.setRange(0, target)
        self._count_bar.setValue(min(st.accepted_count, target))
        self._set_bar_satisfied(self._count_bar, st.count_ok)
        # --- B1 tilt bar: honest semantics -- reads INCOMPLETE until skew_ok. ---
        # While the gate is unmet the visible fill is capped below full and the
        # label says "need more" (so one mild tilt can't read as a finished bar);
        # once skew_ok it snaps to a full GREEN "tilt: OK".
        if st.skew_ok:
            self._tilt_bar.setValue(100)
            self._tilt_bar.setFormat("tilt: OK")
        else:
            need = max(self._coll.cfg.min_skew_spread, 1e-9)
            tilt_frac = min(1.0, st.coverage.skew_range / need)
            # Cap below full so an incomplete gate never paints a ~100% bar.
            self._tilt_bar.setValue(int(tilt_frac * _TILT_CAP_PCT))
            self._tilt_bar.setFormat("tilt: need more")
        self._set_bar_satisfied(self._tilt_bar, st.skew_ok)
        # Guidance line: the collector's own reason is already operator-facing
        # ("tilt the board more", "too similar", which camera missed it, ...).
        col = theme.GOOD if st.accepted else theme.TEXT_DIM
        self._status.setText(st.reason)
        self._status.setStyleSheet(f"color: {col};")

    def _refresh_liveness(self) -> None:
        """Update the always-visible stream-liveness readout (operator-facing).

        Shows the connection state (running vs stopped), the total frames received
        off the stream, and a detections/sec rate averaged over a short trailing
        window. A live stream with a frozen frame count, or a det/s of 0.0 while
        capturing, is the operator's signal that something is wrong upstream -- which
        the UI process's stdout never reaches them.
        """
        dev = self._device_id or "?"
        state = "live" if self._running else "stopped"
        # det/s = completions in the trailing window / window span (steady, no spikes).
        det_per_s = len(self._detect_times) / _LIVENESS_WINDOW_S
        self._liveness_lbl.setText(
            f"stream[{dev}]: {state}  ·  received {self._frames_received} frames  ·  "
            f"{det_per_s:.1f} detections/s")

    @staticmethod
    def _set_bar_satisfied(bar: QProgressBar, satisfied: bool) -> None:
        """Paint a progress bar's chunk GREEN when satisfied, AMBER while still needed.

        Only re-applies the stylesheet on the satisfied->unsatisfied edge (Qt caches
        the string), so this is cheap to call every tick.
        """
        qss = _BAR_GOOD if satisfied else _BAR_WARN
        if bar.styleSheet() != qss:
            bar.setStyleSheet(qss)

    def _tick_accept_confirmation(self, st) -> None:
        """B2: latch + decay the green "view banked" preview-border confirmation.

        On the accept EDGE (this frame's status reports ``accepted``) the latch is
        (re)armed to ``_ACCEPT_HOLD_TICKS`` and the border flips green ONCE. Each
        subsequent tick decrements the latch; on the revert edge (latch hits 0) the
        border flips back to neutral ONCE. Styles are touched only on those two
        edges, never mid-hold, so there is no per-frame stylesheet churn.
        """
        if st is not None and st.accepted:
            # Accept edge: arm/re-arm the hold and (if not already lit) light it.
            if self._accept_hold == 0:
                self._preview.setStyleSheet(_PREVIEW_QSS_BANKED)
            self._accept_hold = _ACCEPT_HOLD_TICKS
            return
        if self._accept_hold > 0:
            self._accept_hold -= 1
            if self._accept_hold == 0:
                # Revert edge: drop back to the neutral idle border.
                self._preview.setStyleSheet(_PREVIEW_QSS_IDLE)

    def _show_left(self, gray: np.ndarray, corners) -> None:
        """Blit the LEFT gray to the preview, drawing any detected corners on it.

        Converts the gray to an RGB buffer (so the green markers are visible),
        stamps a small square at each subpixel corner, and scales the pixmap to the
        preview width. The RGB buffer is retained on ``self`` so the QImage backing
        store stays alive until the next frame (a freed buffer renders as garbage).
        """
        g = np.ascontiguousarray(gray)
        h, w = g.shape[:2]
        rgb = np.repeat(g[:, :, None], 3, axis=2)        # gray -> RGB
        if corners is not None:
            r, gc, b = _CORNER_RGB
            for x, y in corners:
                xi, yi = int(round(float(x))), int(round(float(y)))
                # 3x3 stamp clamped to the image -- a cheap, dependency-free marker.
                y0, y1 = max(0, yi - 1), min(h, yi + 2)
                x0, x1 = max(0, xi - 1), min(w, xi + 2)
                rgb[y0:y1, x0:x1] = (r, gc, b)
        rgb = np.ascontiguousarray(rgb)
        self._preview_buf = rgb                          # keep alive for QImage
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img)
        target = self._preview.size()
        if target.width() > 1 and target.height() > 1:
            pix = pix.scaled(target, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                             QtCore.Qt.TransformationMode.SmoothTransformation)
        self._preview.setPixmap(pix)

    # -- solve (off the UI thread) ----------------------------------------- #
    def _begin_solve(self) -> None:
        """Kick off the OFF-thread solve once the dataset is complete.

        Always passes a STABLE debug-dump path (suffixed by the accepted view count,
        not a timestamp, so re-runs overwrite the same file) so the operator's REAL
        captured corners are saved for offline reproduction of a failure. The path is
        shown in the status line so the operator can send us that one file.
        """
        n_views = self._coll.accepted_count
        # Stable, per-view-count path (no Date.now): re-solving the same dataset
        # overwrites the same file, and the operator always knows where it is.
        self._dump_path = f"/tmp/oakd_calib_views_{n_views}.npz"
        self._status.setText(
            f"Captured {n_views} diverse views — solving… "
            f"(debug dump → {self._dump_path})")
        self._status.setStyleSheet(f"color: {theme.ACCENT};")
        square_m = float(self._square_spin.value()) / 1000.0   # mm -> m
        self._solver = _SolveWorker(
            self._coll.views, self._coll.pattern_cols, self._coll.pattern_rows,
            square_m, self._image_size, dump_path=self._dump_path, parent=self)
        self._solver.done.connect(self._on_solved)
        self._solver.failed.connect(self._on_solve_failed)
        self._solver.start()

    def _on_solve_failed(self, msg: str) -> None:
        self._status.setText(f"⚠ {msg}")
        self._status.setStyleSheet(f"color: {theme.BAD};")
        # Allow the operator to recapture / retry from the same dialog.
        self._start_btn.setText("START")
        self._set_phase_idle()

    def _on_solve_did_not_converge(self, result) -> None:
        """The solve ran but the result is implausible (``ok=False``) -- recapture.

        Shows the numbers (so the operator can SEE the runaway focal / diverged RMS)
        plus the honest reason and the debug-dump path, and leaves Save DISABLED so a
        non-converged calibration can never be persisted. Mirrors a hard-fail state but
        is reached via the solver's own ``ok`` verdict rather than an exception.
        """
        self._verdict = ("FAIL", result.failure_reason)
        self._result_lbl.setText(
            f"K_l fx,fy = {result.K_l[0, 0]:.1f}, {result.K_l[1, 1]:.1f} px&nbsp;&nbsp; "
            f"stereo RMS = {result.stereo_rms:.3g} px&nbsp;&nbsp; "
            f"baseline = {result.baseline_m * 1000:.2f} mm "
            f"<span style='color:{theme.TEXT_DIM}'>"
            f"(n_views={result.n_views_used})</span>")
        self._verdict_lbl.setText(
            f"calibration did not converge — {result.failure_reason}")
        self._verdict_lbl.setStyleSheet(f"color: {theme.BAD};")
        dump = (f"  Debug data saved to {self._dump_path} — send us this file."
                if self._dump_path else "")
        self._status.setText(
            "⚠ Calibration did not converge — recapture (check the board size and "
            "that the whole board is visible and well-lit, and TILT it more)." + dump)
        self._status.setStyleSheet(f"color: {theme.BAD};")
        self._save_btn.setEnabled(False)
        self._start_btn.setText("START")
        self._set_phase_idle()

    def _on_solved(self, result) -> None:
        """Show RMS / baseline / K, then run calib_check and show its verdict.

        H3 + B3: the two key metrics (stereo RMS, baseline) are severity-coloured
        against sane bounds, the EXPECTED baseline (~75 mm for an OAK-D) is echoed
        next to the measured one, and the ENTERED square size is echoed too -- so a
        25.00-vs-25.40 mm typo or a bad solve is obvious on the results line itself.
        The label renders rich text so individual metrics can carry their own colour.

        If the solve flagged ``ok=False`` (a wide-FOV runaway focal length, a diverged
        stereo RMS, or too few clean views), the wizard shows an HONEST "did not
        converge — recapture" message and leaves Save disabled rather than persisting
        garbage. The debug dump was still written, so the operator can send it to us.
        """
        self._result = result
        if not getattr(result, "ok", True):
            self._on_solve_did_not_converge(result)
            return
        baseline_mm = result.baseline_m * 1000.0
        square_mm = float(self._square_spin.value())
        # Severity colours: GOOD inside the sane band, WARN/BAD outside (a plain hint,
        # not a hard gate -- the verdict line still owns the PASS/WARN/FAIL call).
        rms_col = theme.GOOD if result.stereo_rms <= _RMS_GOOD_PX else theme.WARN
        if _BASELINE_GOOD_MM[0] <= baseline_mm <= _BASELINE_GOOD_MM[1]:
            base_col = theme.GOOD
        else:
            base_col = theme.BAD
        self._result_lbl.setText(
            f"RMS  mono L/R = {result.rms_l:.3f}/{result.rms_r:.3f} px&nbsp;&nbsp; "
            f"stereo = <b><span style='color:{rms_col}'>"
            f"{result.stereo_rms:.3f} px</span></b>&nbsp;&nbsp; "
            f"baseline = <b><span style='color:{base_col}'>"
            f"{baseline_mm:.2f} mm</span></b> "
            f"<span style='color:{theme.TEXT_DIM}'>(expect ~{_BASELINE_EXPECT_MM:.0f} "
            f"for OAK-D)</span>&nbsp;&nbsp; "
            f"<span style='color:{theme.TEXT_DIM}'>(square={square_mm:.2f} mm)</span>"
            f"<br>K_l fx,fy,cx,cy = {result.K_l[0, 0]:.1f}, {result.K_l[1, 1]:.1f}, "
            f"{result.K_l[0, 2]:.1f}, {result.K_l[1, 2]:.1f}")
        verdict, text = self._run_calib_check(result)
        self._verdict = (verdict, text)
        col = {"PASS": theme.GOOD, "WARN": theme.WARN}.get(verdict, theme.BAD)
        self._verdict_lbl.setText(f"calib_check: {verdict} — {text}")
        self._verdict_lbl.setStyleSheet(f"color: {col};")
        self._status.setText("Solve complete. Review the verdict, then Save.")
        self._status.setStyleSheet(f"color: {theme.GOOD};")
        self._save_btn.setEnabled(True)
        self._start_btn.setText("START")
        self._set_phase_idle()

    def _run_calib_check(self, result) -> tuple[str, str]:
        """Grade a solved result with the shipped calib_check suite.

        Reloads the result through the EXACT pipeline loader (write the calib dict
        in memory -> ``StereoCalib.from_json``) so the graded values are byte-
        identical to what the live pipeline would consume (this is also what
        validates the cm/m translation factor). Returns the worst status across
        the suite (FAIL > WARN > PASS) + a one-line reason for the operator.
        """
        from imu_camera.io.reader import StereoCalib
        from imu_camera.tools.calib_check import FAIL, WARN, run_checks
        from sky.calib.writer import calib_to_dict

        calib = StereoCalib.from_json(calib_to_dict(result, self._image_size))
        checks = run_checks(calib, reader=None)
        fails = [c for c in checks if c.status == FAIL]
        warns = [c for c in checks if c.status == WARN]
        if fails:
            return "FAIL", f"{len(fails)} failing: {fails[0].name} ({fails[0].note})"
        if warns:
            return "WARN", f"{len(warns)} warning(s): {warns[0].name}"
        return "PASS", f"all {len(checks)} checks clean"

    # -- save (Phase 3 writer) --------------------------------------------- #
    def _on_save(self) -> None:
        """Persist the solved calib to THIS device's store + export a file.

        Two persistences happen here:

        * the per-device STORE save (the load-bearing one) -- writes the calib to the
          ``.cache`` keyed by ``self._device_id``. The live pipeline uses the trusted
          FACTORY calib by DEFAULT; this saved calib is applied only when capture runs
          with ``--use-camera-calib``. This is the camera-side mirror of how the IMU
          calib dialogs save per-device IMU calib.
        * a calib.json FILE export (kept for replay / sharing across machines).

        A non-PASS ``calib_check`` verdict warns the operator before EITHER persistence,
        since an unvalidated calib applied via ``--use-camera-calib`` can silently
        degrade the trajectory. The store save runs even if the operator then cancels
        the file picker -- persisting the solve is the primary intent and must not be
        lost.
        """
        if self._result is None:
            return
        # Honest warning: a non-PASS verdict means the live pipeline may be poisoned
        # by this calib, so make the operator confirm before persisting it.
        verdict = self._verdict[0] if self._verdict else "FAIL"
        if verdict != "PASS":
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.warning(
                self, "calib_check did not PASS",
                f"calib_check returned {verdict}. Saving an unvalidated "
                f"calibration can silently degrade the trajectory.\n\nSave anyway?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if reply != QMessageBox.StandardButton.Save:
                return

        # Auto-apply: save to the per-device store FIRST so it sticks even if the
        # operator cancels the file picker below. Build the SAME calib dict the file
        # export writes (translation in cm, the loader's convention) and key it by the
        # wizard's device id -- device-agnostic, no OAK-D specifics in the UI.
        from sky.calib.writer import calib_to_dict, write_calib_json
        from imu_camera.device.camera_calib_store import save_camera_calib

        applied = ""
        if self._device_id is not None:
            calib_dict = calib_to_dict(self._result, self._image_size)
            store_path = save_camera_calib(self._device_id, calib_dict)
            applied = (
                f"Saved for device {self._device_id}. The live pipeline uses "
                f"FACTORY calib by default; run with --use-camera-calib to apply "
                f"this one.  [{store_path}]  ")

        path, _ = QFileDialog.getSaveFileName(
            self, "Save calib.json", "calib.json", "JSON (*.json)")
        if not path:
            # The store save already took effect; reflect that even with no file export.
            if applied:
                self._status.setText(applied.rstrip())
                self._status.setStyleSheet(f"color: {theme.GOOD};")
            return
        written = write_calib_json(self._result, self._image_size, path)
        self._status.setText(f"{applied}Exported {written}")
        self._status.setStyleSheet(f"color: {theme.GOOD};")

    # -- error / lifecycle ------------------------------------------------- #
    def _on_error(self, msg: str) -> None:
        """Surface a stream error (mono guard / connect failure) and stop capture."""
        self._stop_stream()
        self._status.setText(f"⚠ {msg}")
        self._status.setStyleSheet(f"color: {theme.BAD};")
        self._start_btn.setText("START")
        self._set_phase_idle()

    def closeEvent(self, event) -> None:                              # noqa: N802
        """Stop the drain timer + wait out any running detect / solve on close."""
        # _stop_stream already stops the timer and waits out any in-flight detection,
        # so no _DetectWorker thread is left running when the dialog is destroyed.
        self._stop_stream()
        # Tear down the fullscreen board too, so closing the wizard never leaves an
        # orphaned board window covering the screen.
        self._close_board_window()
        solver = self._solver
        if solver is not None and solver.isRunning():
            # Let the (short) solve finish so the worker thread is not destroyed
            # while running -- it touches no UI, so this is a brief, safe wait.
            solver.wait(5000)
        super().closeEvent(event)
