"""Visual theme — military dark palette.

Mirrors the cockpit-MFD / olive-drab look used in
``flight-controller/tools/_ui.py`` so the OAK-D viewer matches the rest of the
skydev tool family.
"""
from __future__ import annotations


# ---- palette ---------------------------------------------------------------

BG          = "#0d1117"   # GitHub dark — pure neutral dark
PANEL       = "#161b22"   # neutral raised panel
PANEL_EDGE  = "#4a5236"   # OLIVE DRAB khaki — military border
GRID        = "#2a323d"
TEXT        = "#e6edf3"
TEXT_DIM    = "#8b949e"
TEXT_FAINT  = "#484f58"

ACCENT      = "#c9b97f"   # khaki / desert sand — titles
GOOD        = "#7cff5c"   # NVG green
WARN        = "#ffb000"   # caution amber
BAD         = "#ff3b30"   # master warning red

TRACE_PATH  = "#7cff5c"   # trajectory line (NVG green)
AXIS_N      = "#ff3b30"   # North/Forward — red
AXIS_E      = "#7cff5c"   # East/Right    — green
AXIS_U      = "#5ce1ff"   # Up            — HUD cyan

BTN_BG      = "#1a2010"
BTN_HOV     = "#2a3320"
BTN_PRIMARY = "#3d6a1f"


# ---- Qt stylesheet ---------------------------------------------------------

QSS = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 11px;
}}

QFrame#Panel {{
    background-color: {PANEL};
    border: 1px solid {PANEL_EDGE};
    border-radius: 4px;
}}

QLabel#PanelTitle {{
    color: {ACCENT};
    font-weight: bold;
    font-size: 10px;
    letter-spacing: 1.5px;
    padding: 2px 4px 4px 4px;
    border-bottom: 1px solid {PANEL_EDGE};
}}

QLabel#FieldLabel  {{ color: {TEXT_DIM}; }}
QLabel#FieldValue  {{ color: {TEXT};     font-weight: bold; }}
QLabel#FieldGood   {{ color: {GOOD};     font-weight: bold; }}
QLabel#FieldWarn   {{ color: {WARN};     font-weight: bold; }}
QLabel#FieldBad    {{ color: {BAD};      font-weight: bold; }}

QLabel#HeaderTitle {{
    color: {ACCENT};
    font-size: 14px;
    font-weight: bold;
    letter-spacing: 2px;
}}
QLabel#HeaderSub {{
    color: {TEXT_DIM};
    font-size: 10px;
    letter-spacing: 1px;
}}

QWidget#ImuCamWindow {{ background-color: {PANEL}; }}
QLabel#ImuCamView {{
    background-color: #000000;
    border: 1px solid {PANEL_EDGE};
    border-radius: 4px;
}}
QLabel#ImuCamStatus {{
    color: {TEXT_DIM};
    font-size: 11px;
    padding: 2px 4px;
}}

QPushButton {{
    background-color: {BTN_BG};
    color: {TEXT};
    border: 1px solid {PANEL_EDGE};
    border-radius: 3px;
    padding: 5px 10px;
    min-width: 56px;
}}
QPushButton:hover    {{ background-color: {BTN_HOV};     border-color: {ACCENT}; }}
QPushButton:pressed  {{ background-color: {BTN_PRIMARY}; }}
QPushButton:checked  {{ background-color: {BTN_PRIMARY}; border-color: {GOOD}; }}

QToolBar {{
    background-color: {PANEL};
    border-bottom: 1px solid {PANEL_EDGE};
    spacing: 6px;
    padding: 4px;
}}

QStatusBar {{
    background-color: {PANEL};
    color: {TEXT_DIM};
    border-top: 1px solid {PANEL_EDGE};
}}

QMenuBar {{
    background-color: {PANEL};
    color: {TEXT};
    border-bottom: 1px solid {PANEL_EDGE};
    padding: 2px;
}}
QMenuBar::item {{ background: transparent; padding: 4px 10px; }}
QMenuBar::item:selected {{ background-color: {BTN_HOV}; color: {ACCENT}; }}
QMenu {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {PANEL_EDGE};
}}
QMenu::item {{ padding: 5px 22px; }}
QMenu::item:selected {{ background-color: {BTN_PRIMARY}; color: {TEXT}; }}
QMenu::separator {{ height: 1px; background: {PANEL_EDGE}; margin: 4px 8px; }}

QDialog {{ background-color: {BG}; }}

QProgressBar {{
    background-color: {BTN_BG};
    border: 1px solid {PANEL_EDGE};
    border-radius: 3px;
    text-align: center;
    color: {TEXT};
    height: 16px;
}}
QProgressBar::chunk {{ background-color: {BTN_PRIMARY}; }}

QLabel#FaceDone   {{ color: {GOOD}; font-weight: bold; }}
QLabel#FaceTodo   {{ color: {TEXT_FAINT}; }}
QLabel#DialogHint {{ color: {TEXT_DIM}; }}
QLabel#DialogMono {{ color: {TEXT}; font-weight: bold; }}
"""