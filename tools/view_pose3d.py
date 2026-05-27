#!/usr/bin/env python3
"""Entry point — launch the OAK-D 3D pose viewer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication       # noqa: E402

from oakd.pose import PoseHistory              # noqa: E402
from oakd.sources import FakePoseSource        # noqa: E402
from oakd.ui.mainwindow import MainWindow      # noqa: E402


def _build_source(name: str):
    name = name.lower()
    if name == "fake":
        return FakePoseSource(rate_hz=100.0, radius_m=3.0, period_s=12.0)
    if name == "oak":
        from oakd.sources.depthai_vo import OakBasaltVioSource
        return OakBasaltVioSource()
    raise SystemExit(f"unknown --source '{name}' (expected: fake|oak)")


def main() -> int:
    ap = argparse.ArgumentParser(description="OAK-D 3D pose viewer")
    ap.add_argument("--source", default="fake", choices=("fake", "oak"),
                    help="pose provider")
    args = ap.parse_args()

    history = PoseHistory(capacity=8192)
    source = _build_source(args.source)
    source.start(history.push)

    app = QApplication(sys.argv)
    win = MainWindow(history, source, source_name=args.source)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
