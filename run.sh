#!/usr/bin/env bash
# Launcher for the 3D pose viewer of *our* from-scratch VIO.
# (For the DepthAI/Basalt baseline viewer, run baseline/tools/view_pose3d.py.)
#
# Modes:
#   ./run.sh ...                    -- single-process viewer (default; offline-safe)
#   ./run.sh --proc ...             -- 4-process pipeline (capture + vio + slam + ui)
#                                      see docs/PROC4_ARCHITECTURE.md
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "[run.sh] .venv missing — bootstrap with:" >&2
  echo "  python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# --proc selects the 4-process launcher; strip the flag and forward the rest.
if [ "${1:-}" = "--proc" ]; then
  shift
  exec .venv/bin/python -m ours.proc.launcher --auto-suffix "$@"
fi

exec .venv/bin/python ours/tools/view_pose3d.py "$@"
