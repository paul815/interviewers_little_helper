#!/bin/bash
# ===========================================================================
#  Interviewer's Little Helper - macOS / Linux launcher.
#
#  Double-click in Finder (or run from a terminal). Creates .venv via setup.py
#  on first run, then starts the app. macOS has no .lnk: this file is the
#  double-clickable entry point, the Windows twin is tools/launch_win.bat.
#
#  The app opens its own 600x400 always-on-top window. This terminal window is
#  its console - closing it quits the app.
# ===========================================================================
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

VENV_PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

log_event() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >>"$LOG_DIR/launcher.log"; }

if [ ! -x "$VENV_PY" ]; then
    echo
    echo "  No Python environment yet (.venv is missing)."
    echo "  Running setup.py - this downloads dependencies and ASR weights."
    echo
    if command -v python3 >/dev/null 2>&1; then
        python3 "$ROOT/setup.py"
    else
        echo "  Python 3.11+ was not found. Install it, then run this again."
        read -r -p "  Press Enter to close..." _
        exit 1
    fi
fi

if [ ! -x "$VENV_PY" ]; then
    echo
    echo "  Setup did not finish - .venv still has no interpreter."
    echo "  Re-run it by hand to see the error:  python3 setup.py"
    read -r -p "  Press Enter to close..." _
    exit 1
fi

PORT="$("$VENV_PY" -m tools.launcher_port 2>/dev/null || echo 8756)"
URL="http://127.0.0.1:$PORT"

if "$VENV_PY" -c "import socket,sys;s=socket.socket();s.settimeout(0.5);sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null; then
    echo
    echo "  Interviewer's Little Helper is already running on $URL."
    echo "  Its window is small and always-on-top - it may be hiding behind Zoom."
    open "$URL" >/dev/null 2>&1
    sleep 3
    exit 0
fi

echo
echo "  Starting Interviewer's Little Helper on $URL..."
log_event "starting application"
"$VENV_PY" -m app.main
STATUS=$?
log_event "application exited with code $STATUS"

if [ "$STATUS" -ne 0 ]; then
    echo
    echo "  The app exited with code $STATUS."
    echo "  Launcher log:    $LOG_DIR/launcher.log"
    echo "  Application log: $LOG_DIR/app.log"
    echo
    read -r -p "  Press Enter to close..." _
fi
exit "$STATUS"
