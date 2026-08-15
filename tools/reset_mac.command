#!/bin/bash
# ===========================================================================
#  Interviewer's Little Helper - factory reset (macOS / Linux).
#
#  Twin of tools/reset_win.bat. Returns the working copy to a pristine source
#  tree: stops the app, then removes everything at the repository root that is
#  not part of the sources (.venv, logs, sessions, projects, guides,
#  config.json, caches).
#
#  NOT removed - these live outside the repository and cost hours to refetch:
#    * ASR weights in the HuggingFace cache (~/.cache/huggingface)
#    * Ollama models
# ===========================================================================
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# --- Options ---------------------------------------------------------------
# --keep-venv   DEV ONLY. Keeps .venv, so the multi-gigabyte stack survives the
#               reset and the next run starts in seconds. Because the launcher
#               only runs setup.py when .venv is missing, this ALSO skips the
#               installer - so it does not exercise first-run setup. Must be OFF
#               when verifying a release.
KEEP_VENV=0
while [ $# -gt 0 ]; do
    case "$1" in
        --keep-venv) KEEP_VENV=1; shift ;;
        *) echo; echo "  Unknown option: $1"; echo "  Usage: reset_mac.command [--keep-venv]"; exit 2 ;;
    esac
done

# Source entries that survive the reset. Everything else at the root goes.
KEEP=(
    .claude .editorconfig .git .github .gitattributes .gitignore
    .pre-commit-config.yaml .python-version
    AGENTS.md CLAUDE.md LICENSE PLAN.md README.md
    app config.example.json docs examples "Launch ILH.lnk"
    requirements-ci.txt requirements-common.txt requirements-dev.txt
    requirements-macos.txt requirements-windows.txt
    setup.py tests tools
)
[ "$KEEP_VENV" -eq 1 ] && KEEP+=(.venv)

is_kept() {
    local name="$1" k
    for k in "${KEEP[@]}"; do [ "$name" = "$k" ] && return 0; done
    return 1
}

targets() {
    local entry name
    for entry in "$ROOT"/* "$ROOT"/.[!.]*; do
        [ -e "$entry" ] || continue
        name="$(basename "$entry")"
        is_kept "$name" || printf '%s\n' "$name"
    done
}

echo
echo "============================================================"
echo "  WARNING: this PERMANENTLY removes all local app data."
echo "  Projects, sessions, transcripts, reports, guides, logs and"
echo "  your config.json will be deleted. This cannot be undone."
echo "============================================================"
echo
echo "  Root: $ROOT"
echo
echo "  Will be removed:"
if [ -z "$(targets)" ]; then
    echo "    (nothing - the tree is already clean)"
else
    targets | sed 's/^/    - /'
fi
echo
echo "  Kept: sources, git history, ASR weights in the HuggingFace cache,"
echo "        and Ollama models (both live outside this folder)."
if [ "$KEEP_VENV" -eq 1 ]; then
    echo
    echo "  DEV MODE (--keep-venv): .venv is PRESERVED, so setup.py will NOT"
    echo "  re-run on the next launch and first-run setup is not re-tested."
fi
echo
printf 'Type Yes and press Enter to proceed: '
read -r CONFIRM
if [ "$CONFIRM" != "Yes" ]; then
    echo "Reset cancelled."
    exit 0
fi
echo

# Stop ONLY this install's Python processes. Never blanket-kill by name - that
# would also take down the user's unrelated Python. Scoped two ways: the PID we
# recorded, and any process whose command line runs from THIS directory.
echo "Stopping the application..."
PIDFILE="$ROOT/logs/launcher.pid"
if [ -f "$PIDFILE" ]; then
    PID="$(cat "$PIDFILE" 2>/dev/null || true)"
    # Only kill the recorded PID after confirming it still belongs to THIS
    # install: PIDs get recycled, and a stale pid file must never take down an
    # unrelated process.
    if [ -n "${PID:-}" ] && ps -p "$PID" -o command= 2>/dev/null | grep -qF "$ROOT"; then
        kill "$PID" 2>/dev/null || true
    fi
fi
pkill -f "$ROOT/.venv/bin/python" 2>/dev/null || true
sleep 2

FAIL=0
while IFS= read -r name; do
    [ -n "$name" ] || continue
    echo "Removing $name..."
    rm -rf "$ROOT/$name"
    if [ -e "$ROOT/$name" ]; then
        echo "[ERROR] Could not remove $name - files may still be in use."
        FAIL=1
    fi
done <<EOF
$(targets)
EOF

# Purge bytecode and test caches so the reset leaves a pristine source tree.
# With --keep-venv the sweep is scoped to the source folders: recursing into a
# preserved .venv would churn thousands of site-packages caches for no gain.
echo "Removing Python caches..."
if [ "$KEEP_VENV" -eq 1 ]; then
    SWEEP=(app tests tools docs examples)
else
    SWEEP=(.)
fi
for dir in "${SWEEP[@]}"; do
    [ -d "$ROOT/$dir" ] || continue
    find "$ROOT/$dir" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
    find "$ROOT/$dir" -name .pytest_cache -type d -prune -exec rm -rf {} + 2>/dev/null
    find "$ROOT/$dir" \( -name '*.pyc' -o -name '*.pyo' \) -type f -delete 2>/dev/null
done

echo
if [ "$FAIL" -eq 1 ]; then
    echo "Reset INCOMPLETE. Some entries could not be removed."
    echo "Close the app, any terminals and editors holding those files, then retry."
    exit 1
fi
if [ "$KEEP_VENV" -eq 1 ]; then
    echo "Reset complete. .venv preserved - the next launch skips setup.py."
    echo "This is a DEV shortcut - run without --keep-venv before a release."
else
    echo "Reset complete. Run tools/launch_mac.command to set the app up again."
fi
echo "ASR weights and Ollama models were left untouched."
exit 0
