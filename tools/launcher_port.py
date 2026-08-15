"""The server port on one line — for the launch scripts.

The launchers (`tools/launch_win.bat`, `tools/launch_mac.command`) need to know
where to knock in order to tell whether the application came up. Parsing
config.json inside the .bat is not an option: cmd.exe eats the commas inside
`for /f` and a python one-liner silently breaks — hence a separate file.

Read directly, without importing app.config: the launcher calls this before the
dependencies are guaranteed to be in place, and there is nothing here to fail.

    python -m tools.launcher_port    # -> 8756
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 8756


def port() -> int:
    """The port from config.json; on any trouble, the default from config.py."""
    cfg = ROOT / "config.json"
    try:
        # utf-8-sig: Notepad and PowerShell like to prepend a BOM.
        data = json.loads(cfg.read_text(encoding="utf-8-sig"))
        value = int(data["server"]["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_PORT
    return value if 1 <= value <= 65535 else DEFAULT_PORT


if __name__ == "__main__":
    print(port())
