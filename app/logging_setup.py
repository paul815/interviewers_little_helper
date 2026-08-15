"""Logging: the console plus the file logs/app.log. No external telemetry."""
from __future__ import annotations

import logging
import logging.handlers

from .config import ROOT

_configured = False


def configure(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        log_dir = ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        fileh = logging.handlers.RotatingFileHandler(
            log_dir / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        fileh.setFormatter(fmt)
        root.addHandler(fileh)
    except OSError as e:  # the file log is not critical
        root.warning("The file log is unavailable: %s", e)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
