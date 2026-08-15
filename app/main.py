"""The entry point.

By default it opens a native always-on-top window (pywebview): the right half of
the screen at full height — Zoom on the left, the helper on the right. The window
stretches, and its size is remembered between runs. The --browser flag starts
only the server and opens an ordinary browser tab (without always-on-top — see
the README).
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import threading
import time
from pathlib import Path

import uvicorn

from . import APP_NAME
from .config import AppConfig, WindowConfig
from .logging_setup import configure as configure_logging

log = logging.getLogger("ilh.main")


def build_server(cfg: AppConfig) -> uvicorn.Server:
    from .server.app import create_app

    uv_cfg = uvicorn.Config(
        create_app(cfg),
        host=cfg.server.host,
        port=cfg.server.port,
        log_level="warning",
        access_log=False,
    )
    return uvicorn.Server(uv_cfg)


def wait_port(host: str, port: int, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def work_area() -> tuple[int, int, int, int] | None:
    """The screen minus the taskbar: (x, y, width, height).

    A "full height" window must not slide under the taskbar — otherwise the
    bottom lines of the transcript cannot be reached with the mouse. Only Windows
    gives the exact rectangle; on the other platforms it is None and the size
    comes from pywebview.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        SPI_GETWORKAREA = 0x0030
        if not ctypes.windll.user32.SystemParametersInfoW(
            SPI_GETWORKAREA, 0, ctypes.byref(rect), 0
        ):
            return None
    except (ImportError, AttributeError, OSError):
        return None
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def screen_size() -> tuple[int, int] | None:
    """The screen size per pywebview: it is known before the window is created."""
    try:
        import webview

        screen = webview.screens[0]
        return int(screen.width), int(screen.height)
    except (ImportError, IndexError, AttributeError):
        return None


def screen_area() -> tuple[int, int, int, int] | None:
    """The rectangle we fit the window into."""
    area = work_area()
    if area is not None:
        return area
    size = screen_size()
    return (0, 0, size[0], size[1]) if size is not None else None


def default_geometry(win: WindowConfig) -> tuple[int, int]:
    """Half the screen horizontally, the full height vertically."""
    area = screen_area()
    if not win.half_screen or area is None:
        return win.width, win.height
    _, _, full_w, full_h = area
    return max(full_w // 2, win.min_width), max(full_h, win.min_height)


def dock_position(win: WindowConfig, width: int) -> tuple[int | None, int | None]:
    """The right edge of the screen: the left half is left for the call window."""
    area = screen_area()
    if not win.half_screen or area is None:
        return None, None
    x0, y0, full_w, _ = area
    return x0 + max(full_w - width, 0), y0


def restore_size(win: WindowConfig, state_path: Path) -> tuple[int, int]:
    """The size from the previous run, if it exists and does not look like garbage."""
    width, height = default_geometry(win)
    if not win.remember_size:
        return width, height
    try:
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        saved_w, saved_h = int(saved["width"]), int(saved["height"])
    except (OSError, ValueError, KeyError, TypeError):
        return width, height

    # The monitor may have been swapped for a smaller one: a window that does
    # not fit the screen would put its own buttons out of reach.
    max_w, max_h = 100_000, 100_000
    size = screen_size()
    if size is not None:
        max_w, max_h = size
    return (
        min(max(saved_w, win.min_width), max_w),
        min(max(saved_h, win.min_height), max_h),
    )


def track_size(window, state_path: Path):
    """Subscribes to resize; returns the save function (called on exit).

    We write on exit rather than on every pixel of dragging the frame.
    """
    last: dict[str, int] = {}

    def on_resized(width, height) -> None:
        last["width"], last["height"] = int(width), int(height)

    try:
        window.events.resized += on_resized
    except (AttributeError, TypeError):  # a pywebview build without the event
        log.debug("pywebview has no resized event — the window size is not remembered")
        return lambda: None

    def save() -> None:
        if not last:
            return
        try:
            state_path.write_text(json.dumps(last), encoding="utf-8")
        except OSError as e:
            log.warning("Could not remember the window size: %s", e)

    return save


def main() -> None:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--browser", action="store_true",
                        help="no native window: open in the browser (no always-on-top)")
    parser.add_argument("--config", type=Path, default=None, help="path to config.json")
    parser.add_argument("--port", type=int, default=None, help="override the port")
    args = parser.parse_args()

    cfg = AppConfig.load(args.config)
    configure_logging(cfg.log_level)
    if args.port:
        cfg.server.port = args.port
    url = f"http://{cfg.server.host}:{cfg.server.port}"

    probe = socket.socket()
    try:
        probe.bind((cfg.server.host, cfg.server.port))
    except OSError:
        raise SystemExit(
            f"Port {cfg.server.port} is busy — the application seems to be running already "
            f"(the window may be hiding behind Zoom). Or start it with --port <another port>."
        ) from None  # a socket traceback explains nothing to the user
    finally:
        probe.close()

    server = build_server(cfg)

    if args.browser:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        log.info("%s: %s (Ctrl+C to quit)", APP_NAME, url)
        server.run()
        return

    server_thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    server_thread.start()
    if not wait_port(cfg.server.host, cfg.server.port):
        raise SystemExit(f"The server did not come up at {url} — see logs/app.log")

    try:
        import webview
    except ImportError:
        raise SystemExit(
            "pywebview is not installed. Install the dependencies (requirements-*.txt) "
            "or start with the --browser flag."
        ) from None

    win = cfg.window
    width, height = restore_size(win, cfg.window_state_path)
    x, y = dock_position(win, width)
    log.info("%s: a %dx%d window on top of the rest, server at %s", APP_NAME, width, height, url)
    window = webview.create_window(
        APP_NAME, url,
        width=width, height=height, x=x, y=y,
        min_size=(win.min_width, win.min_height),
        resizable=win.resizable, on_top=win.on_top,
    )
    save_size = track_size(window, cfg.window_state_path) if win.remember_size else lambda: None
    try:
        webview.start()  # blocks until the window is closed
    finally:
        save_size()
        # Shut the server down cleanly: the lifespan shutdown stops the active session.
        server.should_exit = True
        server_thread.join(timeout=15)


if __name__ == "__main__":
    main()
