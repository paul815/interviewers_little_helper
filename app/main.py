"""Точка входа.

По умолчанию открывает нативное окно 600×400 поверх других окон (pywebview).
Флаг --browser запускает только сервер и открывает обычную вкладку браузера
(без always-on-top — см. README).
"""
from __future__ import annotations

import argparse
import logging
import socket
import threading
import time
from pathlib import Path

import uvicorn

from . import APP_NAME
from .config import AppConfig
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


def main() -> None:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--browser", action="store_true",
                        help="без нативного окна: открыть в браузере (нет always-on-top)")
    parser.add_argument("--config", type=Path, default=None, help="путь к config.json")
    parser.add_argument("--port", type=int, default=None, help="переопределить порт")
    args = parser.parse_args()

    cfg = AppConfig.load(args.config)
    configure_logging(cfg.log_level)
    if args.port:
        cfg.server.port = args.port
    url = f"http://{cfg.server.host}:{cfg.server.port}"

    server = build_server(cfg)

    if args.browser:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        log.info("%s: %s (Ctrl+C для выхода)", APP_NAME, url)
        server.run()
        return

    server_thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    server_thread.start()
    if not wait_port(cfg.server.host, cfg.server.port):
        raise SystemExit(f"Сервер не поднялся на {url} — смотрите logs/app.log")

    try:
        import webview
    except ImportError:
        raise SystemExit(
            "pywebview не установлен. Поставьте зависимости (requirements-*.txt) "
            "или запустите с флагом --browser."
        )

    log.info("%s: окно 600×400 поверх остальных, сервер на %s", APP_NAME, url)
    webview.create_window(
        APP_NAME, url, width=600, height=400, on_top=True, resizable=False,
    )
    try:
        webview.start()  # блокируется до закрытия окна
    finally:
        # Корректно гасим сервер: lifespan-shutdown остановит активную сессию.
        server.should_exit = True
        server_thread.join(timeout=15)


if __name__ == "__main__":
    main()
