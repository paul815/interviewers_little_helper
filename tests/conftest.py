"""Shared by all the tests."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import AppConfig
from app.server.app import create_app

# The server checks Host (app/server/security.py), so TestClient has to come
# from a real address rather than the default `testserver`: otherwise the tests
# would exercise a path the application never actually takes.
BASE_URL = "http://127.0.0.1:8756"
# websocket_connect does urljoin("ws://testserver", url) and ignores base_url,
# so for websockets the address has to be given in full.
WS_URL = "ws://127.0.0.1:8756/ws"


def local_client(cfg: AppConfig) -> TestClient:
    """A TestClient that looks local to the Origin/Host layer."""
    return TestClient(create_app(cfg), base_url=BASE_URL)


def ws_connect(client: TestClient, **kwargs):
    """A /ws connection with a correct Host."""
    return client.websocket_connect(WS_URL, **kwargs)


def tmp_cfg(tmp_path) -> AppConfig:
    """A config whose entire storage lives in the test's temporary folder."""
    cfg = AppConfig()
    cfg.storage.sessions_dir = str(tmp_path / "sessions")
    cfg.storage.guides_dir = str(tmp_path / "guides")
    cfg.storage.projects_dir = str(tmp_path / "projects")
    cfg.storage.prompts_file = str(tmp_path / "prompts.json")
    return cfg
