"""Perimeter tests: names from a URL must not become arbitrary paths, and a
foreign page must not reach the local server.

Neither scenario is theoretical. The naive check "no / \\ or .." let `C:evil`
through on Windows (a drive-relative path), and /ws without an Origin check
handed the live transcript to any tab open in the browser.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.paths import UnsafeName, safe_child
from app.server.app import create_app
from app.server.security import _host_only
from tests.conftest import BASE_URL, WS_URL, local_client, tmp_cfg

# ---------------------------------------------------------------- safe_child

UNSAFE = [
    "C:foo",        # a drive-relative path: the main hole this all exists for
    "C:",
    "C:Users",
    "\\\\server\\share",
    "a/b",
    "a\\b",
    "../etc",
    "..",
    ".",
    "",
    "   ",
    "x" * 200,      # longer than the limit
    "\x00",
    "a\x00b",
    "con", "NUL", "COM1", "lpt9", "nul.json",  # DOS device names
    ".hidden",      # the first character must be a letter or a digit
    "-flag",
    "~",
    "%2e%2e",
]

SAFE = [
    "20260808-101500-onboarding",
    "2026-08-08_10-15-00",
    "гайд-1",       # non-Latin names stay legal: a guide may be in any language
    "b2b-onboarding-2",
    "Guide.v2",
]


@pytest.mark.parametrize("name", UNSAFE)
def test_safe_child_rejects(tmp_path, name):
    with pytest.raises(UnsafeName):
        safe_child(tmp_path, name, ".json")


@pytest.mark.parametrize("name", SAFE)
def test_safe_child_accepts(tmp_path, name):
    path = safe_child(tmp_path, name, ".json")
    assert path.is_relative_to(tmp_path.resolve())
    assert path.name == f"{name}.json"


def test_safe_child_without_suffix(tmp_path):
    assert safe_child(tmp_path, "proj").name == "proj"


# ---------------------------------------------------------- paths over HTTP

@pytest.fixture()
def client(tmp_path):
    with local_client(tmp_cfg(tmp_path)) as c:
        yield c


@pytest.mark.skipif(os.name != "nt", reason="drive-relative paths exist only on Windows")
def test_drive_relative_name_would_escape(tmp_path):
    """Pinning down the reason the whitelist was needed in the first place.

    `root / "X:evil.json"`, where X differs from the root's drive, gives a path
    on drive X — the join silently loses the root. The repository is usually not
    on C:, so for guides/ this is exactly that case. safe_child must refuse.
    """
    other = "D:" if tmp_path.drive.upper().startswith("C") else "C:"
    assert not (tmp_path / f"{other}evil.json").is_relative_to(tmp_path)
    for name in (f"{other}evil", "C:evil"):
        with pytest.raises(UnsafeName):
            safe_child(tmp_path, name, ".json")


def test_guide_delete_cannot_escape_root(client):
    """DELETE /api/guides/C:evil — refused, and the guide library is unchanged."""
    before = client.get("/api/guides").json()["guides"]
    assert client.delete("/api/guides/C:evil").status_code >= 400
    assert client.get("/api/guides").json()["guides"] == before


@pytest.mark.parametrize("path", [
    "/api/guides/C:evil",
    "/api/guides/..",
    "/api/projects/C:evil",
    "/api/projects/C:evil/sessions",
    "/api/projects/C:evil/coverage",
])
def test_read_endpoints_reject_unsafe_ids(client, path):
    assert client.get(path).status_code >= 400


def test_open_rejects_unsafe_project_id(client):
    r = client.post("/api/projects/C:evil/open", json={})
    assert r.status_code >= 400


def test_project_with_reserved_title_stays_openable(client):
    """The title «CON» yields the slug `con` — a DOS device name. The project must
    still be creatable and openable, or the id and its check disagree."""
    pid = client.post("/api/projects", json={"title": "CON"}).json()["project_id"]
    assert client.get(f"/api/projects/{pid}").status_code == 200


# ---------------------------------------------------------- Host and Origin

def test_host_only_strips_port():
    assert _host_only("127.0.0.1:8756") == "127.0.0.1"
    assert _host_only("localhost") == "localhost"
    assert _host_only("[::1]:8756") == "::1"


def test_foreign_host_rejected(tmp_path):
    """DNS rebinding: the attacker's domain resolves to 127.0.0.1."""
    with TestClient(create_app(tmp_cfg(tmp_path)), base_url="http://evil.com") as c:
        assert c.get("/api/state").status_code == 403


def test_own_host_allowed(client):
    assert client.get("/api/state").status_code == 200


def test_foreign_origin_rejected(client):
    r = client.get("/api/state", headers={"Origin": "https://evil.com"})
    assert r.status_code == 403


def test_own_origin_allowed(client):
    r = client.get("/api/state", headers={"Origin": BASE_URL})
    assert r.status_code == 200


def test_ws_foreign_origin_rejected(client):
    """The main point: a foreign tab does not read the transcript over /ws."""
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(WS_URL, headers={"Origin": "https://evil.com"}):
            pass


def test_ws_own_origin_gets_snapshot(client):
    with client.websocket_connect(WS_URL, headers={"Origin": BASE_URL}) as ws:
        assert ws.receive_json()["type"] == "snapshot"


def test_ws_without_origin_allowed_by_default(client):
    """Not a browser (curl, a script) — let it in: it reads projects/ off the disk anyway."""
    with client.websocket_connect(WS_URL) as ws:
        assert ws.receive_json()["type"] == "snapshot"


def test_require_origin_closes_the_door(tmp_path):
    cfg = tmp_cfg(tmp_path)
    cfg.security.require_origin = True
    with local_client(cfg) as c:
        assert c.get("/api/state").status_code == 403
        assert c.get("/api/state", headers={"Origin": BASE_URL}).status_code == 200
