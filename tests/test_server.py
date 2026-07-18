"""API-тесты через TestClient: только детерминированные сценарии, работающие
и без Ollama/аудиоустройств (например, в CI)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import AppConfig
from app.server.app import create_app

GUIDE = {
    "guide_id": "g", "language": "ru", "title": "Тест",
    "sections": [{"id": "s1", "title": "S", "topics": [{"id": "s1.t1", "question": "Q?"}]}],
}


@pytest.fixture()
def client(tmp_path):
    cfg = AppConfig()
    cfg.storage.sessions_dir = str(tmp_path / "sessions")
    cfg.storage.guides_dir = str(tmp_path / "guides")
    with TestClient(create_app(cfg)) as c:
        yield c


def test_index_and_static(client):
    assert "Interview Helper" in client.get("/").text
    assert len(client.get("/app.css").text) > 500
    assert len(client.get("/app.js").text) > 500


def test_state_snapshot_shape(client):
    snap = client.get("/api/state").json()
    assert snap["state"] == "idle"
    for key in ("flags", "channels", "default_duration_min", "segments", "interval_s"):
        assert key in snap


def test_devices_endpoint_never_500(client):
    data = client.get("/api/devices").json()
    assert "devices" in data and "error" in data


def test_llm_status_shape(client):
    st = client.get("/api/llm/status").json()
    assert isinstance(st["ok"], bool) and "model" in st


def test_start_rejects_same_device(client):
    r = client.post("/api/session/start",
                    json={"mic_index": 1, "system_index": 1, "guide": GUIDE})
    assert r.status_code == 409 and "одно и то же" in r.json()["detail"]


def test_session_actions_require_running_session(client):
    for path, body in [
        ("/api/session/stop", None),
        ("/api/session/analyze", None),
        ("/api/session/flag", {"note": "x"}),
        ("/api/topics/status", {"topic_id": "s1.t1", "status": "covered"}),
        ("/api/recommendations/dismiss", {"topic_id": "s1.t1"}),
    ]:
        r = client.post(path, json=body)
        assert r.status_code == 409, path


def test_guide_library_crud(client):
    r = client.post("/api/guides", json={"guide": GUIDE, "source_text": "raw"})
    assert r.status_code == 200
    fid = r.json()["file_id"]

    listed = client.get("/api/guides").json()["guides"]
    assert any(g["file_id"] == fid for g in listed)

    loaded = client.get(f"/api/guides/{fid}").json()
    assert loaded["source_text"] == "raw"
    assert loaded["guide"]["title"] == "Тест"

    assert client.delete(f"/api/guides/{fid}").json()["ok"] is True
    assert client.get(f"/api/guides/{fid}").status_code == 409


def test_guide_library_rejects_bad_guide(client):
    r = client.post("/api/guides", json={"guide": {"нет": "полей"}, "source_text": ""})
    assert r.status_code == 409


def test_monitor_with_bad_devices_degrades_gracefully(client):
    r = client.post("/api/monitor/start", json={"mic_index": 99998, "system_index": 99999})
    assert r.status_code == 200 and r.json()["ok"] is False
    assert client.post("/api/monitor/stop").json()["ok"] is True


def test_websocket_snapshot(client):
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot" and msg["payload"]["state"] == "idle"
