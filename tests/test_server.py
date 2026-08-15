"""API tests through TestClient: only deterministic scenarios that also work
without Ollama or audio devices (in CI, for instance)."""
from __future__ import annotations

import asyncio

import pytest

from app.config import AppConfig
from tests.conftest import local_client, tmp_cfg, ws_connect

GUIDE = {
    "guide_id": "g", "language": "en", "title": "Test",
    "sections": [{"id": "s1", "title": "S", "topics": [{"id": "s1.t1", "question": "Q?"}]}],
}


@pytest.fixture()
def client(tmp_path):
    with local_client(tmp_cfg(tmp_path)) as c:
        yield c


def test_index_and_static(client):
    assert "Interview Helper" in client.get("/").text
    assert len(client.get("/app.css").text) > 500
    assert len(client.get("/app.js").text) > 500


def test_state_snapshot_shape(client):
    snap = client.get("/api/state").json()
    assert snap["state"] == "idle"
    for key in ("flags", "channels", "default_duration_min", "segments", "interval_s",
                "questions", "guide_text"):
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
    assert r.status_code == 409 and "the same device" in r.json()["detail"]


def test_session_actions_require_running_session(client):
    for path, body in [
        ("/api/session/stop", None),
        ("/api/session/analyze", None),
        ("/api/session/flag", {"note": "x"}),
        ("/api/session/question", {"text": "ask about unsubscribes"}),
        ("/api/session/question/done", {"question_id": 1, "done": True}),
        ("/api/topics/status", {"topic_id": "s1.t1", "status": "covered"}),
        ("/api/recommendations/dismiss", {"topic_id": "s1.t1"}),
    ]:
        r = client.post(path, json=body)
        assert r.status_code == 409, path


def _running_controller(tmp_path):
    """A controller with a session stubbed in: the interview screen is poked without hardware."""
    import time

    from app.server.controller import AppController, SessionRuntime
    from app.server.hub import WsHub
    from app.storage.session_store import SessionStore

    cfg = AppConfig()
    cfg.storage.sessions_dir = str(tmp_path / "sessions")
    cfg.storage.guides_dir = str(tmp_path / "guides")
    cfg.storage.projects_dir = str(tmp_path / "projects")
    controller = AppController(cfg, WsHub())
    rt = SessionRuntime()
    rt.store = SessionStore(tmp_path, "s-live")
    rt.t0_mono = time.monotonic() - 90.0
    rt.guide_text = "Introductions\nGood afternoon!"
    controller.rt = rt
    controller.state = "running"
    return controller


def test_comment_and_questions_round_trip(tmp_path):
    controller = _running_controller(tmp_path)

    flag = asyncio.run(controller.add_flag(
        "confused about the tiers", "Tell me about your channel. What is it about?",
        "The channel and the role of Telegram", "s2.t1",
    ))
    assert flag["anchor_section"] == "The channel and the role of Telegram"
    assert flag["topic_id"] == "s2.t1"
    assert flag["t"] >= 90.0  # the time is measured from session start, not from zero

    q = asyncio.run(controller.add_question("  ask about the second channel  "))
    assert q["text"] == "ask about the second channel" and q["done"] is False
    assert asyncio.run(controller.set_question_done(q["id"], True))["done"] is True

    snap = controller.snapshot()
    assert snap["guide_text"].startswith("Introductions")
    assert snap["flags"][0]["note"] == "confused about the tiers"
    assert snap["questions"][0]["done"] is True
    controller.rt.store.close()


def test_empty_question_rejected(tmp_path):
    controller = _running_controller(tmp_path)
    with pytest.raises(Exception, match="The question is empty"):
        asyncio.run(controller.add_question("   "))
    controller.rt.store.close()


def test_guide_library_crud(client):
    r = client.post("/api/guides", json={"guide": GUIDE, "source_text": "raw"})
    assert r.status_code == 200
    fid = r.json()["file_id"]

    listed = client.get("/api/guides").json()["guides"]
    assert any(g["file_id"] == fid for g in listed)

    loaded = client.get(f"/api/guides/{fid}").json()
    assert loaded["source_text"] == "raw"
    assert loaded["guide"]["title"] == "Test"

    assert client.delete(f"/api/guides/{fid}").json()["ok"] is True
    assert client.get(f"/api/guides/{fid}").status_code == 409


def test_guide_library_rejects_bad_guide(client):
    r = client.post("/api/guides", json={"guide": {"no": "fields"}, "source_text": ""})
    assert r.status_code == 409


def test_monitor_with_bad_devices_degrades_gracefully(client):
    r = client.post("/api/monitor/start", json={"mic_index": 99998, "system_index": 99999})
    assert r.status_code == 200 and r.json()["ok"] is False
    assert client.post("/api/monitor/stop").json()["ok"] is True


def test_websocket_snapshot(client):
    with ws_connect(client) as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot" and msg["payload"]["state"] == "idle"
