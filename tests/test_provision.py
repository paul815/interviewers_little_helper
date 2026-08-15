"""Tests of the model downloads and of loopback capture.

No hardware or network required: Ollama is replaced by a local HTTP server that
emits the same NDJSON stream the real /api/pull does, and the ASR weights load is
stubbed — what is checked is the mechanics of the progress, not HuggingFace itself.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.audio import loopback
from app.config import AppConfig
from app.provision import Provisioner, asr_ready


class FakeHub:
    def __init__(self):
        self.events: list[dict] = []

    def broadcast_threadsafe(self, type_: str, payload: dict) -> None:
        self.events.append({"type": type_, "payload": payload})


class FakeLLM:
    def __init__(self, **status):
        self.status = {"ok": False, "server_up": True, "model_found": False, **status}

    async def check(self) -> dict:
        return dict(self.status)


def make_provisioner(hub=None, llm=None, cfg=None) -> Provisioner:
    return Provisioner(cfg or AppConfig(), hub or FakeHub(), llm or FakeLLM())


# ------------------------------------------------------------- set-up status

def test_refresh_marks_llm_blocked_without_server():
    prov = make_provisioner(llm=FakeLLM(server_up=False))
    snap = asyncio.run(prov.refresh())
    assert snap["items"]["llm"]["state"] == "blocked"
    assert "ollama.com" in snap["items"]["llm"]["message"]


def test_refresh_marks_llm_missing_when_server_up():
    prov = make_provisioner(llm=FakeLLM(server_up=True, model_found=False))
    assert asyncio.run(prov.refresh())["items"]["llm"]["state"] == "missing"


def test_refresh_marks_llm_ok_when_model_present():
    prov = make_provisioner(llm=FakeLLM(server_up=True, model_found=True))
    assert asyncio.run(prov.refresh())["items"]["llm"]["state"] == "ok"


def test_asr_ready_unknown_model_is_not_nagging():
    # An unfamiliar name (a local path) is not our case, so the wizard stays quiet.
    assert asr_ready(None) is True


def test_asr_ready_false_for_missing_and_partial(tmp_path, monkeypatch):
    repo = tmp_path / "models--org--repo"
    monkeypatch.setattr("app.provision._repo_dir", lambda repo_id: repo)

    assert asr_ready("org/repo") is False  # the folder is not there at all

    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc.incomplete").write_bytes(b"x" * 1024)
    assert asr_ready("org/repo") is False  # a download that was never finished

    (blobs / "abc.incomplete").unlink()
    (blobs / "abc").write_bytes(b"x" * (33 * 1024 * 1024))
    assert asr_ready("org/repo") is True


# ------------------------------------------------------- downloading a model

def _ollama_stub(lines: list[dict]):
    """A local server answering /api/pull with an NDJSON stream, as Ollama does."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for line in lines:
                self.wfile.write((json.dumps(line) + "\n").encode())
                self.wfile.flush()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_pull_llm_reports_bytes_and_finishes():
    server = _ollama_stub([
        {"status": "pulling manifest"},
        {"status": "pulling abc", "completed": 100, "total": 400},
        {"status": "pulling abc", "completed": 400, "total": 400},
        {"status": "success"},
    ])
    try:
        hub, cfg = FakeHub(), AppConfig()
        cfg.llm.base_url = f"http://127.0.0.1:{server.server_address[1]}"
        prov = make_provisioner(hub=hub, cfg=cfg, llm=FakeLLM(model_found=False))

        async def run():
            await prov.refresh()
            await prov._run(["llm"])

        asyncio.run(run())
    finally:
        server.shutdown()

    assert prov.snapshot()["items"]["llm"]["state"] == "ok"
    seen = [e["payload"]["items"]["llm"] for e in hub.events if e["type"] == "setup_progress"]
    assert any(it["done_bytes"] == 100 and it["total_bytes"] == 400 for it in seen)
    assert any(it["state"] == "downloading" for it in seen)


def test_pull_llm_error_becomes_item_error():
    server = _ollama_stub([{"error": "model not found"}])
    try:
        cfg = AppConfig()
        cfg.llm.base_url = f"http://127.0.0.1:{server.server_address[1]}"
        prov = make_provisioner(cfg=cfg, llm=FakeLLM(model_found=False))

        async def run():
            await prov.refresh()
            await prov._run(["llm"])

        asyncio.run(run())
    finally:
        server.shutdown()

    item = prov.snapshot()["items"]["llm"]
    assert item["state"] == "error" and "model not found" in item["message"]


def test_fetch_asr_reports_growing_cache(tmp_path, monkeypatch):
    """The ASR weights progress is measured by the growth of the folder in the HF cache."""
    repo = tmp_path / "models--org--repo"
    repo.mkdir()
    monkeypatch.setattr("app.provision.asr_repo_id", lambda name: "org/repo")
    monkeypatch.setattr("app.provision._repo_dir", lambda repo_id: repo)

    def fake_download(self):
        for i in range(1, 4):
            (repo / f"part{i}").write_bytes(b"x" * 1_000_000)
            time.sleep(0.4)

    monkeypatch.setattr(Provisioner, "_load_asr_blocking", fake_download)

    hub = FakeHub()
    prov = make_provisioner(hub=hub)
    asyncio.run(prov._fetch_asr())

    assert prov.snapshot()["items"]["asr"]["state"] == "ok"
    progress = [e["payload"]["items"]["asr"]["done_bytes"]
                for e in hub.events if e["type"] == "setup_progress"]
    assert max(progress) > 0, "the download progress was never updated"


def test_start_is_idempotent_while_running():
    prov = make_provisioner()

    async def run():
        prov._items["asr"]["state"] = "missing"
        monkeyed = asyncio.Event()

        async def slow(_wanted):
            await monkeyed.wait()

        prov._run = slow
        first = prov.start(["asr"])
        assert first["running"] is True
        task = prov._task
        prov.start(["asr"])  # a second click on the button does not spawn more tasks
        assert prov._task is task
        monkeyed.set()
        await task

    asyncio.run(run())


# ------------------------------------------------------------------ loopback

def test_loopback_index_boundary():
    base = loopback.LOOPBACK_INDEX_BASE
    assert loopback.is_loopback_index(None) is False
    assert loopback.is_loopback_index(0) is False
    assert loopback.is_loopback_index(base - 1) is False
    assert loopback.is_loopback_index(base) is True


def test_loopback_list_empty_when_unavailable(monkeypatch):
    monkeypatch.setattr(loopback, "available", lambda: False)
    assert loopback.list_loopback_devices() == []


def test_loopback_unavailable_off_windows(monkeypatch):
    monkeypatch.setattr(loopback.sys, "platform", "darwin")
    assert loopback.available() is False


@pytest.mark.skipif(not loopback.available(), reason="needs Windows with PyAudioWPatch")
def test_loopback_devices_have_offset_indices():
    devices = loopback.list_loopback_devices()
    assert all(d["index"] >= loopback.LOOPBACK_INDEX_BASE for d in devices)
    assert all(d["is_loopback"] and d["max_input_channels"] > 0 for d in devices)
