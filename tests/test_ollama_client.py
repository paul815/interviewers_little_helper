"""The Ollama client against a local HTTP stub.

Ollama itself is not needed here: we bring up our own server on an ephemeral port
and answer with whatever the particular test needs. The focus is on the failure
modes — they decide whether the researcher sees a clear error or a silent screen.
"""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.config import LLMConfig
from app.llm.ollama_client import OllamaClient, OllamaError, robust_json_parse

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def _serve(routes: dict):
    """routes: path -> (status, body), or a list of answers in turn."""
    state = {k: (list(v) if isinstance(v, list) else v) for k, v in routes.items()}
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, path, body_in=None):
            entry = state.get(path)
            if entry is None:
                self.send_error(404)
                return
            if isinstance(entry, list):
                status, body = entry.pop(0) if len(entry) > 1 else entry[0]
            else:
                status, body = entry
            payload = json.dumps(body).encode() if isinstance(body, dict) else body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self._reply(self.path)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            seen.append(json.loads(self.rfile.read(n) or b"{}"))
            self._reply(self.path)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = LLMConfig(base_url=f"http://127.0.0.1:{srv.server_address[1]}", request_timeout_s=5)
    return srv, cfg, seen


@pytest.fixture()
def stub():
    made = []

    def make(routes):
        srv, cfg, seen = _serve(routes)
        made.append(srv)
        return cfg, seen

    yield make
    for srv in made:
        srv.shutdown()


# ---------------------------------------------------------------- robust parse

@pytest.mark.parametrize("raw,expected", [
    ('{"ok": true}', {"ok": True}),
    ('```json\n{"ok": true}\n```', {"ok": True}),
    ('<think>reasoning away</think>{"ok": true}', {"ok": True}),
    ('Here is the answer: {"ok": true} — that is all.', {"ok": True}),
])
def test_robust_json_parse_survives_llm_noise(raw, expected):
    assert robust_json_parse(raw) == expected


def test_robust_json_parse_gives_up_loudly():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        robust_json_parse("no json here at all")


# ------------------------------------------------------------------- check()

def test_check_ok_when_model_present(stub):
    cfg, _ = stub({
        "/api/version": (200, {"version": "0.5.0"}),
        "/api/tags": (200, {"models": [{"name": "qwen3:8b"}]}),
    })
    st = asyncio.run(OllamaClient(cfg).check())
    assert st["ok"] and st["server_up"] and st["model_found"]
    assert st["version"] == "0.5.0" and st["error"] is None


def test_check_matches_model_without_tag(stub):
    """The config says «qwen3», the server reports «qwen3:8b». Same model."""
    cfg, _ = stub({
        "/api/version": (200, {"version": "0.5.0"}),
        "/api/tags": (200, {"models": [{"name": "qwen3:8b"}]}),
    })
    cfg.model = "qwen3"
    assert asyncio.run(OllamaClient(cfg).check())["model_found"]


def test_check_reports_missing_model(stub):
    cfg, _ = stub({
        "/api/version": (200, {"version": "0.5.0"}),
        "/api/tags": (200, {"models": [{"name": "llama3:8b"}]}),
    })
    st = asyncio.run(OllamaClient(cfg).check())
    assert st["server_up"] and not st["model_found"] and not st["ok"]
    assert "ollama pull" in st["error"]


def test_check_reports_server_down():
    cfg = LLMConfig(base_url="http://127.0.0.1:1", request_timeout_s=2)
    st = asyncio.run(OllamaClient(cfg).check())
    assert not st["server_up"] and "Ollama is not answering" in st["error"]


# ---------------------------------------------------------------- chat_json()

def test_chat_json_happy_path(stub):
    cfg, seen = stub({"/api/chat": (200, {
        "message": {"content": '{"ok": true}'}, "eval_count": 12, "prompt_eval_count": 34,
    })})
    parsed, meta = asyncio.run(OllamaClient(cfg).chat_json("sys", "usr", SCHEMA))
    assert parsed == {"ok": True}
    assert meta.eval_count == 12 and meta.prompt_eval_count == 34
    assert meta.retried is False and meta.prompt_chars == len("sys") + len("usr")
    assert seen[0]["think"] is False and seen[0]["format"] == SCHEMA
    assert seen[0]["stream"] is False


def test_chat_json_retries_once_on_garbage(stub):
    cfg, seen = stub({"/api/chat": [
        (200, {"message": {"content": "I am no good at JSON"}}),
        (200, {"message": {"content": '{"ok": true}'}}),
    ]})
    parsed, meta = asyncio.run(OllamaClient(cfg).chat_json("sys", "usr", SCHEMA))
    assert parsed == {"ok": True} and meta.retried is True
    # The retry carries both the bad answer and an explanation of what is wrong with it.
    assert len(seen[1]["messages"]) == 4


def test_chat_json_gives_up_after_second_garbage(stub):
    cfg, _ = stub({"/api/chat": (200, {"message": {"content": "not JSON again"}})})
    with pytest.raises(OllamaError, match="returned invalid JSON twice"):
        asyncio.run(OllamaClient(cfg).chat_json("sys", "usr", SCHEMA))


def test_missing_model_becomes_readable_error(stub):
    cfg, _ = stub({"/api/chat": (404, {"error": "model 'qwen3:8b' not found"})})
    with pytest.raises(OllamaError, match="ollama pull"):
        asyncio.run(OllamaClient(cfg).chat_json("sys", "usr", SCHEMA))


def test_old_ollama_without_think_param_is_retried_without_it(stub):
    """An old Ollama complains about think — the client must drop the flag and retry."""
    cfg, seen = stub({"/api/chat": [
        (400, {"error": "unknown field think"}),
        (200, {"message": {"content": '{"ok": true}'}}),
    ]})
    client = OllamaClient(cfg)
    parsed, _ = asyncio.run(client.chat_json("sys", "usr", SCHEMA))
    assert parsed == {"ok": True}
    assert client._supports_think_param is False
    assert "think" in seen[0] and "think" not in seen[1]


def test_old_ollama_without_schema_format_falls_back_to_json(stub):
    cfg, seen = stub({"/api/chat": [
        (400, {"error": "invalid format value"}),
        (200, {"message": {"content": '{"ok": true}'}}),
    ]})
    client = OllamaClient(cfg)
    parsed, _ = asyncio.run(client.chat_json("sys", "usr", SCHEMA))
    assert parsed == {"ok": True}
    assert client._supports_schema_format is False
    assert seen[0]["format"] == SCHEMA and seen[1]["format"] == "json"


def test_other_http_errors_surface_as_is(stub):
    cfg, _ = stub({"/api/chat": (500, {"error": "boom"})})
    with pytest.raises(OllamaError, match="Ollama error 500"):
        asyncio.run(OllamaClient(cfg).chat_json("sys", "usr", SCHEMA))


def test_server_down_during_chat(stub):
    cfg = LLMConfig(base_url="http://127.0.0.1:1", request_timeout_s=2)
    with pytest.raises(OllamaError, match="Is the server running"):
        asyncio.run(OllamaClient(cfg).chat_json("sys", "usr", SCHEMA))
