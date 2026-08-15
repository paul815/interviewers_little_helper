"""Client for the local Ollama: health check, chat with structured JSON output,
resilient parsing and a retry when the answer is invalid."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

import httpx

from ..config import LLMConfig

log = logging.getLogger("ilh.llm")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


class OllamaError(Exception):
    """An error carrying a message that is ready for the UI."""


def robust_json_parse(text: str) -> dict:
    """Strips <think> blocks and markdown fences, then cuts out the JSON object."""
    cleaned = _THINK_RE.sub("", text)
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("the response contains no JSON object")
    return json.loads(cleaned[start : end + 1])


@dataclass
class ChatMeta:
    duration_s: float
    prompt_chars: int
    raw_response: str
    eval_count: int | None = None
    prompt_eval_count: int | None = None
    retried: bool = False


class OllamaClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._supports_think_param = True
        self._supports_schema_format = True

    async def check(self) -> dict:
        """Server status and model presence; errors in human-readable form."""
        result = {"ok": False, "server_up": False, "model_found": False,
                  "model": self.cfg.model, "version": None, "error": None}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                ver = await client.get(f"{self.cfg.base_url}/api/version")
                ver.raise_for_status()
                result["server_up"] = True
                result["version"] = ver.json().get("version")

                tags = await client.get(f"{self.cfg.base_url}/api/tags")
                tags.raise_for_status()
                names = [m.get("name", "") for m in tags.json().get("models", [])]
                want = self.cfg.model
                result["model_found"] = any(
                    n == want or (":" not in want and n.split(":")[0] == want) for n in names
                )
                if not result["model_found"]:
                    result["error"] = (
                        f"Model «{want}» was not found in Ollama. "
                        f"Run: ollama pull {want}"
                    )
        except (httpx.ConnectError, httpx.TimeoutException):
            result["error"] = (
                f"Ollama is not answering at {self.cfg.base_url}. "
                "Start the Ollama application (or `ollama serve`)."
            )
        except httpx.HTTPError as e:
            result["error"] = f"Error requesting Ollama: {e}"
        result["ok"] = result["server_up"] and result["model_found"]
        return result

    async def chat_json(self, system: str, user: str, schema: dict) -> tuple[dict, ChatMeta]:
        """One chat call with a strict JSON answer. On invalid JSON, one retry
        that carries the error text."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        started = time.monotonic()
        raw, data = await self._chat_once(messages, schema)
        retried = False
        try:
            parsed = robust_json_parse(raw)
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("Invalid JSON from the LLM (%s) — retrying the request", e)
            retried = True
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user",
                 "content": f"Your answer is not valid JSON ({e}). Return ONLY a valid "
                            f"JSON object matching the given schema, with no explanations."},
            ]
            raw, data = await self._chat_once(messages, schema)
            try:
                parsed = robust_json_parse(raw)
            except (ValueError, json.JSONDecodeError) as e2:
                raise OllamaError(f"The LLM returned invalid JSON twice: {e2}") from e2

        meta = ChatMeta(
            duration_s=time.monotonic() - started,
            prompt_chars=len(system) + len(user),
            raw_response=raw,
            eval_count=data.get("eval_count"),
            prompt_eval_count=data.get("prompt_eval_count"),
            retried=retried,
        )
        return parsed, meta

    async def _chat_once(self, messages: list[dict], schema: dict) -> tuple[str, dict]:
        payload: dict = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "format": schema if self._supports_schema_format else "json",
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "num_ctx": self.cfg.num_ctx,
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.num_predict,
            },
        }
        if self._supports_think_param:
            payload["think"] = False  # for thinking models (qwen3): JSON without the reasoning

        timeout = httpx.Timeout(self.cfg.request_timeout_s, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for _ in range(3):  # downgrades for older Ollama versions
                try:
                    resp = await client.post(f"{self.cfg.base_url}/api/chat", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("message", {}).get("content", ""), data
                except httpx.ConnectError as e:
                    raise OllamaError(
                        f"Ollama is not answering at {self.cfg.base_url}. Is the server running?"
                    ) from e
                except httpx.TimeoutException as e:
                    raise OllamaError(
                        f"Ollama did not answer within {self.cfg.request_timeout_s:.0f} s "
                        "(is the model overloaded or offloaded to the CPU?)"
                    ) from e
                except httpx.HTTPStatusError as e:
                    body = e.response.text[:500]
                    if e.response.status_code == 404 and "model" in body.lower():
                        raise OllamaError(
                            f"Model «{self.cfg.model}» was not found. "
                            f"Run: ollama pull {self.cfg.model}"
                        ) from e
                    if (e.response.status_code == 400 and "think" in body.lower()
                            and self._supports_think_param):
                        log.info("Ollama does not accept the think parameter — disabling it")
                        self._supports_think_param = False
                        payload.pop("think", None)
                        continue
                    if (e.response.status_code == 400 and "format" in body.lower()
                            and self._supports_schema_format):
                        log.info(
                            "Ollama does not accept a JSON schema in format — "
                            "falling back to format=json"
                        )
                        self._supports_schema_format = False
                        payload["format"] = "json"
                        continue
                    raise OllamaError(f"Ollama error {e.response.status_code}: {body}") from e
        raise OllamaError("Ollama rejected the request after every attempt")
