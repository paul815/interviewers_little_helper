"""Клиент локального Ollama: health-check, чат со структурированным JSON-выводом,
устойчивый парсинг и повторный запрос при невалидном ответе."""
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
    """Ошибка с готовым для UI сообщением."""


def robust_json_parse(text: str) -> dict:
    """Срезает <think>-блоки и markdown-ограждения, вырезает JSON-объект."""
    cleaned = _THINK_RE.sub("", text)
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("в ответе нет JSON-объекта")
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
        """Статус сервера и наличие модели; ошибки — в человекочитаемом виде."""
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
                        f"Модель «{want}» не найдена в Ollama. "
                        f"Выполните: ollama pull {want}"
                    )
        except (httpx.ConnectError, httpx.TimeoutException):
            result["error"] = (
                f"Ollama не отвечает на {self.cfg.base_url}. "
                "Запустите приложение Ollama (или `ollama serve`)."
            )
        except httpx.HTTPError as e:
            result["error"] = f"Ошибка запроса к Ollama: {e}"
        result["ok"] = result["server_up"] and result["model_found"]
        return result

    async def chat_json(self, system: str, user: str, schema: dict) -> tuple[dict, ChatMeta]:
        """Один диалоговый вызов со строгим JSON-ответом. При невалидном JSON —
        один повторный запрос с текстом ошибки."""
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
            log.warning("Невалидный JSON от LLM (%s) — повторяю запрос", e)
            retried = True
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user",
                 "content": f"Твой ответ не является валидным JSON ({e}). "
                            f"Верни ТОЛЬКО валидный JSON-объект по заданной схеме, без пояснений."},
            ]
            raw, data = await self._chat_once(messages, schema)
            try:
                parsed = robust_json_parse(raw)
            except (ValueError, json.JSONDecodeError) as e2:
                raise OllamaError(f"LLM дважды вернула невалидный JSON: {e2}") from e2

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
            payload["think"] = False  # для thinking-моделей (qwen3): JSON без размышлений

        timeout = httpx.Timeout(self.cfg.request_timeout_s, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for _ in range(3):  # даунгрейды для старых версий Ollama
                try:
                    resp = await client.post(f"{self.cfg.base_url}/api/chat", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("message", {}).get("content", ""), data
                except httpx.ConnectError as e:
                    raise OllamaError(
                        f"Ollama не отвечает на {self.cfg.base_url}. Сервер запущен?"
                    ) from e
                except httpx.TimeoutException as e:
                    raise OllamaError(
                        f"Ollama не ответила за {self.cfg.request_timeout_s:.0f} c "
                        "(модель перегружена или выгружена на CPU?)"
                    ) from e
                except httpx.HTTPStatusError as e:
                    body = e.response.text[:500]
                    if e.response.status_code == 404 and "model" in body.lower():
                        raise OllamaError(
                            f"Модель «{self.cfg.model}» не найдена. Выполните: ollama pull {self.cfg.model}"
                        ) from e
                    if e.response.status_code == 400 and "think" in body.lower() and self._supports_think_param:
                        log.info("Ollama не принимает параметр think — отключаю")
                        self._supports_think_param = False
                        payload.pop("think", None)
                        continue
                    if e.response.status_code == 400 and "format" in body.lower() and self._supports_schema_format:
                        log.info("Ollama не принимает JSON-схему в format — перехожу на format=json")
                        self._supports_schema_format = False
                        payload["format"] = "json"
                        continue
                    raise OllamaError(f"Ошибка Ollama {e.response.status_code}: {body}") from e
        raise OllamaError("Ollama отклонила запрос после всех попыток")
