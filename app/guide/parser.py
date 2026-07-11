"""Парсинг свободного текста гайда в структуру через LLM (один раз на сессию).

Идентификаторы секций/тем назначаются детерминированно на нашей стороне —
LLM возвращает только структуру без id.
"""
from __future__ import annotations

import logging
from datetime import datetime

from ..llm.ollama_client import OllamaClient
from .schemas import Guide, Section, Topic

log = logging.getLogger("ilh.guide")

GUIDE_PARSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "language": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "topics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "notes": {"type": "string"},
                            },
                            "required": ["question"],
                        },
                    },
                },
                "required": ["title", "topics"],
            },
        },
    },
    "required": ["title", "language", "sections"],
}

_SYSTEM = """Ты — ассистент качественного исследователя. Тебе дают текст гайда интервью \
в свободной форме. Преобразуй его в структуру: секции, внутри — темы/вопросы.

Правила:
- Сохраняй язык и формулировки оригинала, ничего не выдумывай и не добавляй от себя.
- Каждый самостоятельный вопрос или тема — отдельный элемент topics. Если в одной строке \
несколько разных вопросов по смыслу — раздели их.
- Пояснения, подсказки интервьюеру («если молчит — спросить про…») клади в notes к теме.
- Если в тексте нет явных секций — создай одну секцию с осмысленным названием.
- title — короткое название всего гайда (придумай по содержанию, на языке гайда).
- language — основной язык гайда кодом ISO 639-1 («ru», «en», …).

Ответ — строго JSON по схеме."""


async def parse_guide_text(llm: OllamaClient, raw_text: str) -> Guide:
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Текст гайда пуст")
    parsed, meta = await llm.chat_json(_SYSTEM, f"ТЕКСТ ГАЙДА:\n\n{raw_text}", GUIDE_PARSE_SCHEMA)
    log.info("Гайд распарсен за %.1f c (повтор: %s)", meta.duration_s, meta.retried)
    return build_guide(parsed)


def build_guide(parsed: dict) -> Guide:
    """Собирает Guide из сырого ответа LLM, назначая детерминированные id."""
    sections = []
    for si, sec in enumerate(parsed.get("sections") or [], start=1):
        topics = []
        for ti, topic in enumerate(sec.get("topics") or [], start=1):
            question = (topic.get("question") or "").strip()
            if not question:
                continue
            notes = (topic.get("notes") or "").strip() or None
            topics.append(Topic(id=f"s{si}.t{ti}", question=question, notes=notes))
        title = (sec.get("title") or f"Секция {si}").strip()
        if topics:
            sections.append(Section(id=f"s{si}", title=title, topics=topics))
    if not sections:
        raise ValueError("Из текста гайда не удалось извлечь ни одной темы")
    return Guide(
        guide_id=datetime.now().strftime("g-%Y%m%d-%H%M%S"),
        language=(parsed.get("language") or "ru").strip().lower()[:5],
        title=(parsed.get("title") or "Гайд интервью").strip(),
        sections=sections,
    )
