"""Parsing free-form guide text into a structure via the LLM (once per session).

Section/topic identifiers are assigned deterministically on our side — the LLM
returns only the structure, without ids.
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

SYSTEM_PARSE = """You are an assistant to a qualitative researcher. You are given the text of an \
interview guide in free form. Turn it into a structure: sections, and topics/questions inside them.

Rules:
- Keep the original language and wording, invent nothing and add nothing of your own.
- Every self-contained question or topic is a separate topics element. If one line holds several \
distinct questions by meaning, split them.
- Explanations and hints for the interviewer ("if they go quiet, ask about…") go into \
the topic's notes.
- If the text has no explicit sections, create one section with a meaningful name.
- title is a short name for the whole guide (make one up from the content, in the guide's language).
- language is the guide's main language as an ISO 639-1 code ("ru", "en", …).

The answer is strictly JSON matching the schema."""


async def parse_guide_text(llm: OllamaClient, raw_text: str, system: str = "") -> Guide:
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("The guide text is empty")
    system = system or SYSTEM_PARSE
    parsed, meta = await llm.chat_json(system, f"GUIDE TEXT:\n\n{raw_text}", GUIDE_PARSE_SCHEMA)
    log.info("Guide parsed in %.1f s (retried: %s)", meta.duration_s, meta.retried)
    return build_guide(parsed)


def build_guide(parsed: dict) -> Guide:
    """Assembles a Guide from the raw LLM response, assigning deterministic ids."""
    sections = []
    for si, sec in enumerate(parsed.get("sections") or [], start=1):
        topics = []
        for ti, topic in enumerate(sec.get("topics") or [], start=1):
            question = (topic.get("question") or "").strip()
            if not question:
                continue
            notes = (topic.get("notes") or "").strip() or None
            topics.append(Topic(id=f"s{si}.t{ti}", question=question, notes=notes))
        title = (sec.get("title") or f"Section {si}").strip()
        if topics:
            sections.append(Section(id=f"s{si}", title=title, topics=topics))
    if not sections:
        raise ValueError("Could not extract a single topic from the guide text")
    return Guide(
        guide_id=datetime.now().strftime("g-%Y%m%d-%H%M%S"),
        language=(parsed.get("language") or "en").strip().lower()[:5],
        title=(parsed.get("title") or "Interview guide").strip(),
        sections=sections,
    )
