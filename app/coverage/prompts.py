"""Промпты цикла анализа. Гайд кладём в системный промпт: он статичен всю
сессию, и Ollama переиспользует KV-кэш этого префикса на каждом цикле."""
from __future__ import annotations

from ..domain import SPEAKER_SHORT_RU, Segment, fmt_ts
from ..guide.schemas import Guide
from .schemas import CoverageState

_SYSTEM_TEMPLATE = """Ты — ассистент качественного исследователя, работающий во время интервью. \
Тебе периодически передают новый фрагмент транскрипта; ты отслеживаешь, какие темы гайда \
уже раскрыты, и подсказываешь, что спросить.

ГАЙД ИНТЕРВЬЮ (язык: {language}):
{guide_block}

ПРАВИЛА:
1. Сопоставляй темы с разговором ПО СМЫСЛУ, а не по совпадению слов. Тема covered, если \
респондент содержательно на неё ответил; partial — если затронута мельком или раскрыта не \
полностью; иначе not_covered.
2. В topic_updates возвращай ТОЛЬКО темы, чей статус улучшился относительно текущего состояния \
(not_covered -> partial -> covered). Статусы никогда не понижаются. Если изменений нет — пустой список.
3. Для каждого обновления давай evidence: короткий (до 15 слов) парафраз того, чем тема закрыта, \
на языке гайда. confidence — число 0..1.
4. В recommendations дай рекомендации типа "coverage_gap" для тем, которые ПОСЛЕ твоих обновлений \
остаются not_covered или partial: краткое note (что именно не раскрыто) и suggested_question — \
готовая естественная формулировка вопроса на языке гайда, которую интервьюер может произнести \
дословно. Для partial-тем вопрос должен добирать именно недостающее.
5. urgency="high" — для полностью непокрытых важных тем; иначе "normal". Не больше {max_recs} \
рекомендаций, в первую очередь not_covered.
6. Транскрипт распознан автоматически и может содержать ошибки и смесь языков — интерпретируй по смыслу.
7. Используй только topic_id из гайда.

Отвечай строго JSON по заданной схеме."""


def guide_block(guide: Guide) -> str:
    lines = []
    for sec in guide.sections:
        lines.append(f"[{sec.id}] {sec.title}")
        for t in sec.topics:
            note = f" (примечание: {t.notes})" if t.notes else ""
            lines.append(f"  ({t.id}) {t.question}{note}")
    return "\n".join(lines)


def system_prompt(guide: Guide, max_recs: int) -> str:
    return _SYSTEM_TEMPLATE.format(
        language=guide.language, guide_block=guide_block(guide), max_recs=max_recs
    )


def state_block(state: CoverageState, guide: Guide) -> str:
    lines = []
    for _, topic in guide.all_topics():
        st = state.topics.get(topic.id)
        if st is None:
            continue
        line = f"{topic.id} = {st.status}"
        if st.status == "partial" and st.evidence:
            line += f" ({st.evidence})"
        lines.append(line)
    return "\n".join(lines)


def delta_block(segments: list[Segment], max_chars: int) -> str:
    """Дельта транскрипта: реплики подряд от одного говорящего склеиваются."""
    lines: list[str] = []
    last_speaker = None
    last_t1 = -10.0
    for seg in segments:
        if seg.speaker == last_speaker and seg.t0 - last_t1 < 2.0 and lines:
            lines[-1] += " " + seg.text
        else:
            lines.append(f"[{fmt_ts(seg.t0)}] {SPEAKER_SHORT_RU[seg.speaker]}: {seg.text}")
        last_speaker, last_t1 = seg.speaker, seg.t1
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = "(…начало фрагмента опущено из-за объёма…)\n" + text[-max_chars:]
    return text


def user_prompt(state: CoverageState, guide: Guide, segments: list[Segment], max_delta_chars: int) -> str:
    return (
        "ТЕКУЩЕЕ СОСТОЯНИЕ ПОКРЫТИЯ:\n"
        f"{state_block(state, guide)}\n\n"
        "НОВЫЙ ФРАГМЕНТ ТРАНСКРИПТА (И — интервьюер, Р — респондент):\n"
        f"{delta_block(segments, max_delta_chars)}\n\n"
        "Обнови покрытие и дай рекомендации."
    )
