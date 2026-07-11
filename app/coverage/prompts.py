"""Промпты цикла анализа. Гайд кладём в системный промпт: он статичен всю
сессию, и Ollama переиспользует KV-кэш этого префикса на каждом цикле."""
from __future__ import annotations

from ..domain import SPEAKER_SHORT_RU, Segment, fmt_ts
from ..guide.schemas import Guide
from .schemas import CoverageState

_SYSTEM_LIVE = """Ты — ассистент качественного исследователя, работающий во время интервью. \
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
остаются not_covered или partial: topic_id, краткое note (что именно не раскрыто) и \
suggested_question — готовая естественная формулировка вопроса на языке гайда, которую интервьюер \
может произнести дословно. Для partial-тем вопрос должен добирать именно недостающее. \
urgency="high" — для полностью непокрытых важных тем; иначе "normal". Не больше {max_recs} таких \
рекомендаций, в первую очередь not_covered.
5. {probes_rule}
6. Транскрипт распознан автоматически и может содержать ошибки и смесь языков — интерпретируй по смыслу.
7. Используй только topic_id из гайда.

Отвечай строго JSON по заданной схеме."""

_PROBES_RULE = """Дополнительно: если в новом фрагменте респондент сказал что-то неожиданное, \
противоречивое, эмоциональное или мельком упомянул потенциально важное для исследования — добавь \
до {max_probes} рекомендаций типа "probe": quote (короткая цитата-триггер из фрагмента), note \
(почему это стоит копнуть) и suggested_question (уточняющий вопрос на языке гайда). Пробы давай \
только по СВЕЖЕМУ фрагменту и только когда есть реальная зацепка; topic_id указывай, если проба \
относится к теме гайда, иначе опусти."""

_PROBES_OFF = 'Рекомендации типа "probe" не используй.'

_SYSTEM_FINAL = """Ты — ассистент качественного исследователя. Интервью закончилось; ты делаешь \
финальную сверку транскрипта с гайдом для отчёта.

ГАЙД ИНТЕРВЬЮ (язык: {language}):
{guide_block}

ПРАВИЛА:
1. Сопоставляй темы с фрагментом ПО СМЫСЛУ. В topic_updates верни темы, чей статус в этом \
фрагменте оказался ЛУЧШЕ текущего состояния (not_covered -> partial -> covered), с evidence \
(короткий парафраз до 15 слов) и confidence 0..1. Особо внимательно проверь темы, которые сейчас \
числятся not_covered — их могли пропустить во время интервью.
2. В findings собери тезисы для отчёта: для каждой темы, содержательно затронутой в этом \
фрагменте, — 1-2 коротких факта «что мы узнали» (конкретика: практики, цифры, цитаты, боли), \
на языке гайда. Каждый тезис — отдельный элемент {{topic_id, finding}}.
3. Транскрипт распознан автоматически и может содержать ошибки — интерпретируй по смыслу.
4. Используй только topic_id из гайда.

Отвечай строго JSON по заданной схеме."""


def guide_block(guide: Guide) -> str:
    lines = []
    for sec in guide.sections:
        lines.append(f"[{sec.id}] {sec.title}")
        for t in sec.topics:
            note = f" (примечание: {t.notes})" if t.notes else ""
            lines.append(f"  ({t.id}) {t.question}{note}")
    return "\n".join(lines)


def system_prompt_live(guide: Guide, max_recs: int, max_probes: int) -> str:
    probes_rule = (
        _PROBES_RULE.format(max_probes=max_probes) if max_probes > 0 else _PROBES_OFF
    )
    return _SYSTEM_LIVE.format(
        language=guide.language,
        guide_block=guide_block(guide),
        max_recs=max_recs,
        probes_rule=probes_rule,
    )


def system_prompt_final(guide: Guide) -> str:
    return _SYSTEM_FINAL.format(language=guide.language, guide_block=guide_block(guide))


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
    """Фрагмент транскрипта: реплики подряд от одного говорящего склеиваются."""
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


_MODE_HEADERS = {
    "delta": "НОВЫЙ ФРАГМЕНТ ТРАНСКРИПТА (И — интервьюер, Р — респондент):",
    "reconcile": (
        "СВЕРОЧНЫЙ ФРАГМЕНТ — повторная проверка последних минут разговора. "
        "Ищи темы, которые могли быть пропущены в предыдущих циклах "
        "(И — интервьюер, Р — респондент):"
    ),
    "final": "ФРАГМЕНТ ПОЛНОГО ТРАНСКРИПТА (И — интервьюер, Р — респондент):",
}


def user_prompt(
    state: CoverageState,
    guide: Guide,
    segments: list[Segment],
    max_delta_chars: int,
    mode: str = "delta",
) -> str:
    tail = (
        "Обнови покрытие, собери findings."
        if mode == "final"
        else "Обнови покрытие и дай рекомендации."
    )
    return (
        "ТЕКУЩЕЕ СОСТОЯНИЕ ПОКРЫТИЯ:\n"
        f"{state_block(state, guide)}\n\n"
        f"{_MODE_HEADERS[mode]}\n"
        f"{delta_block(segments, max_delta_chars)}\n\n"
        f"{tail}"
    )
