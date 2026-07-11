"""Сборка отчёта сессии (report.md).

Детерминированная часть собирается из финального состояния покрытия и
findings, накопленных финальным проходом; краткое резюме — один небольшой
вызов LLM (опционален: без него отчёт всё равно валиден).
"""
from __future__ import annotations

from ..domain import fmt_ts
from ..guide.schemas import Guide
from ..coverage.schemas import CoverageState

STATUS_MARK = {"covered": "✅", "partial": "◐", "not_covered": "❌"}
STATUS_RU = {"covered": "покрыто", "partial": "частично", "not_covered": "не покрыто"}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

SUMMARY_SYSTEM = """Ты — ассистент качественного исследователя. По тезисам, собранным из \
интервью, напиши краткое резюме сессии: 4–7 предложений на языке гайда. Только факты из \
тезисов, без домыслов; выдели главные боли/находки и что осталось нераскрытым. \
Ответ — строго JSON по схеме."""


def summary_user_prompt(guide: Guide, state: CoverageState, findings: dict[str, list[str]]) -> str:
    lines = [f"Гайд: «{guide.title}»", "", "ТЕЗИСЫ ПО ТЕМАМ:"]
    for sec in guide.sections:
        for t in sec.topics:
            for f in findings.get(t.id, []):
                lines.append(f"- [{sec.title} / {t.question}] {f}")
    not_covered = [
        t.question
        for _, t in guide.all_topics()
        if state.topics.get(t.id) and state.topics[t.id].status == "not_covered"
    ]
    if not_covered:
        lines += ["", "НЕ РАСКРЫТО:"] + [f"- {q}" for q in not_covered]
    return "\n".join(lines)


def build_report_markdown(
    session_id: str,
    guide: Guide,
    state: CoverageState,
    findings: dict[str, list[str]],
    flags: list[dict],
    meta: dict,
    summary: str | None,
) -> str:
    counts = state.counts()
    out: list[str] = [
        f"# Отчёт сессии — {session_id}",
        "",
        f"Гайд: «{guide.title}»  ",
        f"Начало: {meta.get('started_at', '—')}  ",
        f"Сегментов транскрипта: {meta.get('segments', '—')}  ",
        f"Покрытие: **{counts['covered']} covered · {counts['partial']} partial · "
        f"{counts['not_covered']} not covered** (всего {counts['total']})",
        "",
    ]

    if summary:
        out += ["## Резюме", "", summary, ""]

    out.append("## Темы")
    for sec in guide.sections:
        out += ["", f"### {sec.title}", ""]
        for t in sec.topics:
            st = state.topics.get(t.id)
            status = st.status if st else "not_covered"
            out.append(f"**{STATUS_MARK[status]} {t.question}** — {STATUS_RU[status]}")
            if st and st.evidence:
                out.append(f"  *({st.evidence})*")
            for f in findings.get(t.id, []):
                out.append(f"- {f}")
            out.append("")

    not_covered = [
        (sec.title, t.question)
        for sec in guide.sections
        for t in sec.topics
        if state.topics.get(t.id) and state.topics[t.id].status == "not_covered"
    ]
    if not_covered:
        out += ["## Не раскрыто (для следующего интервью)", ""]
        out += [f"- {sec}: {q}" for sec, q in not_covered]
        out.append("")

    if flags:
        out += ["## Отмеченные моменты 🚩", ""]
        for fl in flags:
            note = f" — {fl['note']}" if fl.get("note") else ""
            out.append(f"- [{fmt_ts(fl['t'])}]{note}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"
