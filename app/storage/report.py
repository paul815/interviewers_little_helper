"""Building the session report (report.md).

The deterministic part is assembled from the final coverage state and the
findings gathered by the final pass; the short summary is one small LLM call
(optional: the report is still valid without it).
"""
from __future__ import annotations

from ..coverage.prompts import resolve_output_language
from ..coverage.schemas import CoverageState
from ..domain import fmt_ts
from ..guide.schemas import Guide

STATUS_MARK = {"covered": "✅", "partial": "◐", "not_covered": "❌"}
STATUS_LABEL = {"covered": "covered", "partial": "partial", "not_covered": "not covered"}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

SUMMARY_SYSTEM = """You are an assistant to a qualitative researcher. From the points collected \
during the interview, write a short session summary: 4-7 sentences in {language}. Only facts from \
the points, no speculation; highlight the main pain points/findings and what was left uncovered. \
Answer strictly as JSON matching the schema."""


def summary_system_prompt(guide: Guide, output_language: str = "auto") -> str:
    return SUMMARY_SYSTEM.format(language=resolve_output_language(guide, output_language))


def summary_user_prompt(guide: Guide, state: CoverageState, findings: dict[str, list[str]]) -> str:
    lines = [f"Guide: «{guide.title}»", "", "POINTS BY TOPIC:"]
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
        lines += ["", "NOT COVERED:"] + [f"- {q}" for q in not_covered]
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
        f"# Session report — {session_id}",
        "",
        f"Guide: «{guide.title}»  ",
        f"Started: {meta.get('started_at', '—')}  ",
        f"Transcript segments: {meta.get('segments', '—')}  ",
        f"Coverage: **{counts['covered']} covered · {counts['partial']} partial · "
        f"{counts['not_covered']} not covered** (of {counts['total']})",
        "",
    ]

    if summary:
        out += ["## Summary", "", summary, ""]

    out.append("## Topics")
    for sec in guide.sections:
        out += ["", f"### {sec.title}", ""]
        for t in sec.topics:
            st = state.topics.get(t.id)
            status = st.status if st else "not_covered"
            out.append(f"**{STATUS_MARK[status]} {t.question}** — {STATUS_LABEL[status]}")
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
        out += ["## Not covered (for the next interview)", ""]
        out += [f"- {sec}: {q}" for sec, q in not_covered]
        out.append("")

    if flags:
        out += ["## Flagged moments 🚩", ""]
        for fl in flags:
            note = f" — {fl['note']}" if fl.get("note") else ""
            out.append(f"- [{fmt_ts(fl['t'])}]{note}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"
