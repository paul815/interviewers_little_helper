"""Prompts for the analysis cycle. The guide goes into the system prompt: it is
static for the whole session, so Ollama reuses the KV cache of that prefix on
every cycle.

The texts below are defaults. The researcher can rewrite them in "Settings"
(app/storage/prompt_store.py) and add project-level instructions on top; both
arrive here as the templates/extra parameters.

Two languages are in play and they are not the same one. `guide_language` is
the language the guide itself is written in; `language` is the language the
model must answer in, resolved from analysis.output_language ("auto" follows
the guide).
"""
from __future__ import annotations

from collections.abc import Mapping

from ..domain import SPEAKER_SHORT, Segment, fmt_ts
from ..guide.schemas import Guide
from .schemas import CoverageState

SYSTEM_LIVE = """You are an assistant to a qualitative researcher, working during an interview. \
You are periodically handed a new fragment of the transcript; you track which guide topics have \
already been covered and suggest what to ask next.

INTERVIEW GUIDE (language: {guide_language}):
{guide_block}

RULES:
1. Match topics to the conversation BY MEANING, not by word overlap. A topic is covered if the \
respondent answered it substantively; partial if it was touched on in passing or answered \
incompletely; otherwise not_covered.
2. In topic_updates return ONLY topics whose status improved relative to the current state \
(not_covered -> partial -> covered). Statuses never go down. If nothing changed, return an empty list.
3. For every update give evidence: a short (up to 15 words) paraphrase of what closed the topic, \
written in {language}. confidence is a number in 0..1.
4. In recommendations give "coverage_gap" recommendations for topics that AFTER your updates \
remain not_covered or partial: topic_id, a brief note (what exactly is missing) and \
suggested_question — a ready, natural wording in {language} that the interviewer can say verbatim. \
For partial topics the question must pick up exactly what is missing. \
urgency="high" for important topics that are completely uncovered; otherwise "normal". No more than \
{max_recs} such recommendations, not_covered topics first.
5. {probes_rule}
6. The transcript is recognised automatically and may contain errors and a mix of languages — \
interpret it by meaning.
7. Use only topic_id values from the guide.

Answer strictly as JSON matching the given schema."""

PROBES_RULE = """Additionally: if in the new fragment the respondent said something unexpected, \
contradictory, emotional, or mentioned in passing something potentially important for the research \
— add up to {max_probes} recommendations of type "probe": quote (a short trigger quote from the \
fragment), note (why it is worth digging into) and suggested_question (a follow-up question in \
{language}). Give probes only for the FRESH fragment and only when there is a real hook; set \
topic_id if the probe relates to a guide topic, otherwise omit it."""

PROBES_OFF = 'Do not use recommendations of type "probe".'

SYSTEM_FINAL = """You are an assistant to a qualitative researcher. The interview is over; you are \
doing a final reconciliation of the transcript against the guide for the report.

INTERVIEW GUIDE (language: {guide_language}):
{guide_block}

RULES:
1. Match topics to the fragment BY MEANING. In topic_updates return topics whose status in this \
fragment turned out BETTER than the current state (not_covered -> partial -> covered), with \
evidence (a short paraphrase, up to 15 words, in {language}) and confidence in 0..1. Pay particular \
attention to topics currently listed as not_covered — they may have been missed during the interview.
2. In findings collect the points for the report: for every topic substantively touched on in this \
fragment, 1-2 short facts of "what we learned" (specifics: practices, numbers, quotes, pain points), \
written in {language}. Each point is a separate {{topic_id, finding}} element.
3. The transcript is recognised automatically and may contain errors — interpret it by meaning.
4. Use only topic_id values from the guide.

Answer strictly as JSON matching the given schema."""


def resolve_output_language(guide: Guide, output_language: str = "auto") -> str:
    """Language the model must answer in.

    "auto" (the default) follows the guide, which is what a researcher running
    an interview in their own language expects. Any other value is passed to the
    model as-is, so both "en" and "English" work.
    """
    value = (output_language or "auto").strip()
    if not value or value.lower() == "auto":
        return guide.language
    return value


def guide_block(guide: Guide) -> str:
    lines = []
    for sec in guide.sections:
        lines.append(f"[{sec.id}] {sec.title}")
        for t in sec.topics:
            note = f" (note: {t.notes})" if t.notes else ""
            lines.append(f"  ({t.id}) {t.question}{note}")
    return "\n".join(lines)


_EXTRA_HEADER = (
    "ADDITIONAL INSTRUCTIONS FOR THIS PROJECT (treat them on a par with the rules above; "
    "they do not change the response format — JSON matching the schema is still required):"
)


def extra_block(text: str) -> str:
    """Project instructions are appended AFTER the template rather than
    substituted into it: that way they are not lost if the researcher rewrites
    the template itself."""
    text = (text or "").strip()
    return f"\n\n{_EXTRA_HEADER}\n{text}" if text else ""


def system_prompt_live(
    guide: Guide,
    max_recs: int,
    max_probes: int,
    templates: Mapping[str, str] | None = None,
    extra: str = "",
    output_language: str = "auto",
) -> str:
    tpl = templates or {}
    language = resolve_output_language(guide, output_language)
    probes_rule = (
        tpl.get("probes_rule", PROBES_RULE).format(max_probes=max_probes, language=language)
        if max_probes > 0
        else PROBES_OFF
    )
    body = tpl.get("system_live", SYSTEM_LIVE).format(
        language=language,
        guide_language=guide.language,
        guide_block=guide_block(guide),
        max_recs=max_recs,
        probes_rule=probes_rule,
    )
    return body + extra_block(extra)


def system_prompt_final(
    guide: Guide,
    templates: Mapping[str, str] | None = None,
    extra: str = "",
    output_language: str = "auto",
) -> str:
    body = (templates or {}).get("system_final", SYSTEM_FINAL).format(
        language=resolve_output_language(guide, output_language),
        guide_language=guide.language,
        guide_block=guide_block(guide),
    )
    return body + extra_block(extra)


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
    """A transcript fragment: consecutive utterances by the same speaker are merged."""
    lines: list[str] = []
    last_speaker = None
    last_t1 = -10.0
    for seg in segments:
        if seg.speaker == last_speaker and seg.t0 - last_t1 < 2.0 and lines:
            lines[-1] += " " + seg.text
        else:
            lines.append(f"[{fmt_ts(seg.t0)}] {SPEAKER_SHORT[seg.speaker]}: {seg.text}")
        last_speaker, last_t1 = seg.speaker, seg.t1
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = "(…the start of the fragment was dropped for length…)\n" + text[-max_chars:]
    return text


_MODE_HEADERS = {
    "delta": "NEW TRANSCRIPT FRAGMENT (I — interviewer, R — respondent):",
    "reconcile": (
        "RECONCILIATION FRAGMENT — a re-check of the last few minutes of the conversation. "
        "Look for topics that may have been missed in previous cycles "
        "(I — interviewer, R — respondent):"
    ),
    "final": "FRAGMENT OF THE FULL TRANSCRIPT (I — interviewer, R — respondent):",
}


def user_prompt(
    state: CoverageState,
    guide: Guide,
    segments: list[Segment],
    max_delta_chars: int,
    mode: str = "delta",
) -> str:
    tail = (
        "Update the coverage, collect findings."
        if mode == "final"
        else "Update the coverage and give recommendations."
    )
    return (
        "CURRENT COVERAGE STATE:\n"
        f"{state_block(state, guide)}\n\n"
        f"{_MODE_HEADERS[mode]}\n"
        f"{delta_block(segments, max_delta_chars)}\n\n"
        f"{tail}"
    )
