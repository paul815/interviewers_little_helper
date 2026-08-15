"""User edits to the analysis prompts.

The default texts live in code (app/coverage/prompts.py, app/guide/parser.py)
and change with the application version. Only the overrides a researcher wrote
by hand are stored here: in a single prompts.json, so an edit can be carried to
another machine and "Reset" restores exactly the current default.

Placeholders like {guide_block} are substituted by the code. That is why the
text is validated before it is saved: an unfamiliar curly brace would take the
analysis down in production, in the middle of an interview — whereas here it is
visible straight away, with an explanation.
"""
from __future__ import annotations

import json
import logging
import os
import string
import threading
from dataclasses import dataclass
from pathlib import Path

from ..coverage import prompts as coverage_prompts
from ..guide import parser as guide_parser

log = logging.getLogger("ilh.prompts")


class PromptError(Exception):
    """An error carrying a message that is ready for the UI."""


@dataclass(frozen=True)
class PromptSpec:
    key: str
    title: str
    hint: str
    default: str
    allowed: frozenset[str]   # which placeholders are substituted at all
    required: frozenset[str]  # the ones without which the prompt is meaningless


_PLACEHOLDER_HINTS = {
    "language": "language of the answer (analysis.output_language, or the guide's)",
    "guide_language": "language the guide itself is written in (ru, en…)",
    "guide_block": "the guide's sections and topics with their ids",
    "max_recs": "recommendation limit from the settings",
    "probes_rule": "the probe-question rule (template below)",
    "max_probes": "probe-per-cycle limit from the settings",
}

SPECS: tuple[PromptSpec, ...] = (
    PromptSpec(
        key="system_live",
        title="Analysis during the interview",
        hint=(
            "System prompt of the analysis cycle: when a topic counts as covered and "
            "what recommendations to give the interviewer."
        ),
        default=coverage_prompts.SYSTEM_LIVE,
        allowed=frozenset(
            {"language", "guide_language", "guide_block", "max_recs", "probes_rule"}
        ),
        required=frozenset({"guide_block", "probes_rule"}),
    ),
    PromptSpec(
        key="probes_rule",
        title="Probe-question rule",
        hint=(
            "Substituted into {probes_rule}. Not used when probes are switched off "
            "(max_probes = 0)."
        ),
        default=coverage_prompts.PROBES_RULE,
        allowed=frozenset({"max_probes", "language"}),
        required=frozenset(),
    ),
    PromptSpec(
        key="system_final",
        title="Final reconciliation on «Stop»",
        hint=(
            "A pass over the whole transcript: picking up missed topics and "
            "points for the report."
        ),
        default=coverage_prompts.SYSTEM_FINAL,
        allowed=frozenset({"language", "guide_language", "guide_block"}),
        required=frozenset({"guide_block"}),
    ),
    PromptSpec(
        key="guide_parse",
        title="Parsing the guide text",
        hint="How free-form guide text is turned into sections and topics.",
        default=guide_parser.SYSTEM_PARSE,
        allowed=frozenset(),
        required=frozenset(),
    ),
)

_BY_KEY = {spec.key: spec for spec in SPECS}


def spec(key: str) -> PromptSpec:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise PromptError(f"Unknown prompt: {key}") from None


def defaults() -> dict[str, str]:
    return {s.key: s.default for s in SPECS}


def _fields(text: str) -> set[str]:
    """Placeholders of the text; {{…}} is an escaped brace and is not one."""
    try:
        return {
            name for _, name, _, _ in string.Formatter().parse(text) if name is not None
        }
    except ValueError as e:  # unclosed brace
        raise PromptError(f"Error in the curly braces: {e}") from e


def validate(key: str, text: str) -> str:
    """Returns the normalised text, or explains what is wrong with it."""
    sp = spec(key)
    text = (text or "").strip()
    if not text:
        raise PromptError("A prompt cannot be empty")

    used = _fields(text)
    unknown = sorted(f for f in used if f not in sp.allowed)
    if unknown:
        known = ", ".join(f"{{{f}}}" for f in sorted(sp.allowed)) or "none"
        raise PromptError(
            f"Unknown substitutions: {', '.join('{' + f + '}' for f in unknown)}. "
            f"Available here: {known}. If you need curly braces as literal text, "
            f"double them: {{{{ and }}}}."
        )
    missing = sorted(sp.required - used)
    if missing:
        raise PromptError(
            "Without "
            + ", ".join("{" + f + "}" for f in missing)
            + " the prompt will not work — put those substitutions back into the text."
        )

    # A trial render: catches {0}, {a[b]} and anything else that survives the parse above.
    try:
        text.format(**{f: "" for f in sp.allowed})
    except (IndexError, KeyError, ValueError) as e:
        raise PromptError(f"The text does not render: {e}") from e
    return text


class PromptStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def overrides(self) -> dict[str, str]:
        """Only what a human rewrote. A broken file must not take startup down."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as e:
            log.error("Cannot read %s: %s — using the default prompts", self.path, e)
            return {}
        if not isinstance(data, dict):
            log.error("Expected an object in %s — using the default prompts", self.path)
            return {}

        out: dict[str, str] = {}
        for key, text in data.items():
            if key not in _BY_KEY or not isinstance(text, str):
                log.warning("Skipping prompt %r from %s", key, self.path)
                continue
            try:
                out[key] = validate(key, text)
            except PromptError as e:
                # A default is better than analysis that dies mid-interview.
                log.error("Prompt %s is broken (%s) — using the default", key, e)
        return out

    def templates(self) -> dict[str, str]:
        """The full set for rendering: defaults with the edits on top."""
        return {**defaults(), **self.overrides()}

    def describe(self) -> list[dict]:
        """The composition for the UI: default, current text and an "edited" flag."""
        current = self.overrides()
        return [
            {
                "key": s.key,
                "title": s.title,
                "hint": s.hint,
                "default": s.default,
                "text": current.get(s.key, s.default),
                "customized": s.key in current,
                "placeholders": [
                    {"name": f, "hint": _PLACEHOLDER_HINTS.get(f, "")}
                    for f in sorted(s.allowed)
                ],
            }
            for s in SPECS
        ]

    def save(self, key: str, text: str) -> dict:
        """Text that matches the default is not stored: let it ride the version."""
        sp = spec(key)
        text = validate(key, text)
        with self._lock:
            data = self.overrides()
            if text == sp.default.strip():
                data.pop(key, None)
            else:
                data[key] = text
            self._write(data)
        log.info("Prompt %s updated", key)
        return {"key": key, "text": text, "customized": key in data}

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            data = {} if key is None else self.overrides()
            if key is not None:
                spec(key)
                data.pop(key, None)
            self._write(data)
        log.info("Prompt %s reset to the default", key or "(all)")

    def _write(self, data: dict[str, str]) -> None:
        if not data:
            self.path.unlink(missing_ok=True)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
