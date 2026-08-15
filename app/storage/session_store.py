"""Session persistence: one folder per session, appended as it goes for crash
resilience, readable JSON plus a markdown transcript.

The session folder lives either inside a project (projects/<project>/sessions/,
see project_store) or in the shared sessions/ — when an interview is recorded
without a project.

sessions/2026-07-11_14-30-00/
  meta.json               — session parameters
  guide.json              — the confirmed guide structure
  transcript.jsonl        — segments, appended after every recognition
  transcript.md           — readable transcript (regenerated)
  audio.wav               — the interview recording, stereo 16 kHz: L — interviewer,
                            R — respondent (storage.save_audio, see audio/recorder)
  coverage_state.json     — current coverage state (rewritten atomically)
  recommendations.jsonl   — history of recommendations per cycle
  analysis_log.jsonl      — raw LLM exchanges for debugging
  flags.jsonl             — moments flagged by the researcher
  report.md               — the session report (after "Stop")
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path

from ..config import AppConfig
from ..domain import SPEAKER_FULL, Segment, Speaker, fmt_ts
from ..guide.schemas import Guide

log = logging.getLogger("ilh.storage")


def _fsync(f) -> None:
    """Push what was written all the way to disk. Without this the file lives in
    the OS cache and disappears on a power cut — while surviving an ordinary
    process crash."""
    f.flush()
    try:
        os.fsync(f.fileno())
    except OSError as e:  # network drive, a yanked USB stick
        log.debug("fsync did not work: %s", e)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        _fsync(f)  # otherwise os.replace may expose an empty file after a crash
    os.replace(tmp, path)


def _flag_block(flag: dict) -> str:
    """A flag without text is just a bookmark; with text it is a comment.

    The comment carries the guide question that was selected at the moment of
    typing: without it, a week later there is no telling what the line "and
    here they are lying" referred to.
    """
    note = (flag.get("note") or "").strip()
    head = "Researcher comment" if note else "Researcher flag"
    lines = [f"> 🚩 **[{fmt_ts(flag['t'])}] {head}.**" + (f" {note}" if note else "")]
    anchor = (flag.get("anchor") or "").strip()
    if anchor:
        section = (flag.get("anchor_section") or "").strip()
        where = f"{section} → {anchor}" if section else anchor
        lines.append(f"> <sub>on guide question: {where}</sub>")
    return "\n".join(lines)


def _speaker_name(value: str) -> str:
    try:
        return SPEAKER_FULL[Speaker(value)]
    except ValueError:  # written by another version — a raw name beats a crash
        return value


def build_transcript_markdown(
    session_id: str,
    guide_title: str,
    segments: list[dict],
    flags: list[dict],
    questions: list[dict],
) -> str:
    """A readable transcript from the raw records.

    Works on dicts rather than `Segment`: the same lines are assembled both
    during the session and when rebuilding from transcript.jsonl after a crash
    (see recovery).
    """
    segments = sorted(segments, key=lambda s: s["t0"])
    flags = sorted(flags, key=lambda f: f["t"])
    blocks: list[str] = []
    last_speaker = None
    last_t1 = -10.0
    fi = 0
    for seg in segments:
        while fi < len(flags) and flags[fi]["t"] <= seg["t0"]:
            blocks.append(_flag_block(flags[fi]))
            last_speaker = None  # a flag breaks the merging of utterances
            fi += 1
        if seg["speaker"] == last_speaker and seg["t0"] - last_t1 < 2.0 and blocks:
            blocks[-1] += " " + seg["text"]
        else:
            blocks.append(
                f"**[{fmt_ts(seg['t0'])}] {_speaker_name(seg['speaker'])}:** {seg['text']}"
            )
        last_speaker, last_t1 = seg["speaker"], seg["t1"]
    blocks.extend(_flag_block(f) for f in flags[fi:])
    text = (
        f"# Interview transcript — {session_id}\n\n"
        f"Guide: «{guide_title}»\n\n" + "\n\n".join(blocks) + "\n"
    )
    if questions:
        rows = "\n".join(
            f"- [{'x' if q['done'] else ' '}] [{fmt_ts(q['t'])}] {q['text']}" for q in questions
        )
        text += f"\n## Extra questions\n\n{rows}\n"
    return text


class SessionStore:
    def __init__(self, root: Path, session_id: str):
        self.session_id = session_id
        self.dir = root / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._transcript_f = open(self.dir / "transcript.jsonl", "a", encoding="utf-8")
        self.flags: list[dict] = []
        self.questions: list[dict] = []
        self._q_seq = 0

    @staticmethod
    def create(cfg: AppConfig, meta: dict, root: Path | None = None) -> SessionStore:
        session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        store = SessionStore(root or cfg.sessions_path, session_id)
        store.save_meta(
            {
                "session_id": session_id,
                "started_at": datetime.now().astimezone().isoformat(),
                "analysis_interval_s": cfg.analysis.interval_s,
                "llm_model": cfg.llm.model,
                "asr_model": cfg.asr.model,
                **meta,
            }
        )
        log.info("Session folder: %s", store.dir)
        return store

    def save_meta(self, meta: dict) -> None:
        with self._lock:
            self._meta = meta
            _atomic_write(self.dir / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))

    def save_guide(self, guide: Guide) -> None:
        with self._lock:
            _atomic_write(
                self.dir / "guide.json",
                json.dumps(guide.model_dump(), ensure_ascii=False, indent=2),
            )

    def save_guide_text(self, text: str) -> None:
        """The guide exactly as the researcher pasted it.

        guide.json is the LLM's parse: it loses wording and sometimes whole
        questions. The interview screen and the post-hoc review of the recording
        need the original.
        """
        if not text.strip():
            return
        with self._lock:
            _atomic_write(self.dir / "guide_source.txt", text)

    def append_segment(self, seg: Segment) -> None:
        with self._lock:
            self._transcript_f.write(json.dumps(seg.to_dict(), ensure_ascii=False) + "\n")
            _fsync(self._transcript_f)

    def save_coverage(self, state) -> None:
        with self._lock:
            _atomic_write(
                self.dir / "coverage_state.json",
                json.dumps(state.model_dump(), ensure_ascii=False, indent=2),
            )

    def append_recommendations(self, iteration: int, items: list[dict]) -> None:
        entry = {
            "iteration": iteration,
            "ts": datetime.now().astimezone().isoformat(),
            "recommendations": items,
        }
        with self._lock:
            with open(self.dir / "recommendations.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def append_analysis_log(self, entry: dict) -> None:
        with self._lock:
            with open(self.dir / "analysis_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def add_flag(
        self,
        t: float,
        note: str = "",
        anchor: str = "",
        anchor_section: str = "",
        topic_id: str | None = None,
    ) -> dict:
        flag = {
            "t": round(t, 1),
            "note": note,
            "anchor": anchor,
            "anchor_section": anchor_section,
            "topic_id": topic_id,
            "ts": datetime.now().astimezone().isoformat(),
        }
        with self._lock:
            self.flags.append(flag)
            with open(self.dir / "flags.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(flag, ensure_ascii=False) + "\n")
                _fsync(f)  # a flag is placed by hand — losing it to a crash stings most
        return flag

    # ------------------------------------------------- extra "ask" questions

    def _write_questions(self) -> None:
        """The file is rewritten whole: a checkbox gets unticked and ticked again."""
        _atomic_write(
            self.dir / "questions.jsonl",
            "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in self.questions),
        )

    def add_question(self, t: float, text: str) -> dict:
        with self._lock:
            self._q_seq += 1
            q = {
                "id": self._q_seq,
                "t": round(t, 1),
                "text": text,
                "done": False,
                "ts": datetime.now().astimezone().isoformat(),
            }
            self.questions.append(q)
            self._write_questions()
        return q

    def set_question_done(self, question_id: int, done: bool) -> dict | None:
        with self._lock:
            for q in self.questions:
                if q["id"] == question_id:
                    q["done"] = done
                    self._write_questions()
                    return q
        return None

    def save_report(self, markdown: str) -> None:
        with self._lock:
            _atomic_write(self.dir / "report.md", markdown)

    def render_markdown(self, segments: list[Segment], guide: Guide) -> None:
        with self._lock:
            flags = list(self.flags)
            questions = list(self.questions)
        text = build_transcript_markdown(
            self.session_id, guide.title, [s.to_dict() for s in segments], flags, questions
        )
        with self._lock:
            _atomic_write(self.dir / "transcript.md", text)

    def finalize(self, extra_meta: dict) -> None:
        with self._lock:
            meta = getattr(self, "_meta", {"session_id": self.session_id})
            meta.update(extra_meta)
        self.save_meta(meta)
        self.close()

    def close(self) -> None:
        with self._lock:
            try:
                _fsync(self._transcript_f)
                self._transcript_f.close()
            except ValueError:
                pass  # already closed
