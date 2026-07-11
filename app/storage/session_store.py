"""Персистентность сессии: отдельная папка на сессию, append-по-ходу для
устойчивости к сбоям, читаемые JSON + markdown-транскрипт.

sessions/2026-07-11_14-30-00/
  meta.json               — параметры сессии
  guide.json              — подтверждённая структура гайда
  transcript.jsonl        — сегменты, append после каждого распознавания
  transcript.md           — читаемый транскрипт (перегенерируется)
  coverage_state.json     — текущее состояние покрытия (перезапись атомарно)
  recommendations.jsonl   — история рекомендаций по циклам
  analysis_log.jsonl      — сырые обмены с LLM для отладки
  flags.jsonl             — моменты, отмеченные исследователем
  report.md               — отчёт сессии (после «Стоп»)
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path

from ..config import AppConfig
from ..domain import SPEAKER_FULL_RU, Segment, fmt_ts
from ..guide.schemas import Guide

log = logging.getLogger("ilh.storage")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _flag_block(flag: dict) -> str:
    note = f" {flag['note']}" if flag.get("note") else ""
    return f"> 🚩 **[{fmt_ts(flag['t'])}] Отметка исследователя.**{note}"


class SessionStore:
    def __init__(self, root: Path, session_id: str):
        self.session_id = session_id
        self.dir = root / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._transcript_f = open(self.dir / "transcript.jsonl", "a", encoding="utf-8")
        self.flags: list[dict] = []

    @staticmethod
    def create(cfg: AppConfig, meta: dict) -> "SessionStore":
        session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        store = SessionStore(cfg.sessions_path, session_id)
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
        log.info("Папка сессии: %s", store.dir)
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

    def append_segment(self, seg: Segment) -> None:
        with self._lock:
            self._transcript_f.write(json.dumps(seg.to_dict(), ensure_ascii=False) + "\n")
            self._transcript_f.flush()

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

    def add_flag(self, t: float, note: str = "") -> dict:
        flag = {
            "t": round(t, 1),
            "note": note,
            "ts": datetime.now().astimezone().isoformat(),
        }
        with self._lock:
            self.flags.append(flag)
            with open(self.dir / "flags.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(flag, ensure_ascii=False) + "\n")
        return flag

    def save_report(self, markdown: str) -> None:
        with self._lock:
            _atomic_write(self.dir / "report.md", markdown)

    def render_markdown(self, segments: list[Segment], guide: Guide) -> None:
        segments = sorted(segments, key=lambda s: s.t0)
        with self._lock:
            flags = sorted(self.flags, key=lambda f: f["t"])
        blocks: list[str] = []
        last_speaker = None
        last_t1 = -10.0
        fi = 0
        for seg in segments:
            while fi < len(flags) and flags[fi]["t"] <= seg.t0:
                blocks.append(_flag_block(flags[fi]))
                last_speaker = None  # флаг разрывает склейку реплик
                fi += 1
            if seg.speaker == last_speaker and seg.t0 - last_t1 < 2.0 and blocks:
                blocks[-1] += " " + seg.text
            else:
                blocks.append(f"**[{fmt_ts(seg.t0)}] {SPEAKER_FULL_RU[seg.speaker]}:** {seg.text}")
            last_speaker, last_t1 = seg.speaker, seg.t1
        blocks.extend(_flag_block(f) for f in flags[fi:])
        text = (
            f"# Транскрипт интервью — {self.session_id}\n\n"
            f"Гайд: «{guide.title}»\n\n" + "\n\n".join(blocks) + "\n"
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
                self._transcript_f.flush()
                self._transcript_f.close()
            except ValueError:
                pass  # уже закрыт
