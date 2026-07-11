"""Офлайн-прогон движка покрытия без аудио: текстовый транскрипт скармливается
батчами, как будто идут 4-минутные циклы. Нужен работающий Ollama.

Формат транскрипта — строки вида:
    И: Расскажите, как вы обычно ...
    Р: Ну, обычно я ...
(вместо И/Р можно INTERVIEWER/RESPONDENT)

Пример:
    python -m tools.simulate --guide examples/guide.txt --transcript examples/interview.txt
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import AppConfig  # noqa: E402
from app.coverage.engine import CoverageEngine  # noqa: E402
from app.domain import Speaker  # noqa: E402
from app.guide.parser import parse_guide_text  # noqa: E402
from app.llm.ollama_client import OllamaClient  # noqa: E402
from app.logging_setup import configure  # noqa: E402
from app.transcript.store import TranscriptStore  # noqa: E402

PREFIXES = {
    "и": Speaker.INTERVIEWER, "interviewer": Speaker.INTERVIEWER,
    "р": Speaker.RESPONDENT, "respondent": Speaker.RESPONDENT,
}


def parse_lines(path: Path) -> list[tuple[Speaker, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        prefix, text = line.split(":", 1)
        speaker = PREFIXES.get(prefix.strip().lower())
        if speaker and text.strip():
            out.append((speaker, text.strip()))
    return out


async def notify(type_: str, payload: dict) -> None:
    if type_ == "coverage":
        c = payload["counts"]
        print(f"\n  покрытие: {c['covered']} covered / {c['partial']} partial / "
              f"{c['not_covered']} not_covered")
    elif type_ == "recommendations":
        for r in payload["items"]:
            print(f"  [{r['urgency']}] {r.get('section_title') or ''} — {r.get('note')}")
            print(f"      → {r.get('suggested_question')}")
    elif type_ in ("status", "error"):
        print(f"  ({payload.get('message')})")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--batch-lines", type=int, default=12,
                        help="реплик на один цикл анализа (по умолчанию 12)")
    args = parser.parse_args()

    cfg = AppConfig.load()
    configure(cfg.log_level)
    llm = OllamaClient(cfg.llm)

    check = await llm.check()
    if not check["ok"]:
        raise SystemExit(f"LLM недоступна: {check['error']}")

    print("Парсинг гайда…")
    guide = await parse_guide_text(llm, args.guide.read_text(encoding="utf-8"))
    print(f"«{guide.title}» ({guide.language}):")
    for sec in guide.sections:
        print(f"  [{sec.id}] {sec.title}")
        for t in sec.topics:
            print(f"    ({t.id}) {t.question}")

    transcript = TranscriptStore()
    engine = CoverageEngine(
        guide=guide, transcript=transcript, llm=llm,
        analysis_cfg=cfg.analysis, llm_cfg=cfg.llm,
        notify=notify, session_store=None, session_id="simulate",
    )

    lines = parse_lines(args.transcript)
    if not lines:
        raise SystemExit("В транскрипте не найдено реплик формата «И: …» / «Р: …»")
    t = 0.0
    for start in range(0, len(lines), args.batch_lines):
        batch = lines[start : start + args.batch_lines]
        for speaker, text in batch:
            transcript.add(speaker, t, t + 5.0, text, guide.language)
            t += 6.0
        print(f"\n=== Цикл {start // args.batch_lines + 1}: +{len(batch)} реплик ===")
        await engine.analyze(manual=True)

    print("\nГотово.")


if __name__ == "__main__":
    asyncio.run(main())
