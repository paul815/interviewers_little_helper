"""Движок покрытия: running-состояние тем, планировщик с ручным запуском,
вызов LLM и монотонное слияние результатов.

Три режима анализа:
- delta      — обычный цикл: только новые реплики с прошлого анализа;
- reconcile  — каждый N-й цикл: окно с прошлой сверки (ловит темы,
               пропущенные в отдельных дельтах, не раздувая контекст);
- final      — на «Стоп»: весь транскрипт последовательными окнами,
               плюс сбор findings (тезисов) для отчёта.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Awaitable, Callable

from pydantic import ValidationError

from ..config import AnalysisConfig, LLMConfig
from ..domain import Segment
from ..guide.schemas import Guide
from ..llm.ollama_client import OllamaClient, OllamaError
from ..transcript.store import TranscriptStore
from . import prompts
from .schemas import (
    STATUS_RANK,
    AnalysisResponse,
    CoverageState,
    Finding,
    Recommendation,
    TopicState,
    TopicUpdate,
    build_final_schema,
    build_live_schema,
)

log = logging.getLogger("ilh.coverage")

Notify = Callable[[str, dict], Awaitable[None]]


class CoverageEngine:
    def __init__(
        self,
        guide: Guide,
        transcript: TranscriptStore | None,
        llm: OllamaClient | None,
        analysis_cfg: AnalysisConfig,
        llm_cfg: LLMConfig,
        notify: Notify,
        session_store=None,
        session_id: str = "",
    ):
        self.guide = guide
        self.transcript = transcript
        self.llm = llm
        self.cfg = analysis_cfg
        self.llm_cfg = llm_cfg
        self.notify = notify
        self.store = session_store

        self.state = CoverageState(
            session_id=session_id,
            topics={tid: TopicState() for tid in guide.topic_ids()},
        )
        self._topic_meta = {
            t.id: {"order": i, "question": t.question, "section_title": s.title}
            for i, (s, t) in enumerate(guide.all_topics())
        }
        self._system_live = prompts.system_prompt_live(
            guide, analysis_cfg.max_recommendations, analysis_cfg.max_probes
        )
        self._system_final = prompts.system_prompt_final(guide)
        self._live_schema = build_live_schema()
        self._final_schema = build_final_schema()

        self._recs: dict[str, Recommendation] = {}  # coverage_gap, ключ = topic_id
        self._probes: list[Recommendation] = []     # свежие пробы, заменяются каждый цикл
        self._findings: dict[str, list[str]] = {}   # topic_id -> тезисы (финальный проход)
        self._cursor = 0
        self._reconcile_cursor = 0
        self._lock = asyncio.Lock()
        self.analyzing = False
        self.next_analysis_at: float | None = None

    # ------------------------------------------------------------- планировщик

    async def run(self, stop_event: asyncio.Event, manual_event: asyncio.Event) -> None:
        log.info(
            "Планировщик анализа запущен: интервал %d c, сверка каждый %s-й цикл",
            self.cfg.interval_s, self.cfg.reconcile_every or "—",
        )
        try:
            while not stop_event.is_set():
                self.next_analysis_at = time.time() + self.cfg.interval_s
                await self._notify_timer()
                manual = await self._wait_for_trigger(stop_event, manual_event)
                if stop_event.is_set():
                    break
                try:
                    await self.analyze(manual=manual)
                except Exception:
                    # Планировщик должен пережить любой сбой одного цикла.
                    log.exception("Непредвиденная ошибка цикла анализа")
                    await self.notify("error", {"message": "Внутренняя ошибка анализа — подробности в logs/app.log"})
                    await self.notify("analysis", {"phase": "failed"})
        except asyncio.CancelledError:
            log.info("Планировщик анализа отменён")
            raise
        finally:
            self.next_analysis_at = None

    async def _wait_for_trigger(self, stop_event: asyncio.Event, manual_event: asyncio.Event) -> bool:
        """Ждёт до next_analysis_at; True — если сработал ручной запуск."""
        remaining = (self.next_analysis_at or 0) - time.time()
        if remaining <= 0:
            return False
        stop_t = asyncio.create_task(stop_event.wait())
        manual_t = asyncio.create_task(manual_event.wait())
        try:
            done, _ = await asyncio.wait(
                {stop_t, manual_t}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for task in (stop_t, manual_t):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stop_t, manual_t, return_exceptions=True)
        if manual_t in done and not manual_t.cancelled():
            manual_event.clear()
            return not stop_event.is_set()
        return False

    async def _notify_timer(self) -> None:
        await self.notify(
            "timer",
            {"next_analysis_at": self.next_analysis_at, "interval_s": self.cfg.interval_s},
        )

    # ----------------------------------------------------------------- анализ

    def _pick_mode(self) -> str:
        next_iter = self.state.analysis_iteration + 1
        if self.cfg.reconcile_every > 0 and next_iter % self.cfg.reconcile_every == 0:
            return "reconcile"
        return "delta"

    async def analyze(self, manual: bool = False) -> None:
        if self._lock.locked():
            await self.notify("status", {"message": "Анализ уже выполняется"})
            return
        async with self._lock:
            mode = self._pick_mode()
            cursor = self._reconcile_cursor if mode == "reconcile" else self._cursor
            segments, new_cursor = self.transcript.delta_since(cursor)
            if not segments:
                await self.notify("status", {"message": "Нет новых реплик — анализ пропущен"})
                return
            ok = await self._run_analysis(segments, mode, manual)
            if ok:
                self._cursor = max(self._cursor, new_cursor)
                if mode == "reconcile":
                    self._reconcile_cursor = new_cursor

    async def final_pass(self) -> bool:
        """Финальная сверка всего транскрипта окнами; собирает findings.
        Вызывается на «Стоп» после того, как ASR дожал очередь."""
        if self.transcript is None or self.llm is None:
            return False
        segments = sorted(self.transcript.all_segments(), key=lambda s: s.t0)
        if not segments:
            return False
        windows = self._split_windows(segments, self.llm_cfg.max_delta_chars)
        log.info("Финальная сверка: %d окон, %d сегментов", len(windows), len(segments))
        for i, window in enumerate(windows, 1):
            await self.notify(
                "status",
                {"message": f"Финальная сверка транскрипта: {i}/{len(windows)}…"},
            )
            async with self._lock:
                ok = await self._run_analysis(window, "final", manual=False)
            if not ok:
                log.warning("Финальная сверка прервана на окне %d/%d", i, len(windows))
                return False
        self._cursor = self._reconcile_cursor = len(self.transcript)
        return True

    @staticmethod
    def _split_windows(segments: list[Segment], max_chars: int) -> list[list[Segment]]:
        windows: list[list[Segment]] = [[]]
        size = 0
        for seg in segments:
            cost = len(seg.text) + 24  # префикс «[MM:SS] И: »
            if windows[-1] and size + cost > max_chars:
                windows.append([])
                size = 0
            windows[-1].append(seg)
            size += cost
        return [w for w in windows if w]

    async def _run_analysis(self, segments: list[Segment], mode: str, manual: bool) -> bool:
        final = mode == "final"
        self.analyzing = True
        started = time.monotonic()
        log_entry: dict = {
            "iteration": self.state.analysis_iteration + 1,
            "ts": datetime.now().astimezone().isoformat(),
            "mode": mode,
            "manual": manual,
            "segments": len(segments),
        }
        try:
            await self.notify(
                "analysis",
                {"phase": "started", "mode": mode, "manual": manual, "new_segments": len(segments)},
            )
            system = self._system_final if final else self._system_live
            schema = self._final_schema if final else self._live_schema
            user = prompts.user_prompt(
                self.state, self.guide, segments, self.llm_cfg.max_delta_chars, mode
            )
            try:
                parsed, meta = await self.llm.chat_json(system, user, schema)
                log_entry.update(
                    duration_s=round(meta.duration_s, 1),
                    prompt_chars=meta.prompt_chars,
                    eval_count=meta.eval_count,
                    prompt_eval_count=meta.prompt_eval_count,
                    retried=meta.retried,
                    raw_response=meta.raw_response[:20000],
                )
            except OllamaError as e:
                # Курсоры не двигаем: фрагмент попадёт в следующий цикл.
                log_entry["error"] = str(e)
                self._persist_log(log_entry)
                log.error("Цикл анализа (%s) не удался: %s", mode, e)
                await self.notify("error", {"message": f"Анализ не удался: {e}"})
                await self.notify("analysis", {"phase": "failed"})
                return False

            response = self._validate(parsed)
            applied = self._apply_response(response, mode)
            self.state.analysis_iteration += 1
            self.state.updated_at = datetime.now().astimezone().isoformat()
        finally:
            self.analyzing = False

        self._persist(log_entry)
        counts = self.state.counts()
        log.info(
            "Анализ #%d (%s): %d обновлений, %d рекомендаций, %d проб, %.1f c (покрыто %d/%d)",
            self.state.analysis_iteration, mode, applied, len(self._recs), len(self._probes),
            time.monotonic() - started, counts["covered"], counts["total"],
        )
        await self.notify("coverage", self.coverage_payload())
        if not final:
            await self.notify("recommendations", self.recommendations_payload())
        await self.notify(
            "analysis",
            {"phase": "done", "mode": mode, "iteration": self.state.analysis_iteration,
             "duration_s": round(time.monotonic() - started, 1), "applied": applied},
        )
        return True

    # ------------------------------------------------------- валидация/слияние

    def _validate(self, parsed: dict) -> AnalysisResponse:
        """Повреждённые элементы отбрасываются по одному, остальное сохраняем."""
        updates: list[TopicUpdate] = []
        recs: list[Recommendation] = []
        findings: list[Finding] = []
        for item in parsed.get("topic_updates") or []:
            try:
                updates.append(TopicUpdate.model_validate(item))
            except ValidationError as e:
                log.warning("Отброшено невалидное обновление темы %r: %s", item, e)
        for item in parsed.get("recommendations") or []:
            try:
                recs.append(Recommendation.model_validate(item))
            except ValidationError as e:
                log.warning("Отброшена невалидная рекомендация %r: %s", item, e)
        for item in parsed.get("findings") or []:
            try:
                findings.append(Finding.model_validate(item))
            except ValidationError as e:
                log.warning("Отброшен невалидный finding %r: %s", item, e)
        return AnalysisResponse(topic_updates=updates, recommendations=recs, findings=findings)

    def _apply_response(self, resp: AnalysisResponse, mode: str = "delta") -> int:
        iteration = self.state.analysis_iteration + 1
        applied = 0
        unknown: list[str] = []
        for upd in resp.topic_updates:
            st = self.state.topics.get(upd.topic_id)
            if st is None:
                unknown.append(upd.topic_id)
                continue
            if STATUS_RANK[upd.status] <= STATUS_RANK[st.status]:
                continue  # монотонность: статусы не понижаются и не «мигают»
            st.status = upd.status
            st.confidence = upd.confidence
            st.evidence = upd.evidence or st.evidence
            st.last_update_iteration = iteration
            applied += 1
        if unknown:
            log.warning("LLM вернула неизвестные topic_id (игнорирую): %s", unknown)

        if mode == "final":
            for f in resp.findings:
                if f.topic_id not in self.state.topics or not f.finding.strip():
                    continue
                bucket = self._findings.setdefault(f.topic_id, [])
                if f.finding not in bucket:
                    bucket.append(f.finding.strip())
        else:
            self._merge_recommendations(resp.recommendations)

        for key in list(self._recs):
            tid = self._recs[key].topic_id
            if tid and self.state.topics.get(tid) and self.state.topics[tid].status == "covered":
                del self._recs[key]
        return applied

    def _merge_recommendations(self, recs: list[Recommendation]) -> None:
        probes: list[Recommendation] = []
        for rec in recs:
            if rec.topic_id and rec.topic_id not in self.state.topics:
                log.warning("Рекомендация для неизвестной темы %s — отвязываю", rec.topic_id)
                rec.topic_id = None
            if rec.type == "probe":
                probes.append(rec)
                continue
            if not rec.topic_id:
                log.warning("coverage_gap без topic_id — игнорирую")
                continue
            if self.state.topics[rec.topic_id].status == "covered":
                continue
            self._recs[rec.topic_id] = rec
        # Пробы живут один цикл: устаревшие подсказки «копнуть» только мешают.
        self._probes = probes[: self.cfg.max_probes]

    # ------------------------------------------------------------ payload/диск

    def coverage_payload(self) -> dict:
        return {
            "topics": {tid: st.model_dump() for tid, st in self.state.topics.items()},
            "counts": self.state.counts(),
            "iteration": self.state.analysis_iteration,
            "updated_at": self.state.updated_at,
        }

    def recommendations_payload(self) -> dict:
        return {
            "items": self.current_recommendations(),
            "probes": self.current_probes(),
            "iteration": self.state.analysis_iteration,
        }

    def current_recommendations(self) -> list[dict]:
        items = []
        for rec in self._recs.values():
            meta = self._topic_meta.get(rec.topic_id or "", {})
            status = (
                self.state.topics[rec.topic_id].status
                if rec.topic_id in self.state.topics
                else None
            )
            items.append(
                {
                    **rec.model_dump(),
                    "topic_question": meta.get("question"),
                    "section_title": meta.get("section_title"),
                    "status": status,
                    "_order": meta.get("order", 10**6),
                }
            )
        items.sort(key=lambda r: (r["urgency"] != "high", r["status"] != "not_covered", r["_order"]))
        for it in items:
            it.pop("_order", None)
        return items[: self.cfg.max_recommendations]

    def current_probes(self) -> list[dict]:
        out = []
        for rec in self._probes:
            meta = self._topic_meta.get(rec.topic_id or "", {})
            out.append({**rec.model_dump(), "topic_question": meta.get("question")})
        return out

    def findings(self) -> dict[str, list[str]]:
        return {tid: list(items) for tid, items in self._findings.items()}

    def _persist(self, log_entry: dict) -> None:
        if self.store is None:
            return
        try:
            self.store.save_coverage(self.state)
            self.store.append_recommendations(
                self.state.analysis_iteration, self.current_recommendations() + self.current_probes()
            )
            self._persist_log(log_entry)
            if self.transcript is not None:
                self.store.render_markdown(self.transcript.all_segments(), self.guide)
        except Exception:
            log.exception("Не удалось сохранить результаты анализа на диск")

    def _persist_log(self, entry: dict) -> None:
        if self.store is None:
            return
        try:
            self.store.append_analysis_log(entry)
        except Exception:
            log.exception("Не удалось записать лог анализа")
