"""Движок покрытия: running-состояние тем, 4-минутный планировщик с ручным
запуском, вызов LLM и монотонное слияние результатов."""
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
    Recommendation,
    TopicState,
    TopicUpdate,
    build_analysis_schema,
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
        self._system = prompts.system_prompt(guide, analysis_cfg.max_recommendations)
        self._schema = build_analysis_schema()
        self._recs: dict[str, Recommendation] = {}
        self._cursor = 0
        self._lock = asyncio.Lock()
        self.analyzing = False
        self.next_analysis_at: float | None = None

    # ------------------------------------------------------------- планировщик

    async def run(self, stop_event: asyncio.Event, manual_event: asyncio.Event) -> None:
        log.info("Планировщик анализа запущен: интервал %d c", self.cfg.interval_s)
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
            done, pending = await asyncio.wait(
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

    async def analyze(self, manual: bool = False) -> None:
        if self._lock.locked():
            await self.notify("status", {"message": "Анализ уже выполняется"})
            return
        async with self._lock:
            delta, next_cursor = self.transcript.delta_since(self._cursor)
            if not delta:
                await self.notify("status", {"message": "Нет новых реплик — анализ пропущен"})
                return

            self.analyzing = True
            try:
                await self.notify(
                    "analysis",
                    {"phase": "started", "manual": manual, "new_segments": len(delta)},
                )
                started = time.monotonic()
                log_entry: dict = {
                    "iteration": self.state.analysis_iteration + 1,
                    "ts": datetime.now().astimezone().isoformat(),
                    "manual": manual,
                    "delta_segments": len(delta),
                }
                try:
                    user = prompts.user_prompt(self.state, self.guide, delta, self.llm_cfg.max_delta_chars)
                    parsed, meta = await self.llm.chat_json(self._system, user, self._schema)
                    log_entry.update(
                        duration_s=round(meta.duration_s, 1),
                        prompt_chars=meta.prompt_chars,
                        eval_count=meta.eval_count,
                        prompt_eval_count=meta.prompt_eval_count,
                        retried=meta.retried,
                        raw_response=meta.raw_response[:20000],
                    )
                except OllamaError as e:
                    # Курсор не двигаем: эта дельта попадёт в следующий цикл.
                    log_entry["error"] = str(e)
                    self._persist_log(log_entry)
                    log.error("Цикл анализа не удался: %s", e)
                    await self.notify("error", {"message": f"Анализ не удался: {e}"})
                    await self.notify("analysis", {"phase": "failed"})
                    return

                response = self._validate(parsed)
                applied = self._apply_response(response)
                self._cursor = next_cursor
                self.state.analysis_iteration += 1
                self.state.updated_at = datetime.now().astimezone().isoformat()
            finally:
                self.analyzing = False

            self._persist(log_entry)
            counts = self.state.counts()
            log.info(
                "Анализ #%d: %d обновлений, %d рекомендаций, %.1f c (покрыто %d/%d)",
                self.state.analysis_iteration, applied, len(self._recs),
                time.monotonic() - started, counts["covered"], counts["total"],
            )
            await self.notify("coverage", self.coverage_payload())
            await self.notify("recommendations", self.recommendations_payload())
            await self.notify(
                "analysis",
                {"phase": "done", "iteration": self.state.analysis_iteration,
                 "duration_s": round(time.monotonic() - started, 1), "applied": applied},
            )

    # ------------------------------------------------------- валидация/слияние

    def _validate(self, parsed: dict) -> AnalysisResponse:
        """Повреждённые элементы отбрасываются по одному, остальное сохраняем."""
        updates: list[TopicUpdate] = []
        recs: list[Recommendation] = []
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
        return AnalysisResponse(topic_updates=updates, recommendations=recs)

    def _apply_response(self, resp: AnalysisResponse) -> int:
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

        for rec in resp.recommendations:
            if rec.topic_id:
                st = self.state.topics.get(rec.topic_id)
                if st is None:
                    log.warning("Рекомендация для неизвестной темы %s — игнорирую", rec.topic_id)
                    continue
                if st.status == "covered":
                    continue
                self._recs[rec.topic_id] = rec
            else:
                self._recs[f"{rec.type}#{len(self._recs)}"] = rec

        for key in list(self._recs):
            tid = self._recs[key].topic_id
            if tid and self.state.topics.get(tid) and self.state.topics[tid].status == "covered":
                del self._recs[key]
        return applied

    # ------------------------------------------------------------ payload/диск

    def coverage_payload(self) -> dict:
        return {
            "topics": {tid: st.model_dump() for tid, st in self.state.topics.items()},
            "counts": self.state.counts(),
            "iteration": self.state.analysis_iteration,
            "updated_at": self.state.updated_at,
        }

    def recommendations_payload(self) -> dict:
        return {"items": self.current_recommendations(), "iteration": self.state.analysis_iteration}

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

    def _persist(self, log_entry: dict) -> None:
        if self.store is None:
            return
        try:
            self.store.save_coverage(self.state)
            self.store.append_recommendations(
                self.state.analysis_iteration, self.current_recommendations()
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
