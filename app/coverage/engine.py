"""Coverage engine: running topic state, a scheduler with a manual trigger,
the LLM call and monotonic merging of the results.

Three analysis modes:
- delta      — the usual cycle: only utterances new since the last analysis;
- reconcile  — every Nth cycle: the window since the last reconcile (catches
               topics missed by individual deltas without inflating the context);
- final      — on "Stop": the whole transcript in consecutive windows, plus
               collecting findings for the report.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime

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
        templates: Mapping[str, str] | None = None,
        instructions: str = "",
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
            guide,
            analysis_cfg.max_recommendations,
            analysis_cfg.max_probes,
            templates=templates,
            extra=instructions,
            output_language=analysis_cfg.output_language,
        )
        self._system_final = prompts.system_prompt_final(
            guide,
            templates=templates,
            extra=instructions,
            output_language=analysis_cfg.output_language,
        )
        self._live_schema = build_live_schema()
        self._final_schema = build_final_schema()

        self._recs: dict[str, Recommendation] = {}  # coverage_gap, keyed by topic_id
        self._probes: list[Recommendation] = []     # fresh probes, replaced every cycle
        self._findings: dict[str, list[str]] = {}   # topic_id -> points (final pass)
        self._muted: set[str] = set()               # topics with hidden recommendations
        self._cursor = 0
        self._reconcile_cursor = 0
        self._lock = asyncio.Lock()
        self.analyzing = False
        self.next_analysis_at: float | None = None

    # --------------------------------------------------------------- scheduler

    async def run(self, stop_event: asyncio.Event, manual_event: asyncio.Event) -> None:
        log.info(
            "Analysis scheduler started: interval %d s, reconcile every %s cycles",
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
                    # The scheduler must survive any failure of a single cycle.
                    log.exception("Unexpected error in the analysis cycle")
                    await self.notify("error", {
                        "message": "Internal analysis error — details in logs/app.log"
                    })
                    await self.notify("analysis", {"phase": "failed"})
        except asyncio.CancelledError:
            log.info("Analysis scheduler cancelled")
            raise
        finally:
            self.next_analysis_at = None

    async def _wait_for_trigger(
        self, stop_event: asyncio.Event, manual_event: asyncio.Event
    ) -> bool:
        """Waits until next_analysis_at; True if the manual trigger fired."""
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

    # ---------------------------------------------------------------- analysis

    def _pick_mode(self) -> str:
        next_iter = self.state.analysis_iteration + 1
        if self.cfg.reconcile_every > 0 and next_iter % self.cfg.reconcile_every == 0:
            return "reconcile"
        return "delta"

    async def analyze(self, manual: bool = False) -> None:
        if self._lock.locked():
            await self.notify("status", {"message": "Analysis is already running"})
            return
        async with self._lock:
            mode = self._pick_mode()
            cursor = self._reconcile_cursor if mode == "reconcile" else self._cursor
            segments, new_cursor = self.transcript.delta_since(cursor)
            if not segments:
                await self.notify("status", {"message": "No new utterances — analysis skipped"})
                return
            ok = await self._run_analysis(segments, mode, manual)
            if ok:
                self._cursor = max(self._cursor, new_cursor)
                if mode == "reconcile":
                    self._reconcile_cursor = new_cursor

    async def final_pass(self) -> bool:
        """Final reconciliation of the whole transcript in windows; collects
        findings. Called on "Stop" once ASR has drained its queue."""
        if self.transcript is None or self.llm is None:
            return False
        segments = sorted(self.transcript.all_segments(), key=lambda s: s.t0)
        if not segments:
            return False
        windows = self._split_windows(segments, self.llm_cfg.max_delta_chars)
        log.info("Final reconciliation: %d windows, %d segments", len(windows), len(segments))
        for i, window in enumerate(windows, 1):
            await self.notify(
                "status",
                {"message": f"Final transcript reconciliation: {i}/{len(windows)}…"},
            )
            async with self._lock:
                ok = await self._run_analysis(window, "final", manual=False)
            if not ok:
                log.warning("Final reconciliation aborted at window %d/%d", i, len(windows))
                return False
        self._cursor = self._reconcile_cursor = len(self.transcript)
        return True

    @staticmethod
    def _split_windows(segments: list[Segment], max_chars: int) -> list[list[Segment]]:
        windows: list[list[Segment]] = [[]]
        size = 0
        for seg in segments:
            cost = len(seg.text) + 24  # the "[MM:SS] I: " prefix
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
                # Cursors stay put: the fragment will come round in the next cycle.
                log_entry["error"] = str(e)
                self._persist_log(log_entry)
                log.error("Analysis cycle (%s) failed: %s", mode, e)
                await self.notify("error", {"message": f"Analysis failed: {e}"})
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
            "Analysis #%d (%s): %d updates, %d recommendations, %d probes, %.1f s (covered %d/%d)",
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

    # ------------------------------------------------------- validation/merging

    def _validate(self, parsed: dict) -> AnalysisResponse:
        """Broken items are dropped one by one; everything else is kept."""
        updates: list[TopicUpdate] = []
        recs: list[Recommendation] = []
        findings: list[Finding] = []
        for item in parsed.get("topic_updates") or []:
            try:
                updates.append(TopicUpdate.model_validate(item))
            except ValidationError as e:
                log.warning("Dropped invalid topic update %r: %s", item, e)
        for item in parsed.get("recommendations") or []:
            try:
                recs.append(Recommendation.model_validate(item))
            except ValidationError as e:
                log.warning("Dropped invalid recommendation %r: %s", item, e)
        for item in parsed.get("findings") or []:
            try:
                findings.append(Finding.model_validate(item))
            except ValidationError as e:
                log.warning("Dropped invalid finding %r: %s", item, e)
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
            if st.manual:
                continue  # the researcher's manual mark has the last word
            if STATUS_RANK[upd.status] <= STATUS_RANK[st.status]:
                continue  # monotonicity: statuses never go down and never flicker
            st.status = upd.status
            st.confidence = upd.confidence
            st.evidence = upd.evidence or st.evidence
            st.last_update_iteration = iteration
            applied += 1
        if unknown:
            log.warning("The LLM returned unknown topic_ids (ignoring): %s", unknown)

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
                log.warning("Recommendation for unknown topic %s — detaching it", rec.topic_id)
                rec.topic_id = None
            if rec.type == "probe":
                probes.append(rec)
                continue
            if not rec.topic_id:
                log.warning("coverage_gap without a topic_id — ignoring")
                continue
            if self.state.topics[rec.topic_id].status == "covered":
                continue
            self._recs[rec.topic_id] = rec
        # Probes live for one cycle: stale "dig into this" hints only get in the way.
        self._probes = probes[: self.cfg.max_probes]

    # ------------------------------------------------------ manual control (UI)

    def set_manual_status(self, topic_id: str, status: str | None) -> None:
        """status=None clears the manual mark (the status stays, and the LLM may
        update it again); otherwise the status is pinned by the researcher."""
        st = self.state.topics.get(topic_id)
        if st is None:
            raise ValueError(f"Unknown topic: {topic_id}")
        if status is None:
            st.manual = False
        else:
            if status not in STATUS_RANK:
                raise ValueError(f"Invalid status: {status}")
            st.status = status  # a manual edit may lower the status as well
            st.manual = True
            st.confidence = None
            st.evidence = "marked manually"
            if status == "covered":
                self._recs.pop(topic_id, None)
        st.last_update_iteration = self.state.analysis_iteration
        self._muted.discard(topic_id)
        self.state.updated_at = datetime.now().astimezone().isoformat()
        if self.store is not None:
            try:
                self.store.save_coverage(self.state)
            except Exception:
                log.exception("Could not save the state after a manual edit")

    def dismiss_recommendation(self, topic_id: str) -> None:
        """Hides the hint for a topic until its status is changed manually."""
        if topic_id not in self.state.topics:
            raise ValueError(f"Unknown topic: {topic_id}")
        self._muted.add(topic_id)
        self._recs.pop(topic_id, None)

    # ------------------------------------------------------------ payload/disk

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
            if rec.topic_id in self._muted:
                continue
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
        items.sort(
            key=lambda r: (r["urgency"] != "high", r["status"] != "not_covered", r["_order"])
        )
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
                self.state.analysis_iteration,
                self.current_recommendations() + self.current_probes(),
            )
            self._persist_log(log_entry)
            if self.transcript is not None:
                self.store.render_markdown(self.transcript.all_segments(), self.guide)
        except Exception:
            log.exception("Could not persist the analysis results to disk")

    def _persist_log(self, entry: dict) -> None:
        if self.store is None:
            return
        try:
            self.store.append_analysis_log(entry)
        except Exception:
            log.exception("Could not write the analysis log")
