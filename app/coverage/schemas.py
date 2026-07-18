"""Схемы движка покрытия: состояние, ответ LLM, рекомендации.

Расширяемость: новые типы рекомендаций (например, пробинг-вопросы) добавляются
в RECOMMENDATION_TYPES — схема ответа LLM собирается из этого списка, движок
трактует type как открытую категорию, фронт рендерит неизвестные типы общим видом.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["not_covered", "partial", "covered"]
STATUS_RANK: dict[str, int] = {"not_covered": 0, "partial": 1, "covered": 2}

RECOMMENDATION_TYPES: list[str] = ["coverage_gap", "probe"]


class TopicState(BaseModel):
    status: Status = "not_covered"
    confidence: float | None = None
    evidence: str | None = None
    last_update_iteration: int = 0
    manual: bool = False  # выставлено исследователем вручную — LLM не переопределяет


class CoverageState(BaseModel):
    session_id: str
    analysis_iteration: int = 0
    updated_at: str = Field(default_factory=lambda: datetime.now().astimezone().isoformat())
    topics: dict[str, TopicState] = Field(default_factory=dict)

    def counts(self) -> dict:
        c = {"covered": 0, "partial": 0, "not_covered": 0}
        for st in self.topics.values():
            c[st.status] += 1
        c["total"] = len(self.topics)
        return c


class TopicUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topic_id: str
    status: Status
    confidence: float | None = None
    evidence: str | None = None


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = "coverage_gap"
    topic_id: str | None = None
    urgency: Literal["high", "normal"] = "normal"
    note: str = ""
    suggested_question: str | None = None
    quote: str | None = None  # для probe: реплика-триггер


class Finding(BaseModel):
    """Тезис «что узнали по теме» — собирается финальным проходом для отчёта."""

    model_config = ConfigDict(extra="ignore")

    topic_id: str
    finding: str


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topic_updates: list[TopicUpdate] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)


_TOPIC_UPDATES_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "topic_id": {"type": "string"},
            "status": {"type": "string", "enum": ["not_covered", "partial", "covered"]},
            "confidence": {"type": "number"},
            "evidence": {"type": "string"},
        },
        "required": ["topic_id", "status"],
    },
}


def build_live_schema() -> dict:
    """Схема ответа обычного/сверочного цикла (без $ref — совместимее)."""
    return {
        "type": "object",
        "properties": {
            "topic_updates": _TOPIC_UPDATES_SCHEMA,
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": RECOMMENDATION_TYPES},
                        "topic_id": {"type": "string"},
                        "urgency": {"type": "string", "enum": ["high", "normal"]},
                        "note": {"type": "string"},
                        "suggested_question": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ["type", "note", "suggested_question"],
                },
            },
        },
        "required": ["topic_updates", "recommendations"],
    }


def build_final_schema() -> dict:
    """Схема финального прохода: обновления статусов + тезисы для отчёта."""
    return {
        "type": "object",
        "properties": {
            "topic_updates": _TOPIC_UPDATES_SCHEMA,
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic_id": {"type": "string"},
                        "finding": {"type": "string"},
                    },
                    "required": ["topic_id", "finding"],
                },
            },
        },
        "required": ["topic_updates", "findings"],
    }
