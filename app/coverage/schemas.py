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

RECOMMENDATION_TYPES: list[str] = ["coverage_gap"]  # MVP; будущее: "probe"


class TopicState(BaseModel):
    status: Status = "not_covered"
    confidence: float | None = None
    evidence: str | None = None
    last_update_iteration: int = 0


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


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topic_updates: list[TopicUpdate] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)


def build_analysis_schema() -> dict:
    """JSON-схема для structured output Ollama (без $ref — совместимее)."""
    return {
        "type": "object",
        "properties": {
            "topic_updates": {
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
            },
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
                    },
                    "required": ["type", "topic_id", "note", "suggested_question"],
                },
            },
        },
        "required": ["topic_updates", "recommendations"],
    }
