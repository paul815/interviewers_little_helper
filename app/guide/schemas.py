"""Pydantic-модели структурированного гайда."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Topic(BaseModel):
    id: str
    question: str
    notes: str | None = None
    priority: Literal["must", "nice"] = "must"


class Section(BaseModel):
    id: str
    title: str
    topics: list[Topic] = Field(default_factory=list)


class Guide(BaseModel):
    guide_id: str
    language: str = "ru"
    title: str = ""
    sections: list[Section] = Field(default_factory=list)

    def all_topics(self) -> list[tuple[Section, Topic]]:
        return [(s, t) for s in self.sections for t in s.topics]

    def topic_ids(self) -> list[str]:
        return [t.id for _, t in self.all_topics()]
