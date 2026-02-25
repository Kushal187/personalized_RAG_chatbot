from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    person_name: str
    source_file: str
    snippet: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    target_person: Optional[str] = None


class RejectedFile(BaseModel):
    filename: str
    reason: str


class BuildResponse(BaseModel):
    processed_count: int
    rejected_count: int
    rejected_files: list[RejectedFile] = Field(default_factory=list)
    candidate_count: int
    chunk_count: int
    candidates: list[str] = Field(default_factory=list)
    target_person: Optional[str] = None
    target_loaded: bool


class StatusResponse(BaseModel):
    candidate_count: int
    chunk_count: int
    candidates: list[str] = Field(default_factory=list)
    target_person: Optional[str] = None
    target_loaded: bool
