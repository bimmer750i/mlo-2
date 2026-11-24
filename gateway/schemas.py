from __future__ import annotations

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    class_id: int
    label: str
    score: float


class VisionResponse(BaseModel):
    model: str
    predictions: list[Prediction]
    latency_ms: float


class ChatRequest(BaseModel):
    messages: list[dict] = Field(..., min_length=1)
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False