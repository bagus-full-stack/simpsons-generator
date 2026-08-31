from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TextGenerationRequest(BaseModel):
    character_name: str = Field(min_length=1, max_length=200)
    steps: int = Field(default=30, ge=1, le=75)
    turbo_mode: bool = False


class JobOut(BaseModel):
    id: str
    mode: str
    prompt: str
    status: str
    image_url: str | None = None
    error: str | None = None
    duration_ms: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
