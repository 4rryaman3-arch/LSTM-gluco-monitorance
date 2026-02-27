from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class ForecastRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(
        ..., description="User selected timestamp for the forecast starting point."
    )
    current_glucose: float = Field(..., ge=40, le=500, description="mg/dL")
    horizon_hours: int = Field(default=6, ge=1, le=24)
    step_minutes: int = Field(default=5, ge=5, le=60)
    insulin_units: float | None = Field(default=None, ge=0, le=50)
    include_without_insulin: bool = True

    @model_validator(mode="after")
    def validate_step_alignment(self) -> "ForecastRequest":
        if (self.horizon_hours * 60) % self.step_minutes != 0:
            raise ValueError("horizon_hours * 60 must be divisible by step_minutes")
        return self


class ForecastPoint(BaseModel):
    timestamp: datetime
    cgm: float
    status: Literal["hypo", "euglycemia", "hyper"]
    insulin_effect: float


class ModelInfo(BaseModel):
    model_active: bool
    model_name: str
    notes: str


class InsulinRecommendation(BaseModel):
    units: float
    expected_time_in_range_pct: float
    expected_min_cgm: float
    expected_max_cgm: float
    objective_score: float
    note: str


class ForecastResponse(BaseModel):
    request_id: str
    input: ForecastRequest
    with_insulin: list[ForecastPoint] | None
    without_insulin: list[ForecastPoint] | None
    recommended_insulin: InsulinRecommendation
    model: ModelInfo
