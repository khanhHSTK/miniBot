"""Pydantic schemas cho API dự đoán giá nhà Ames."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class HouseFeatures(RootModel[dict[str, float | int]]):
    """Các đặc trưng số của căn nhà; cột còn thiếu lấy giá trị nền của tập train."""

    root: dict[str, float | int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_values(self) -> "HouseFeatures":
        if not self.root:
            raise ValueError("Cần cung cấp ít nhất một đặc trưng.")
        for name, value in self.root.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"'{name}' phải là số hữu hạn.")
        return self


class BatchHouseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: list[HouseFeatures] = Field(..., min_length=1, max_length=5_000)


class PredictionResponse(BaseModel):
    predicted_price_usd: float = Field(description="Giá nhà dự đoán, đơn vị USD")
    estimated_error_usd: float = Field(description="RMSE trên tập kiểm thử")
    prediction_interval_95_usd: tuple[float, float]
    model_version: str
    confidence_note: str


class BatchPredictionResponse(BaseModel):
    predictions_usd: list[float]
    intervals_95_usd: list[tuple[float, float]]
    count: int
    model_version: str


class HistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    input_features: dict[str, Any]
    predicted_price_usd: float
    lower_bound_usd: float
    upper_bound_usd: float
    model_version: str
    timestamp: datetime
