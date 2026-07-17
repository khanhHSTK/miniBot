"""Bảng SQLAlchemy dùng để lưu lịch sử dự đoán giá nhà Ames."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class HousePredictionHistory(Base):
    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    input_features_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    predicted_price_usd: Mapped[float] = mapped_column(Float, nullable=False)
    lower_bound_usd: Mapped[float] = mapped_column(Float, nullable=False)
    upper_bound_usd: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False, default="unknown")
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    @property
    def input_features(self) -> dict[str, Any]:
        try:
            return json.loads(self.input_features_json)
        except (TypeError, ValueError):
            return {}

    @input_features.setter
    def input_features(self, value: dict[str, Any]) -> None:
        self.input_features_json = json.dumps(value, ensure_ascii=False)
