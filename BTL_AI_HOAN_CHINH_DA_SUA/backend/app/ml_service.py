"""Tải model Ames Housing, xử lý dữ liệu đầu vào và thực hiện dự đoán.

Model tốt nhất được huấn luyện bởi ``file_train/train_and_tune_models.py``
trên dữ liệu Ames Housing đã làm sạch trong ``cleaned_data``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
CLEANED_DATA_DIR = PROJECT_ROOT / "cleaned_data"
TRAINING_CSV = CLEANED_DATA_DIR / "AmesHousing_basic_cleaned_for_training.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "file_train" / "best_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "file_train" / "model_metadata.json"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()
METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", str(DEFAULT_METADATA_PATH))).resolve()
TARGET_COL = "saleprice"

KEY_INPUT_FIELDS = [
    "overall_qual", "overall_cond", "year_built", "year_remod_add", "yr_sold",
    "gr_liv_area", "total_bsmt_sf", "1st_flr_sf", "2nd_flr_sf", "totalsf",
    "full_bath", "half_bath", "bsmt_full_bath", "bsmt_half_bath", "totalbath",
    "garage_cars", "garage_area", "fireplaces", "lot_area", "lot_frontage",
    "bedroom_abvgr", "kitchen_abvgr", "totrms_abvgrd", "houseage", "remodage",
    "hasgarage", "hasbasement", "hasfireplace",
]

FIELD_RANGES: dict[str, tuple[float, float]] = {
    "overall_qual": (1, 10),
    "overall_cond": (1, 10),
    "year_built": (1800, 2035),
    "year_remod_add": (1800, 2035),
    "yr_sold": (2000, 2035),
    "gr_liv_area": (100, 10000),
    "total_bsmt_sf": (0, 10000),
    "1st_flr_sf": (100, 10000),
    "2nd_flr_sf": (0, 10000),
    "totalsf": (100, 20000),
    "full_bath": (0, 10),
    "half_bath": (0, 10),
    "bsmt_full_bath": (0, 10),
    "bsmt_half_bath": (0, 10),
    "totalbath": (0, 20),
    "garage_cars": (0, 10),
    "garage_area": (0, 5000),
    "fireplaces": (0, 10),
    "lot_area": (100, 500000),
    "lot_frontage": (0, 1000),
    "bedroom_abvgr": (0, 20),
    "kitchen_abvgr": (0, 10),
    "totrms_abvgrd": (1, 30),
    "houseage": (0, 300),
    "remodage": (0, 300),
    "hasgarage": (0, 1),
    "hasbasement": (0, 1),
    "hasfireplace": (0, 1),
}

if str(CLEANED_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(CLEANED_DATA_DIR))


class ModelService:
    """Bọc model Ames và pipeline tiền xử lý, nạp một lần khi khởi động app."""

    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.estimator: Any = None
        self.metadata: dict[str, Any] = {}
        self._feature_columns: list[str] | None = None
        self._template_row: dict[str, Any] | None = None

    def load(self) -> "ModelService":
        if not self.model_path.exists():
            raise RuntimeError(
                f"Không tìm thấy model tại '{self.model_path}'. "
                "Chạy file_train/train_and_tune_models.py trước để tạo best_model.joblib."
            )
        try:
            self.estimator = joblib.load(self.model_path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Không thể tải model '{self.model_path}': {exc}") from exc
        if not hasattr(self.estimator, "predict"):
            raise RuntimeError("Model artifact không có phương thức predict().")

        if not TRAINING_CSV.exists():
            raise RuntimeError(
                f"Không tìm thấy '{TRAINING_CSV}'. File này dùng để xác định đúng "
                "danh sách đặc trưng của model."
            )
        full = pd.read_csv(TRAINING_CSV)
        self._feature_columns = [c for c in full.columns if c != TARGET_COL]
        # Median tạo bộ giá trị nền đại diện hơn so với lấy ngẫu nhiên dòng đầu tiên.
        self._template_row = full[self._feature_columns].median(numeric_only=True).to_dict()

        if self.metadata_path.exists():
            try:
                self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Không đọc được metadata model: {exc}") from exc
        else:
            self.metadata = {
                "model_name": type(self.estimator).__name__,
                "model_version": "unknown",
                "data_version": "unknown",
                "metrics": {},
            }

        expected = getattr(self.estimator, "n_features_in_", None)
        if expected is not None and int(expected) != len(self._feature_columns):
            raise RuntimeError(
                f"Model yêu cầu {expected} đặc trưng nhưng dữ liệu chuẩn có "
                f"{len(self._feature_columns)} đặc trưng."
            )
        return self

    @property
    def feature_columns(self) -> list[str]:
        if self._feature_columns is None:
            raise RuntimeError("Model chưa được tải (gọi load() trước).")
        return self._feature_columns

    @property
    def template_row(self) -> dict[str, Any]:
        if self._template_row is None:
            raise RuntimeError("Model chưa được tải (gọi load() trước).")
        return self._template_row

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    @property
    def rmse(self) -> float:
        value = self.metadata.get("metrics", {}).get("rmse", 0.0)
        return float(value or 0.0)

    def _validate_overrides(self, overrides: dict[str, Any]) -> None:
        unknown = [k for k in overrides if k not in self.feature_columns]
        if unknown:
            raise ValueError(f"Đặc trưng không hợp lệ: {unknown}")
        for name, value in overrides.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"'{name}' phải là số.") from exc
            if not np.isfinite(numeric):
                raise ValueError(f"'{name}' phải là số hữu hạn.")
            if name in FIELD_RANGES:
                low, high = FIELD_RANGES[name]
                if numeric < low or numeric > high:
                    raise ValueError(f"'{name}' phải nằm trong khoảng {low:g}–{high:g}.")

    @staticmethod
    def _synchronise_engineered_features(values: dict[str, Any], supplied: set[str]) -> None:
        """Giữ các đặc trưng dẫn xuất nhất quán với các trường thành phần."""
        if {"yr_sold", "year_built"} & supplied and "yr_sold" in values and "year_built" in values:
            values["houseage"] = max(0.0, float(values["yr_sold"]) - float(values["year_built"]))
        if {"yr_sold", "year_remod_add"} & supplied and "yr_sold" in values and "year_remod_add" in values:
            values["remodage"] = max(0.0, float(values["yr_sold"]) - float(values["year_remod_add"]))

        area_parts = {"total_bsmt_sf", "1st_flr_sf", "2nd_flr_sf"}
        if area_parts & supplied and area_parts.issubset(values):
            values["totalsf"] = sum(float(values[c]) for c in area_parts)

        bath_parts = {"full_bath", "half_bath", "bsmt_full_bath", "bsmt_half_bath"}
        if bath_parts & supplied and bath_parts.issubset(values):
            values["totalbath"] = (
                float(values["full_bath"])
                + 0.5 * float(values["half_bath"])
                + float(values["bsmt_full_bath"])
                + 0.5 * float(values["bsmt_half_bath"])
            )

        if {"garage_area", "garage_cars"} & supplied:
            values["hasgarage"] = int(float(values.get("garage_area", 0)) > 0 or float(values.get("garage_cars", 0)) > 0)
        if "total_bsmt_sf" in supplied:
            values["hasbasement"] = int(float(values.get("total_bsmt_sf", 0)) > 0)
        if "fireplaces" in supplied:
            values["hasfireplace"] = int(float(values.get("fireplaces", 0)) > 0)

    def predict_from_partial(self, overrides: dict[str, Any]) -> float:
        """Dự đoán từ các đặc trưng chính; cột còn lại lấy giá trị nền trung vị."""
        if not overrides:
            raise ValueError("Cần cung cấp ít nhất một đặc trưng đầu vào.")
        self._validate_overrides(overrides)

        values = dict(self.template_row)
        values.update({k: float(v) for k, v in overrides.items()})
        self._synchronise_engineered_features(values, set(overrides))
        frame = pd.DataFrame([values])[self.feature_columns].astype(float)
        prediction = float(self.estimator.predict(frame)[0])
        if not np.isfinite(prediction):
            raise RuntimeError("Model trả về NaN hoặc giá trị vô hạn.")
        return prediction

    def prediction_interval_95(self, prediction: float) -> tuple[float, float]:
        """Khoảng tham khảo từ RMSE hold-out, không phải khoảng tin cậy đã hiệu chuẩn."""
        margin = 1.96 * self.rmse
        return max(0.0, prediction - margin), prediction + margin

    def predict_from_raw_dataframe(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Dự đoán hàng loạt từ CSV thô cùng định dạng AmesHousing.csv gốc."""
        from clean_ames_housing_dataset import clean_ames_housing, make_model_ready  # noqa: E402

        df = raw_df.copy()
        has_target = any(c.strip().lower() == "saleprice" for c in df.columns)
        if not has_target:
            df["SalePrice"] = 0.0

        required_raw_cols = {
            "Garage Yr Blt", "Year Built", "Year Remod/Add", "Neighborhood",
            "Lot Frontage", "Electrical", "Gr Liv Area", "Total Bsmt SF",
            "1st Flr SF", "2nd Flr SF", "Full Bath", "Half Bath",
            "Bsmt Full Bath", "Bsmt Half Bath", "Garage Area", "Fireplaces",
            "Yr Sold",
        }
        missing_raw = sorted(required_raw_cols - set(df.columns))
        if missing_raw:
            raise ValueError(
                "CSV thiếu các cột bắt buộc theo định dạng AmesHousing.csv gốc: "
                f"{missing_raw}. Vui lòng dùng file raw 80+ cột như "
                "cleaned_data/AmesHousing.csv."
            )

        try:
            df_clean = clean_ames_housing(df)
            X, _ = make_model_ready(df_clean)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Không xử lý được CSV Ames: {exc}") from exc

        missing = sorted(set(self.feature_columns) - set(X.columns))
        if missing:
            X = pd.concat([X, pd.DataFrame(0, index=X.index, columns=missing)], axis=1)
        extra = set(X.columns) - set(self.feature_columns)
        if extra:
            X = X.drop(columns=list(extra))
        X = X[self.feature_columns].astype(float)

        predictions = np.asarray(self.estimator.predict(X), dtype=float).reshape(-1)
        if predictions.shape[0] != len(X):
            raise RuntimeError("Model trả về số lượng dự đoán không khớp dữ liệu đầu vào.")
        if not np.all(np.isfinite(predictions)):
            raise RuntimeError("Model trả về NaN hoặc giá trị vô hạn.")
        return predictions
