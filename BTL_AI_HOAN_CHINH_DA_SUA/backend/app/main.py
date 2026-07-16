from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .ml_service import KEY_INPUT_FIELDS, ModelService
from .models import HousePredictionHistory
from .schemas import (
    BatchHouseFeatures,
    BatchPredictionResponse,
    HistoryResponse,
    HouseFeatures,
    PredictionResponse,
)

logger = logging.getLogger(__name__)
model_service: ModelService | None = None
CONFIDENCE_NOTE = (
    "Khoảng 95% là ước lượng tham khảo từ RMSE trên tập kiểm thử, "
    "không phải xác suất bảo đảm cho từng căn nhà."
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model_service
    Base.metadata.create_all(bind=engine)
    model_service = ModelService().load()
    logger.info("Đã tải model Ames từ %s", model_service.model_path)
    yield


app = FastAPI(
    title="AI House Price Prediction API",
    description="Dự đoán giá nhà theo bộ đặc trưng Ames Housing.",
    version="3.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DbSession = Annotated[Session, Depends(get_db)]


def _require_model() -> ModelService:
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")
    return model_service


def _record_for(
    service: ModelService,
    input_features: dict[str, Any],
    prediction: float,
) -> HousePredictionHistory:
    lower, upper = service.prediction_interval_95(prediction)
    record = HousePredictionHistory(
        predicted_price_usd=prediction,
        lower_bound_usd=lower,
        upper_bound_usd=upper,
        model_version=service.model_version,
    )
    record.input_features = input_features
    return record


@app.get("/", tags=["System"])
def root():
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["System"])
def health_check(db: DbSession):
    try:
        db.execute(text("SELECT 1"))
        database_status = "connected"
    except SQLAlchemyError:
        database_status = "error"

    ready = model_service is not None and database_status == "connected"
    return {
        "status": "healthy" if ready else "degraded",
        "model_loaded": model_service is not None,
        "database": database_status,
    }


@app.get("/model-info", tags=["System"])
def model_info():
    service = _require_model()
    metadata = service.metadata
    return {
        "model_name": type(service.estimator).__name__,
        "model_version": service.model_version,
        "data_version": metadata.get("data_version", "unknown"),
        "n_features": len(service.feature_columns),
        "dataset": "Ames Housing (dữ liệu thật, đã làm sạch trong thư mục cleaned_data)",
        "training_rows": metadata.get("training_rows"),
        "target": "SalePrice",
        "target_unit": "USD",
        "framework": "scikit-learn / xgboost",
        "selection_metric": metadata.get("selection_metric", "MAE"),
        "metrics": metadata.get("metrics", {}),
        "confidence_method": metadata.get("confidence_method", CONFIDENCE_NOTE),
        "artifact": "file_train/best_model.joblib",
        "training_data": "cleaned_data/AmesHousing_basic_cleaned_for_training.csv",
    }


@app.get("/key-fields", tags=["System"])
def key_fields():
    service = _require_model()
    return {
        "key_fields": KEY_INPUT_FIELDS,
        "defaults": {k: service.template_row.get(k) for k in KEY_INPUT_FIELDS},
    }


@app.post("/predict", response_model=PredictionResponse, tags=["ML Prediction"])
def predict_house_price(features: HouseFeatures, db: DbSession):
    service = _require_model()
    try:
        predicted_price = service.predict_from_partial(features.root)
        lower, upper = service.prediction_interval_95(predicted_price)
        db.add(_record_for(service, features.root, predicted_price))
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Lỗi lưu lịch sử dự đoán")
        raise HTTPException(status_code=500, detail="Không thể lưu lịch sử dự đoán") from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lỗi không lường trước khi dự đoán đơn lẻ")
        raise HTTPException(status_code=400, detail=f"Không dự đoán được: {exc}") from exc

    return PredictionResponse(
        predicted_price_usd=round(predicted_price, 2),
        estimated_error_usd=round(service.rmse, 2),
        prediction_interval_95_usd=(round(lower, 2), round(upper, 2)),
        model_version=service.model_version,
        confidence_note=CONFIDENCE_NOTE,
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["ML Prediction"])
def predict_batch(batch: BatchHouseFeatures, db: DbSession):
    service = _require_model()
    try:
        predictions = [service.predict_from_partial(item.root) for item in batch.inputs]
        db.add_all(
            _record_for(service, item.root, prediction)
            for item, prediction in zip(batch.inputs, predictions, strict=True)
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Lỗi lưu lịch sử batch")
        raise HTTPException(status_code=500, detail="Không thể lưu lịch sử batch") from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lỗi không lường trước khi dự đoán batch")
        raise HTTPException(status_code=400, detail=f"Không dự đoán được: {exc}") from exc

    intervals = [service.prediction_interval_95(float(v)) for v in predictions]
    return BatchPredictionResponse(
        predictions_usd=[round(float(v), 2) for v in predictions],
        intervals_95_usd=[(round(lo, 2), round(hi, 2)) for lo, hi in intervals],
        count=len(predictions),
        model_version=service.model_version,
    )


@app.post("/predict-batch-csv", response_model=BatchPredictionResponse, tags=["ML Prediction"])
async def predict_batch_csv(db: DbSession, file: UploadFile = File(...)):
    service = _require_model()
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file CSV.")
    try:
        raw_df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Không đọc được CSV: {exc}") from exc

    if raw_df.empty:
        raise HTTPException(status_code=400, detail="CSV không có dòng dữ liệu.")
    if len(raw_df) > 5_000:
        raise HTTPException(status_code=400, detail="Mỗi lần chỉ xử lý tối đa 5.000 dòng.")

    try:
        predictions = service.predict_from_raw_dataframe(raw_df)
    except (ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lỗi không lường trước khi xử lý batch CSV")
        raise HTTPException(status_code=400, detail=f"Không xử lý được CSV: {exc}") from exc

    try:
        db.add_all(
            _record_for(service, {"source": file.filename, "row_index": int(i)}, float(p))
            for i, p in enumerate(predictions)
        )
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Lỗi lưu lịch sử batch CSV")

    intervals = [service.prediction_interval_95(float(v)) for v in predictions]
    return BatchPredictionResponse(
        predictions_usd=[round(float(v), 2) for v in predictions],
        intervals_95_usd=[(round(lo, 2), round(hi, 2)) for lo, hi in intervals],
        count=len(predictions),
        model_version=service.model_version,
    )


@app.get("/history", response_model=list[HistoryResponse], tags=["Data"])
def get_history(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=1_000)] = 20,
):
    records = (
        db.query(HousePredictionHistory)
        .order_by(HousePredictionHistory.id.desc())
        .limit(limit)
        .all()
    )
    return [
        HistoryResponse(
            id=r.id,
            input_features=r.input_features,
            predicted_price_usd=r.predicted_price_usd,
            lower_bound_usd=r.lower_bound_usd,
            upper_bound_usd=r.upper_bound_usd,
            model_version=r.model_version,
            timestamp=r.timestamp,
        )
        for r in records
    ]
