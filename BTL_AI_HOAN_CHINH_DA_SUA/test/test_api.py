from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

TEST_DB = Path(__file__).with_name("test_house_history.db")
if TEST_DB.exists():
    TEST_DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB.as_posix()}"

from backend.app.main import app  # noqa: E402
from backend.app.ml_service import ModelService  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_AMES_CSV = PROJECT_ROOT / "cleaned_data" / "AmesHousing.csv"

VALID_PAYLOAD = {
    "overall_qual": 7,
    "overall_cond": 6,
    "year_built": 2000,
    "year_remod_add": 2005,
    "yr_sold": 2010,
    "gr_liv_area": 1800,
    "total_bsmt_sf": 1000,
    "1st_flr_sf": 1200,
    "2nd_flr_sf": 600,
    "full_bath": 2,
    "half_bath": 1,
    "bsmt_full_bath": 0,
    "bsmt_half_bath": 0,
    "garage_cars": 2,
    "garage_area": 480,
    "fireplaces": 1,
    "lot_area": 8000,
    "lot_frontage": 60,
    "bedroom_abvgr": 3,
    "kitchen_abvgr": 1,
    "totrms_abvgrd": 7,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client
    if TEST_DB.exists():
        TEST_DB.unlink()


def test_model_artifact_and_training_contract():
    service = ModelService().load()
    assert type(service.estimator).__name__ == "XGBRegressor"
    assert len(service.feature_columns) == 263
    assert service.model_version == "1.0.0"
    assert service.rmse == pytest.approx(18668.790641067248)


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
        "database": "connected",
    }


def test_model_info(client: TestClient):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "XGBRegressor"
    assert data["model_version"] == "1.0.0"
    assert data["data_version"] == "ames-clean-v1.0"
    assert data["n_features"] == 263
    assert data["training_rows"] == 2925
    assert data["metrics"]["r2"] == pytest.approx(0.9509799480438232)
    assert "cleaned_data" in data["training_data"]


def test_key_fields(client: TestClient):
    response = client.get("/key-fields")
    assert response.status_code == 200
    data = response.json()
    assert "overall_qual" in data["key_fields"]
    assert "total_bsmt_sf" in data["key_fields"]
    assert data["defaults"]["overall_qual"] is not None


def test_predict_and_store_history(client: TestClient):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert 20_000 <= data["predicted_price_usd"] <= 1_000_000
    assert data["estimated_error_usd"] == pytest.approx(18668.79)
    assert len(data["prediction_interval_95_usd"]) == 2
    assert data["prediction_interval_95_usd"][0] < data["predicted_price_usd"]
    assert data["prediction_interval_95_usd"][1] > data["predicted_price_usd"]
    assert data["model_version"] == "1.0.0"

    history = client.get("/history?limit=5")
    assert history.status_code == 200
    records = history.json()
    assert len(records) == 1
    assert records[0]["input_features"]["overall_qual"] == 7
    assert records[0]["model_version"] == "1.0.0"
    assert records[0]["lower_bound_usd"] < records[0]["upper_bound_usd"]


def test_predict_batch(client: TestClient):
    second = {**VALID_PAYLOAD, "overall_qual": 3, "gr_liv_area": 800}
    response = client.post("/predict-batch", json={"inputs": [VALID_PAYLOAD, second]})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions_usd"]) == 2
    assert len(data["intervals_95_usd"]) == 2
    assert data["model_version"] == "1.0.0"


def test_predict_rejects_unknown_feature(client: TestClient):
    invalid = {**VALID_PAYLOAD, "cot_khong_ton_tai": 123}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 400


def test_predict_rejects_out_of_range_value(client: TestClient):
    invalid = {**VALID_PAYLOAD, "overall_qual": 99}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 400
    assert "overall_qual" in response.json()["detail"]


def test_predict_rejects_wrong_type(client: TestClient):
    invalid = {**VALID_PAYLOAD, "gr_liv_area": "khong-phai-so"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_history_limit_validation(client: TestClient):
    response = client.get("/history?limit=0")
    assert response.status_code == 422


@pytest.mark.skipif(not RAW_AMES_CSV.exists(), reason="Không có file dữ liệu Ames gốc")
def test_predict_batch_csv_matches_real_data_shape(client: TestClient):
    raw = pd.read_csv(RAW_AMES_CSV).head(5)
    csv_bytes = raw.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/predict-batch-csv",
        files={"file": ("raw.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 5
    assert len(data["predictions_usd"]) == 5
    assert len(data["intervals_95_usd"]) == 5
    for price in data["predictions_usd"]:
        assert 20_000 <= price <= 1_000_000
