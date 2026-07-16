# AI House — Hệ thống dự đoán giá nhà Ames

Ứng dụng học thuật phục vụ bài tập lớn/midterm Machine Learning:

- **Frontend:** Streamlit, gồm 5 màn hình.
- **Backend:** FastAPI, có REST API dự đoán và lịch sử.
- **Database:** SQLite + SQLAlchemy.
- **Model:** XGBoost tuned, lưu bằng `joblib`.
- **Dữ liệu:** Ames Housing trong thư mục `cleaned_data`.
- **Kiểm thử:** Pytest + FastAPI TestClient.
- **Triển khai:** Docker Compose.

## Kết quả model đã kiểm tra

Model `file_train/best_model.joblib` sử dụng đúng **263 đặc trưng** từ file
`cleaned_data/AmesHousing_basic_cleaned_for_training.csv` gồm **2.925 dòng**.
Kết quả tái tính trên tập test 20%, `random_state=42`:

| Độ đo | Kết quả |
|---|---:|
| MAE | 12,268.87 USD |
| RMSE | 18,668.79 USD |
| R² | 0.95098 |
| MAPE | 6.98% |

Với bài toán hồi quy, hệ thống không dùng “xác suất lớp”. Thay vào đó API trả
khoảng dự đoán tham khảo 95% bằng `prediction ± 1.96 × RMSE`. Khoảng này chỉ
mang tính minh họa học thuật, chưa phải khoảng tin cậy đã hiệu chuẩn.

## Chạy nhanh trên Windows

Yêu cầu Python **3.11**. Tại thư mục gốc dự án, nhấp đúp:

```text
start_windows.bat
```

Hoặc chạy thủ công:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
$env:API_URL="http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501
```

Địa chỉ:

- Giao diện: `http://127.0.0.1:8501`
- Swagger API: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## REST API chính

| Method | Endpoint | Chức năng |
|---|---|---|
| POST | `/predict` | Dự đoán đơn lẻ, trả giá + khoảng tham khảo |
| POST | `/predict-batch` | Dự đoán nhiều JSON |
| POST | `/predict-batch-csv` | Upload CSV raw Ames |
| GET | `/health` | Kiểm tra model và CSDL |
| GET | `/model-info` | Version, dữ liệu, độ đo |
| GET | `/history?limit=20` | Xem lịch sử dự đoán |

## Chạy kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Kiểm tra độ phủ:

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=backend.app --cov-report=term-missing
```

Kết quả bản bàn giao: **12 passed**.

## Dự đoán hàng loạt

Upload CSV cùng định dạng 82 cột raw của Ames Housing, ví dụ
`Overall Qual`, `Gr Liv Area`, `Neighborhood`, ... Hệ thống tự gọi pipeline
`cleaned_data/clean_ames_housing_dataset.py`, căn chỉnh đúng 263 đặc trưng và
xuất các cột dự đoán, cận dưới, cận trên. Tối đa 5.000 dòng/lần.

## Huấn luyện lại

Có thể chạy từ **bất kỳ thư mục hiện hành nào** vì đường dẫn được xác định theo
vị trí file script:

```powershell
.\.venv\Scripts\python.exe file_train/train_and_tune_models.py
```

Script tạo:

- `ridge_baseline.joblib`
- `random_forest_tuned.joblib`
- `xgboost_tuned.joblib`
- `best_model.joblib`
- `model_comparison.csv`
- `model_metadata.json`

## Tạo lại biểu đồ EDA

```powershell
.\.venv\Scripts\python.exe eda_charts/eda_visualization.py
```

## Chạy bằng Docker

```bash
docker compose up --build
```

- Backend expose cổng `8000` và đã copy cả `cleaned_data` lẫn `file_train`.
- Frontend expose cổng `8501`, gọi backend qua `http://backend:8000`.
- SQLite được giữ trong volume `house-data`.

Dừng hệ thống:

```bash
docker compose down
```

Xóa cả lịch sử trong volume:

```bash
docker compose down -v
```

## Cấu trúc chính

```text
BTL_AI_HOAN_CHINH/
├── backend/app/                 # FastAPI, model service, SQLAlchemy
├── cleaned_data/                # raw, cleaned, model-ready và script clean
├── eda_charts/                  # 4 biểu đồ EDA + script
├── file_train/                  # script train, model, metrics, metadata
├── frontend/app.py              # 5 màn hình Streamlit
├── test/test_api.py             # 12 kiểm thử
├── docker-compose.yml
├── requirements.txt
├── start_windows.bat
└── run_tests_windows.bat
```

## Lưu ý học thuật

- Form đơn lẻ chỉ nhập các đặc trưng quan trọng; các cột còn lại dùng giá trị
  trung vị của tập huấn luyện.
- Model phản ánh thị trường Ames trong dữ liệu lịch sử, không nên xem là công cụ
  định giá thương mại cho nhà ở Việt Nam hoặc thị trường hiện tại.
- Không đóng gói `.venv` vào file ZIP; mỗi máy tự tạo môi trường ảo để tránh lỗi
  khác hệ điều hành và giảm dung lượng.
  ### chạy local
cd C:\Users\66tie\Downloads\BTL_AI_HOAN_CHINH\BTL_AI_HOAN_CHINH
Remove-Item -Recurse -Force .\.venv

py -3.11 -m venv .venv

.\.venv\Scripts\python.exe --version

.\.venv\Scripts\python.exe -m pip install --upgrade pip

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\start_windows.bat
