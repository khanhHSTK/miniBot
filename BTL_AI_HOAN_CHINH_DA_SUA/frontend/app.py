"""Giao diện Streamlit cho hệ thống dự đoán giá nhà Ames Housing."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st
from requests import Response
from requests.exceptions import RequestException

API_URL = os.getenv("API_URL", "http://localhost:8000").strip().rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15"))

st.set_page_config(
    page_title="Hệ thống dự đoán giá nhà Ames",
    page_icon="🏠",
    layout="wide",
)


def api_request(method: str, endpoint: str, **kwargs: Any) -> Response | None:
    try:
        return requests.request(
            method,
            f"{API_URL}{endpoint}",
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
    except RequestException as exc:
        st.error(f"Không thể kết nối backend tại {API_URL}: {exc}")
        return None


def parse_error(response: Response) -> str:
    try:
        payload = response.json()
        return str(payload.get("detail", payload))
    except ValueError:
        return response.text or "Backend không trả nội dung lỗi."


def format_usd(value: float) -> str:
    return f"${value:,.0f} USD"


@st.cache_data(ttl=60)
def fetch_key_fields() -> dict[str, Any] | None:
    response = api_request("GET", "/key-fields")
    return response.json() if response is not None and response.ok else None


@st.cache_data(ttl=60)
def fetch_model_info() -> dict[str, Any] | None:
    response = api_request("GET", "/model-info")
    return response.json() if response is not None and response.ok else None


st.sidebar.title("Menu chức năng")
st.sidebar.caption(f"Backend: {API_URL}")
choice = st.sidebar.radio(
    "Chọn màn hình",
    [
        "1. Trang chủ / Dashboard",
        "2. Dự đoán đơn lẻ",
        "3. Thông tin model",
        "4. Lịch sử dự đoán",
        "5. Dự đoán hàng loạt (CSV thô)",
    ],
)

if choice == "1. Trang chủ / Dashboard":
    st.title("Hệ thống AI dự đoán giá nhà Ames Housing")
    st.write(
        "Ứng dụng học thuật gồm Streamlit, FastAPI, SQLite và XGBoost, "
        "huấn luyện trên dữ liệu Ames Housing đã làm sạch."
    )

    health_response = api_request("GET", "/health")
    info = fetch_model_info()
    if health_response is not None and health_response.ok:
        health = health_response.json()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trạng thái", health.get("status", "unknown").upper())
        col2.metric("Model", "Đã tải" if health.get("model_loaded") else "Chưa tải")
        col3.metric("Database", health.get("database", "unknown"))
        col4.metric("Phiên bản", (info or {}).get("model_version", "unknown"))
    elif health_response is not None:
        st.error(f"Backend lỗi {health_response.status_code}: {parse_error(health_response)}")

    if info:
        metrics = info.get("metrics", {})
        st.subheader("Kết quả mô hình trên tập kiểm thử")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", format_usd(float(metrics.get("mae", 0))))
        m2.metric("RMSE", format_usd(float(metrics.get("rmse", 0))))
        m3.metric("R²", f"{float(metrics.get('r2', 0)):.4f}")
        m4.metric("MAPE", f"{float(metrics.get('mape_percent', 0)):.2f}%")
        st.caption(
            "Model tốt nhất: XGBoost tuned, chọn theo MAE. "
            "Dữ liệu đầu vào chuẩn có 263 đặc trưng sau mã hóa."
        )

elif choice == "2. Dự đoán đơn lẻ":
    st.title("Dự đoán giá nhà")
    st.caption(
        "Nhập các thuộc tính chính. Hệ thống tự tính TotalSF, TotalBath, tuổi nhà "
        "và các cờ có garage/tầng hầm/lò sưởi để dữ liệu đầu vào nhất quán."
    )

    key_info = fetch_key_fields()
    defaults = (key_info or {}).get("defaults", {})

    def d(name: str, fallback: float) -> float:
        value = defaults.get(name)
        return float(value) if value is not None else fallback

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Chất lượng và thời gian")
            overall_qual = st.slider("Chất lượng tổng thể", 1, 10, int(d("overall_qual", 6)))
            overall_cond = st.slider("Tình trạng tổng thể", 1, 10, int(d("overall_cond", 5)))
            year_built = st.number_input("Năm xây", 1800, 2035, int(d("year_built", 1990)))
            year_remod_add = st.number_input("Năm sửa gần nhất", 1800, 2035, int(d("year_remod_add", 2000)))
            yr_sold = st.number_input("Năm định giá/bán", 2000, 2035, int(d("yr_sold", 2010)))
            lot_area = st.number_input("Diện tích lô đất (sqft)", 100.0, 500000.0, d("lot_area", 9000.0))
            lot_frontage = st.number_input("Mặt tiền lô đất (ft)", 0.0, 1000.0, d("lot_frontage", 65.0))
        with col2:
            st.subheader("Diện tích và số phòng")
            gr_liv_area = st.number_input("Diện tích sinh hoạt (sqft)", 100.0, 10000.0, d("gr_liv_area", 1500.0))
            total_bsmt_sf = st.number_input("Diện tích tầng hầm (sqft)", 0.0, 10000.0, d("total_bsmt_sf", 900.0))
            first_flr_sf = st.number_input("Diện tích tầng 1 (sqft)", 100.0, 10000.0, d("1st_flr_sf", 1100.0))
            second_flr_sf = st.number_input("Diện tích tầng 2 (sqft)", 0.0, 10000.0, d("2nd_flr_sf", 0.0))
            bedroom_abvgr = st.number_input("Số phòng ngủ", 0, 20, int(d("bedroom_abvgr", 3)))
            totrms_abvgrd = st.number_input("Tổng số phòng", 1, 30, int(d("totrms_abvgrd", 6)))
            kitchen_abvgr = st.number_input("Số bếp", 0, 10, int(d("kitchen_abvgr", 1)))
        with col3:
            st.subheader("Tiện nghi")
            full_bath = st.number_input("Phòng tắm đầy đủ", 0, 10, int(d("full_bath", 2)))
            half_bath = st.number_input("Phòng tắm nửa", 0, 10, int(d("half_bath", 0)))
            bsmt_full_bath = st.number_input("Phòng tắm đầy đủ ở tầng hầm", 0, 10, int(d("bsmt_full_bath", 0)))
            bsmt_half_bath = st.number_input("Phòng tắm nửa ở tầng hầm", 0, 10, int(d("bsmt_half_bath", 0)))
            garage_cars = st.number_input("Sức chứa garage (xe)", 0, 10, int(d("garage_cars", 2)))
            garage_area = st.number_input("Diện tích garage (sqft)", 0.0, 5000.0, d("garage_area", 480.0))
            fireplaces = st.number_input("Số lò sưởi", 0, 10, int(d("fireplaces", 1)))

        submitted = st.form_submit_button("Thực hiện dự đoán", type="primary")

    if submitted:
        if year_remod_add < year_built:
            st.error("Năm sửa gần nhất không được nhỏ hơn năm xây.")
        elif yr_sold < year_built:
            st.error("Năm định giá/bán không được nhỏ hơn năm xây.")
        else:
            payload = {
                "overall_qual": overall_qual,
                "overall_cond": overall_cond,
                "year_built": year_built,
                "year_remod_add": year_remod_add,
                "yr_sold": yr_sold,
                "gr_liv_area": gr_liv_area,
                "total_bsmt_sf": total_bsmt_sf,
                "1st_flr_sf": first_flr_sf,
                "2nd_flr_sf": second_flr_sf,
                "full_bath": full_bath,
                "half_bath": half_bath,
                "bsmt_full_bath": bsmt_full_bath,
                "bsmt_half_bath": bsmt_half_bath,
                "garage_cars": garage_cars,
                "garage_area": garage_area,
                "fireplaces": fireplaces,
                "lot_area": lot_area,
                "lot_frontage": lot_frontage,
                "bedroom_abvgr": bedroom_abvgr,
                "totrms_abvgrd": totrms_abvgrd,
                "kitchen_abvgr": kitchen_abvgr,
            }
            response = api_request("POST", "/predict", json=payload)
            if response is not None and response.ok:
                data = response.json()
                lower, upper = data["prediction_interval_95_usd"]
                c1, c2 = st.columns(2)
                c1.metric("Giá nhà dự kiến", format_usd(data["predicted_price_usd"]))
                c2.metric("RMSE tham khảo", format_usd(data["estimated_error_usd"]))
                st.success(
                    f"Khoảng dự đoán tham khảo 95%: {format_usd(lower)} – {format_usd(upper)}"
                )
                st.caption(data["confidence_note"])
                st.info(f"Model version: {data['model_version']} · Đã lưu lịch sử vào CSDL.")
            elif response is not None:
                st.error(f"Dự đoán thất bại ({response.status_code}): {parse_error(response)}")

elif choice == "3. Thông tin model":
    st.title("Thông tin model")
    info = fetch_model_info()
    if info:
        st.subheader(f"{info['model_name']} · version {info['model_version']}")
        st.write(info["dataset"])
        left, right = st.columns(2)
        with left:
            st.json({
                "data_version": info.get("data_version"),
                "training_rows": info.get("training_rows"),
                "n_features": info.get("n_features"),
                "target": info.get("target"),
                "artifact": info.get("artifact"),
                "training_data": info.get("training_data"),
            })
        with right:
            st.json(info.get("metrics", {}))
        st.warning(info.get("confidence_method", ""))
    else:
        st.error("Không đọc được thông tin model từ backend.")

elif choice == "4. Lịch sử dự đoán":
    st.title("Lịch sử dự đoán")
    limit = st.slider("Số bản ghi", min_value=5, max_value=200, value=20, step=5)
    response = api_request("GET", f"/history?limit={limit}")
    if response is not None and response.ok:
        records = response.json()
        if not records:
            st.info("Chưa có dữ liệu lịch sử.")
        else:
            frame = pd.DataFrame(records)
            for col in ["predicted_price_usd", "lower_bound_usd", "upper_bound_usd"]:
                frame[col] = frame[col].map(format_usd)
            st.dataframe(frame, width="stretch", hide_index=True)
    elif response is not None:
        st.error(f"Không tải được lịch sử: {parse_error(response)}")

elif choice == "5. Dự đoán hàng loạt (CSV thô)":
    st.title("Dự đoán hàng loạt bằng CSV thô Ames Housing")
    st.write(
        "Upload CSV cùng định dạng với `cleaned_data/AmesHousing.csv` (80+ cột raw). "
        "Cột SalePrice không bắt buộc. Hệ thống tự làm sạch, mã hóa và dự đoán."
    )
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            preview = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Không đọc được CSV: {exc}")
            preview = None

        if preview is not None:
            if preview.empty:
                st.error("CSV không có dòng dữ liệu.")
            elif len(preview) > 5_000:
                st.error("Mỗi lần chỉ xử lý tối đa 5.000 dòng.")
            else:
                st.dataframe(preview.head(10), width="stretch")
                if st.button("Chạy dự đoán hàng loạt", type="primary"):
                    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                    response = api_request("POST", "/predict-batch-csv", files=files)
                    if response is not None and response.ok:
                        data = response.json()
                        result = preview.copy()
                        result["PredictedPriceUSD"] = data["predictions_usd"]
                        result["Lower95USD"] = [x[0] for x in data["intervals_95_usd"]]
                        result["Upper95USD"] = [x[1] for x in data["intervals_95_usd"]]
                        st.success(
                            f"Đã dự đoán {data['count']} dòng bằng model {data['model_version']}."
                        )
                        st.dataframe(result, width="stretch")
                        st.download_button(
                            "Tải kết quả CSV",
                            result.to_csv(index=False).encode("utf-8-sig"),
                            "predicted_house_prices.csv",
                            "text/csv",
                        )
                    elif response is not None:
                        st.error(
                            f"Dự đoán thất bại ({response.status_code}): {parse_error(response)}"
                        )
