import argparse
import hashlib
import json
import time
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = PROJECT_ROOT / "cleaned_data" / "AmesHousing_basic_cleaned_for_training.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(path: Path):
    df = pd.read_csv(path)
    if 'saleprice' not in df.columns:
        raise ValueError(f"File {path} không có cột 'saleprice'")
    X = df.drop(columns=['saleprice'])
    y = df['saleprice']
    return X, y

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = float(np.mean(np.abs((y_test - preds) / y_test)) * 100)
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

def run(args):
    t0 = time.time()
    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    results = []
    trained_models = {}

    print('\n[1/3] Training Ridge (baseline, co StandardScaler)...')
    ridge = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=10.0, random_state=42)),
    ])
    ridge.fit(X_train, y_train)
    metrics = evaluate(ridge, X_test, y_test)
    joblib.dump(ridge, out_dir / 'ridge_baseline.joblib')
    trained_models['Ridge'] = ridge
    results.append({'model': 'Ridge', 'tuned': False, **metrics})
    print(f"MAE={metrics['mae']:,.0f}  R2={metrics['r2']:.4f}")

    print('\n[2/3] Training RandomForest baseline...')
    rf_base = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_base.fit(X_train, y_train)
    metrics_base = evaluate(rf_base, X_test, y_test)
    results.append({'model': 'RandomForest (baseline)', 'tuned': False, **metrics_base})
    print(f"  MAE={metrics_base['mae']:,.0f}  R2={metrics_base['r2']:.4f}")

    print('Tuning RandomForest bang RandomizedSearchCV (3-fold CV)...')
    rf_param_dist = {
        'n_estimators': [150, 250, 350],
        'max_depth': [None, 15, 25],
        'max_features': [0.3, 0.5, 0.8],
        'min_samples_leaf': [1, 2, 4],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),  # n_jobs=1 o day de tranh oversubscribe voi n_jobs=-1 ben duoi
        param_distributions=rf_param_dist,
        n_iter=8,
        scoring='neg_mean_absolute_error',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    metrics_tuned = evaluate(rf_best, X_test, y_test)
    joblib.dump(rf_best, out_dir / 'random_forest_tuned.joblib')
    trained_models['RandomForest (tuned)'] = rf_best
    results.append({'model': 'RandomForest (tuned)', 'tuned': True, **metrics_tuned})
    print(f"  Best params: {rf_search.best_params_}")
    print(f"  MAE={metrics_tuned['mae']:,.0f}  R2={metrics_tuned['r2']:.4f}")

    print('\n[3/3] Training XGBoost baseline...')
    xgb_base = XGBRegressor(n_estimators=300, random_state=42, verbosity=0, n_jobs=-1)
    xgb_base.fit(X_train, y_train)
    metrics_base_xgb = evaluate(xgb_base, X_test, y_test)
    results.append({'model': 'XGBoost (baseline)', 'tuned': False, **metrics_base_xgb})
    print(f"  MAE={metrics_base_xgb['mae']:,.0f}  R2={metrics_base_xgb['r2']:.4f}")

    print('Tuning XGBoost bang RandomizedSearchCV (3-fold CV)...')
    xgb_param_dist = {
        'n_estimators': [150, 250, 350],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.7, 0.85, 1.0],
        'colsample_bytree': [0.7, 0.85, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=42, verbosity=0, n_jobs=1),
        param_distributions=xgb_param_dist,
        n_iter=8,
        scoring='neg_mean_absolute_error',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    xgb_search.fit(X_train, y_train)
    xgb_best = xgb_search.best_estimator_
    metrics_tuned_xgb = evaluate(xgb_best, X_test, y_test)
    joblib.dump(xgb_best, out_dir / 'xgboost_tuned.joblib')
    trained_models['XGBoost (tuned)'] = xgb_best
    results.append({'model': 'XGBoost (tuned)', 'tuned': True, **metrics_tuned_xgb})
    print(f"  Best params: {xgb_search.best_params_}")
    print(f"  MAE={metrics_tuned_xgb['mae']:,.0f}  R2={metrics_tuned_xgb['r2']:.4f}")

    df_res = pd.DataFrame(results).sort_values('mae').reset_index(drop=True)
    df_res.to_csv(out_dir / 'model_comparison.csv', index=False)

    best_row = df_res.iloc[0]
    best_name = best_row['model']
    print(f"\n=== Model tot nhat theo MAE: {best_name} (MAE={best_row['mae']:,.0f}, R2={best_row['r2']:.4f}) ===")

    if best_name in trained_models:
        joblib.dump(trained_models[best_name], out_dir / 'best_model.joblib')
    else:
        fallback = {'RandomForest (baseline)': rf_base, 'XGBoost (baseline)': xgb_base}.get(best_name)
        if fallback is not None:
            joblib.dump(fallback, out_dir / 'best_model.joblib')

    best_model_obj = trained_models.get(best_name, {'RandomForest (baseline)': rf_base, 'XGBoost (baseline)': xgb_base}.get(best_name))
    if best_model_obj is not None and hasattr(best_model_obj, 'feature_importances_'):
        imp_df = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model_obj.feature_importances_
        }).sort_values('importance', ascending=False)
        imp_df.to_csv(out_dir / 'feature_importance.csv', index=False)
        print('\nTop 10 feature importance:')
        print(imp_df.head(10).to_string(index=False))

    best_model_path = out_dir / 'best_model.joblib'
    metadata = {
        'model_name': type(best_model_obj).__name__ if best_model_obj is not None else str(best_name),
        'model_version': args.model_version,
        'data_version': args.data_version,
        'dataset': 'Ames Housing',
        'target': 'SalePrice',
        'target_unit': 'USD',
        'training_rows': int(len(X)),
        'n_features': int(X.shape[1]),
        'split': '80/20, random_state=42',
        'selection_metric': 'MAE',
        'metrics': {
            'mae': float(best_row['mae']),
            'rmse': float(best_row['rmse']),
            'r2': float(best_row['r2']),
            'mape_percent': float(best_row['mape']),
        },
        'artifact_sha256': sha256_file(best_model_path) if best_model_path.exists() else None,
        'training_data_sha256': sha256_file(data_path),
        'created_at': date.today().isoformat(),
        'confidence_method': 'Khoang tham khao 95% = du doan ± 1.96 × RMSE tren tap test; day khong phai xac suat bao dam.',
    }
    (out_dir / 'model_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    print(f"\nHoan tat trong {time.time()-t0:.1f}s. Ket qua luu tai: {out_dir}/")
    print(df_res.to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(DEFAULT_DATA))
    p.add_argument('--out', default=str(SCRIPT_DIR))
    p.add_argument('--model-version', default='1.0.0')
    p.add_argument('--data-version', default='ames-clean-v1.0')
    args = p.parse_args()
    run(args)