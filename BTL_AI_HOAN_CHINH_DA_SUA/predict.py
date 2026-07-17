import argparse
import sys
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'cleaned_data'
MODEL_DIR = ROOT / 'file_train'

TARGET_COL = 'saleprice'

def load_model(model_path=None):
    if model_path is None:
        model_path = MODEL_DIR / 'best_model.joblib'
    path = Path(model_path)
    if not path.exists():
        print(f"Khong tim thay file mo hinh: {path}")
        sys.exit(1)
    model = joblib.load(path)
    print(f"Da load mo hinh: {path} ({type(model).__name__})")
    return model

def get_feature_columns():
    path = DATA_DIR / 'AmesHousing_basic_cleaned_for_training.csv'
    if not path.exists():
        print(f"Khong tim thay file chuan: {path}")
        sys.exit(1)
    df = pd.read_csv(path, nrows=0)
    cols = [c for c in df.columns if c != TARGET_COL]
    print(f"Doc cau truc: {len(cols)} features")
    return cols

def load_clean_pipeline():
    sys.path.insert(0, str(ROOT / 'cleaned_data'))
    from clean_ames_housing_dataset import clean_ames_housing, make_model_ready
    return clean_ames_housing, make_model_ready

def preprocess_csv(raw_path):
    df = pd.read_csv(raw_path)
    print(f"Doc raw: {len(df)} dong, {len(df.columns)} cot")

    clean_fn, ready_fn = load_clean_pipeline()
    df_clean = clean_fn(df)
    X, _ = ready_fn(df_clean)

    feature_cols = get_feature_columns()
    missing = set(feature_cols) - set(X.columns)
    for c in missing:
        X[c] = 0
    extra = set(X.columns) - set(feature_cols)
    if extra:
        X = X.drop(columns=list(extra))
    X = X[feature_cols]

    print(f"Sau xu ly: {len(X)} dong, {len(X.columns)} features")
    return X

def predict_csv(model, input_path, output_path):
    X = preprocess_csv(input_path)
    preds = model.predict(X)
    out_df = pd.DataFrame({'predicted_saleprice': np.round(preds, 2)})
    out_df.to_csv(output_path, index=False)
    print(f"\nDa luu ket qua: {output_path}")
    print(f"Thong ke du doan:")
    print(f"  Min: ${preds.min():,.2f}")
    print(f"  Max: ${preds.max():,.2f}")
    print(f"  Mean: ${preds.mean():,.2f}")
    print(f"  Median: ${np.median(preds):,.2f}")
    return out_df

def get_template_row():
    path = DATA_DIR / 'AmesHousing_basic_cleaned_for_training.csv'
    df = pd.read_csv(path)
    row = df.drop(columns=[TARGET_COL]).iloc[0].to_dict()
    return row

def interactive_predict(model):
    feature_cols = get_feature_columns()
    template = get_template_row()

    print("\n=== DU DOAN GIA NHA (TUONG TAC) ===")
    print("Nhap gia tri (Enter de giu nguyen gia tri mac dinh).")
    print(f"Co {len(feature_cols)} features, nhap mot so cot chinh:\n")

    key_cols = [
        'overall_qual', 'overall_cond', 'year_built', 'year_remod_add',
        'gr_liv_area', 'totalsf', 'totalbath', 'garage_cars', 'garage_area',
        'fireplaces', 'lot_area', 'lot_frontage', 'bsmtfin_sf_1',
        'total_bsmt_sf', '1st_flr_sf', '2nd_flr_sf', 'full_bath', 'half_bath',
        'bedroom_abvgr', 'kitchen_abvgr', 'totrms_abvgrd',
        'houseage', 'remodage', 'hasgarage', 'hasbasement', 'hasfireplace',
        'central_air_Y', 'street_Pave', 'paved_drive_Y',
    ]

    values = template.copy()
    for col in key_cols:
        default = values.get(col, 0)
        inp = input(f"  {col} [{default}]: ").strip()
        if inp != "":
            try:
                values[col] = float(inp) if isinstance(default, (int, float)) else inp
            except ValueError:
                values[col] = inp

    df = pd.DataFrame([values])
    df = df[feature_cols]
    pred = model.predict(df)[0]
    print(f"\n=> GIÁ NHÀ DỰ ĐÓAN: ${pred:,.2f}")
    return pred

def demo_predict(model):
    feature_cols = get_feature_columns()
    template = get_template_row()
    values = template.copy()

    overrides = {
        'overall_qual': 7, 'overall_cond': 6, 'year_built': 2000,
        'year_remod_add': 2005, 'gr_liv_area': 1800, 'totalsf': 2500,
        'totalbath': 2.5, 'garage_cars': 2, 'garage_area': 480,
        'fireplaces': 1, 'lot_area': 8000, 'lot_frontage': 60,
        'bsmtfin_sf_1': 900, 'total_bsmt_sf': 1000,
        '1st_flr_sf': 1200, '2nd_flr_sf': 600,
        'full_bath': 2, 'half_bath': 1,
        'bedroom_abvgr': 3, 'kitchen_abvgr': 1, 'totrms_abvgrd': 7,
        'houseage': 24, 'remodage': 19, 'hasgarage': 1,
        'hasbasement': 1, 'hasfireplace': 1,
        'mo_sold': 6, 'yr_sold': 2024,
        'central_air_Y': 1, 'street_Pave': 1, 'paved_drive_Y': 1,
        'ms_zoning_RL': 1, 'neighborhood_NAmes': 1,
        'bldg_type_1Fam': 1, 'house_style_1Story': 1,
        'foundation_PConc': 1, 'garage_type_Attchd': 1,
        'exterior_1st_VinylSd': 1, 'exterior_2nd_VinylSd': 1,
        'sale_condition_Normal': 1, 'sale_type_WD': 1,
        'functional_Typ': 1, 'roof_style_Gable': 1,
        'heating_GasA': 1, 'electrical_SBrkr': 1,
    }
    values.update(overrides)

    df = pd.DataFrame([values])
    df = df[feature_cols]
    pred = model.predict(df)[0]
    print(f"\n=== DEMO ===")
    print(f" Mẫu nhà: 1800sqft, 3PN, 2.5 phòng tắm, garage 2 xe, OverallQual=7")
    print(f" => Giá dự đoán: ${pred:,.2f}")
    return pred

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Du doan gia nha bang mo hinh .joblib")
    p.add_argument('--input', help='File CSV raw dau vao')
    p.add_argument('--output', default='ket_qua_du_doan.csv')
    p.add_argument('--model', default=str(MODEL_DIR / 'best_model.joblib'))
    p.add_argument('--interactive', action='store_true',
                   help='Che do nhap tuong tac')
    p.add_argument('--demo', action='store_true',
                   help='Du doan nhanh voi du lieu mau')
    args = p.parse_args()

    model = load_model(args.model)

    if args.interactive:
        interactive_predict(model)
    elif args.input:
        predict_csv(model, args.input, args.output)
    else:
        demo_predict(model)