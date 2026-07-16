import sys
import pandas as pd

NA_VALUES = ['NA', '']

def load_ames_csv(path: str):
    return pd.read_csv(path, keep_default_na=False, na_values=NA_VALUES)


NONE_CATEGORY_COLS = [
    'pool_qc', 'misc_feature', 'alley', 'fence', 'fireplace_qu',
    'garage_type', 'garage_finish', 'garage_qual', 'garage_cond',
    'bsmt_qual', 'bsmt_cond', 'bsmt_exposure', 'bsmtfin_type_1', 'bsmtfin_type_2',
    'mas_vnr_type',
]
NO_FEATURE_LABEL = 'NoFeature'

ZERO_FILL_COLS = [
    'mas_vnr_area', 'bsmtfin_sf_1', 'bsmtfin_sf_2', 'bsmt_unf_sf', 'total_bsmt_sf',
    'bsmt_full_bath', 'bsmt_half_bath', 'garage_cars', 'garage_area',
]


ORDINAL_QUALITY_MAP = {
    'NoFeature': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5,
}
ORDINAL_QUALITY_COLS = [
    'exter_qual', 'exter_cond', 'bsmt_qual', 'bsmt_cond', 'heating_qc',
    'kitchen_qual', 'fireplace_qu', 'garage_qual', 'garage_cond', 'pool_qc',
]


def encode_ordinal_features(df: pd.DataFrame):    
    df = df.copy()
    for c in ORDINAL_QUALITY_COLS:
        if c in df.columns:
            df[c] = df[c].map(ORDINAL_QUALITY_MAP).astype(int)
    return df

def clean_column_names(df: pd.DataFrame):
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace('/', '_', regex=False)
        .str.replace(' ', '_', regex=False).str.lower()
    )
    return df

def normalize_text_columns(df: pd.DataFrame):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('string')
        df[col] = df[col].str.strip()
        df[col] = df[col].replace('', pd.NA)
    return df

def engineer_features(df: pd.DataFrame):
    df = df.copy()

    df['houseage'] = df['yr_sold'] - df['year_built']
    df['remodage'] = df['yr_sold'] - df['year_remod_add']
    df['houseage'] = df['houseage'].clip(lower=0)
    df['remodage'] = df['remodage'].clip(lower=0)

    df['totalsf'] = df['total_bsmt_sf'] + df['1st_flr_sf'] + df['2nd_flr_sf']

    df['totalbath'] = (
        df['full_bath'] + 0.5 * df['half_bath']
        + df['bsmt_full_bath'] + 0.5 * df['bsmt_half_bath']
    )

    df['hasgarage'] = (df['garage_area'] > 0).astype(int)
    df['hasbasement'] = (df['total_bsmt_sf'] > 0).astype(int)
    df['hasfireplace'] = (df['fireplaces'] > 0).astype(int)

    return df

def clean_ames_housing(df: pd.DataFrame):
    df = clean_column_names(df)
    df = normalize_text_columns(df)

    df.loc[df['garage_yr_blt'] == 2207, 'garage_yr_blt'] = 2007
    mask = df['year_remod_add'] < df['year_built']
    df.loc[mask, 'year_remod_add'] = df.loc[mask, 'year_built']

    for c in NONE_CATEGORY_COLS:
        df[c] = df[c].fillna(NO_FEATURE_LABEL)
    for c in ZERO_FILL_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['garage_yr_blt'] = pd.to_numeric(df['garage_yr_blt'], errors='coerce').fillna(df['year_built'])
    df['lot_frontage'] = df.groupby('neighborhood')['lot_frontage'] \
        .transform(lambda s: s.fillna(s.median()))
    df['lot_frontage'] = df['lot_frontage'].fillna(df['lot_frontage'].median())
    df['electrical'] = df['electrical'].fillna(df['electrical'].mode(dropna=True).iloc[0])

    df['ms_subclass'] = df['ms_subclass'].astype(str)
    outlier_mask = df['gr_liv_area'] > 4000
    df = df[~outlier_mask].reset_index(drop=True)

    for c in df.select_dtypes(include=['object', 'string']).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(NO_FEATURE_LABEL)

    for c in df.select_dtypes(include=['number']).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    df['mas_vnr_type'] = df['mas_vnr_type'].replace('None', 'NoMasVnr')

    df = encode_ordinal_features(df)
    df = engineer_features(df)

    return df

def make_model_ready(df: pd.DataFrame):
    model_df = df.copy()
    target_col = 'saleprice'
    if target_col not in model_df.columns:
        raise KeyError(f"Target column '{target_col}' not found")

    X = model_df.drop(columns=[target_col])
    y = model_df[target_col]
    X = pd.get_dummies(X, drop_first=True)
    return X, y

if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else 'cleaned_data/AmesHousing.csv'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'cleaned_data/AmesHousing_basic_cleaned.csv'
    model_ready_path = sys.argv[3] if len(sys.argv) > 3 else out_path.replace('.csv', '_for_training.csv')

    raw = load_ames_csv(in_path)
    clean = clean_ames_housing(raw)
    print(f"Raw shape: {raw.shape}")
    print(f"Cleaned shape: {clean.shape}")
    print(f"Missing values remaining: {clean.isnull().sum().sum()}")
    print(f"Duplicate rows: {clean.duplicated().sum()}")
    print("New arrays:",
          ['houseage', 'remodage', 'totalsf', 'totalbath', 'hasgarage', 'hasbasement', 'hasfireplace'])
    clean.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    model_ready_X, model_ready_y = make_model_ready(clean)
    model_ready_df = pd.concat([model_ready_X, model_ready_y], axis=1)
    model_ready_df.to_csv(model_ready_path, index=False)
    print(f"Model-ready data saved to {model_ready_path}")