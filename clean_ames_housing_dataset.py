import sys
import pandas as pd

NA_VALUES = ['NA', '']


def load_ames_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=NA_VALUES)


NONE_CATEGORY_COLS = [
    'pool_qc', 'misc_feature', 'alley', 'fence', 'fireplace_qu',
    'garage_type', 'garage_finish', 'garage_qual', 'garage_cond',
    'bsmt_qual', 'bsmt_cond', 'bsmt_exposure', 'bsmtfin_type_1', 'bsmtfin_type_2',
    'mas_vnr_type',
]

ZERO_FILL_COLS = [
    'mas_vnr_area', 'bsmtfin_sf_1', 'bsmtfin_sf_2', 'bsmt_unf_sf', 'total_bsmt_sf',
    'bsmt_full_bath', 'bsmt_half_bath', 'garage_cars', 'garage_area',
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace('/', '_', regex=False)
        .str.replace(' ', '_', regex=False)
        .str.lower()
    )
    return df


def clean_ames_housing(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_column_names(df)
    str_cols = df.select_dtypes(include=['object']).columns
    for c in str_cols:
        df[c] = df[c].str.strip()
    df.loc[df['garage_yr_blt'] == 2207, 'garage_yr_blt'] = 2007
    mask = df['year_remod_add'] < df['year_built']
    df.loc[mask, 'year_remod_add'] = df.loc[mask, 'year_built']
    for c in NONE_CATEGORY_COLS:
        df[c] = df[c].fillna('None')
    for c in ZERO_FILL_COLS:
        df[c] = df[c].fillna(0)

    df['garage_yr_blt'] = df['garage_yr_blt'].fillna(df['year_built'])
    df['lot_frontage'] = df.groupby('neighborhood')['lot_frontage'] \
        .transform(lambda s: s.fillna(s.median()))
    df['lot_frontage'] = df['lot_frontage'].fillna(df['lot_frontage'].median())
    df['electrical'] = df['electrical'].fillna(df['electrical'].mode().iloc[0])
    df['ms_subclass'] = df['ms_subclass'].astype(str)

    outlier_mask = (df['gr_liv_area'] > 4000) & (df['saleprice'] < 300000)
    df = df[~outlier_mask].reset_index(drop=True)
    return df


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else 'AmesHousing.csv'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'AmesHousing_cleaned.csv'
    raw = load_ames_csv(in_path)
    clean = clean_ames_housing(raw)
    print(f"Raw shape:     {raw.shape}")
    print(f"Cleaned shape: {clean.shape}")
    print(f"Missing values remaining: {clean.isnull().sum().sum()}")
    print(f"Duplicate rows: {clean.duplicated().sum()}")
    clean.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print("Reminder: load this cleaned file with load_ames_csv() too, "
          "not plain pd.read_csv() - otherwise 'None' category values "
          "(Pool QC, Alley, Fence, Garage Type, ...) turn back into NaN.")