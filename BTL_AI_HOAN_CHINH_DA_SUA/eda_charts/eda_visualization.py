import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / 'cleaned_data' / 'AmesHousing_basic_cleaned.csv'
OUTPUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else SCRIPT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
sns.histplot(df['saleprice'], kde=True, ax=axes[0], color='#4C72B0')
axes[0].set_title('Phan bo SalePrice (goc)')
axes[0].set_xlabel('SalePrice (USD)')

sns.histplot(np.log1p(df['saleprice']), kde=True, ax=axes[1], color='#55A868')
axes[1].set_title('Phan bo log(SalePrice)')
axes[1].set_xlabel('log(1 + SalePrice)')

skew_raw = df['saleprice'].skew()
skew_log = np.log1p(df['saleprice']).skew()
fig.suptitle(f"Hinh 3.1 - Phan bo bien muc tieu (skew goc={skew_raw:.2f}, skew log={skew_log:.2f})")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'hinh_3_1_phan_bo_saleprice.png', dpi=150)
plt.close(fig)

outlier_cols = ['gr_liv_area', 'total_bsmt_sf', 'lot_area', 'saleprice']
fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
for ax, col in zip(axes, outlier_cols):
    sns.boxplot(y=df[col], ax=ax, color='#DD8452')
    ax.set_title(col)
fig.suptitle('Hinh 3.2 - Boxplot phat hien outlier o cac bien so quan trong')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'hinh_3_2_boxplot_outlier.png', dpi=150)
plt.close(fig)

numeric_df = df.select_dtypes(include=['number'])
top_corr = numeric_df.corr(numeric_only=True)['saleprice'].abs().sort_values(ascending=False)
top_features = top_corr.index[1:13]  # top 12 (bo saleprice ra khoi danh sach)
corr_matrix = numeric_df[list(top_features) + ['saleprice']].corr()

fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
            annot_kws={'size': 8})
ax.set_title('Hinh 3.3 - Tuong quan giua top 12 bien so va SalePrice')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'hinh_3_3_correlation_heatmap.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.scatterplot(data=df, x='gr_liv_area', y='saleprice', alpha=0.4, ax=axes[0], color='#4C72B0')
axes[0].set_title('GrLivArea vs SalePrice')

sns.boxplot(data=df, x='overall_qual', y='saleprice', ax=axes[1], color='#55A868')
axes[1].set_title('OverallQual vs SalePrice')
axes[1].tick_params(axis='x', rotation=0)

fig.suptitle('Hinh 3.4 - Quan he giua dac trung quan trong va bien muc tieu')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'hinh_3_4_feature_vs_target.png', dpi=150)
plt.close(fig)

print('Đã tạo 4 biểu đồ:')
print(' - hinh_3_1_phan_bo_saleprice.png')
print(' - hinh_3_2_boxplot_outlier.png')
print(' - hinh_3_3_correlation_heatmap.png')
print(' - hinh_3_4_feature_vs_target.png')
print()
print('Top 12 bien tuong quan manh nhat voi SalePrice:')
print(top_corr.iloc[1:13].to_string())