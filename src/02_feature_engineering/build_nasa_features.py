"""
Build NASA POWER features from cached JSON files and merge with existing parquet datasets.
Run after download_nasa.py has completed.
"""

import json, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb

SEED = 42
DATA  = Path('data')
CACHE = DATA / 'nasa_cache'

train = pd.read_csv(DATA / 'Train.csv', parse_dates=['deathdate'])
test  = pd.read_csv(DATA / 'Test.csv',  parse_dates=['deathdate'])

locs = (
    pd.concat([train[['latitude','longitude']], test[['latitude','longitude']]])
    .drop_duplicates().reset_index(drop=True).round(6)
)


# ── Load all cached time series ────────────────────────────────────────────
def load_ts(lat, lon, start_year, end_year):
    frames = []
    for year in range(start_year, end_year + 1):
        cp = CACHE / f'{lat:.6f}_{lon:.6f}_{year}.json'
        if not cp.exists():
            continue
        with open(cp) as f:
            data = json.load(f)
        if not data or 'properties' not in data:
            continue
        props = data['properties']['parameter']
        df_y = pd.DataFrame(props)
        df_y.index = pd.to_datetime(df_y.index, format='%Y%m%d')
        frames.append(df_y)
    if not frames:
        return pd.DataFrame()
    ts = pd.concat(frames).sort_index()
    ts = ts.replace(-999.0, np.nan)
    return ts

all_dates  = pd.concat([train['deathdate'], test['deathdate']])
START_YEAR = (all_dates.min() - pd.Timedelta(days=95)).year
END_YEAR   = all_dates.max().year

print(f'Loading {len(locs)} time series from cache...')
ts_store = {}
for _, row in locs.iterrows():
    lat = round(row['latitude'],  6)
    lon = round(row['longitude'], 6)
    ts  = load_ts(lat, lon, START_YEAR, END_YEAR)
    ts_store[(lat, lon)] = ts
    print(f'  ({lat:.4f}, {lon:.4f}) → {len(ts)} days', flush=True)
print('Done loading.')


# ── Climatological monthly means per location ──────────────────────────────
clim_store = {}
for (lat, lon), ts in ts_store.items():
    if ts.empty:
        continue
    cols = [c for c in ['RH2M', 'WS2M', 'T2M', 'T2M_MAX', 'PRECTOTCORR'] if c in ts.columns]
    monthly = ts[cols].copy()
    monthly['month'] = monthly.index.month
    clim = monthly.groupby('month').mean()
    clim.columns = [f'clim_{c}' for c in clim.columns]
    clim_store[(lat, lon)] = clim


# ── Steadman heat index ────────────────────────────────────────────────────
def steadman_heat_index(T, RH):
    return (
        -8.78469475556
        + 1.61139411   * T
        + 2.33854883889 * RH
        - 0.14611605   * T  * RH
        - 0.012308094  * T  * T
        - 0.0164248278 * RH * RH
        + 0.002211732  * T  * T  * RH
        + 0.00072546   * T  * RH * RH
        - 0.000003582  * T  * T  * RH * RH
    )


# ── Feature extraction per record ─────────────────────────────────────────
def compute_features_for_record(ts, target_date):
    w90 = ts[ts.index <= target_date].tail(90)
    w30 = w90.tail(30)
    w14 = w90.tail(14)
    w7  = w90.tail(7)

    feat = {}

    # Relative humidity
    feat['rh2m_7d']  = w7['RH2M'].mean()  if len(w7)  > 0 else np.nan
    feat['rh2m_14d'] = w14['RH2M'].mean() if len(w14) > 0 else np.nan
    feat['rh2m_30d'] = w30['RH2M'].mean() if len(w30) > 0 else np.nan
    feat['rh2m_90d'] = w90['RH2M'].mean() if len(w90) > 0 else np.nan

    # Wind speed
    feat['ws2m_7d']  = w7['WS2M'].mean()  if len(w7)  > 0 else np.nan
    feat['ws2m_30d'] = w30['WS2M'].mean() if len(w30) > 0 else np.nan

    # Solar radiation
    feat['solar_7d']  = w7['ALLSKY_SFC_SW_DWN'].mean()  if len(w7)  > 0 else np.nan
    feat['solar_30d'] = w30['ALLSKY_SFC_SW_DWN'].mean() if len(w30) > 0 else np.nan

    # Heat index
    if len(w30) > 0 and w30['T2M_MAX'].notna().any() and w30['RH2M'].notna().any():
        hi = steadman_heat_index(w30['T2M_MAX'], w30['RH2M'])
        feat['heat_index_30d'] = hi.mean()
        feat['heat_index_max'] = hi.max()
    else:
        feat['heat_index_30d'] = np.nan
        feat['heat_index_max'] = np.nan

    # RH anomaly short vs long
    feat['rh2m_anomaly_7v30']  = feat['rh2m_7d']  - feat['rh2m_30d']
    feat['rh2m_anomaly_30v90'] = feat['rh2m_30d'] - feat['rh2m_90d']

    # High humidity flags
    feat['is_high_humidity_30d'] = int(feat['rh2m_30d'] > 60) if not np.isnan(feat['rh2m_30d']) else np.nan
    feat['is_high_humidity_7d']  = int(feat['rh2m_7d']  > 60) if not np.isnan(feat['rh2m_7d'])  else np.nan

    # NASA cross-check temperatures
    feat['nasa_tmax_30d'] = w30['T2M_MAX'].mean()     if len(w30) > 0 else np.nan
    feat['nasa_tmin_30d'] = w30['T2M_MIN'].mean()     if len(w30) > 0 else np.nan
    feat['nasa_rain_30d'] = w30['PRECTOTCORR'].sum()  if len(w30) > 0 else np.nan

    return feat


def get_clim_anomaly(lat, lon, target_date, rh2m_30d):
    feat = {}
    key = (round(lat, 6), round(lon, 6))
    if key not in clim_store:
        feat['rh2m_clim_month']   = np.nan
        feat['rh2m_clim_anomaly'] = np.nan
        return feat
    clim = clim_store[key]
    month = target_date.month
    if month in clim.index and 'clim_RH2M' in clim.columns:
        feat['rh2m_clim_month']   = clim.loc[month, 'clim_RH2M']
        feat['rh2m_clim_anomaly'] = rh2m_30d - feat['rh2m_clim_month']
    else:
        feat['rh2m_clim_month']   = np.nan
        feat['rh2m_clim_anomaly'] = np.nan
    return feat


def build_nasa_features(df):
    rows = []
    for _, row in df.iterrows():
        lat   = round(row['latitude'],  6)
        lon   = round(row['longitude'], 6)
        ddate = row['deathdate']
        key   = (lat, lon)

        if key not in ts_store or ts_store[key].empty:
            rows.append({'ID': row['ID']})
            continue

        ts   = ts_store[key]
        feat = compute_features_for_record(ts, ddate)
        clim = get_clim_anomaly(lat, lon, ddate, feat.get('rh2m_30d', np.nan))
        feat.update(clim)
        feat['ID'] = row['ID']
        rows.append(feat)
    return pd.DataFrame(rows)


print('\nBuilding NASA features for train...')
nasa_train = build_nasa_features(train)
print(f'Train NASA features: {nasa_train.shape}')

print('Building NASA features for test...')
nasa_test  = build_nasa_features(test)
print(f'Test  NASA features: {nasa_test.shape}')


# ── NaN imputation ─────────────────────────────────────────────────────────
nasa_feat_cols = [c for c in nasa_train.columns if c != 'ID']
nan_rates = nasa_train[nasa_feat_cols].isnull().mean()
print('\nNaN rates (train):')
print(nan_rates[nan_rates > 0].sort_values(ascending=False).to_string())

for col in nasa_feat_cols:
    if nasa_train[col].isnull().mean() < 0.15:
        med = nasa_train[col].median()
        nasa_train[col] = nasa_train[col].fillna(med)
        nasa_test[col]  = nasa_test[col].fillna(med)

print(f'\nNaN after imputation: train={nasa_train[nasa_feat_cols].isnull().sum().sum()}, test={nasa_test[nasa_feat_cols].isnull().sum().sum()}')


# ── Correlation with target ─────────────────────────────────────────────────
y = pd.read_parquet(DATA / 'y_train.parquet')['is_climate_sensitive']
corrs = {col: nasa_train[col].corr(y) for col in nasa_feat_cols}
print('\nCorrelation with target:')
print(pd.Series(corrs).sort_values(key=abs, ascending=False).to_string())


# ── Merge with existing features ───────────────────────────────────────────
X_train_old = pd.read_parquet(DATA / 'X_train.parquet')
X_test_old  = pd.read_parquet(DATA / 'X_test.parquet')
X_train_old['ID'] = train['ID'].values
X_test_old['ID']  = test['ID'].values

X_train_v2 = X_train_old.merge(nasa_train, on='ID', how='left').drop(columns=['ID'])
X_test_v2  = X_test_old.merge(nasa_test,   on='ID', how='left').drop(columns=['ID'])

new_cols = [c for c in X_train_v2.columns if c not in X_train_old.drop(columns=['ID']).columns]
print(f'\nNew features ({len(new_cols)}): {new_cols}')
print(f'X_train_v2: {X_train_v2.shape}, NaN: {X_train_v2.isnull().sum().sum()}')
print(f'X_test_v2:  {X_test_v2.shape},  NaN: {X_test_v2.isnull().sum().sum()}')


# ── CV comparison ─────────────────────────────────────────────────────────
CAT_COLS = ['zone', 'gender']
groups   = train['location'].values

LGBM_PARAMS = dict(
    objective='binary', metric='binary_logloss', verbosity=-1,
    learning_rate=0.0351, num_leaves=117, min_child_samples=13,
    feature_fraction=0.771, bagging_fraction=0.995, bagging_freq=5,
    reg_alpha=0.00255, reg_lambda=0.0651,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    n_estimators=2000, random_state=SEED,
)


def run_cv(X, label, n_splits=5):
    Xc = X.copy()
    for col in CAT_COLS:
        if col in Xc.columns:
            le = LabelEncoder()
            Xc[col] = le.fit_transform(Xc[col].astype(str))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xc, y, groups)):
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            Xc.iloc[tr_idx], y.iloc[tr_idx],
            eval_X=Xc.iloc[va_idx], eval_y=y.iloc[va_idx],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=-1)],
        )
        oof[va_idx] = model.predict_proba(Xc.iloc[va_idx])[:, 1]
        print(f'  Fold {fold+1}: best_iteration={model.best_iteration_}', flush=True)
    f1  = f1_score(y, (oof >= 0.5).astype(int))
    auc = roc_auc_score(y, oof)
    score = 0.6 * f1 + 0.4 * auc
    print(f'[{label}] F1={f1:.4f}  AUC={auc:.4f}  Score={score:.4f}')
    return score, oof


print('\n=== CV baseline (v1, 59 features) ===')
score_v1, oof_v1 = run_cv(X_train_old.drop(columns=['ID']), 'v1_baseline')

print('\n=== CV with NASA features (v2) ===')
score_v2, oof_v2 = run_cv(X_train_v2, 'v2_nasa')

print(f'\nDelta: {score_v2 - score_v1:+.4f}')


# ── Save enriched datasets ─────────────────────────────────────────────────
X_train_v2.to_parquet(DATA / 'X_train_v2.parquet')
X_test_v2.to_parquet(DATA / 'X_test_v2.parquet')
print('\nSaved X_train_v2.parquet and X_test_v2.parquet')


# ── Generate submission if improved ───────────────────────────────────────
if score_v2 > score_v1:
    print(f'\nScore improved (+{score_v2 - score_v1:.4f}) — generating submission v6...')
    Xc_tr = X_train_v2.copy()
    Xc_te = X_test_v2.copy()
    for col in CAT_COLS:
        le = LabelEncoder()
        Xc_tr[col] = le.fit_transform(Xc_tr[col].astype(str))
        Xc_te[col] = le.transform(Xc_te[col].astype(str))
    params_sub = {**LGBM_PARAMS, 'n_estimators': 1000}
    model_sub = lgb.LGBMClassifier(**params_sub)
    model_sub.fit(Xc_tr, y)
    proba = model_sub.predict_proba(Xc_te)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    test_ids = pd.read_parquet(DATA / 'test_ids.parquet')['ID']
    sub = pd.DataFrame({'ID': test_ids, 'TargetF1': pred, 'TargetRAUC': proba})
    out = DATA / f'submission_v6_nasa_{score_v2:.4f}.csv'
    sub.to_csv(out, index=False)
    print(f'Submission saved: {out}')
    print(f'TargetF1 distribution: {sub["TargetF1"].value_counts().to_dict()}')
else:
    print(f'\nNo improvement ({score_v2:.4f} vs {score_v1:.4f}) — no submission generated.')
