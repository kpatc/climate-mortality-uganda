"""
Features v4 — MAP PfPR + ENSO + calibration isotonique.
Sources :
- Malaria Atlas Project WCS : PfPR 2020, Pf Incidence 2020, Pf Mortality 2020
- NOAA ONI (Oceanic Niño Index) : Niño 3.4 mensuel depuis 1982
Benchmark : LB = 0.8210, CV = 0.8016.
"""

import io, requests, time
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

SEED = 42
DATA = Path('data')
BEST_CV = 0.8003

train = pd.read_csv(DATA / 'Train.csv', parse_dates=['deathdate'])
test  = pd.read_csv(DATA / 'Test.csv',  parse_dates=['deathdate'])
y     = pd.read_parquet(DATA / 'y_train.parquet')['is_climate_sensitive']
groups = train['location'].values

# ── 1. MAP PfPR, Pf Incidence, Pf Mortality ───────────────────────────────
COVERAGES = {
    'pfpr_2020':       'Explorer__2020_Global_PfPR',
    'pf_incidence_2020': 'Explorer__2020_Global_Pf_Incidence',
    'pf_mortality_2020': 'Explorer__2020_Global_Pf_Mortality_Rate',
    'pfpr_2019':       'Explorer__2019_Global_PfPR',
}

locs = (
    pd.concat([train[['latitude','longitude']], test[['latitude','longitude']]])
    .drop_duplicates().reset_index(drop=True).round(6)
)

MAP_CACHE = DATA / 'map_cache.parquet'

if MAP_CACHE.exists():
    map_df = pd.read_parquet(MAP_CACHE)
    print(f'MAP cache chargé : {map_df.shape}')
else:
    print(f'Téléchargement MAP pour {len(locs)} lieux...')
    rows = []
    for _, row in locs.iterrows():
        lat = round(row['latitude'],  6)
        lon = round(row['longitude'], 6)
        feat = {'latitude': lat, 'longitude': lon}
        for col, cov_id in COVERAGES.items():
            delta = 0.05
            url = (
                f'https://data.malariaatlas.org/geoserver/Explorer/ows'
                f'?service=WCS&version=2.0.1&request=GetCoverage'
                f'&coverageId={cov_id}'
                f'&subset=Long({lon-delta},{lon+delta})'
                f'&subset=Lat({lat-delta},{lat+delta})'
                f'&format=image/tiff'
            )
            for attempt in range(3):
                try:
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200:
                        with rasterio.open(io.BytesIO(r.content)) as ds:
                            val = list(ds.sample([(lon, lat)]))[0][0]
                            feat[col] = float(val) if val != ds.nodata else np.nan
                    else:
                        feat[col] = np.nan
                    break
                except Exception:
                    time.sleep(2 ** attempt)
                    feat[col] = np.nan
            time.sleep(0.5)
        rows.append(feat)
        print(f"  ({lat:.4f},{lon:.4f}) pfpr={feat.get('pfpr_2020', 'err'):.3f}", flush=True)
    map_df = pd.DataFrame(rows)
    map_df.to_parquet(MAP_CACHE)
    print('MAP cache sauvegardé.')

print('\nStats MAP features :')
print(map_df[['pfpr_2020','pf_incidence_2020','pf_mortality_2020']].describe().round(4))

# Merge MAP features
train = train.merge(map_df.round(6), on=['latitude','longitude'], how='left')
test  = test.merge(map_df.round(6),  on=['latitude','longitude'], how='left')

# Feature dérivée : différence PfPR entre années (tendance)
train['pfpr_trend'] = train['pfpr_2020'] - train['pfpr_2019']
test['pfpr_trend']  = test['pfpr_2020']  - test['pfpr_2019']


# ── 2. ENSO (ONI — Niño 3.4 mensuel) ─────────────────────────────────────
print('\nTéléchargement ENSO...')
oni_url = 'https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices'
r = requests.get(oni_url, timeout=30)
from io import StringIO
oni_df = pd.read_csv(StringIO(r.text), sep=r'\s+')
oni_df.columns = ['year','month','nino12','nino12_anom','nino3','nino3_anom',
                   'nino4','nino4_anom','nino34','nino34_anom']
oni_df = oni_df[['year','month','nino34_anom']].copy()
oni_df['year']  = oni_df['year'].astype(int)
oni_df['month'] = oni_df['month'].astype(int)
print(f'ENSO chargé : {len(oni_df)} mois ({oni_df.year.min()}-{oni_df.year.max()})')

def add_enso(df):
    df = df.copy()
    df['year']  = df['deathdate'].dt.year
    df['month'] = df['deathdate'].dt.month

    # ONI au moment du décès
    df = df.merge(oni_df.rename(columns={'nino34_anom':'oni_0m'}),
                  on=['year','month'], how='left')

    # ONI 3 mois avant (lag malaria East Africa)
    lag3 = df[['year','month']].copy()
    lag3['month'] -= 3
    lag3.loc[lag3['month'] <= 0, 'year']  -= 1
    lag3.loc[lag3['month'] <= 0, 'month'] += 12
    lag3 = lag3.merge(oni_df.rename(columns={'nino34_anom':'oni_3m'}),
                      on=['year','month'], how='left')
    df['oni_3m'] = lag3['oni_3m'].values

    # ONI 6 mois avant
    lag6 = df[['year','month']].copy()
    lag6['month'] -= 6
    lag6.loc[lag6['month'] <= 0, 'year']  -= 1
    lag6.loc[lag6['month'] <= 0, 'month'] += 12
    lag6 = lag6.merge(oni_df.rename(columns={'nino34_anom':'oni_6m'}),
                      on=['year','month'], how='left')
    df['oni_6m'] = lag6['oni_6m'].values

    # El Niño actif (>0.5) ou La Niña (<-0.5) au lag 3m
    df['is_el_nino_3m'] = (df['oni_3m'] >  0.5).astype(int)
    df['is_la_nina_3m'] = (df['oni_3m'] < -0.5).astype(int)

    return df

train = add_enso(train)
test  = add_enso(test)

print('Corrélations ENSO avec target :')
for col in ['oni_0m','oni_3m','oni_6m','is_el_nino_3m','is_la_nina_3m']:
    print(f'  {col}: {train[col].corr(y):.4f}')

print('Corrélations MAP avec target :')
for col in ['pfpr_2020','pf_incidence_2020','pf_mortality_2020','pfpr_trend']:
    print(f'  {col}: {train[col].corr(y):.4f}')


# ── 3. Assemblage features v4 ─────────────────────────────────────────────
X_train_v1 = pd.read_parquet(DATA / 'X_train.parquet')
X_test_v1  = pd.read_parquet(DATA / 'X_test.parquet')
X_train_v1['ID'] = train['ID'].values
X_test_v1['ID']  = test['ID'].values

map_enso_cols = ['pfpr_2020','pf_incidence_2020','pf_mortality_2020','pfpr_trend',
                 'oni_0m','oni_3m','oni_6m','is_el_nino_3m','is_la_nina_3m']

for col in map_enso_cols:
    X_train_v1[col] = train[col].values
    X_test_v1[col]  = test[col].values

X_train_v4 = X_train_v1.drop(columns=['ID'])
X_test_v4  = X_test_v1.drop(columns=['ID'])

# Imputation NaN
for col in map_enso_cols:
    med = X_train_v4[col].median()
    X_train_v4[col] = X_train_v4[col].fillna(med)
    X_test_v4[col]  = X_test_v4[col].fillna(med)

print(f'\nFeatures v4 : {X_train_v4.shape[1]} (+{len(map_enso_cols)} nouvelles)')
print(f'NaN : {X_train_v4.isnull().sum().sum()}')

X_train_v4.to_parquet(DATA / 'X_train_v4.parquet')
X_test_v4.to_parquet(DATA / 'X_test_v4.parquet')


# ── 4. CV v5_params + v4_features ─────────────────────────────────────────
CAT_COLS = ['zone','gender']
Xc = X_train_v4.copy()
Xc_te = X_test_v4.copy()
for col in CAT_COLS:
    le = LabelEncoder()
    Xc[col]    = le.fit_transform(Xc[col].astype(str))
    Xc_te[col] = le.transform(Xc_te[col].astype(str))

PARAMS = dict(
    objective='binary', metric='binary_logloss', verbosity=-1,
    learning_rate=0.0351, num_leaves=117, min_child_samples=13,
    feature_fraction=0.771, bagging_fraction=0.995, bagging_freq=5,
    reg_alpha=0.00255, reg_lambda=0.0651,
    scale_pos_weight=(y==0).sum()/(y==1).sum(),
    n_estimators=2000, random_state=SEED,
)

gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(y))
test_preds = np.zeros(len(X_test_v4))

print('\nCV v5_params + v4_features...')
for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xc, y, groups)):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(Xc.iloc[tr_idx], y.iloc[tr_idx],
          eval_X=Xc.iloc[va_idx], eval_y=y.iloc[va_idx],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=-1)])
    oof[va_idx] = m.predict_proba(Xc.iloc[va_idx])[:, 1]
    test_preds += m.predict_proba(Xc_te)[:, 1] / 5
    print(f'  Fold {fold+1}: iter={m.best_iteration_}')

f1    = f1_score(y, (oof>=0.5).astype(int))
auc   = roc_auc_score(y, oof)
score = 0.6*f1 + 0.4*auc
print(f'\nCV v4 : F1={f1:.4f}  AUC={auc:.4f}  Score={score:.4f}')
print(f'Benchmark v5 : {BEST_CV:.4f}  Delta : {score-BEST_CV:+.4f}')


# ── 5. Calibration isotonique (test en parallèle) ─────────────────────────
print('\n--- Calibration isotonique sur OOF v4 ---')
# Calibration OOF via cross_val_predict avec IsotonicRegression
from sklearn.model_selection import cross_val_predict
iso = IsotonicRegression(out_of_bounds='clip')
# Utiliser les OOF comme proba d'entrée
oof_cal = np.zeros(len(y))
for tr_idx, va_idx in gkf.split(Xc, y, groups):
    iso_fold = IsotonicRegression(out_of_bounds='clip')
    iso_fold.fit(oof[tr_idx], y.iloc[tr_idx])
    oof_cal[va_idx] = iso_fold.predict(oof[va_idx])

f1_cal  = f1_score(y, (oof_cal>=0.5).astype(int))
auc_cal = roc_auc_score(y, oof_cal)
score_cal = 0.6*f1_cal + 0.4*auc_cal
print(f'Après calibration : F1={f1_cal:.4f}  AUC={auc_cal:.4f}  Score={score_cal:.4f}')
print(f'Delta calibration : {score_cal - score:+.4f}')


# ── 6. Génération soumission ───────────────────────────────────────────────
# Utiliser le meilleur score (avec ou sans calibration)
best = max(score, score_cal)
if best > BEST_CV:
    if score_cal > score:
        # Calibrer les prédictions test aussi
        iso_full = IsotonicRegression(out_of_bounds='clip')
        iso_full.fit(oof, y)
        final_proba = iso_full.predict(test_preds)
        tag = 'calibrated'
    else:
        final_proba = test_preds
        tag = 'raw'

    pred = (final_proba >= 0.5).astype(int)
    test_ids = pd.read_parquet(DATA / 'test_ids.parquet')['ID']
    sub = pd.DataFrame({'ID': test_ids, 'TargetF1': pred, 'TargetRAUC': final_proba})
    out = DATA / f'submission_v11_map_enso_{best:.4f}_{tag}.csv'
    sub.to_csv(out, index=False)
    print(f'\n✓ Soumission : {out}')
    print(f'TargetF1 : {sub["TargetF1"].value_counts().to_dict()}')

    # Feature importance
    m_full = lgb.LGBMClassifier(**{**PARAMS, 'n_estimators':1000}, random_state=SEED)
    m_full.fit(Xc, y)
    imp = pd.Series(m_full.feature_importances_, index=Xc.columns)
    print('\nTop 15 features :')
    print(imp.sort_values(ascending=False).head(15).to_string())
else:
    print(f'\n✗ Pas d\'amélioration vs v5 ({best:.4f} <= {BEST_CV:.4f}).')

# Log
exp = pd.read_csv('notes/experiments.csv')
new = pd.DataFrame([{
    'date': str(pd.Timestamp.today().date()),
    'model': 'LGBM_v5params_v4features_MAP_ENSO',
    'cv_strategy': 'GroupKFold(5)_by_location',
    'cv_score': round(best, 4),
    'cv_f1': round(max(f1,f1_cal), 4),
    'cv_auc': round(max(auc,auc_cal), 4),
    'lb_public': '',
    'notes': f'MAP PfPR+Incidence+Mortality + ENSO ONI 0/3/6m. Delta={best-BEST_CV:+.4f}',
}])
pd.concat([exp, new], ignore_index=True).to_csv('notes/experiments.csv', index=False)
print('Expérience loguée.')
