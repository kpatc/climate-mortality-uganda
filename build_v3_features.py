"""
Feature engineering v3 — basé sur la littérature épidémiologique exacte.

Sources :
- PMC12676583 : lag malaria/pluie 2-8 semaines, pic à 4 semaines, seuil 200mm/semaine
- Patterns Zindi gagnants : target encoding OOF + ensemble

Nouvelles features :
1. Lags épidémiologiques précis depuis NASA cache : 14/21/28/42/56j
2. Flags hebdomadaires : rain > 200mm à J-14, J-28, J-42, J-56
3. Target encoding OOF par location, zone, season × location
4. Pseudo-features : max weekly rain sur 8 semaines, n semaines >200mm
"""

import json
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

BEST_CV = 0.8003  # v5 — benchmark strict

train = pd.read_csv(DATA / 'Train.csv', parse_dates=['deathdate'])
test  = pd.read_csv(DATA / 'Test.csv',  parse_dates=['deathdate'])
y     = pd.read_parquet(DATA / 'y_train.parquet')['is_climate_sensitive']

locs = (
    pd.concat([train[['latitude','longitude']], test[['latitude','longitude']]])
    .drop_duplicates().reset_index(drop=True).round(6)
)

# ── Load NASA cache time series ────────────────────────────────────────────
def load_ts(lat, lon):
    frames = []
    for year in range(2007, 2023):
        cp = CACHE / f'{lat:.6f}_{lon:.6f}_{year}.json'
        if not cp.exists():
            continue
        with open(cp) as f:
            data = json.load(f)
        if not data or 'properties' not in data:
            continue
        df_y = pd.DataFrame(data['properties']['parameter'])
        df_y.index = pd.to_datetime(df_y.index, format='%Y%m%d')
        frames.append(df_y)
    if not frames:
        return pd.DataFrame()
    ts = pd.concat(frames).sort_index().replace(-999.0, np.nan)
    return ts

print('Chargement des séries temporelles NASA...')
ts_store = {}
for _, row in locs.iterrows():
    lat = round(row['latitude'], 6)
    lon = round(row['longitude'], 6)
    ts_store[(lat, lon)] = load_ts(lat, lon)
print(f'{len(ts_store)} lieux chargés.')


# ── Feature extraction épidémiologique ────────────────────────────────────
def epi_lag_features(ts, target_date):
    """
    Features basées sur PMC12676583 :
    - Lags précis 14/21/28/42/56 jours (fenêtres hebdomadaires)
    - Flags pluie > 200mm à chaque lag (seuil du papier)
    - Max pluie hebdomadaire sur 8 semaines
    - Nombre de semaines > 200mm sur 8 semaines
    """
    feat = {}
    ts_past = ts[ts.index <= target_date]

    for lag_days, lag_name in [(14,'2w'), (21,'3w'), (28,'4w'), (42,'6w'), (56,'8w')]:
        # Fenêtre de 7 jours centrée sur J-lag
        end   = target_date - pd.Timedelta(days=lag_days - 3)
        start = target_date - pd.Timedelta(days=lag_days + 3)
        window = ts[(ts.index >= start) & (ts.index <= end)]
        rain_week = window['PRECTOTCORR'].sum() if len(window) > 0 else np.nan
        feat[f'rain_lag_{lag_name}'] = rain_week
        feat[f'rain_over200_lag_{lag_name}'] = int(rain_week > 200) if not np.isnan(rain_week) else 0

    # Rolling cumuls précis (14/28j) — pas disponibles dans climate_features.csv
    for days in [14, 21, 28, 42, 56]:
        w = ts_past.tail(days)
        feat[f'rain_sum_{days}d'] = w['PRECTOTCORR'].sum() if len(w) > 0 else np.nan
        feat[f'tmax_mean_{days}d'] = w['T2M_MAX'].mean() if len(w) > 0 else np.nan

    # Semaines sur 8 semaines (56j) — statistiques hebdomadaires
    weekly_rains = []
    for week in range(8):
        start = target_date - pd.Timedelta(days=(week+1)*7)
        end   = target_date - pd.Timedelta(days=week*7)
        w = ts[(ts.index > start) & (ts.index <= end)]
        weekly_rains.append(w['PRECTOTCORR'].sum() if len(w) > 0 else 0)

    feat['max_weekly_rain_8w']   = max(weekly_rains) if weekly_rains else np.nan
    feat['n_weeks_over200_8w']   = sum(r > 200 for r in weekly_rains)
    feat['n_weeks_over100_8w']   = sum(r > 100 for r in weekly_rains)
    feat['weekly_rain_std_8w']   = np.std(weekly_rains) if weekly_rains else np.nan
    feat['weeks_since_peak_rain'] = next(
        (i for i, r in enumerate(weekly_rains) if r == max(weekly_rains)), 7
    )

    return feat


def build_epi_features(df):
    rows = []
    for _, row in df.iterrows():
        lat  = round(row['latitude'],  6)
        lon  = round(row['longitude'], 6)
        key  = (lat, lon)
        ts   = ts_store.get(key, pd.DataFrame())
        feat = {'ID': row['ID']}
        if not ts.empty:
            feat.update(epi_lag_features(ts, row['deathdate']))
        rows.append(feat)
    return pd.DataFrame(rows)

print('Calcul des features épidémiologiques (train)...')
epi_train = build_epi_features(train)
print('Calcul des features épidémiologiques (test)...')
epi_test  = build_epi_features(test)
print(f'Features épi : {len([c for c in epi_train.columns if c != "ID"])} nouvelles colonnes')


# ── NaN check et imputation ────────────────────────────────────────────────
epi_cols = [c for c in epi_train.columns if c != 'ID']
nan_rates = epi_train[epi_cols].isnull().mean()
print('NaN rates (max 5):')
print(nan_rates.sort_values(ascending=False).head(5))

for col in epi_cols:
    med = epi_train[col].median()
    epi_train[col] = epi_train[col].fillna(med)
    epi_test[col]  = epi_test[col].fillna(med)


# ── Corrélation avec target ────────────────────────────────────────────────
corrs = {col: epi_train[col].corr(y) for col in epi_cols}
print('\nTop corrélations (abs) avec target :')
print(pd.Series(corrs).sort_values(key=abs, ascending=False).head(15).to_string())


# ── Target encoding OOF par location / zone / season ─────────────────────
# OOF pour éviter toute fuite — même split GroupKFold que le modèle
print('\nTarget encoding OOF...')
groups = train['location'].values
gkf    = GroupKFold(n_splits=5)

# Location target mean (OOF)
loc_te_train = np.zeros(len(train))
global_mean  = y.mean()
for tr_idx, va_idx in gkf.split(train, y, groups):
    loc_means = train.iloc[tr_idx].groupby('location')['location'].transform(
        lambda x: y.iloc[tr_idx][x.index].mean()
    )
    # Pour chaque validation, utiliser la moyenne du fold train
    fold_map = {}
    for loc in train.iloc[tr_idx]['location'].unique():
        mask = train.iloc[tr_idx]['location'] == loc
        fold_map[loc] = y.iloc[tr_idx][mask].mean()
    for i in va_idx:
        loc = train.iloc[i]['location']
        loc_te_train[i] = fold_map.get(loc, global_mean)

# Test : utiliser la moyenne sur tout le train
loc_map_full = train.groupby('location').apply(lambda g: y.iloc[g.index].mean())
loc_te_test  = test['location'].map(loc_map_full).fillna(global_mean).values

train['loc_target_mean'] = loc_te_train
test['loc_target_mean']  = loc_te_test

# Zone target mean (OOF)
zone_te_train = np.zeros(len(train))
for tr_idx, va_idx in gkf.split(train, y, groups):
    zone_map = {}
    for zone in train.iloc[tr_idx]['zone'].unique():
        mask = train.iloc[tr_idx]['zone'] == zone
        zone_map[zone] = y.iloc[tr_idx][mask].mean()
    for i in va_idx:
        zone_te_train[i] = zone_map.get(train.iloc[i]['zone'], global_mean)

zone_map_full = train.groupby('zone').apply(lambda g: y.iloc[g.index].mean())
zone_te_test  = test['zone'].map(zone_map_full).fillna(global_mean).values

train['zone_target_mean'] = zone_te_train
test['zone_target_mean']  = zone_te_test

# Season × location target mean
season_map = {1:4,2:4,3:1,4:1,5:1,6:2,7:2,8:3,9:3,10:3,11:3,12:4}
train['season'] = train['deathdate'].dt.month.map(season_map)
test['season']  = test['deathdate'].dt.month.map(season_map)

train['loc_season'] = train['location'].astype(str) + '_' + train['season'].astype(str)
test['loc_season']  = test['location'].astype(str) + '_' + test['season'].astype(str)

loc_season_te_train = np.zeros(len(train))
for tr_idx, va_idx in gkf.split(train, y, groups):
    ls_map = {}
    for ls in train.iloc[tr_idx]['loc_season'].unique():
        mask = train.iloc[tr_idx]['loc_season'] == ls
        ls_map[ls] = y.iloc[tr_idx][mask].mean()
    for i in va_idx:
        loc_season_te_train[i] = ls_map.get(train.iloc[i]['loc_season'], global_mean)

ls_map_full = train.groupby('loc_season').apply(lambda g: y.iloc[g.index].mean())
loc_season_te_test = test['loc_season'].map(ls_map_full).fillna(global_mean).values

train['loc_season_target_mean'] = loc_season_te_train
test['loc_season_target_mean']  = loc_season_te_test

print('Target encoding terminé.')
print(f'  loc_target_mean  corr={pd.Series(loc_te_train).corr(y):.3f}')
print(f'  zone_target_mean corr={pd.Series(zone_te_train).corr(y):.3f}')
print(f'  loc_season_te    corr={pd.Series(loc_season_te_train).corr(y):.3f}')


# ── Merge tout ────────────────────────────────────────────────────────────
X_train_v1 = pd.read_parquet(DATA / 'X_train.parquet')
X_test_v1  = pd.read_parquet(DATA / 'X_test.parquet')
X_train_v1['ID'] = train['ID'].values
X_test_v1['ID']  = test['ID'].values

X_train_v3 = (
    X_train_v1
    .merge(epi_train, on='ID', how='left')
    .drop(columns=['ID'])
)
X_test_v3 = (
    X_test_v1
    .merge(epi_test, on='ID', how='left')
    .drop(columns=['ID'])
)

# Ajouter target encoding
X_train_v3['loc_target_mean']       = train['loc_target_mean'].values
X_train_v3['zone_target_mean']      = train['zone_target_mean'].values
X_train_v3['loc_season_target_mean']= train['loc_season_target_mean'].values
X_test_v3['loc_target_mean']        = test['loc_target_mean'].values
X_test_v3['zone_target_mean']       = test['zone_target_mean'].values
X_test_v3['loc_season_target_mean'] = test['loc_season_target_mean'].values

new_cols = [c for c in X_train_v3.columns if c not in X_train_v1.drop(columns=['ID']).columns]
print(f'\nTotal features v3 : {X_train_v3.shape[1]} (+{len(new_cols)} nouvelles)')
print(f'NaN v3 train : {X_train_v3.isnull().sum().sum()}')

X_train_v3.to_parquet(DATA / 'X_train_v3.parquet')
X_test_v3.to_parquet(DATA / 'X_test_v3.parquet')
print('Sauvegardé : X_train_v3.parquet, X_test_v3.parquet')


# ── CV comparison v5 params sur v3 features ───────────────────────────────
CAT_COLS = ['zone', 'gender']
Xc_train = X_train_v3.copy()
Xc_test  = X_test_v3.copy()
for col in CAT_COLS:
    le = LabelEncoder()
    Xc_train[col] = le.fit_transform(Xc_train[col].astype(str))
    Xc_test[col]  = le.transform(Xc_test[col].astype(str))

PARAMS_V5 = dict(
    objective='binary', metric='binary_logloss', verbosity=-1,
    learning_rate=0.0351, num_leaves=117, min_child_samples=13,
    feature_fraction=0.771, bagging_fraction=0.995, bagging_freq=5,
    reg_alpha=0.00255, reg_lambda=0.0651,
    scale_pos_weight=(y==0).sum()/(y==1).sum(),
    n_estimators=2000, random_state=SEED,
)

oof = np.zeros(len(y))
print('\nCV v5_params + v3_features (GroupKFold 5)...')
for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xc_train, y, groups)):
    model = lgb.LGBMClassifier(**PARAMS_V5)
    model.fit(
        Xc_train.iloc[tr_idx], y.iloc[tr_idx],
        eval_X=Xc_train.iloc[va_idx], eval_y=y.iloc[va_idx],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=-1)],
    )
    oof[va_idx] = model.predict_proba(Xc_train.iloc[va_idx])[:, 1]
    print(f'  Fold {fold+1}: best_iter={model.best_iteration_}')

f1    = f1_score(y, (oof>=0.5).astype(int))
auc   = roc_auc_score(y, oof)
score = 0.6*f1 + 0.4*auc
print(f'\nCV v3 : F1={f1:.4f}  AUC={auc:.4f}  Score={score:.4f}')
print(f'Benchmark v5 : {BEST_CV:.4f}')
print(f'Delta : {score - BEST_CV:+.4f}')

if score > BEST_CV:
    print(f'\n✓ Amélioration ! Génération soumission v8...')
    model_full = lgb.LGBMClassifier(**{**PARAMS_V5, 'n_estimators':1000}, random_state=SEED)
    model_full.fit(Xc_train, y)
    proba = model_full.predict_proba(Xc_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    test_ids = pd.read_parquet(DATA / 'test_ids.parquet')['ID']
    sub = pd.DataFrame({'ID': test_ids, 'TargetF1': pred, 'TargetRAUC': proba})
    out = DATA / f'submission_v8_v3feat_{score:.4f}.csv'
    sub.to_csv(out, index=False)
    print(f'Soumission : {out}')
    print(f'TargetF1 : {sub["TargetF1"].value_counts().to_dict()}')

    # Feature importance
    imp = pd.Series(model_full.feature_importances_, index=Xc_train.columns)
    print('\nTop 20 features (gain) :')
    print(imp.sort_values(ascending=False).head(20).to_string())
else:
    print(f'\n✗ Pas d\'amélioration. Garder v5 (LB=0.8113).')
    imp = None

# Log
exp = pd.read_csv('notes/experiments.csv')
new = pd.DataFrame([{
    'date': str(pd.Timestamp.today().date()),
    'model': 'LGBM_v5params_v3features',
    'cv_strategy': 'GroupKFold(5)_by_location',
    'cv_score': round(score, 4),
    'cv_f1': round(f1, 4),
    'cv_auc': round(auc, 4),
    'lb_public': '',
    'notes': f'v3 features: epi lags 14/21/28/42/56j + weekly stats + target encoding OOF. Delta={score-BEST_CV:+.4f}',
}])
pd.concat([exp, new], ignore_index=True).to_csv('notes/experiments.csv', index=False)
print('\nExpérience loguée.')
