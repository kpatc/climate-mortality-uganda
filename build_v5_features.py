"""
Features v5 — year×age interactions + quadratic year term.

Key insight: U5 CS rate dropped from 97% (2014) to 49% (2019), a massive temporal×age
signal not captured by year and age_log alone as separate features. Tree splits on year
then age may miss this interaction with limited data. Explicit interactions help.

New features:
  year_centered         = year - 2014
  year_sq               = year_centered²
  year_x_u5             = year_centered × is_child_u5
  year_x_infant         = year_centered × is_infant
  year_x_neonate        = year_centered × is_neonate
  year_x_elderly        = year_centered × is_elderly
  year_x_adult          = year_centered × is_adult
  year_x_agelog         = year_centered × age_log
  year_x_rain30         = year_centered × rain_sum_30d (normalized)
  year_x_tavg30         = year_centered × tavg_30d (normalized)
  rain30_x_u5_yrtrend   = rain_sum_30d × is_child_u5 × year_centered

Benchmark: CV=0.8027 (CatBoost v10, v1 features).
"""

import numpy as np
import pandas as pd
import catboost as cb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42
DATA = Path('data')
BEST_CV = 0.8027

X_train = pd.read_parquet(DATA / 'X_train.parquet')
X_test  = pd.read_parquet(DATA / 'X_test.parquet')
y       = pd.read_parquet(DATA / 'y_train.parquet')['is_climate_sensitive']
train   = pd.read_csv(DATA / 'Train.csv')
groups  = train['location'].values

YEAR_REF = 2014


def add_v5_features(X):
    X = X.copy()
    yc = X['year'] - YEAR_REF
    X['year_centered'] = yc
    X['year_sq']       = yc ** 2
    X['year_x_u5']     = yc * X['is_child_u5']
    X['year_x_infant'] = yc * X['is_infant']
    X['year_x_neonate']= yc * X['is_neonate']
    X['year_x_elderly']= yc * X['is_elderly']
    X['year_x_adult']  = yc * X['is_adult']
    X['year_x_agelog'] = yc * X['age_log']

    rain_norm = (X['rain_sum_30d'] - X['rain_sum_30d'].mean()) / (X['rain_sum_30d'].std() + 1e-9)
    tavg_norm = (X['tavg_30d']    - X['tavg_30d'].mean())     / (X['tavg_30d'].std()     + 1e-9)
    X['year_x_rain30']      = yc * rain_norm
    X['year_x_tavg30']      = yc * tavg_norm
    X['rain30_x_u5_yrtrend']= rain_norm * X['is_child_u5'] * yc
    return X


X_train_v5 = add_v5_features(X_train)
X_test_v5  = add_v5_features(X_test)

# Fix normalisation statistics on train only for test
rain_mean = X_train['rain_sum_30d'].mean()
rain_std  = X_train['rain_sum_30d'].std()
tavg_mean = X_train['tavg_30d'].mean()
tavg_std  = X_train['tavg_30d'].std()
for Xv in [X_train_v5, X_test_v5]:
    yc = Xv['year'] - YEAR_REF
    rain_n = (Xv['rain_sum_30d'] - rain_mean) / (rain_std + 1e-9)
    tavg_n = (Xv['tavg_30d']     - tavg_mean) / (tavg_std  + 1e-9)
    Xv['year_x_rain30']       = yc * rain_n
    Xv['year_x_tavg30']       = yc * tavg_n
    Xv['rain30_x_u5_yrtrend'] = rain_n * Xv['is_child_u5'] * yc

print(f'Features v5 : {X_train_v5.shape[1]} (+{X_train_v5.shape[1]-X_train.shape[1]} nouvelles)')
new_cols = [c for c in X_train_v5.columns if c not in X_train.columns]
print(f'Nouvelles : {new_cols}')

# Correlations with target
print('\nCorr nouvelles features vs target :')
for col in new_cols:
    print(f'  {col}: {X_train_v5[col].corr(y):.4f}')

# Encode categoricals
le_zone   = LabelEncoder().fit(X_train_v5['zone'].astype(str))
le_gender = LabelEncoder().fit(X_train_v5['gender'].astype(str))
for Xv in [X_train_v5, X_test_v5]:
    Xv['zone']   = le_zone.transform(Xv['zone'].astype(str))
    Xv['gender'] = le_gender.transform(Xv['gender'].astype(str))

SPW  = (y==0).sum() / (y==1).sum()
gkf  = GroupKFold(n_splits=5)
folds = list(gkf.split(X_train_v5, y, groups))

# CatBoost v10 params (best found in grid search)
PARAMS = dict(
    loss_function='Logloss', eval_metric='F1',
    learning_rate=0.02, depth=7, l2_leaf_reg=1.0,
    subsample=0.8, colsample_bylevel=0.8,
    scale_pos_weight=SPW, iterations=2000,
    random_seed=SEED, verbose=False,
)

print('\nCV CatBoost v10 params + v5 features...')
oof = np.zeros(len(y))
test_preds = np.zeros(len(X_test_v5))
for tr_idx, va_idx in folds:
    m = cb.CatBoostClassifier(**PARAMS)
    m.fit(X_train_v5.iloc[tr_idx], y.iloc[tr_idx],
          eval_set=(X_train_v5.iloc[va_idx], y.iloc[va_idx]),
          use_best_model=True)
    oof[va_idx] = m.predict_proba(X_train_v5.iloc[va_idx])[:, 1]
    test_preds += m.predict_proba(X_test_v5)[:, 1] / 5

f1    = f1_score(y, (oof >= 0.5).astype(int))
auc   = roc_auc_score(y, oof)
score = 0.6 * f1 + 0.4 * auc
print(f'CV v5 (CB params) : F1={f1:.4f}  AUC={auc:.4f}  Score={score:.4f}')
print(f'Benchmark v10     : {BEST_CV:.4f}')
print(f'Delta             : {score - BEST_CV:+.4f}')

if score > BEST_CV:
    print(f'\n✓ Amélioration (+{score - BEST_CV:.4f}) — génération soumission...')

    X_train_v5.to_parquet(DATA / 'X_train_v5.parquet')
    X_test_v5.to_parquet(DATA / 'X_test_v5.parquet')

    test_ids = pd.read_parquet(DATA / 'test_ids.parquet')['ID']
    pred = (test_preds >= 0.5).astype(int)
    sub = pd.DataFrame({'ID': test_ids, 'TargetF1': pred, 'TargetRAUC': test_preds})

    sample = pd.read_csv(DATA / 'SampleSubmission.csv')
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample)

    out = DATA / f'submission_v13_cb_v5feat_{score:.4f}.csv'
    sub.to_csv(out, index=False)
    print(f'Soumission : {out}')
    print(f'TargetF1 : {sub["TargetF1"].value_counts().to_dict()}')

    np.save(DATA / 'oof_cb_v5feat.npy', oof)
    np.save(DATA / 'test_cb_v5feat.npy', test_preds)

else:
    print(f'\n✗ Pas d\'amélioration vs v10 ({score:.4f} <= {BEST_CV:.4f}).')
    print('Sauvegarde parquet quand même pour Optuna ultérieur...')
    X_train_v5.to_parquet(DATA / 'X_train_v5.parquet')
    X_test_v5.to_parquet(DATA / 'X_test_v5.parquet')

# Feature importance
m_full = cb.CatBoostClassifier(**{**PARAMS, 'iterations': 1000}, random_seed=SEED, verbose=False)
m_full.fit(X_train_v5, y)
imp = pd.Series(m_full.get_feature_importance(), index=X_train_v5.columns)
print('\nTop 25 features :')
print(imp.sort_values(ascending=False).head(25).to_string())

# Log
exp = pd.read_csv('notes/experiments.csv')
new = pd.DataFrame([{
    'date': str(pd.Timestamp.today().date()),
    'model': 'CatBoost_v10params_v5features',
    'cv_strategy': 'GroupKFold(5)_by_location',
    'cv_score': round(score, 4),
    'cv_f1': round(f1, 4),
    'cv_auc': round(auc, 4),
    'lb_public': '',
    'notes': f'CB v10 params + year×age interactions. Delta={score - BEST_CV:+.4f}. Features={X_train_v5.shape[1]}',
}])
pd.concat([exp, new], ignore_index=True).to_csv('notes/experiments.csv', index=False)
print('Expérience loguée.')
