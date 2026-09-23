"""
Stacking v15 : 4 modèles niveau 1 + LogReg meta-learner.

Niveau 1 :
  - CatBoost v10 (SPW=0.537, AUC-optimisé)   → oof_cb_v10
  - CatBoost bestSPW (SPW optimal)            → oof_cb_bestspw
  - LGBM v5 params                            → re-généré ici
  - CatBoost old (oof_cat)                    → oof_cat

Niveau 2 :
  - LogisticRegression C=0.01 avec features clés pour conditioning
  - GroupKFold(5) pour éviter la fuite

Insight :
  - CB v10 OOF AUC = 0.8166 (bon ranking)
  - CB bestSPW OOF F1 = 0.82+ (bon seuil 0.5)
  - Les deux ensemble capturent différents aspects de la distribution
"""

import numpy as np
import pandas as pd
import catboost as cb
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42
DATA = Path('data')
BEST_CV = 0.8027

y     = pd.read_parquet(DATA / 'y_train.parquet')['is_climate_sensitive']
train = pd.read_csv(DATA / 'Train.csv')
groups = train['location'].values
X_train = pd.read_parquet(DATA / 'X_train.parquet')
X_test  = pd.read_parquet(DATA / 'X_test.parquet')

le_zone   = LabelEncoder().fit(X_train['zone'].astype(str))
le_gender = LabelEncoder().fit(X_train['gender'].astype(str))
for Xv in [X_train, X_test]:
    Xv['zone']   = le_zone.transform(Xv['zone'].astype(str))
    Xv['gender'] = le_gender.transform(Xv['gender'].astype(str))

SPW   = (y==0).sum() / (y==1).sum()
gkf   = GroupKFold(n_splits=5)
folds = list(gkf.split(X_train, y, groups))


def score(oof):
    f1  = f1_score(y, (oof >= 0.5).astype(int))
    auc = roc_auc_score(y, oof)
    return 0.6*f1 + 0.4*auc, f1, auc


# ── Niveau 1 : CB v10 (déjà calculé) ──────────────────────────────────────
oof_cb10   = np.load(DATA / 'oof_cb_v10.npy')
test_cb10  = np.load(DATA / 'test_cb_v10.npy')
s10, f10, a10 = score(oof_cb10)
print(f'CB v10     : Score={s10:.4f}  F1={f10:.4f}  AUC={a10:.4f}')

# ── Niveau 1 : CB bestSPW (déjà calculé ou à recalculer) ──────────────────
if (DATA / 'oof_cb_bestspw.npy').exists():
    oof_bestspw  = np.load(DATA / 'oof_cb_bestspw.npy')
    test_bestspw = np.load(DATA / 'test_cb_bestspw.npy')
    sb, fb, ab = score(oof_bestspw)
    print(f'CB bestSPW : Score={sb:.4f}  F1={fb:.4f}  AUC={ab:.4f}')
else:
    print('oof_cb_bestspw.npy non trouvé — recalcul SPW=1.2...')
    SPW_BEST = 1.2
    CB_PARAMS = dict(
        loss_function='Logloss', eval_metric='F1',
        learning_rate=0.02, depth=7, l2_leaf_reg=1.0,
        subsample=0.8, colsample_bylevel=0.8, scale_pos_weight=SPW_BEST,
        iterations=2000, random_seed=SEED, verbose=False,
    )
    oof_bestspw = np.zeros(len(y))
    test_bestspw = np.zeros(len(X_test))
    for tr_idx, va_idx in folds:
        m = cb.CatBoostClassifier(**CB_PARAMS)
        m.fit(X_train.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=(X_train.iloc[va_idx], y.iloc[va_idx]),
              use_best_model=True)
        oof_bestspw[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
        test_bestspw += m.predict_proba(X_test)[:, 1] / 5
    np.save(DATA / 'oof_cb_bestspw.npy', oof_bestspw)
    np.save(DATA / 'test_cb_bestspw.npy', test_bestspw)
    sb, fb, ab = score(oof_bestspw)
    print(f'CB bestSPW : Score={sb:.4f}  F1={fb:.4f}  AUC={ab:.4f}')

# ── Niveau 1 : LGBM v5 params (re-généré) ─────────────────────────────────
LGBM_PARAMS = dict(
    objective='binary', metric='binary_logloss', verbosity=-1,
    learning_rate=0.0351, num_leaves=117, min_child_samples=13,
    feature_fraction=0.771, bagging_fraction=0.995, bagging_freq=5,
    reg_alpha=0.00255, reg_lambda=0.0651,
    scale_pos_weight=SPW, n_estimators=2000, random_state=SEED,
)

oof_lgbm5 = np.zeros(len(y))
test_lgbm5 = np.zeros(len(X_test))
for tr_idx, va_idx in folds:
    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(X_train.iloc[tr_idx], y.iloc[tr_idx],
          eval_set=[(X_train.iloc[va_idx], y.iloc[va_idx])],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=-1)])
    oof_lgbm5[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
    test_lgbm5 += m.predict_proba(X_test)[:, 1] / 5

sl, fl, al = score(oof_lgbm5)
print(f'LGBM v5    : Score={sl:.4f}  F1={fl:.4f}  AUC={al:.4f}')

# ── Niveau 2 : stacking LogReg ─────────────────────────────────────────────
meta_features_cols = ['age_log', 'year', 'is_infant', 'is_child_u5', 'is_neonate',
                      'is_elderly', 'is_adult', 'rain_sum_30d', 'tavg_30d',
                      'ndvi_30d', 'u5_x_rain30', 'elderly_x_temp_anom', 'age_log_x_rain30']

print('\n=== Stacking L2 ===')
best_s_stack, best_meta_oof, best_meta_test = -np.inf, None, None
for C in [0.005, 0.01, 0.05, 0.1]:
    oof_meta = np.zeros(len(y))
    test_meta = np.zeros(len(X_test))
    for tr_idx, va_idx in folds:
        X_meta_tr = np.column_stack([
            oof_cb10[tr_idx], oof_bestspw[tr_idx], oof_lgbm5[tr_idx],
            X_train.iloc[tr_idx][meta_features_cols].values
        ])
        X_meta_va = np.column_stack([
            oof_cb10[va_idx], oof_bestspw[va_idx], oof_lgbm5[va_idx],
            X_train.iloc[va_idx][meta_features_cols].values
        ])
        X_meta_te = np.column_stack([
            test_cb10, test_bestspw, test_lgbm5,
            X_test[meta_features_cols].values
        ])
        sc = StandardScaler()
        X_meta_tr = sc.fit_transform(X_meta_tr)
        X_meta_va = sc.transform(X_meta_va)
        X_meta_te_s = sc.transform(X_meta_te)

        m = LogisticRegression(C=C, max_iter=1000, random_state=SEED)
        m.fit(X_meta_tr, y.iloc[tr_idx])
        oof_meta[va_idx] = m.predict_proba(X_meta_va)[:, 1]
        test_meta += m.predict_proba(X_meta_te_s)[:, 1] / 5

    ss, fs, as_ = score(oof_meta)
    print(f'  C={C:.3f}: F1={fs:.4f}  AUC={as_:.4f}  Score={ss:.4f}')
    if ss > best_s_stack:
        best_s_stack = ss
        best_meta_oof = oof_meta.copy()
        best_meta_test = test_meta.copy()

print(f'\nMeilleur stacking : {best_s_stack:.4f}')
print(f'Benchmark v10     : {BEST_CV:.4f}')
print(f'Delta             : {best_s_stack - BEST_CV:+.4f}')

# Génération soumission si meilleur
if best_s_stack > BEST_CV:
    test_ids = pd.read_parquet(DATA / 'test_ids.parquet')['ID']
    pred = (best_meta_test >= 0.5).astype(int)
    sub = pd.DataFrame({'ID': test_ids, 'TargetF1': pred, 'TargetRAUC': best_meta_test})

    sample = pd.read_csv(DATA / 'SampleSubmission.csv')
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample)

    f_final  = f1_score(y, best_meta_oof >= 0.5)
    a_final  = roc_auc_score(y, best_meta_oof)
    out = DATA / f'submission_v15_stacking_{best_s_stack:.4f}.csv'
    sub.to_csv(out, index=False)
    print(f'\n✓ Soumission : {out}')
    print(f'TargetF1 : {sub["TargetF1"].value_counts().to_dict()}')

    np.save(DATA / 'oof_stacking_v15.npy', best_meta_oof)
    np.save(DATA / 'test_stacking_v15.npy', best_meta_test)

# Log
exp = pd.read_csv('notes/experiments.csv')
new = pd.DataFrame([{
    'date': str(pd.Timestamp.today().date()),
    'model': 'Stacking_L2_CBv10+CBbestSPW+LGBM',
    'cv_strategy': 'GroupKFold(5)_by_location',
    'cv_score': round(best_s_stack, 4),
    'cv_f1': round(f1_score(y, best_meta_oof >= 0.5), 4),
    'cv_auc': round(roc_auc_score(y, best_meta_oof), 4),
    'lb_public': '',
    'notes': f'LogReg stacking on CB_v10+CB_bestSPW+LGBM OOF. Delta={best_s_stack-BEST_CV:+.4f}',
}])
pd.concat([exp, new], ignore_index=True).to_csv('notes/experiments.csv', index=False)
print('Expérience loguée.')
