"""
Ensemble v9 : LGBM + XGBoost + CatBoost sur features v1 (59 features).
+ Pseudo-labeling sur prédictions haute confiance (>0.85).
Benchmark : v5 CV=0.8003, LB=0.8113.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42
DATA = Path('data')
BEST_CV = 0.8003

X_train = pd.read_parquet(DATA / 'X_train.parquet')
X_test  = pd.read_parquet(DATA / 'X_test.parquet')
y       = pd.read_parquet(DATA / 'y_train.parquet')['is_climate_sensitive']
train   = pd.read_csv(DATA / 'Train.csv')
groups  = train['location'].values

CAT_COLS = ['zone', 'gender']
gkf = GroupKFold(n_splits=5)

# ── Encode catégorielles ───────────────────────────────────────────────────
le_zone   = LabelEncoder().fit(X_train['zone'].astype(str))
le_gender = LabelEncoder().fit(X_train['gender'].astype(str))

def encode(X):
    Xc = X.copy()
    Xc['zone']   = le_zone.transform(Xc['zone'].astype(str))
    Xc['gender'] = le_gender.transform(Xc['gender'].astype(str))
    return Xc

Xc_train = encode(X_train)
Xc_test  = encode(X_test)

SPW = (y==0).sum() / (y==1).sum()

# ── Hyperparams (v5 régularisés — prouvés sur LB) ─────────────────────────
LGBM_PARAMS = dict(
    objective='binary', metric='binary_logloss', verbosity=-1,
    learning_rate=0.0351, num_leaves=117, min_child_samples=13,
    feature_fraction=0.771, bagging_fraction=0.995, bagging_freq=5,
    reg_alpha=0.00255, reg_lambda=0.0651,
    scale_pos_weight=SPW, n_estimators=2000, random_state=SEED,
)

XGB_PARAMS = dict(
    objective='binary:logistic', eval_metric='logloss',
    learning_rate=0.03, max_depth=6, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.01, reg_lambda=1.0,
    scale_pos_weight=SPW, n_estimators=2000,
    random_state=SEED, verbosity=0,
    early_stopping_rounds=200,
)

CB_PARAMS = dict(
    loss_function='Logloss', eval_metric='F1',
    learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
    subsample=0.8, colsample_bylevel=0.8,
    scale_pos_weight=SPW,
    iterations=2000, random_seed=SEED,
    verbose=False,
)

# ── CV Ensemble ────────────────────────────────────────────────────────────
oof_lgbm = np.zeros(len(y))
oof_xgb  = np.zeros(len(y))
oof_cb   = np.zeros(len(y))
test_lgbm = np.zeros(len(X_test))
test_xgb  = np.zeros(len(X_test))
test_cb   = np.zeros(len(X_test))

print('=== Ensemble CV (GroupKFold 5) ===')
for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xc_train, y, groups)):
    X_tr, X_va = Xc_train.iloc[tr_idx], Xc_train.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    print(f'\nFold {fold+1}:')

    # LGBM
    m_lgbm = lgb.LGBMClassifier(**LGBM_PARAMS)
    m_lgbm.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va,
               callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=-1)])
    oof_lgbm[va_idx] = m_lgbm.predict_proba(X_va)[:, 1]
    test_lgbm += m_lgbm.predict_proba(Xc_test)[:, 1] / 5
    print(f'  LGBM  iter={m_lgbm.best_iteration_}  fold_score={0.6*f1_score(y_va,(oof_lgbm[va_idx]>=0.5).astype(int))+0.4*roc_auc_score(y_va,oof_lgbm[va_idx]):.4f}')

    # XGBoost
    m_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[va_idx] = m_xgb.predict_proba(X_va)[:, 1]
    test_xgb += m_xgb.predict_proba(Xc_test)[:, 1] / 5
    print(f'  XGB   iter={m_xgb.best_iteration}  fold_score={0.6*f1_score(y_va,(oof_xgb[va_idx]>=0.5).astype(int))+0.4*roc_auc_score(y_va,oof_xgb[va_idx]):.4f}')

    # CatBoost
    m_cb = cb.CatBoostClassifier(**CB_PARAMS)
    m_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    oof_cb[va_idx] = m_cb.predict_proba(X_va)[:, 1]
    test_cb += m_cb.predict_proba(Xc_test)[:, 1] / 5
    print(f'  CB    iter={m_cb.best_iteration_}  fold_score={0.6*f1_score(y_va,(oof_cb[va_idx]>=0.5).astype(int))+0.4*roc_auc_score(y_va,oof_cb[va_idx]):.4f}')

# ── Scores individuels ─────────────────────────────────────────────────────
def score(oof):
    f1  = f1_score(y, (oof>=0.5).astype(int))
    auc = roc_auc_score(y, oof)
    return 0.6*f1 + 0.4*auc, f1, auc

s_lgbm, f1_lgbm, auc_lgbm = score(oof_lgbm)
s_xgb,  f1_xgb,  auc_xgb  = score(oof_xgb)
s_cb,   f1_cb,   auc_cb    = score(oof_cb)

print(f'\n=== Scores individuels ===')
print(f'LGBM : F1={f1_lgbm:.4f}  AUC={auc_lgbm:.4f}  Score={s_lgbm:.4f}')
print(f'XGB  : F1={f1_xgb:.4f}  AUC={auc_xgb:.4f}  Score={s_xgb:.4f}')
print(f'CB   : F1={f1_cb:.4f}  AUC={auc_cb:.4f}  Score={s_cb:.4f}')

# ── Recherche des poids optimaux ───────────────────────────────────────────
best_w, best_score = (1/3, 1/3, 1/3), -np.inf
for w1 in np.arange(0.2, 0.7, 0.1):
    for w2 in np.arange(0.1, 0.6, 0.1):
        w3 = 1 - w1 - w2
        if w3 < 0.05:
            continue
        oof_ens = w1*oof_lgbm + w2*oof_xgb + w3*oof_cb
        s, _, _ = score(oof_ens)
        if s > best_score:
            best_score = s
            best_w = (w1, w2, w3)

w1, w2, w3 = best_w
oof_ensemble = w1*oof_lgbm + w2*oof_xgb + w3*oof_cb
s_ens, f1_ens, auc_ens = score(oof_ensemble)
print(f'\n=== Ensemble (w_lgbm={w1:.1f}, w_xgb={w2:.1f}, w_cb={w3:.1f}) ===')
print(f'Score={s_ens:.4f}  F1={f1_ens:.4f}  AUC={auc_ens:.4f}')
print(f'Benchmark v5 : {BEST_CV:.4f}')
print(f'Delta : {s_ens - BEST_CV:+.4f}')

# ── Pseudo-labeling (si l'ensemble est meilleur que v5) ───────────────────
test_ensemble = w1*test_lgbm + w2*test_xgb + w3*test_cb

# Enregistrer OOF pour pseudo-labeling
np.save(DATA / 'oof_ensemble_v9.npy',  oof_ensemble)
np.save(DATA / 'test_ensemble_v9.npy', test_ensemble)

# Pseudo-labels haute confiance
conf_mask   = (test_ensemble >= 0.85) | (test_ensemble <= 0.15)
pseudo_pct  = conf_mask.mean() * 100
print(f'\nPseudo-labeling : {conf_mask.sum()} exemples test à haute confiance ({pseudo_pct:.1f}%)')

if conf_mask.sum() > 50 and s_ens > BEST_CV:
    print('Ajout des pseudo-labels et re-CV...')
    pseudo_X = Xc_test[conf_mask].copy()
    pseudo_y = pd.Series((test_ensemble[conf_mask] >= 0.5).astype(int))

    X_pl = pd.concat([Xc_train, pseudo_X], ignore_index=True)
    y_pl = pd.concat([y.reset_index(drop=True), pseudo_y], ignore_index=True)

    oof_pl = np.zeros(len(y))
    test_pl_lgbm = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xc_train, y, groups)):
        # Train sur train + pseudo, valider sur train uniquement
        tr_pl_idx = list(tr_idx) + [len(Xc_train) + i for i in range(len(pseudo_X))]
        X_tr_pl = X_pl.iloc[tr_pl_idx]
        y_tr_pl = y_pl.iloc[tr_pl_idx]
        X_va    = Xc_train.iloc[va_idx]
        y_va    = y.iloc[va_idx]

        m = lgb.LGBMClassifier(**LGBM_PARAMS)
        m.fit(X_tr_pl, y_tr_pl, eval_X=X_va, eval_y=y_va,
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=-1)])
        oof_pl[va_idx] = m.predict_proba(X_va)[:, 1]
        test_pl_lgbm  += m.predict_proba(Xc_test)[:, 1] / 5

    s_pl, f1_pl, auc_pl = score(oof_pl)
    print(f'CV avec pseudo-labels (LGBM) : F1={f1_pl:.4f}  AUC={auc_pl:.4f}  Score={s_pl:.4f}')

    # Choisir la meilleure prédiction test
    if s_pl > s_ens:
        final_proba = test_pl_lgbm
        final_score = s_pl
        print(f'→ Pseudo-labeling améliore (+{s_pl-s_ens:.4f})')
    else:
        final_proba = test_ensemble
        final_score = s_ens
        print(f'→ Pseudo-labeling n\'améliore pas, on garde l\'ensemble')
else:
    final_proba = test_ensemble
    final_score = s_ens

# ── Génération soumission ─────────────────────────────────────────────────
if final_score > BEST_CV:
    pred = (final_proba >= 0.5).astype(int)
    test_ids = pd.read_parquet(DATA / 'test_ids.parquet')['ID']
    sub = pd.DataFrame({'ID': test_ids, 'TargetF1': pred, 'TargetRAUC': final_proba})
    out = DATA / f'submission_v9_ensemble_{final_score:.4f}.csv'
    sub.to_csv(out, index=False)
    print(f'\nSoumission : {out}')
    print(f'TargetF1 : {sub["TargetF1"].value_counts().to_dict()}')
else:
    print(f'\n✗ Pas d\'amélioration vs v5 ({final_score:.4f} <= {BEST_CV:.4f}). Garder v5.')

# Log
exp = pd.read_csv('notes/experiments.csv')
new = pd.DataFrame([{
    'date': str(pd.Timestamp.today().date()),
    'model': f'Ensemble_LGBM+XGB+CB_w({w1:.1f},{w2:.1f},{w3:.1f})',
    'cv_strategy': 'GroupKFold(5)_by_location',
    'cv_score': round(final_score, 4),
    'cv_f1': round(f1_ens, 4),
    'cv_auc': round(auc_ens, 4),
    'lb_public': '',
    'notes': f'Ensemble 3 modèles v1_features. LGBM={s_lgbm:.4f} XGB={s_xgb:.4f} CB={s_cb:.4f} Ens={s_ens:.4f}',
}])
pd.concat([exp, new], ignore_index=True).to_csv('notes/experiments.csv', index=False)
print('Expérience loguée.')
