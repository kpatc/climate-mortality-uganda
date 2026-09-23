"""
Optuna sur CatBoost — le modèle le plus fort (0.8007 sans tuning vs LGBM 0.7951).
Benchmark : v5 CV=0.8003, LB=0.8113.
CatBoost gère les catégorielles nativement — zone et gender passent sans encoding.
"""

import numpy as np
import pandas as pd
import optuna
import catboost as cb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
DATA = Path('data')
BEST_CV = 0.8003

X_train = pd.read_parquet(DATA / 'X_train.parquet')
X_test  = pd.read_parquet(DATA / 'X_test.parquet')
y       = pd.read_parquet(DATA / 'y_train.parquet')['is_climate_sensitive']
train   = pd.read_csv(DATA / 'Train.csv')
groups  = train['location'].values

# CatBoost gère les catégorielles nativement — on encode quand même pour éviter les bugs
le_zone   = LabelEncoder().fit(X_train['zone'].astype(str))
le_gender = LabelEncoder().fit(X_train['gender'].astype(str))
X_train['zone']   = le_zone.transform(X_train['zone'].astype(str))
X_train['gender'] = le_gender.transform(X_train['gender'].astype(str))
X_test['zone']    = le_zone.transform(X_test['zone'].astype(str))
X_test['gender']  = le_gender.transform(X_test['gender'].astype(str))

SPW  = (y==0).sum() / (y==1).sum()
gkf  = GroupKFold(n_splits=5)
folds = list(gkf.split(X_train, y, groups))


def cv_catboost(params):
    oof = np.zeros(len(y))
    for tr_idx, va_idx in folds:
        m = cb.CatBoostClassifier(**params, random_seed=SEED, verbose=False)
        m.fit(
            X_train.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=(X_train.iloc[va_idx], y.iloc[va_idx]),
            use_best_model=True,
        )
        oof[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
    f1  = f1_score(y, (oof>=0.5).astype(int))
    auc = roc_auc_score(y, oof)
    return 0.6*f1 + 0.4*auc, oof


def objective(trial):
    params = dict(
        loss_function='Logloss',
        eval_metric='F1',
        iterations=2000,
        scale_pos_weight=SPW,
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        depth=trial.suggest_int('depth', 4, 8),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bylevel=trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 5, 50),
        bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.0),
    )
    s, _ = cv_catboost(params)
    return s


study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
print(f'Optuna CatBoost — benchmark : {BEST_CV}')
print('50 trials...\n')
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_params.update(dict(
    loss_function='Logloss', eval_metric='F1',
    iterations=2000, scale_pos_weight=SPW,
))

print(f'\nMeilleur trial : {study.best_value:.4f}')
print(f'Benchmark v5   : {BEST_CV:.4f}')
print(f'Delta          : {study.best_value - BEST_CV:+.4f}')
print(f'Params         : {study.best_params}')

final_score, oof_final = cv_catboost(best_params)
f1_final  = f1_score(y, (oof_final>=0.5).astype(int))
auc_final = roc_auc_score(y, oof_final)
print(f'\nCV final : F1={f1_final:.4f}  AUC={auc_final:.4f}  Score={final_score:.4f}')

if final_score > BEST_CV:
    print(f'\n✓ Amélioration (+{final_score - BEST_CV:.4f}) — génération soumission...')
    m_full = cb.CatBoostClassifier(**best_params, random_seed=SEED, verbose=False)
    m_full.fit(X_train, y)
    proba = m_full.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    test_ids = pd.read_parquet(DATA / 'test_ids.parquet')['ID']
    sub = pd.DataFrame({'ID': test_ids, 'TargetF1': pred, 'TargetRAUC': proba})
    out = DATA / f'submission_v10_catboost_{final_score:.4f}.csv'
    sub.to_csv(out, index=False)
    print(f'Soumission : {out}')
    print(f'TargetF1 : {sub["TargetF1"].value_counts().to_dict()}')

    # Sauvegarde OOF pour stacking ultérieur
    np.save(DATA / 'oof_catboost_v10.npy', oof_final)
    np.save(DATA / 'test_catboost_v10.npy', proba)
else:
    print(f'\n✗ Pas d\'amélioration vs v5. Garder v5/v9.')

exp = pd.read_csv('notes/experiments.csv')
new = pd.DataFrame([{
    'date': str(pd.Timestamp.today().date()),
    'model': 'CatBoost_Optuna50',
    'cv_strategy': 'GroupKFold(5)_by_location',
    'cv_score': round(final_score, 4),
    'cv_f1': round(f1_final, 4),
    'cv_auc': round(auc_final, 4),
    'lb_public': '',
    'notes': f'Optuna 50 trials CatBoost. Best params: {study.best_params}. Delta={final_score-BEST_CV:+.4f}',
}])
pd.concat([exp, new], ignore_index=True).to_csv('notes/experiments.csv', index=False)
print('Expérience loguée.')
