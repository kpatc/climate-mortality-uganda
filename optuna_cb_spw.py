"""
Optuna CatBoost avec scale_pos_weight comme hyperparamètre.

Insight : SPW=1.5 donne CV=0.8217 vs 0.8027 avec SPW=0.537 standard.
Le SPW optimal peut changer selon depth/lr — on laisse Optuna trouver.

Benchmark : CV=0.8217 (CB SPW=1.5, depth=7, l2=1.0, lr=0.02).
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
BEST_CV = 0.8217

X_train = pd.read_parquet(DATA / 'X_train.parquet')
X_test  = pd.read_parquet(DATA / 'X_test.parquet')
y       = pd.read_parquet(DATA / 'y_train.parquet')['is_climate_sensitive']
train   = pd.read_csv(DATA / 'Train.csv')
groups  = train['location'].values

le_zone   = LabelEncoder().fit(X_train['zone'].astype(str))
le_gender = LabelEncoder().fit(X_train['gender'].astype(str))
for Xv in [X_train, X_test]:
    Xv['zone']   = le_zone.transform(Xv['zone'].astype(str))
    Xv['gender'] = le_gender.transform(Xv['gender'].astype(str))

gkf   = GroupKFold(n_splits=5)
folds = list(gkf.split(X_train, y, groups))


def cv_cb(params):
    oof = np.zeros(len(y))
    for tr_idx, va_idx in folds:
        m = cb.CatBoostClassifier(**params, random_seed=SEED, verbose=False)
        m.fit(X_train.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=(X_train.iloc[va_idx], y.iloc[va_idx]),
              use_best_model=True)
        oof[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
    f1  = f1_score(y, (oof >= 0.5).astype(int))
    auc = roc_auc_score(y, oof)
    return 0.6*f1 + 0.4*auc, oof


def objective(trial):
    params = dict(
        loss_function='Logloss',
        eval_metric='F1',
        iterations=2000,
        scale_pos_weight=trial.suggest_float('scale_pos_weight', 1.0, 2.5),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        depth=trial.suggest_int('depth', 5, 8),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.5, 5.0),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bylevel=trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 5, 30),
        bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.0),
    )
    s, _ = cv_cb(params)
    return s


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED)
)

# Warm start: SPW=1.5 params that worked
study.enqueue_trial({
    'scale_pos_weight': 1.5,
    'learning_rate': 0.02,
    'depth': 7,
    'l2_leaf_reg': 1.0,
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    'min_data_in_leaf': 10,
    'bagging_temperature': 0.5,
})

print(f'Optuna CatBoost+SPW — benchmark : {BEST_CV}')
print('50 trials...\n')
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_params.update(dict(
    loss_function='Logloss', eval_metric='F1', iterations=2000,
))

print(f'\nMeilleur trial : {study.best_value:.4f}')
print(f'Benchmark      : {BEST_CV:.4f}')
print(f'Delta          : {study.best_value - BEST_CV:+.4f}')
print(f'Params         : {study.best_params}')

final_score, oof_final = cv_cb(best_params)
f1_final  = f1_score(y, (oof_final >= 0.5).astype(int))
auc_final = roc_auc_score(y, oof_final)
print(f'\nCV final : F1={f1_final:.4f}  AUC={auc_final:.4f}  Score={final_score:.4f}')

if final_score > BEST_CV:
    print(f'\n✓ Amélioration (+{final_score - BEST_CV:.4f})')

    test_preds = np.zeros(len(X_test))
    for tr_idx, va_idx in folds:
        m = cb.CatBoostClassifier(**best_params, random_seed=SEED, verbose=False)
        m.fit(X_train.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=(X_train.iloc[va_idx], y.iloc[va_idx]),
              use_best_model=True)
        test_preds += m.predict_proba(X_test)[:, 1] / 5

    test_ids = pd.read_parquet(DATA / 'test_ids.parquet')['ID']
    pred = (test_preds >= 0.5).astype(int)
    sub = pd.DataFrame({'ID': test_ids, 'TargetF1': pred, 'TargetRAUC': test_preds})

    sample = pd.read_csv(DATA / 'SampleSubmission.csv')
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample)

    out = DATA / f'submission_v17_optuna_spw_{final_score:.4f}.csv'
    sub.to_csv(out, index=False)
    print(f'Soumission : {out}')
    print(f'TargetF1 : {sub["TargetF1"].value_counts().to_dict()}')

    np.save(DATA / 'oof_optuna_spw.npy', oof_final)
    np.save(DATA / 'test_optuna_spw.npy', test_preds)
else:
    print(f'\n✗ Pas d\'amélioration ({final_score:.4f} <= {BEST_CV:.4f}).')

exp = pd.read_csv('notes/experiments.csv')
new = pd.DataFrame([{
    'date': str(pd.Timestamp.today().date()),
    'model': 'CatBoost_Optuna50_SPW',
    'cv_strategy': 'GroupKFold(5)_by_location',
    'cv_score': round(final_score, 4),
    'cv_f1': round(f1_final, 4),
    'cv_auc': round(auc_final, 4),
    'lb_public': '',
    'notes': f'Optuna 50 trials CB avec SPW libre. Best: {study.best_params}. Delta={final_score-BEST_CV:+.4f}',
}])
pd.concat([exp, new], ignore_index=True).to_csv('notes/experiments.csv', index=False)
print('Expérience loguée.')
