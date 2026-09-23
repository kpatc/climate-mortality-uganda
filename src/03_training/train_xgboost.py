"""
Optuna XGBoost — AUC=0.8169 sans tuning, meilleur AUC de l'ensemble.
Benchmark : CV=0.8003, LB=0.8210.
Espace de recherche conservateur pour éviter l'overfitting géographique.
"""

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
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

for col in ['zone', 'gender']:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col]  = le.transform(X_test[col].astype(str))

SPW   = (y==0).sum() / (y==1).sum()
gkf   = GroupKFold(n_splits=5)
folds = list(gkf.split(X_train, y, groups))


def cv_xgb(params):
    oof = np.zeros(len(y))
    for tr_idx, va_idx in folds:
        m = xgb.XGBClassifier(**params, random_state=SEED, verbosity=0,
                               early_stopping_rounds=200)
        m.fit(X_train.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X_train.iloc[va_idx], y.iloc[va_idx])],
              verbose=False)
        oof[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
    f1  = f1_score(y, (oof >= 0.5).astype(int))
    auc = roc_auc_score(y, oof)
    return 0.6 * f1 + 0.4 * auc, oof


def objective(trial):
    params = dict(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=2000,
        scale_pos_weight=SPW,
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 7),
        min_child_weight=trial.suggest_int('min_child_weight', 3, 30),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        colsample_bylevel=trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        gamma=trial.suggest_float('gamma', 0.0, 2.0),
    )
    s, _ = cv_xgb(params)
    return s


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED)
)
print(f'Optuna XGBoost — benchmark à battre : {BEST_CV}')
print('60 trials...\n')
study.optimize(objective, n_trials=60, show_progress_bar=True)

best_params = study.best_params
best_params.update(dict(
    objective='binary:logistic', eval_metric='logloss',
    n_estimators=2000, scale_pos_weight=SPW,
))

print(f'\nMeilleur trial Optuna : {study.best_value:.4f}')
print(f'Benchmark v5          : {BEST_CV:.4f}')
print(f'Delta                 : {study.best_value - BEST_CV:+.4f}')
print(f'Params                : {study.best_params}')

final_score, oof_final = cv_xgb({**best_params, 'early_stopping_rounds': 200})
f1_final  = f1_score(y, (oof_final >= 0.5).astype(int))
auc_final = roc_auc_score(y, oof_final)
print(f'\nCV final : F1={f1_final:.4f}  AUC={auc_final:.4f}  Score={final_score:.4f}')

if final_score > BEST_CV:
    print(f'\n✓ Amélioration (+{final_score - BEST_CV:.4f}) — génération soumission...')

    # Full retrain
    m_full = xgb.XGBClassifier(
        **{k: v for k, v in best_params.items() if k != 'early_stopping_rounds'},
        n_estimators=1000, random_state=SEED, verbosity=0
    )
    m_full.fit(X_train, y)
    proba = m_full.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    test_ids = pd.read_parquet(DATA / 'test_ids.parquet')['ID']
    sub = pd.DataFrame({'ID': test_ids, 'TargetF1': pred, 'TargetRAUC': proba})

    sample = pd.read_csv(DATA / 'SampleSubmission.csv')
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample)

    out = DATA / f'submission_v12_xgb_{final_score:.4f}.csv'
    sub.to_csv(out, index=False)
    print(f'Soumission : {out}')
    print(f'TargetF1 : {sub["TargetF1"].value_counts().to_dict()}')

    np.save(DATA / 'oof_xgb_v12.npy', oof_final)
    np.save(DATA / 'test_xgb_v12.npy', proba)

    # Ensemble XGB + LGBM v5
    oof_lgbm = np.load(DATA / 'oof_lgbm.npy')
    test_lgbm = np.load(DATA / 'test_lgbm.npy')

    for w_xgb in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w_lgbm = 1 - w_xgb
        oof_ens = w_xgb * oof_final + w_lgbm * oof_lgbm
        f1e  = f1_score(y, (oof_ens >= 0.5).astype(int))
        auce = roc_auc_score(y, oof_ens)
        se   = 0.6 * f1e + 0.4 * auce
        print(f'  Ensemble XGB({w_xgb:.1f})+LGBM({w_lgbm:.1f}) → {se:.4f}')

else:
    print(f'\n✗ Pas d\'amélioration vs v5 ({final_score:.4f} <= {BEST_CV:.4f}).')

# Log
exp = pd.read_csv('notes/experiments.csv')
new = pd.DataFrame([{
    'date': str(pd.Timestamp.today().date()),
    'model': 'XGBoost_Optuna60',
    'cv_strategy': 'GroupKFold(5)_by_location',
    'cv_score': round(final_score, 4),
    'cv_f1': round(f1_final, 4),
    'cv_auc': round(auc_final, 4),
    'lb_public': '',
    'notes': f'Optuna 60 trials XGB v1_features. Delta={final_score - BEST_CV:+.4f}. Params={study.best_params}',
}])
pd.concat([exp, new], ignore_index=True).to_csv('notes/experiments.csv', index=False)
print('Expérience loguée.')
