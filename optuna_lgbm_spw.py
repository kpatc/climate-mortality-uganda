"""
Optuna LGBM avec scale_pos_weight libre.
Benchmark : CB Optuna SPW = 0.8240.
"""
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
DATA = Path('data')
BEST_CV = 0.8240

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


def cv_lgbm(params):
    oof = np.zeros(len(y))
    for tr_idx, va_idx in folds:
        m = lgb.LGBMClassifier(**params, random_state=SEED)
        m.fit(X_train.iloc[tr_idx], y.iloc[tr_idx],
              eval_X=X_train.iloc[va_idx], eval_y=y.iloc[va_idx],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=-1)])
        oof[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
    f1  = f1_score(y, (oof >= 0.5).astype(int))
    auc = roc_auc_score(y, oof)
    return 0.6*f1 + 0.4*auc, oof


def objective(trial):
    params = dict(
        objective='binary', metric='binary_logloss', verbosity=-1,
        n_estimators=2000,
        scale_pos_weight=trial.suggest_float('scale_pos_weight', 0.8, 2.5),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        num_leaves=trial.suggest_int('num_leaves', 31, 200),
        min_child_samples=trial.suggest_int('min_child_samples', 10, 50),
        feature_fraction=trial.suggest_float('feature_fraction', 0.5, 1.0),
        bagging_fraction=trial.suggest_float('bagging_fraction', 0.6, 1.0),
        bagging_freq=trial.suggest_int('bagging_freq', 1, 7),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
    )
    s, _ = cv_lgbm(params)
    return s


study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
# Warm start avec les meilleurs params v5 + SPW=1.5
study.enqueue_trial({
    'scale_pos_weight': 1.5,
    'learning_rate': 0.0351,
    'num_leaves': 117,
    'min_child_samples': 13,
    'feature_fraction': 0.771,
    'bagging_fraction': 0.995,
    'bagging_freq': 5,
    'reg_alpha': 0.00255,
    'reg_lambda': 0.0651,
})

print(f'Optuna LGBM+SPW — benchmark : {BEST_CV}')
print('50 trials...\n')
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_params.update(dict(objective='binary', metric='binary_logloss', verbosity=-1, n_estimators=2000))

print(f'\nMeilleur trial : {study.best_value:.4f}')
print(f'Benchmark CB   : {BEST_CV:.4f}')
print(f'Delta          : {study.best_value - BEST_CV:+.4f}')
print(f'Params         : {study.best_params}')

final_score, oof_final = cv_lgbm(best_params)
f1_final  = f1_score(y, (oof_final >= 0.5).astype(int))
auc_final = roc_auc_score(y, oof_final)
print(f'\nCV final : F1={f1_final:.4f}  AUC={auc_final:.4f}  Score={final_score:.4f}')

np.save(DATA / 'oof_lgbm_spw.npy', oof_final)

# Test predictions
test_preds = np.zeros(len(X_test))
for tr_idx, va_idx in folds:
    m = lgb.LGBMClassifier(**best_params, random_state=SEED)
    m.fit(X_train.iloc[tr_idx], y.iloc[tr_idx],
          eval_X=X_train.iloc[va_idx], eval_y=y.iloc[va_idx],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=-1)])
    test_preds += m.predict_proba(X_test)[:, 1] / 5
np.save(DATA / 'test_lgbm_spw.npy', test_preds)

# Blend LGBM + CB Optuna
oof_cb = np.load(DATA / 'oof_optuna_spw.npy')
test_cb = np.load(DATA / 'test_optuna_spw.npy')
print('\nBlend LGBM_spw + CB_Optuna:')
best_blend_s = final_score
best_blend_w = 0
for w in np.arange(0.05, 0.5, 0.05):
    blend = w * oof_final + (1-w) * oof_cb
    f1  = f1_score(y, (blend >= 0.5).astype(int))
    auc = roc_auc_score(y, blend)
    s   = 0.6*f1 + 0.4*auc
    print(f'  w_lgbm={w:.2f}: F1={f1:.4f}  AUC={auc:.4f}  Score={s:.4f}')
    if s > best_blend_s:
        best_blend_s = s
        best_blend_w = w

if best_blend_s > BEST_CV:
    blend_test = best_blend_w * test_preds + (1-best_blend_w) * test_cb
    test_ids = pd.read_parquet(DATA / 'test_ids.parquet')['ID']
    pred = (blend_test >= 0.5).astype(int)
    sub = pd.DataFrame({'ID': test_ids, 'TargetF1': pred, 'TargetRAUC': blend_test})
    out = DATA / f'submission_v19_lgbm_cb_{best_blend_s:.4f}.csv'
    sub.to_csv(out, index=False)
    print(f'\n✓ Soumission blend: {out}')

exp = pd.read_csv('notes/experiments.csv')
new = pd.DataFrame([{
    'date': str(pd.Timestamp.today().date()),
    'model': 'LGBM_Optuna50_SPW',
    'cv_strategy': 'GroupKFold(5)_by_location',
    'cv_score': round(final_score, 4),
    'cv_f1': round(f1_final, 4),
    'cv_auc': round(auc_final, 4),
    'lb_public': '',
    'notes': f'Optuna 50 trials LGBM avec SPW libre. Delta={final_score-BEST_CV:+.4f}. Best: {study.best_params}',
}])
pd.concat([exp, new], ignore_index=True).to_csv('notes/experiments.csv', index=False)
print('Expérience loguée.')
