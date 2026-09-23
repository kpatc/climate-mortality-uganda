# Étape 4 — Modélisation et évaluation

## Log des expériences

| Version | Modèle | CV OOF | LB Public | Notes |
|---------|--------|--------|-----------|-------|
| v1 | LGBM Optuna (class_weight=balanced) | 0.7982 | **0.8113** | Baseline soumis |
| v2 | LGBM Optuna (scale_pos_weight) | 0.8003 | à soumettre | +63% pred positives |

## Diagnostic du gap leaderboard

- Notre LB v1 : **0.8113**
- Leader : **0.8757**
- Gap : **0.064**

Le gap n'est PAS dû au seuil (confirmé : seuil 0.5 obligatoire par Zindi).

**Simulation** : pour atteindre 0.876, il faut ~AUC=0.87 + F1=0.88.  
Notre AUC LB est ~0.82. Il manque **0.05 de AUC** — cela ne se corrige pas avec de l'optimisation d'hyperparamètres. Ce sont les **features** qui manquent.

## Meilleurs hyperparamètres (Optuna, 50 trials)

```python
learning_rate     = 0.0351
num_leaves        = 117
min_child_samples = 13
feature_fraction  = 0.771
bagging_fraction  = 0.995
bagging_freq      = 5
reg_alpha         = 0.00255
reg_lambda        = 0.0651
scale_pos_weight  = 0.537  # neg/pos ratio (mieux que class_weight='balanced')
```

## Insights features (SHAP / importance gain)

Top features (gain) — contre-intuitivement, les features climatiques dominent :

| Rang | Feature | Type |
|------|---------|------|
| 1 | rain_sum_90d | Fournie |
| 2 | ndvi_90d | Fournie |
| 3 | temp_range_daily | Engineerée |
| 5 | age_log_x_tavg30 | Interaction |
| 6 | temp_anomaly_7v90 | Engineerée |
| 25 | age (raw) | Original |

Les interactions (age × climat) capturent plus de signal que l'âge brut.

## Plan d'amélioration — Vers le top 3

### URGENT : Données externes manquantes

Le challenge dit : *"participants are encouraged to use publicly available climate datasets"*.
Les leaders ont probablement des variables que nous n'avons pas.

**Variables à télécharger pour chaque (lat, lon, date) :**

| Variable | Source | Raison |
|---------|--------|--------|
| MODIS LST nocturne (MOD11A1) | NASA APPEEARS / GEE | Prédicteur malaria > température diurne |
| ERA5 Relative Humidity | GEE / CDS | Condition propice aux vecteurs |
| ERA5 Soil Moisture | GEE / CDS | Gîtes larvaires |
| ERA5 Wind Speed | GEE / CDS | Dispersion vecteurs |
| WorldPop Density 2010-2020 | WorldPop.org | Densité population = exposition |
| Distance to water (HydroSHEDS) | HydroSHEDS | Habitat Anopheles |
| CHIRPS 180d, 365d | CHIRPS | Anomalie saisonnière longue |

### Approche recommandée pour le download

```python
# Google Earth Engine (Python API) : télécharge les valeurs
# pour chaque point (lat, lon, date) du dataset
import ee
ee.Authenticate()
ee.Initialize()
```

### Autres pistes
1. **Pseudo-labeling** : entraîner sur prédictions test haute confiance (>0.9)
2. **Stacking** : meta-learner sur les 3 modèles (LGBM, XGB, CatBoost)
3. **Optuna 200 trials** : explorer l'espace plus loin
4. **Neural net** : TabNet ou MLP avec embeddings géographiques
