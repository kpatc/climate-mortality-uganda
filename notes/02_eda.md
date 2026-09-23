# Étape 2 — EDA et nettoyage des données

## Structure des fichiers

| Fichier | Lignes | Colonnes | Rôle |
|---------|--------|----------|------|
| Train.csv | 3 146 | 13 | Apprentissage (target inclus) |
| Test.csv | 1 030 | 12 | Soumission |
| climate_features.csv | 4 176 | 18 | Features satellites (couvre train + test) |
| SampleSubmission.csv | — | 3 (ID, TargetF1, TargetRAUC) | Format de sortie |

**Jointure :** `climate_features` se merge sur `ID` — couverture parfaite (0 ID manquant).

## Qualité des données

- **Valeurs manquantes :** 0 dans tous les fichiers
- **Doublons :** aucun (IDs uniques dans train et test, 0 overlap)
- **Incohérences :** aucune détectée

## Variable cible

- **Déséquilibre modéré :** 65.1% classe 1 (climate-sensitive), 34.9% classe 0
- → `class_weight='balanced'` ou `scale_pos_weight` recommandé

## Signal dominant : l'âge

La corrélation de Pearson avec la cible est **-0.44** — de loin la variable la plus discriminante.

| Groupe d'âge | Taux climate-sensitive | N |
|-------------|----------------------|---|
| 1–4 ans | **92.3%** | 621 |
| < 1 an | 78.8% | 1 112 |
| 5–14 ans | 68.4% | 117 |
| 15–59 ans | 49.8% | 650 |
| 60+ ans | **30.0%** | 646 |

→ Les enfants de ≤5 ans représentent **55.8%** du dataset. Leurs décès sont presque tous climate-sensitive.

## Variables climatiques : signal faible en linéaire

| Feature | Corrélation Pearson |
|---------|-------------------|
| age | **-0.44** |
| ndvi_30d | 0.04 |
| rain_sum_30d | -0.04 |
| avg_temperature | -0.03 |
| tavg_30d | -0.02 |
| elevation | ~0 |
| hot_days_30d | NaN (constant à 0) |

→ Les features climatiques auront leur valeur dans les **interactions avec l'âge**, pas en linéaire.

## Structure temporelle

- Train et Test couvrent les **mêmes années (2007–2022)** avec 466 dates communes
- → **Pas de fuite temporelle** : split aléatoire justifié (pas de leakage via date)
- Saisonnalité visible mais faible (taux cible : 0.61 en sept-oct vs 0.69 en nov-avril)

## Structure géographique — RISQUE MAJEUR

| | Train | Test |
|--|-------|------|
| Nb locations | 39 | 11 |
| Locations en commun | **1 seulement** |

→ **Quasi aucun overlap géographique** entre train et test.
→ **Ne jamais utiliser `location` comme feature** (overfitting garanti).
→ **Target encoding par location = fuite** sur le test.
→ Utiliser `latitude`/`longitude` en continu, `elevation`, `zone` à la place.
→ Validation croisée : **GroupKFold par location** recommandé pour simuler la généralisation géographique.

## Feature `hot_days_30d` = inutilisable

Toutes les observations ont `hot_days_30d = 0`. L'Ouganda ne dépasse pas 35°C aux altitudes et latitudes de ces localités. **À exclure absolument.**

## Décisions de nettoyage

1. Drop `hot_days_30d` (variance nulle)
2. Drop `location` (quasi-aucun overlap train/test)
3. Garder `latitude`, `longitude`, `elevation` comme proxies géographiques
4. Parser `deathdate` → `month`, `day_of_year`, `year`
5. Créer des buckets d'âge comme features supplémentaires
