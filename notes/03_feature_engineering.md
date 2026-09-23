# Feature Engineering — Climate Risk & Health

## Features existantes dans climate_features.csv (Étape 0 — inventaire)

Source : données fournies par Zindi (pré-extraites)

| Feature | Source | Granularité | Fenêtre lag |
|---------|--------|-------------|-------------|
| avg_temperature, max_temperature, min_temperature | ERA5-Land | jour J | 0 |
| precipitation | ERA5-Land / CHIRPS | jour J | 0 |
| tavg_7d, tavg_30d, tavg_90d | ERA5-Land | moyennes glissantes | 7/30/90j |
| tmax_30d, tmin_30d | ERA5-Land | moyennes glissantes | 30j |
| temp_range_mean_30d | ERA5-Land | écart moyen | 30j |
| rain_sum_7d, rain_sum_30d, rain_sum_90d | CHIRPS | cumuls | 7/30/90j |
| max_daily_rain_30d | CHIRPS | max journalier | 30j |
| rain_days_30d | CHIRPS | nb jours > seuil | 30j |
| ndvi_30d, ndvi_90d | MODIS | moyennes glissantes | 30/90j |
| elevation | SRTM | statique | - |
| slope | SRTM | statique (exclu: r=0.993 avec elevation) | - |
| hot_days_30d | ERA5-Land | compte jours chauds (exclu: variance=0) | 30j |

**Ce qui manque** : humidité relative (RH), vitesse du vent, rayonnement solaire,
anomalies climatologiques réelles (vs baseline mensuelle locale).

---

## Features engineerées (src/02_feature_engineering.ipynb)

### Temporelles
- month, day_of_year, year : encodage calendaire
- season_uganda : 4 saisons Ouganda (Long Rains, Long Dry, Short Rains, Short Dry)
- month_sin, month_cos, doy_sin, doy_cos : cyclicité (évite discontinuité jan-déc)

### Âge
- is_neonate (<1an), is_infant (<2), is_child_u5 (<5), is_child (1-14), is_adult (15-59), is_elderly (≥60)
- age_log : log1p(age) — compresse la queue droite
- age_bucket : bins [0,1), [1,5), [5,15), [15,60), [60+) → labels 0-4

### Climatiques dérivées
- temp_range_daily = max_temperature - min_temperature (stress thermique)
- rain_anomaly_7v30 = rain_sum_7d - rain_sum_30d/(30/7)
- rain_anomaly_30v90 = rain_sum_30d - rain_sum_90d/3
- temp_anomaly_7v90 = tavg_7d - tavg_90d (rupture climatique courte)
- temp_anomaly_30v90 = tavg_30d - tavg_90d
- rain_intensity_30d = rain_sum_30d / rain_days_30d (intensité, pas volume)
- is_rainy_week : rain_sum_7d > 35mm
- is_rainy_day : precipitation > 0
- ndvi_delta = ndvi_30d - ndvi_90d (accélération de la végétation)
- ndvi_x_rain30 = ndvi_30d × rain_sum_30d (habitat Anopheles)

### Basées sur littérature (Iganga-Mayuge HDSS — PMC12676583)
- is_heavy_rain_week : rain_sum_7d > 200mm (seuil risque malaria Uganda)
- is_malaria_peak_season : mois ∈ {5,6,7,11,12,1} (lag 4-6 sem post saisons des pluies)
- temp_in_malaria_range : 20°C ≤ tavg_30d ≤ 32°C (plage optimale Plasmodium)

### Interactions âge × climat
- u5_x_rain30, u5_x_ndvi, u5_x_rainy_wk : nourrisson × conditions paludiques
- age_log_x_rain30, age_log_x_tavg30 : effet continu âge × climat
- elderly_x_temp_anom : âgé × anomalie chaleur (cardiovasculaire)
- rain_season_x_u5 : saison humide × enfant < 5 ans

**Total : 59 features (après exclusion location, hot_days_30d, slope, deathdate, ID)**

---

## Enrichissement externe — NASA POWER (Étape 1)

**Source** : NASA POWER (Prediction Of Worldwide Energy Resources)  
**URL API** : `https://power.larc.nasa.gov/api/temporal/daily/point`  
**Paramètres** : T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, RH2M, WS2M, ALLSKY_SFC_SW_DWN  
**Couverture** : 55 lieux uniques (lat, lon arrondi à 6 décimales)  
**Fenêtre temporelle** : 2007-06-29 → 2022-12-13 (déathdate min - 95j → max)  
**Format cache** : JSON par (lat, lon, année), stocké dans `data/nasa_cache/`  
**Date d'extraction** : 2026-09-10  
**Authentication** : aucune (API publique)  
**Rate limiting** : 1.2s entre appels, retry exponentiel sur 429/500  

### Nouvelles features calculées (src/04_external_data_download.ipynb)

| Feature | Description | Hypothèse |
|---------|-------------|-----------|
| rh2m_7d, rh2m_14d, rh2m_30d, rh2m_90d | Humidité relative moyenne (%) | Anopheles survit si RH > 60% |
| ws2m_7d, ws2m_30d | Vitesse vent 2m (m/s) | Dispersion des vecteurs |
| solar_7d, solar_30d | Rayonnement solaire (MJ/m²) | Stress thermique indirect |
| heat_index_30d, heat_index_max | Indice de chaleur Steadman (°C) | Stress thermique combiné T+RH |
| rh2m_anomaly_7v30 | rh2m_7d - rh2m_30d | Accélération humidité courte |
| rh2m_anomaly_30v90 | rh2m_30d - rh2m_90d | Tendance humidité longue |
| rh2m_clim_month | Moyenne mensuelle historique RH | Baseline climatologique locale |
| rh2m_clim_anomaly | rh2m_30d - rh2m_clim_month | Anomalie vs normale saisonnière |
| is_high_humidity_30d | rh2m_30d > 60 (binaire) | Seuil survie moustique |
| is_high_humidity_7d | rh2m_7d > 60 (binaire) | Seuil survie moustique court terme |
| nasa_tmax_30d, nasa_tmin_30d | Températures NASA (cross-val) | Validation vs ERA5 existant |
| nasa_rain_30d | Précipitations NASA (cross-val) | Validation vs CHIRPS existant |

**Formule heat index (Steadman simplifié)** :
```
HI = -8.78 + 1.61·T + 2.34·RH - 0.15·T·RH - 0.012·T² - 0.016·RH² + ...
```
où T en °C, RH en %, valide pour T > 27°C.

### Validation croisée Open-Meteo (Étape 2)
- Source : `https://archive-api.open-meteo.com/v1/archive` (ERA5, gratuit, sans clé)  
- Comparaison sur 10 lieux, fenêtre jan. 2015  
- Biais typique NASA vs ERA5 : voir log de la cellule Open-Meteo du notebook  
- Décision : si biais < 1°C et < 5mm/mois, utiliser NASA POWER comme source unique  

---

## Décisions d'exclusion

| Feature | Raison |
|---------|--------|
| location | 39 lieux train / 11 test, 1 commun → fuite géographique si encodée |
| hot_days_30d | Variance nulle = 0 pour tous les enregistrements |
| slope | r = 0.993 avec elevation (redondant) |
