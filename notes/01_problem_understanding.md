# Étape 1 — Compréhension du problème (avec recherche littérature)

## Contexte métier

**Données :** Registres de décès en Ouganda (2007–2022)  
**Site exact :** Iganga-Mayuge Health and Demographic Surveillance System (HDSS), est de l'Ouganda — confirmé par la littérature (PubMed PMC12676583).  
**Géographie :** Est principalement (Iganga, Mayuge, Bugweri, Mbale) + quelques sites Ouest (Kalangala, Mitooma).

## Qu'est-ce qu'un décès "climate-sensitive" ?

Catégories de maladies climate-sensitive au niveau OMS / Sub-Saharan Africa :

| Maladie | Mécanisme climatique | Groupe à risque |
|---------|---------------------|-----------------|
| **Paludisme (P. falciparum)** | Pluie → gîtes larvaires anophèles ; Temp 20-32°C optimal | **Enfants < 5 ans** (mortalité dominante) |
| **Diarrhées / déshydratation** | Contamination eau par ruissellement pluvial | Enfants < 5 ans, nourrissons |
| **Infections respiratoires aiguës** | Variations de température, saison fraîche | Nourrissons, enfants, âgés |
| **Malnutrition** | Sécheresse → choc agricole → famine | Enfants < 5 ans |
| **Choléra/maladies hydriques** | Inondations, contamination | Toutes tranches |
| Fièvre jaune | Vecteurs liés à la végétation humide | Adultes non vaccinés |

**Décès NON climate-sensitive :** maladies cardiovasculaires, cancers, accidents, maladies chroniques — prédominants chez **les adultes et âgés**.

## Insights épidémiologiques clés (littérature, Iganga-Mayuge)

### Lags temporels confirmés pour le paludisme
Source : [Quantifying the Lagged Effects of Climate Variables on Malaria Risk in Eastern Uganda, PMC12676583]
- **Lag pluie → paludisme : 2 à 8 semaines** (rain_sum_7d et rain_sum_30d capturent bien cette fenêtre)
- Seuil critique : >200 mm/semaine de pluie déclenche le risque
- **Pas d'association significative température-paludisme** sur ce site spécifique (températures toujours dans la plage de transmission 20-25°C → hot_days_30d = 0 partout)

### Saisons Uganda (bimodal)
- **Saison des pluies 1 (Long Rains) :** mars–mai → pic paludisme **mai–juillet**
- **Saison sèche 1 :** juin–août
- **Saison des pluies 2 (Short Rains) :** août–novembre → pic paludisme **novembre–janvier**
- **Saison sèche 2 :** décembre–février

### Vulnérabilité par âge
Source : [Climate-driven malaria mortality among children in malaria-endemic areas of Uganda, PMC12360017]
- Enfants < 5 ans : mortalité malaria quasi-exclusivement climate-sensitive
- Mortalité plus sensible à la **température** pour les 5-14 ans qu'aux précipitations
- Garçons 5-14 ans : plus vulnérables à la chaleur que les filles

## Hypothèses de features prioritaires (avant modélisation)

1. **is_child_u5** — flag binaire le plus discriminant (taux 92% vs 30%)
2. **is_malaria_peak_season** — mai-juil ou nov-jan (lag de 4-6 sem après pic pluies)
3. **rain_sum_7d** — capture exactement la fenêtre de lag 2-8 sem
4. **age_log × rain_sum_30d** — interaction continue enfant × pluie
5. **ndvi_30d × rain_sum_30d** — habitat moustique composite
6. **temp_anomaly_7v90** — rupture climatique locale (plus informative que valeur absolue)
7. **elevation** — faible altitude = zones de plaine propices aux gîtes

## Décisions stratégiques

- `hot_days_30d` = 0 partout → confirme la littérature (Ouganda ne dépasse pas 35°C)
- Seuil météo pertinent en Ouganda : **pluie > 200mm/semaine** → `is_heavy_rain_week` à créer
- Focus optimization : les features climatiques n'ont pas de sens sans **interaction avec l'âge**
- GroupKFold par location = correct (le test set simule de nouvelles zones géographiques)

## Ce qui rendrait ce modèle utile au-delà du score

Un modèle qui identifie correctement les décès d'enfants < 5 ans comme climate-sensitive = signal d'alerte précoce pour les programmes de lutte contre le paludisme (aspersion, moustiquaires) dans les zones à forte pluviométrie.
