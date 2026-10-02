# COPD baseline clinical EDA

**41 patients** in the sheet, all analysed. 40 have watch data; no watch data: c034. Watch-only patients not in the sheet: none.

## Cohort

- Age **7–83**, median 65
- Sex: male 37, female 4
- BMI median 24.6 (range 16.2–39.9)
- GOLD: A 19, B 19, E 3
- mMRC median 2.5; CAT median 14; BODE median 2
- Ever-smoker 30/41, biomass exposure 16/41, comorbidity 21/41, **sleep disturbance 8/41**

## Correlations worth reporting (Spearman, n in brackets)

| pair | r | n |
|---|---|---|
| cat vs bode | -0.04 | 40 |
| mmrc vs walk_dist | -0.18 | 40 |
| bode vs walk_dist | +0.01 | 40 |
| fev1_pct vs bode | -0.70 | 24 |
| age vs walk_dist | +0.00 | 39 |
| spo2 vs walk_spo2_min | +0.61 | 40 |
| smoke_index vs fev1_pct | -0.14 | 24 |
| bmi vs bode | -0.30 | 41 |
| exac_12m vs cat | +0.27 | 34 |

## Clinic vs wearable

- clinic pulse vs wearable median HR: r = **+0.40** (n=39), median difference -10.0
- clinic SpO2 vs wearable median SpO2: r = **-0.05** (n=38), median difference +4.0

