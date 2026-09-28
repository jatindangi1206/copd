# COPD baseline clinical EDA

**33 patients** — every wearable patient has a datasheet row (33/33, no id casualties). 8 further sheet patients (c034–c041) have no wearable data and are excluded.

## Cohort

- Age **7–83**, median 65
- Sex: male 31, female 2
- BMI median 24.6 (range 16.2–39.9)
- GOLD: A 15, B 15, E 3
- mMRC median 2.5; CAT median 13; BODE median 3
- Ever-smoker 26/33, biomass exposure 13/33, comorbidity 18/33, **sleep disturbance 7/33**

## Correlations worth reporting (Spearman, n in brackets)

| pair | r | n |
|---|---|---|
| cat vs bode | -0.04 | 32 |
| mmrc vs walk_dist | -0.09 | 32 |
| bode vs walk_dist | +0.01 | 32 |
| fev1_pct vs bode | -0.63 | 18 |
| age vs walk_dist | -0.04 | 32 |
| spo2 vs walk_spo2_min | +0.61 | 32 |
| smoke_index vs fev1_pct | -0.15 | 18 |
| bmi vs bode | -0.25 | 33 |
| exac_12m vs cat | +0.43 | 27 |

## Clinic vs wearable

- clinic pulse vs wearable median HR: r = **+0.36** (n=32), median difference -8.8
- clinic SpO2 vs wearable median SpO2: r = **-0.18** (n=31), median difference +5.0

