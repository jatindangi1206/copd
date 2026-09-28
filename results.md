# results

one section per script, in run order. figures go to figs/ (figs_dark/ with THEME=dark), tables to numbers/. the html report is built from these.

## codings.py --check

```
.venv/bin/python copd_clinical/codings.py --check
```

- ok: yes/no remap, ND/NK as missing, mMRC midpoints, mixed unit durations, BODE reconstruction
- sheet codes yes=1 no=2 none=0, remapped to 1/0/0
- ND / NK / NA treated as blank
- systolic/diastolic labels swapped in the sheet (systolic < diastolic in all 32 rows), higher value used as systolic

## 02_clinical_eda.py

```
.venv/bin/python copd_clinical/02_clinical_eda.py
```

figs: clin_cohort, clin_severity, clin_corr, clin_vs_wearable

- 33 patients with sheet + watch data. c034-c041 in the sheet but no watch data, not used
- 31 men, 2 women
- age 7-83 as recorded, median 65. c025 recorded as 7, kept as recorded
- GOLD A 15, B 15, E 3
- 21 of 33 have no dated exacerbation, 17 episodes in 12 patients
- CAT vs BODE r +0.07, mMRC vs walk distance r +0.06
- spearman: resting SpO2 vs lowest walk SpO2 +0.61, FEV1 vs lowest walk SpO2 +0.60, FEV1 vs BODE -0.63, rest mostly weak
- clinic pulse vs watch median HR r +0.36, watch ~9 bpm lower
- clinic vs watch SpO2 r -0.18 but watch has very few SpO2 readings so ignore

## 03_data_presence.py

```
.venv/bin/python copd_clinical/03_data_presence.py
```

figs: 03a-readings-per-patient, 03b-totals-and-days. table: numbers/data_presence.csv

- heart rate 1,276,237 readings
- HRV 101,955
- temp 181,405
- steps 56,468
- SpO2 1,698
- recording span per patient from about a week to ~11 months

## 04_hrv_per_patient.py

```
.venv/bin/python copd_clinical/04_hrv_per_patient.py
```

figs: 04-hrv-over-time-per-patient

- no HRV at all for c002, c019, c031, c032, c033
- almost every patient shows a low band and a top band near 120-129
- long breaks between recording periods

## 05_missingness.py

```
.venv/bin/python copd_clinical/05_missingness.py
```

figs: 05a-gap-between-readings, 05b-coverage-per-patient, 05c-readings-per-day, 05d-daily-coverage. table: numbers/hrv_missingness.csv

- 86.9% of gaps between HRV readings are exactly 10 min
- gaps over 30 min 5.5%, over 60 min 3.4%, over 3 h 1.3%, over 1 day 0.1%
- median coverage 36%, 9 of 28 patients reach 50%
- dense mid may to early july, thinner after for many patients
- most days either nearly empty or partly covered, almost no full day
- minute difference = time of each hrv reading minus time of the previous one (only rows where hrv came), per patient
- 87.2% of gaps <= 10 min, 92.7% <= 20, 96.6% <= 60
- falls off smoothly after the 10 min peak, small peaks at every missed 10 min reading
- 98.7% of gaps <= 3 h, 99.4% <= 6 h
- fig 05e-minute-difference: histogram up to 3 h (5 min bins), 6 h and 12 h (10 min bins)

## 06_hrv_distributions.py

```
.venv/bin/python copd_clinical/06_hrv_distributions.py
```

figs: 06a-hrv-curves-per-patient, 06b-hrv-histogram, 06c-hrv-curve-each-patient, 06d-median-hrv-per-patient

- 101,955 readings, 28 patients, 26 with at least 200
- HRV is SDNN, ~every 10 min, no unit in export
- two groups: peak in low 30s, block 120-129 (129 = max)
- 23% of readings are >= 120, count jumps at 120
- same shape in every patient, balance differs (c010, c024 mostly low; c004, c020 mostly top)
- per patient medians 41-107

## 07_sleep_and_events.py

```
.venv/bin/python copd_clinical/07_sleep_and_events.py
```

figs: 07a-hrv-sleep, 07b-hrv-around-exacerbations. table: numbers/hrv_sleep.csv

- HRV lower inside recorded sleep for 20 of 27 patients
- 17 dated exacerbations in 12 patients, 3 undated (c005, c032, c037)
- 12 of 17 dates have HRV within 2 weeks
- no change around the date that repeats across events, daily median swings a lot

## 08_sleep_stages.py

```
.venv/bin/python copd_clinical/08_sleep_stages.py
```

figs: 08a-sleep-stage-split, 08b-hrv-deep-vs-light. tables: numbers/sleep_blocks.csv, numbers/sleep_stage_hrv.csv

- 33 patients, 7,393 sleep blocks
- stage share deep 37.9%, light 59.1%, almost awake 2.9%
- deep share per patient 23-55%
- stage minutes cover a median 74% of each block
- HRV lower in deep than light blocks for 12 of 25 patients, no consistent stage effect
- stages come per block not per minute so this is block level

## 09_clinical_profile.py

```
.venv/bin/python copd_clinical/09_clinical_profile.py
```

figs: 09a-datasheet-completeness, 09b-symptoms-and-history, 09c-lung-and-walk. tables: numbers/datasheet_sections.csv, comorbidities.csv, imaging.csv, treatment.csv, treatment_count.csv

- sections 85-100% filled: symptoms, history, exam, imaging, walk test, pre-BD spirometry, CAT, BODE
- blood tests 39-63%, post-BD spirometry 48%, treatment 39%, IgE 3%
- symptoms: cough 21, breathlessness 20, sputum 14, wheeze 11
- history: smoking 26, occupational 20, comorbidity 18, biomass 13, family hx 10, sleep disturbance 7, TB 4
- smoking now: stopped 22, continues 4, never 7
- etiotype C 26, P 7
- comorbidities: hypertension 17, diabetes 10, OSA 5
- FEV1 median 43% pred, FVC 58%
- walk distance median 490 m, lowest walk SpO2 median 92%, min 67%
- imaging available: CXR 10, CT 16, ECG 10, echo 9. result coded 0 = not available

## 11_model_data.py

```
.venv/bin/python copd_clinical/11_model_data.py
```

figs: 11a-segment-threshold, 11b-segments. files: model_data/

- X = 18 slots (180 min); 249 segments, 25 patients; 314,128 rows; 12,384 random-masked, 13,192 block-masked
- ok: masks only on observed HRV inside segments; test always after train; no run > X inside
- x = 18 missing readings in a row (180 min) for this pilot, segments >= 24 h. to review with physiology experts
- 61% of HRV kept in segments, 29% of segment slots missing
- forecasting split 80/20 inside each segment: 69,454 train / 17,483 test slots
- daily split 80/20 per patient: 1,160 / 301 days

## 10_report.py

```
.venv/bin/python copd_clinical/10_report.py
```

- wrote COPD_EDA_Report.html, 24 figures from figs/
- wrote COPD_EDA_Report_dark.html, 24 figures from figs_dark/

## models

not random. 10 hypotheses about hrv characteristics from literature (shape, positive and bounded, circadian rhythm etc), tested on our data, 8 held. built a corpus of methods with assumptions / strengths / weaknesses, kept only ones that dont violate the 8, natively or with a published modification.

- general ml: xgboost, catboost
- neural sequence: rnn, lstm
- latent / state space: hmm, nonlinear state space model, rs-dpf, particle filter / particle smoother
- bayesian / irregular time: gru-ode-bayes, cd-gamma-dglm
- signal decomposition: ossa
- hybrid / research: physiology informed neural ode, gap aware state space transformer (own design)
- foundation model: timesfm 3

characteristics (first tab of the research doc), kept / dropped:

- H1 bounded and positive - kept
- H2 two modes, day and night - kept
- H3 strong lag-1 memory - dropped, lag-1 ~0.41 is moderate not strong
- H4 drifting baseline - kept
- H5 circadian rhythm - kept
- H6 heavy tailed spikes - dropped, watch drops bad readings instead of recording spikes
- H7 irregular timing - kept
- H8 random + behavioural missingness - kept
- H9 real time, no data after t - kept (our constraint). applies to forecasting, imputation here is bidirectional
- H10 per patient baseline - kept

research doc: https://docs.google.com/document/d/1V8rGaJFxRt5ccTJUKq4JcynNGbu9lqcs7h0J0KMzWDA/edit?usp=sharing

## imputation

## forecasting
