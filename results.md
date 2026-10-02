# results

one section per script, in run order. figures go to figs/ (figs_dark/ with THEME=dark), tables to numbers/. the html report is built from these.

## data (export of 29 sep 2026)

```
# data/copd/<pid>/<vital>/<pid>_<vital>.csv  ->  data/wearable/<pid>.csv.gz (1-minute file per patient)
cd .. && .venv/bin/python -c "import curate_dataset as cd; ..."   # cd.master(pid, folder), same function that built the old files
```

- sheet: 41 patients, c001-c041
- new export in data/copd: 41 entries = 40 patient folders (c001-c033, c035-c041) + processed_users.txt (a list of c035-c041). no c034 folder, so c034 has no watch data. readings up to 29 sep 2026. old export (data/raw, to 18 sep) kept but no longer used
- checked first: cd.master() on the old export gives back the old c001 file exactly
- heart rate, hrv, steps, temperature, spo2: same timestamps as the old export where both have the reading
- sleep for c001 and c027 is 5 h 30 min earlier than in the old export (same blocks, moved). in the new export 76% (c001) and 84% (c027) of those blocks start between 21:00 and 08:00, vs 53% before, so the new times look right. all other patients' sleep unchanged
- cohort = all 41 patients in the sheet. watch analyses use the 40 with a watch file; c034 is shown and reported as having no watch data, not dropped

## codings.py --check

```
python codings.py --check
```

- ok: yes/no remap, ND/NK as missing, mMRC midpoints, mixed unit durations, BODE reconstruction
- sheet codes yes=1 no=2 none=0, remapped to 1/0/0
- ND / NK / NA treated as blank
- systolic/diastolic labels swapped in the sheet (systolic < diastolic in all 32 rows), higher value used as systolic

## 02_clinical_eda.py

```
python 02_clinical_eda.py
```

figs: clin_cohort, clin_severity, clin_corr, clin_vs_wearable

- 41 patients in the sheet, all analysed. 40 have watch data, c034 has none
- 37 men, 4 women
- age 7-83 as recorded, median 65. c025 recorded as 7, kept as recorded
- GOLD A 19, B 19, E 3
- 29 of 41 have no dated exacerbation, 17 episodes in 12 patients
- CAT vs BODE r +0.05, mMRC vs walk distance r -0.02
- spearman: FEV1 vs BODE -0.70 (n 24), resting SpO2 vs lowest walk SpO2 +0.61 (n 40), CAT vs HbA1c +0.53 (n 23), FEV1 vs lowest walk SpO2 +0.52 (n 24), rest mostly weak
- clinic pulse vs watch median HR r +0.40 (n 39), watch ~10 bpm lower
- clinic vs watch SpO2 r -0.05 but watch has very few SpO2 readings so ignore

## 03_data_presence.py

```
python 03_data_presence.py
```

figs: 03a-readings-per-patient, 03b-totals-and-days, 03c-time-since-enrolment, 03d-readings-since-enrolment, 03e-readings-per-day-since-enrolment. table: numbers/data_presence.csv

- all 41 sheet patients in the figures; c034 shows zero readings, labelled no watch data
- heart rate 1,332,839 readings
- HRV 106,457
- temp 189,815
- steps 56,468
- SpO2 2,044
- recording span per patient 25 to 419 days (first to last reading of anything)
- enrolment = date the smart watch was provided (wearables section of the sheet), sheet has no enrolment date field
- c022 watch date reads 01-06-20260, not a date, so the treatment plan date next to it is used (1 jun 2026)
- c029 watch date is 12 may but its walk test and spirometry are 12 jun and daily data starts around mid june. used as recorded
- enrolled 12 may to 10 sep 2026, export runs to 29 sep 2026
- 23 patients have some data before enrolment, 1,286 minutes in all. left out of 03c-03e. this is what stretches some spans past a year (c035-c041 have stray readings months before enrolment, back to aug 2025 for five of them)
- 1,686,330 of 1,687,623 readings are on or after enrolment
- watch sent data on a median 71% of enrolled days (c034 counts as 0%)
- c035-c041 (enrolled 24 aug - 10 sep) send very little: 2-116 heart rate readings each, no HRV
- per day since enrolment: densest in the first weeks for many patients, then patchy or stops

## 04_hrv_per_patient.py

```
python 04_hrv_per_patient.py
```

figs: 04-hrv-over-time-per-patient

- watch data but no HRV: c002, c019, c031, c032, c033, c035, c036, c037, c038, c039, c040, c041
- c034: no watch data at all (panel says so)
- almost every patient shows a low band and a top band near 120-129
- long breaks between recording periods

## 05_missingness.py

```
python 05_missingness.py
```

figs: 05a-gap-between-readings, 05b-coverage-per-patient, 05c-readings-per-day, 05d-daily-coverage, 05e-minute-difference. table: numbers/hrv_missingness.csv

- 86.7% of gaps between HRV readings are exactly 10 min
- gaps over 30 min 5.5%, over 60 min 3.4%, over 3 h 1.3%, over 1 day 0.1%
- median coverage 35%, 7 of 28 patients reach 50%
- dense mid may to early july, thinner after for many patients
- most days either nearly empty or partly covered, almost no full day
- minute difference = time of each hrv reading minus time of the previous one (only rows where hrv came), per patient
- 87.1% of gaps <= 10 min, 92.6% <= 20, 96.6% <= 60
- falls off smoothly after the 10 min peak, small peaks at every missed 10 min reading
- 98.7% of gaps <= 3 h, 99.4% <= 6 h
- fig 05e-minute-difference: histogram up to 3 h (5 min bins), 6 h and 12 h (10 min bins)

## 06_hrv_distributions.py

```
python 06_hrv_distributions.py
```

figs: 06a-hrv-curves-per-patient, 06b-hrv-histogram, 06c-hrv-curve-each-patient, 06d-median-hrv-per-patient

- 106,457 readings, 28 patients, 26 with at least 200
- HRV is SDNN, ~every 10 min, no unit in export
- two groups: peak in low 30s, block 120-129 (129 = max)
- 23% of readings are >= 120, count jumps at 120
- same shape in every patient, balance differs (c010, c024 mostly low; c004, c020 mostly top)
- per patient medians 41-107

## 07_sleep_and_events.py

```
python 07_sleep_and_events.py
```

figs: 07a-hrv-sleep, 07b-hrv-around-exacerbations. tables: numbers/hrv_sleep.csv, numbers/exacerbation_day0.csv

- HRV lower inside recorded sleep for 21 of 27 patients
- 17 dated exacerbations in 12 patients, 3 undated (c005, c032, c037)
- 12 of 17 dates have HRV within 2 weeks
- no change around the date that repeats across events, daily median swings a lot
- 07b also has steps (median of the day's 10-min step readings) and hours of recorded sleep per day, each with the patient's usual median as a dashed line
- on the recorded date itself vs usual median: hrv above in 8 of 11, steps above in 7 of 12, sleep above in 7 of 14. few events and the same patients repeat, so not an effect
- c001's sleep on its three event dates changed with the new export (sleep times moved, see data)

## 08_sleep_stages.py

```
python 08_sleep_stages.py
```

figs: 08a-sleep-stage-split, 08b-hrv-deep-vs-light. tables: numbers/sleep_blocks.csv, numbers/sleep_stage_hrv.csv

- 40 patients, 8,111 sleep blocks
- stage share deep 38.0%, light 59.1%, almost awake 2.9%
- deep share per patient 14-55%
- stage minutes cover a median 74% of each block
- HRV lower in deep than light blocks for 13 of 25 patients, no consistent stage effect
- stages come per block not per minute so this is block level

## 09_clinical_profile.py

```
python 09_clinical_profile.py
```

figs: 09a-datasheet-completeness, 09b-symptoms-and-history, 09c-lung-and-walk. tables: numbers/datasheet_sections.csv, comorbidities.csv, imaging.csv, treatment.csv, treatment_count.csv

- all 41 sheet patients
- sections 85-100% filled: symptoms, history, exam, imaging, walk test, pre-BD spirometry, CAT, BODE
- blood tests 35-60%, post-BD spirometry 49%, oscillometry 75%, treatment 38%, IgE 2%
- symptoms (of 41): cough 27, breathlessness 21, sputum 15, wheeze 13
- history: smoking 30, occupational 23, comorbidity 21, biomass 16, family hx 11, sleep disturbance 8 (of 40), TB 4
- smoking now: stopped 23, continues 7, never 11
- etiotype C 30, P 10, 1 blank
- comorbidities: hypertension 19, diabetes 11, OSA 5
- FEV1 median 43% pred, FVC 64%
- walk distance median 486 m, lowest walk SpO2 median 92%, min 67%
- imaging available: CXR 11, CT 16, ECG 11, echo 10. result coded 0 = not available

## 11_model_data.py

```
python 11_model_data.py
```

figs: 11a-segment-threshold, 11b-segments. files: model_data/

- X = 18 slots (180 min); 259 segments, 26 patients; 339,350 rows; 12,677 random-masked, 13,478 block-masked
- ok: masks only on observed HRV inside segments; test always after train; no run > X inside
- x = 18 missing readings in a row (180 min) for this pilot, segments >= 24 h. to review with physiology experts
- 60% of HRV kept in segments, 29% of segment slots missing
- segments 24-363 h long, median 44. c015, c029 have no segment
- forecasting split 80/20 inside each segment: 71,206 train / 17,926 test slots
- daily split 80/20 per patient: 1,233 / 319 days

## 12_model_results.py

```
python 12_model_results.py
```

- reads results/summary.csv (run_models.py) and results/exac_monitoring/ (12_exac_classify.py); writes numbers/model_results.csv, numbers/exac_*.csv, figs 12a, 12b, 12c (light and dark)
- a run is not usable if it never finished, its error is not finite, any prediction is <= 0, or more than 5% are above 129 (the watch's maximum). run_models.py records those counts, nothing is flagged by hand
- rerun on 2 oct on the 29 sep model_data (259 segments, 26 patients; 28 patients have any hrv). every model gave usable values in all four tests
- random test (12,677 hidden): xgboost 26.3, rnn 26.3, lstm 26.4, catboost 26.4, timesfm3 26.5, gast 26.6, nlssm 27.0; straight line 27.1; hmm 32.2
- block test (13,478 hidden): lstm 27.7, rnn 28.0, gast 28.1, xgboost 28.9, catboost 28.9 ... straight line 30.9, hmm 32.4. 3 methods within 1.0 of the best
- 10-min forecast (13,254): xgboost 30.3, catboost 30.4, lstm 30.5, rnn 30.7, pinode 30.8, rsdpf 30.9; patient median 31.5; 9 of 14 methods beat the median; ossa 34.8; last reading 41.1
- daily (318 patient-days): gru_ode_bayes 17.7, catboost 17.8, timesfm3 17.9, xgboost 18.1; patient median 19.2; last reading 19.5; ossa 26.3
- fixes that mattered: rnn/lstm/gast forecast 30.5-31.3 (was 56-73, direct multi-step); pinode impute 28.2 / 30.2 (reversed-time pass); dglm impute and ossa no longer blow up; nlssm forecast 32.8 with no negative values
- gast forecast gave all NaN on its first run and worked when rerun unchanged (31.3); cause not found. the NaN row is still in results/summary.csv, 12_model_results.py uses the last row

## 10_report.py

```
python 10_report.py
```

- wrote COPD_EDA_Report.html, 31 figures from figs/
- wrote COPD_EDA_Report_dark.html, 31 figures from figs_dark/
- sections 13-15 now hold the model results as what we did / input / output / results, no interpretation. section 16 conclusions
- clinical captions (correlations, clinic vs watch, spans, patient counts) are now computed, not typed, so they follow the data
- headline is the 41 patients in the sheet; what is missing is listed per patient (c034 no watch data; 12 with watch data but no HRV)

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

`run_models.py impute <model> --mask random|block`, one process per model (logs/sweep.sh). numbers are in 12_model_results.py above

## forecasting

`run_models.py forecast <model>` and `daily <model>`, training gaps filled by the straight line first

## exacerbation classification

`12_exac_classify.py` (needs numbers/data_presence.csv from 03). target = dated exacerbation during monitoring

- 37 patients, 12 yes, 25 no. left out: c005, c032, c037 (event with no date) and c034 (no watch data). c035-c041 have a few watch readings each so they count as monitored negatives
- followup days 25 to 419 (recording span; stray old timestamps stretch some)
- all 44 values: xgboost 0.63, naive bayes 0.62, random forest 0.61. shuffled answers as high or higher 39 of 200
- reduced 27: xgboost 0.68, random forest 0.65, naive bayes 0.65. shuffled 23 of 200
- 5 of 24 scores below 0.5; always-no check 0.50

## 13_method_figures.py

```
THEME=light python 13_method_figures.py; THEME=dark python 13_method_figures.py
```

- figs 13a (imputation flow), 13b (forecasting flow), 13c (classification flow), 13d (model families: input, idea, output), 13e (classification methods)
- 13a and 13b are drawn on a real segment with the best usable model's real predictions (read from results/summary.csv), so rerun after the models
- report: 13d in section 12, 13a in section 13, 13b in section 14, 13c and 13e in section 15; each of 13-15 has a short "what the numbers mean" list

## tune_models.py

```
python tune_models.py impute|forecast|daily <model>      # logs/tune_sweep.sh runs them in chains
python run_models.py <task> <model> --tuned              # official tests with the best setting -> results/summary_tuned.csv
```

- 42 studies (14 models x impute / forecast / daily), optuna tpe. 40 tries for quick models, 8 for rsdpf, pf, gru_ode_bayes (60 segments), 10 for timesfm3. trial 0 = defaults
- practice set, never the official test: impute = a further 15% of visible readings hidden in runs; forecast / daily = last 20% of the train part
- a setting that gives non-finite, <= 0 or >5% above 129 predictions scores 1000 (dglm impute hit this often, gast once)
- official tests, default -> tuned. block: lstm 27.7 -> 26.8, rnn 28.0 -> 27.2, gast 27.9 -> 27.5, hmm 32.4 -> 30.2. counting > 0.1 as a change: 9 better, 5 same, 0 worse
- random: gast 26.3 -> 25.5, lstm 26.4 -> 25.5, hmm 32.2 -> 27.3
- every 10 min: timesfm3 31.7 -> 30.1 (best tuned), but gast default 29.9 is still the lowest; 6 better, 6 same, 2 worse (gru_ode_bayes 31.5 -> 32.3)
- daily: ossa 26.3 -> 19.4, gast 19.9 -> 18.3, rnn 20.6 -> 19.4; xgboost 17.8 best. practice-set gains of 8-10 for rnn / lstm did not carry over (lstm 20.6 -> 21.1)
- trees barely move (xgboost, catboost within 0.3 everywhere); pf, rsdpf, pinode no gain
- catboost now runs on cpu (it stalled on the shared gpu); its defaults were rerun on cpu
- gast: the fft count of real readings went slightly negative after long gaps (19% of entries, divisor crossed zero in 1.3%), giving the NaN runs. clamped at 0 in models/gast.py; defaults rerun: forecast 31.3 -> 29.9

## shap_analysis.py

```
python shap_analysis.py all --tuned ; THEME=dark python shap_analysis.py figures
```

- treeshap for xgboost / catboost on the scored readings, tuned settings. xgboost shares: impute readings after 28%, nearest reading before/after 24%, readings before 21%, heart rate 8% (vitals 20% in all)
- forecast: recent readings 34%, time of day of the target 32%, vitals 11%. daily: recent readings 30%, heart rate 16%, vitals 46%
- ablation (refit without a vital, block test): all vitals +0.7 to +2.6 (lstm +2.65, rnn +2.33, gast +2.25, catboost +0.94, xgboost +0.90, pinode +0.67); heart rate alone +0.4 to +1.8; steps and temperature ~0
- only xgboost, catboost, rnn, lstm, gast, pinode use vitals; the rest read hrv and time of day
- classification shap (refit on all 37 patients, descriptive only): xgboost reduced set leans on six-minute walk 49%, spirometry 25%
- figs 14a-e, 12d (default vs tuned), 13f (tuning flow); report sections 16 and 17, conclusions now 18
