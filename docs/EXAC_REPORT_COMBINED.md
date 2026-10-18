# Exacerbation experiments: combined set, detection and onset prediction

Run on 17-18 Oct 2026 on the workstation (`run_exac.py`). Set: **combined** (COPD + AIIMS adults + AIIMS
children). Every model ran 25 splits per task:

- references and classical models: always_no, XGBoost, CatBoost, RNN, LSTM;
- TimesFM 3 (frozen) + a learned logistic layer;
- GAST v1 and its four ablations (gast_ssm, gast_attn, gast_nogap, gast_maskonly);
- GAST v1 with 3 and 14 days of history (detection only);
- gast2: GAST v1 with two outputs, trained on both tasks at once.

This file reports process, inputs, outputs and numbers. It does not interpret them. Every decision
behind the design is in `docs/EXAC_DESIGN.md` and `docs/GAST_DESIGN.md`. The copd and copd_clinical sets
also ran (every model except gast_d3, gast_d14 and gast2); they are not in this file.

---

## 1. The two questions

| | Detection | Onset prediction |
|---|---|---|
| question | Is this day an exacerbation day? | At the end of this day, does a new exacerbation episode begin on one of the next 3 days? |
| one sample | one patient-day | one patient-day |
| label = 1 when | the day is inside an exacerbation episode | an episode's first recorded day falls on day+1, day+2 or day+3 |
| which days are samples | every day with at least one HRV reading | days with at least one HRV reading that are **not** inside an episode and are not among the patient's last 3 days of watch data (their next 3 days are not fully observed) |
| an "event" (for event scores) | one episode | one onset |

## 2. What goes in: the data

### 2.1 Cohorts and labels

- **COPD** (this repo's pilot cohort): each dated exacerbation in the clinical datasheet is one
  exacerbation day.
- **AIIMS adults** (a-series): every day of each episode in the adult exacerbation sheet (one date, or a
  date range).
- **AIIMS children** (p-series): the onset day of each episode in the paediatric sheet.
- An episode with no recorded end is that one day. An a- or p-series patient who is in neither sheet had
  no exacerbation: all their days are normal days. A patient with an event whose date cannot be read is
  left out. All patients are treated alike; the groups are used only to report scores separately.

### 2.2 Minutes and signals

- Every minute as exported by the watch. Only minutes with a heart-rate reading between 0 and 300 are
  kept; every other signal is kept only on those minutes. Nothing is averaged, filled in or moved.
- 7 signals per minute: **HRV** (SDNN as the watch reports it, about one value every 10 minutes, no unit),
  **heart rate**, **temperature**, **SpO2**, **steps**, **inside a sleep record** (0/1), **inside a step
  record** (0/1).
- A minute with no reading of a signal is a blank. A gap stays a gap.

### 2.3 The window: what a model sees for one sample

The classified day plus the 6 days before it: 7 calendar days x 1,440 minutes = **10,080 minutes**, each
with the 7 signals. Nothing after the end of the classified day is used. Days or minutes with no data are
blank. (gast_d3 and gast_d14 use 3 and 14 days: 4,320 and 20,160 minutes. The samples are the same days.)

### 2.4 How much data (combined set)

| group | patients | detection days | exacerbation days | episodes | onset days | days before an onset | onsets |
|---|---|---|---|---|---|---|---|
| AIIMS adults | 110 | 9,338 | 125 | 38 | 9,009 | 97 | 36 |
| AIIMS children | 54 | 4,981 | 8 | 8 | 4,881 | 25 | 10 |
| COPD | 27 | 1,372 | 11 | 11 | 1,306 | 29 | 10 |
| **all** | **191** | **15,691** | **144** | **57** | **15,196** | **151** | **56** |

Positive days are under 1% of days in both tasks. Patients counted here are those with at least one HRV day.

## 3. How a run works

### 3.1 Split by patient

- No patient is in both training and test. Patients are grouped into strata (group x whether the patient
  has any exacerbation day) and a share of each stratum is drawn at random for the test set.
- Train:test ratios 80:20, 90:10, 70:30, 60:40, 50:50, each with seeds 0-4: **25 splits per model and
  task**. Every model gets the same patients in the same split, and both tasks use the same patients.
- The networks (RNN, LSTM, TimesFM 3's logistic layer, every GAST model) also hold back 20% of the
  training patients (drawn the same way) for early stopping; they never train on them. The tree models
  train on all training patients.

Average per split (detection): 57.2 test patients, of whom 11.4 have an exacerbation day; 4,881 test
days with 48.4 exacerbation days. Networks: 106.4 training + 27.4 held-back patients; trees: 133.8 training
patients. (Onset: 56.3 test patients, 12.3 with an onset; 4,730 test days with 45.6 positive; networks
104.8 + 26.9, trees 131.7.)

### 3.2 Class balance and threshold

Exacerbation days are rare, so every model is trained with the classes balanced: the trees weight
positive days up; every network trains on batches that are half positive days. Every model outputs a
probability per day. A day is **flagged** when its probability is **0.5 or more**. The threshold is fixed,
not tuned. Every setting is fixed (no tuning).

### 3.3 Training of every network (RNN, LSTM, TimesFM 3's layer, GAST, gast2)

- Adam, learning rate 0.001, batches of 32 days, every batch half positive and half negative days drawn
  at random from the training patients; binary cross-entropy; gradients clipped at 1.0.
- **Early stopping:** every 100 steps the AUROC on the held-back patients' days is computed; training
  stops after 5 checks without improvement (or at 3,000 steps) and the weights from the best check are
  kept (not retrained on all training patients).

## 4. The models

### 4.1 always_no (reference)

- **Design:** says "no exacerbation" for every day.
- **Input:** nothing.
- **Output:** probability 0 for every day; no day is ever flagged.
- **Why it is here:** it shows what the scores look like when no alarm is ever raised. Because positive
  days are under 1%, it already has about 99% accuracy.

### 4.2 XGBoost

- **Design:** a sum of 400 small decision trees (depth 4), each one correcting the ones before it
  (learning rate 0.05; each tree sees a random 80% of the days and 80% of the columns; histogram method;
  GPU). At every split XGBoost learns which side a blank goes to, so nothing is filled in.
- **Input:** one row per sample day = every raw value of its 7-day window side by side:
  10,080 minutes x 7 signals = **70,560 columns**, blanks left blank.
- **Class balance:** exacerbation days weighted up by (normal days / exacerbation days) in the training
  patients, so both classes weigh the same in total.
- **Output:** a probability per day; flagged if 0.5 or more.

### 4.3 CatBoost

- **Design:** a sum of 400 symmetric decision trees (depth 4: every node at one depth uses the same
  split), learning rate 0.05, GPU. CatBoost treats a blank as lower than every reading, so a split can
  separate "no reading" from readings.
- **Input:** the same 70,560-column row as XGBoost.
- **Class balance:** `auto_class_weights = Balanced` (both classes weigh the same in total).
- **Output:** a probability per day; flagged if 0.5 or more.

### 4.4 RNN

- **Input:** the 7-day window as 10,080 minutes x 14 numbers: the 7 readings, each divided by its largest
  value in the training patients (a change of units only, so values are near 0-1), and 7 "reading present"
  flags (a blank becomes 0 with its flag 0).
- **Design, in order:**
  1. **Learned compression:** a 1-D convolution over non-overlapping 10-minute blocks turns each block of
     10 minutes x 14 numbers into 32 numbers, with weights learned during training (not an average):
     10,080 minutes -> 1,008 steps.
  2. **Recurrent network:** `torch.nn.RNN`, 2 layers x 64 units, reading forward in time from 7 days ago
     to the end of the classified day.
  3. **Read-out:** the output after the last 10-minute block -> one number -> sigmoid -> probability.
- **Training:** section 3.3.
- **Output:** a probability per day; flagged if 0.5 or more.

### 4.5 LSTM

Identical to the RNN (same input, same 10-minute compression, same size, same training and early stopping)
with only the recurrent cell changed to `torch.nn.LSTM` (gated cells that can keep a longer memory).

### 4.6 TimesFM 3 (frozen) + a learned logistic layer

- **What TimesFM 3 is:** a time-series foundation model from Google Research (TimesFM 3.0, 331 million
  parameters, 20 transformer layers 1,280 wide), trained by its authors to forecast time series. Here its
  weights are **frozen**: nothing in it is trained on this data.
- **Input:** the 7-day window, every minute, each of the 7 signals as one TimesFM "variate" (TimesFM 3
  attends across variates). 10,080 minutes = 315 patches of 32 minutes. Values as recorded (TimesFM
  normalises each series itself). A blank is marked with TimesFM's own per-point mask: nothing is filled.
- **Design, in order:**
  1. **Frozen TimesFM 3:** reads the window; its attention is causal (each patch sees only earlier ones),
     so the last patch has seen the whole window.
  2. **Read-out:** each variate's transformer output at the last patch: 7 x 1,280 = **8,960 numbers**.
  3. **Learned logistic layer:** one linear layer, 8,960 numbers -> one number -> sigmoid -> probability.
     This layer is the only part that is trained (section 3.3).
- **Cache:** because TimesFM is frozen, each day's 8,960 numbers are the same in every split. They were
  computed once for all 15,691 sample days (during the first split; 42 minutes on the GPU) and stored in
  `model_data_exac/timesfm3_out.npy`.
- **Output:** a probability per day; flagged if 0.5 or more.

### 4.7 GAST v1 (gap-aware state-space transformer; the user's design, `docs/GAST_DESIGN.md`)

- **Input: tokens.** One token per minute of the window that has a heart-rate reading, at its real time.
  No grid and no blocks: the time between tokens is the real elapsed time, so gaps of any length are kept
  as gaps. Each token holds 24 numbers:
  - the 7 readings (each divided by its largest value in the training patients; 0 when blank),
  - 7 "reading present" flags,
  - 7 times since that signal's last reading in the window, as log(1 + minutes),
  - the time since the previous token, as log(1 + minutes),
  - the time of day as sine and cosine.
- **Design, in order:**
  1. **Token layer:** a linear layer, 24 numbers -> 32.
  2. **Gap-aware state space (layer A):** 32 memory channels. At each token, channel j is updated as
     h = exp(-rate_j x minutes since the previous token) x h + gate x (new input). Between tokens the
     memory only fades, by the real elapsed time; when a token arrives it is written in. The rates are
     learned; they start spread from 1/(10 minutes) to 1/(the window), so some channels remember minutes
     and others days. This update is linear in h, so it is computed exactly for all tokens at once.
  3. **Hourly reading:** the memory is read at the end of every hour of the window (168 readings for 7
     days), faded from the last token before that hour. Nothing is invented for hours with no token.
     Each reading -> linear layer -> GELU -> layer norm (32 numbers).
  4. **Time-aware causal attention (layer B):** one self-attention layer over the 168 hourly readings, 2
     heads, each hour sees only earlier hours, with a learned penalty per head that grows with the hours
     between them; then a small feed-forward layer (32 -> 64 -> 32), residual connections, layer norm,
     dropout 0.1.
  5. **Read-out:** the reading at the end of the classified day -> linear -> one number -> sigmoid ->
     probability.
- **Training:** section 3.3.
- **Output:** a probability per day; flagged if 0.5 or more.

### 4.8 The GAST ablations (each one changes one part of 4.7)

| model | what changes | question it answers (GAST_DESIGN 4) |
|---|---|---|
| gast_ssm | no attention (layer B removed): the hourly state is read out directly | does the gap-aware memory alone help? |
| gast_attn | no state space (layer A removed): each hour attends directly to the tokens before it, with a learned time embedding, then the same causal attention | is the state space needed? |
| gast_nogap | no gap features: tokens hold only the 7 readings and the time of day (9 numbers); the memory steps by token count, not by real elapsed time | does knowing the gaps help? |
| gast_maskonly | no readings: tokens hold only the flags, the times since and the time of day (17 numbers) | is the model only reading wear patterns? |
| gast (3 days) / gast (14 days) | the full GAST with a 3- or 14-day window (72 or 336 hourly readings); detection only | does the history length matter? |

### 4.9 gast2: GAST with two outputs

- **Design:** the full GAST (4.7) with two outputs at the end: p(detection) and p(onset in 3 days).
- **Training samples:** every detection day (days with HRV). The detection loss uses every day; the onset
  loss uses only the days valid for onset (not inside an episode, not the last 3 days of data).
  Loss = loss(onset) + 1.0 x loss(detection). Batches are half days positive for either task. Early
  stopping uses the mean of the two held-back AUROCs.
- **Scored:** the detection output on the detection test days, the onset output on the onset test days.

## 5. What is scored, and what each score means

All scores are computed on the **test patients' days only**. "Positive" = exacerbation day (detection) or
day before an onset (onset).

| score | what it is | range / reference |
|---|---|---|
| accuracy | share of days called correctly at the 0.5 threshold | always_no already gets ~0.99 because positives are rare |
| balanced accuracy | (sensitivity + specificity) / 2 | 0.5 = always_no |
| sensitivity | share of positive days that were flagged | 0-1 |
| specificity | share of negative days that were not flagged | 0-1 |
| PPV | share of flagged days that were positive | blank when no day was flagged |
| NPV | share of non-flagged days that were negative | |
| F1 | 2 x PPV x sensitivity / (PPV + sensitivity) | 0-1 |
| AUROC | chance that a random positive day gets a higher probability than a random negative day (ties count half); does not depend on the threshold | 0.5 = no separation, 1 = perfect, below 0.5 = reversed |
| AUPRC | average precision over all thresholds | its no-skill value is the share of positive days (~0.01 here) |
| Brier score | mean of (probability - label)^2 | 0 = perfect; always_no gets the positive share (~0.01) |
| events flagged | events with at least one flagged day (detection: any day of the episode; onset: any of the 3 days before the onset) | shown as caught / total |
| event sensitivity | events flagged / events | 0-1 |
| false alarms / 30 days | flagged negative days per 30 negative days | 0 = never a false alarm |
| lead time (onset) | days from the first flagged day to the onset, median over caught onsets | 1-3 by the task's definition |
| 95% interval | for each split, 1,000 resamples of the test **patients** (a patient's days stay together); the 2.5th and 97.5th percentiles | |
| calibration | days grouped by predicted probability (0-0.1, ..., 0.9-1.0): mean predicted probability vs the share that were actually positive | a calibrated model has the two equal |

**How the tables below are made.** Each score is computed per split; the tables show the **mean over the
25 splits**. "[lo-hi]" is the mean of the 25 per-split 95% intervals. "range" is the lowest and highest
value over the 25 splits. Event counts are **summed over the 25 splits**: a patient is a test patient in
several splits, so the same episode is counted once in each of them. "Pooled" tables put every test day of
every split together (a day appears once for each split it was a test day in). All tables are generated
from `results/exac/` by a script, not typed.

## 6. Results: detection

### 6.1 All groups

| model | AUROC [95%] | AUROC range | AUPRC [95%] | sensitivity [95%] | specificity [95%] | PPV | Brier [95%] | episodes flagged | event sens. | false alarms / 30 d [95%] |
|---|---|---|---|---|---|---|---|---|---|---|
| always_no | 0.500 [0.50-0.50] | 0.500-0.500 | 0.010 [0.003-0.019] | 0.000 [0.00-0.00] | 1.000 [1.00-1.00] | - | 0.010 [0.003-0.019] | 0/434 | 0.000 | 0.00 [0.00-0.00] |
| xgboost | 0.484 [0.34-0.62] | 0.379-0.588 | 0.012 [0.004-0.029] | 0.000 [0.00-0.00] | 1.000 [1.00-1.00] | - | 0.010 [0.003-0.019] | 0/434 | 0.000 | 0.00 [0.00-0.00] |
| catboost | 0.523 [0.42-0.65] | 0.401-0.645 | 0.017 [0.005-0.038] | 0.000 [0.00-0.00] | 1.000 [1.00-1.00] | 0.000 | 0.012 [0.005-0.020] | 0/434 | 0.000 | 0.01 [0.00-0.02] |
| rnn | 0.564 [0.43-0.69] | 0.443-0.656 | 0.018 [0.005-0.038] | 0.370 [0.18-0.55] | 0.734 [0.68-0.78] | 0.014 | 0.180 [0.158-0.204] | 189/434 | 0.458 | 7.97 [6.52-9.47] |
| lstm | 0.599 [0.45-0.74] | 0.461-0.698 | 0.021 [0.006-0.053] | 0.443 [0.22-0.68] | 0.701 [0.63-0.76] | 0.015 | 0.193 [0.168-0.220] | 228/434 | 0.515 | 8.98 [7.10-11.03] |
| timesfm3 | 0.552 [0.44-0.66] | 0.470-0.612 | 0.013 [0.005-0.031] | 0.049 [0.00-0.11] | 0.949 [0.93-0.96] | 0.010 | 0.048 [0.034-0.066] | 41/434 | 0.105 | 1.53 [1.09-2.06] |
| gast | 0.506 [0.35-0.67] | 0.303-0.687 | 0.012 [0.004-0.029] | 0.278 [0.09-0.48] | 0.725 [0.66-0.79] | 0.010 | 0.172 [0.144-0.203] | 171/434 | 0.350 | 8.24 [6.40-10.35] |
| gast_ssm | 0.534 [0.37-0.69] | 0.391-0.743 | 0.013 [0.004-0.033] | 0.367 [0.16-0.59] | 0.658 [0.58-0.73] | 0.011 | 0.193 [0.164-0.225] | 204/434 | 0.451 | 10.27 [8.12-12.60] |
| gast_attn | 0.547 [0.37-0.72] | 0.337-0.783 | 0.015 [0.005-0.034] | 0.326 [0.10-0.54] | 0.725 [0.65-0.79] | 0.010 | 0.178 [0.143-0.217] | 196/434 | 0.407 | 8.25 [6.18-10.56] |
| gast_nogap | 0.579 [0.41-0.74] | 0.385-0.748 | 0.017 [0.006-0.039] | 0.380 [0.14-0.64] | 0.727 [0.64-0.80] | 0.014 | 0.164 [0.125-0.206] | 179/434 | 0.467 | 8.19 [5.91-10.70] |
| gast_maskonly | 0.528 [0.36-0.69] | 0.383-0.795 | 0.013 [0.004-0.032] | 0.321 [0.11-0.55] | 0.710 [0.64-0.78] | 0.010 | 0.180 [0.148-0.216] | 174/434 | 0.387 | 8.69 [6.73-10.80] |
| gast (3 days) | 0.551 [0.40-0.72] | 0.418-0.703 | 0.014 [0.005-0.033] | 0.403 [0.17-0.67] | 0.672 [0.59-0.75] | 0.013 | 0.195 [0.159-0.233] | 221/434 | 0.511 | 9.85 [7.64-12.22] |
| gast (14 days) | 0.469 [0.29-0.69] | 0.174-0.589 | 0.010 [0.004-0.025] | 0.295 [0.08-0.55] | 0.695 [0.62-0.76] | 0.010 | 0.186 [0.151-0.225] | 134/434 | 0.374 | 9.15 [7.18-11.30] |
| gast2 (both tasks) | 0.506 [0.33-0.69] | 0.341-0.695 | 0.012 [0.004-0.029] | 0.027 [0.00-0.08] | 0.962 [0.95-0.97] | 0.006 | 0.059 [0.047-0.073] | 24/434 | 0.053 | 1.13 [0.77-1.60] |

| model | accuracy | balanced accuracy | NPV | F1 |
|---|---|---|---|---|
| always_no | 0.990 | 0.500 | 0.990 | 0.000 |
| xgboost | 0.990 | 0.500 | 0.990 | 0.000 |
| catboost | 0.990 | 0.500 | 0.990 | 0.000 |
| rnn | 0.731 | 0.552 | 0.992 | 0.028 |
| lstm | 0.698 | 0.572 | 0.992 | 0.029 |
| timesfm3 | 0.940 | 0.499 | 0.990 | 0.013 |
| gast | 0.721 | 0.502 | 0.990 | 0.020 |
| gast_ssm | 0.654 | 0.512 | 0.990 | 0.021 |
| gast_attn | 0.721 | 0.525 | 0.991 | 0.019 |
| gast_nogap | 0.723 | 0.553 | 0.992 | 0.027 |
| gast_maskonly | 0.706 | 0.516 | 0.990 | 0.019 |
| gast (3 days) | 0.669 | 0.537 | 0.991 | 0.025 |
| gast (14 days) | 0.691 | 0.495 | 0.990 | 0.017 |
| gast2 (both tasks) | 0.953 | 0.495 | 0.990 | 0.009 |

Pooled over the 25 splits (every test day of every split):

| model | test days | positive days | highest probability | days flagged | flagged and positive |
|---|---|---|---|---|---|
| always_no | 122,035 | 1,209 | 0.000 | 0 | 0 |
| xgboost | 122,035 | 1,209 | 0.393 | 0 | 0 |
| catboost | 122,035 | 1,209 | 0.825 | 13 | 0 |
| rnn | 122,035 | 1,209 | 0.997 | 30,320 | 433 |
| lstm | 122,035 | 1,209 | 0.978 | 39,157 | 585 |
| timesfm3 | 122,035 | 1,209 | 1.000 | 5,022 | 49 |
| gast | 122,035 | 1,209 | 1.000 | 34,261 | 363 |
| gast_ssm | 122,035 | 1,209 | 0.936 | 42,320 | 482 |
| gast_attn | 122,035 | 1,209 | 0.999 | 35,628 | 404 |
| gast_nogap | 122,035 | 1,209 | 1.000 | 29,271 | 406 |
| gast_maskonly | 122,035 | 1,209 | 0.998 | 35,220 | 386 |
| gast (3 days) | 122,035 | 1,209 | 0.997 | 40,222 | 511 |
| gast (14 days) | 122,035 | 1,209 | 1.000 | 31,313 | 308 |
| gast2 (both tasks) | 122,035 | 1,209 | 0.994 | 4,422 | 36 |

### 6.2 Per group

Average test fold per split: COPD 408 days (3.8 positive) from 8.0 patients (1.8 with a positive day); AIIMS adults 2,909 days (42.2 positive) from 33.0 patients (7.8 with a positive day); AIIMS children 1,565 days (2.4 positive) from 16.2 patients (1.8 with a positive day).

| group | model | AUROC [95%] | AUROC range | sensitivity | specificity | episodes flagged | false alarms / 30 d |
|---|---|---|---|---|---|---|---|
| COPD | xgboost | 0.515 [0.35-0.67] | 0.041-0.943 | 0.000 | 1.000 | 0/94 | 0.00 |
| COPD | catboost | 0.524 [0.32-0.68] | 0.068-0.849 | 0.000 | 1.000 | 0/94 | 0.00 |
| COPD | rnn | 0.624 [0.43-0.79] | 0.210-0.963 | 0.463 | 0.703 | 45/94 | 8.90 |
| COPD | lstm | 0.687 [0.47-0.81] | 0.400-0.955 | 0.594 | 0.663 | 59/94 | 10.12 |
| COPD | timesfm3 | 0.649 [0.46-0.75] | 0.333-0.966 | 0.047 | 0.942 | 4/94 | 1.74 |
| COPD | gast | 0.608 [0.44-0.74] | 0.170-0.866 | 0.364 | 0.713 | 39/94 | 8.61 |
| COPD | gast_ssm | 0.671 [0.50-0.78] | 0.460-0.910 | 0.544 | 0.656 | 53/94 | 10.32 |
| COPD | gast_attn | 0.626 [0.42-0.76] | 0.152-0.881 | 0.425 | 0.717 | 45/94 | 8.49 |
| COPD | gast_nogap | 0.651 [0.41-0.81] | 0.305-0.969 | 0.467 | 0.716 | 41/94 | 8.52 |
| COPD | gast_maskonly | 0.639 [0.43-0.78] | 0.357-0.891 | 0.431 | 0.699 | 41/94 | 9.04 |
| COPD | gast (3 days) | 0.654 [0.49-0.76] | 0.442-0.919 | 0.488 | 0.668 | 47/94 | 9.96 |
| COPD | gast (14 days) | 0.650 [0.45-0.77] | 0.305-0.914 | 0.473 | 0.710 | 37/94 | 8.69 |
| COPD | gast2 (both tasks) | 0.662 [0.50-0.76] | 0.383-0.908 | 0.081 | 0.945 | 8/94 | 1.65 |
| AIIMS adults | xgboost | 0.485 [0.35-0.62] | 0.366-0.574 | 0.000 | 1.000 | 0/280 | 0.00 |
| AIIMS adults | catboost | 0.509 [0.40-0.63] | 0.396-0.615 | 0.000 | 1.000 | 0/280 | 0.01 |
| AIIMS adults | rnn | 0.540 [0.38-0.68] | 0.422-0.652 | 0.367 | 0.709 | 130/280 | 8.74 |
| AIIMS adults | lstm | 0.566 [0.39-0.72] | 0.459-0.675 | 0.426 | 0.658 | 155/280 | 10.26 |
| AIIMS adults | timesfm3 | 0.524 [0.40-0.68] | 0.424-0.588 | 0.050 | 0.942 | 37/280 | 1.75 |
| AIIMS adults | gast | 0.472 [0.29-0.66] | 0.273-0.625 | 0.268 | 0.697 | 120/280 | 9.08 |
| AIIMS adults | gast_ssm | 0.492 [0.31-0.68] | 0.299-0.693 | 0.349 | 0.617 | 134/280 | 11.49 |
| AIIMS adults | gast_attn | 0.511 [0.29-0.70] | 0.269-0.733 | 0.313 | 0.687 | 132/280 | 9.39 |
| AIIMS adults | gast_nogap | 0.544 [0.35-0.74] | 0.349-0.672 | 0.365 | 0.691 | 118/280 | 9.26 |
| AIIMS adults | gast_maskonly | 0.487 [0.29-0.68] | 0.322-0.757 | 0.311 | 0.670 | 122/280 | 9.91 |
| AIIMS adults | gast (3 days) | 0.518 [0.35-0.72] | 0.365-0.681 | 0.395 | 0.636 | 153/280 | 10.91 |
| AIIMS adults | gast (14 days) | 0.419 [0.23-0.66] | 0.059-0.527 | 0.278 | 0.660 | 85/280 | 10.19 |
| AIIMS adults | gast2 (both tasks) | 0.462 [0.27-0.66] | 0.292-0.701 | 0.022 | 0.959 | 15/280 | 1.23 |
| AIIMS children | xgboost | 0.320 [0.18-0.48] | 0.060-0.601 | 0.000 | 1.000 | 0/60 | 0.00 |
| AIIMS children | catboost | 0.520 [0.35-0.70] | 0.128-0.970 | 0.000 | 1.000 | 0/60 | 0.00 |
| AIIMS children | rnn | 0.496 [0.33-0.64] | 0.082-0.936 | 0.227 | 0.789 | 14/60 | 6.33 |
| AIIMS children | lstm | 0.450 [0.24-0.62] | 0.042-0.764 | 0.183 | 0.784 | 14/60 | 6.48 |
| AIIMS children | timesfm3 | 0.410 [0.22-0.56] | 0.074-0.660 | 0.000 | 0.963 | 0/60 | 1.11 |
| AIIMS children | gast | 0.418 [0.20-0.58] | 0.045-0.802 | 0.147 | 0.772 | 12/60 | 6.83 |
| AIIMS children | gast_ssm | 0.447 [0.21-0.62] | 0.009-0.838 | 0.230 | 0.727 | 17/60 | 8.19 |
| AIIMS children | gast_attn | 0.495 [0.23-0.70] | 0.121-0.926 | 0.267 | 0.794 | 19/60 | 6.19 |
| AIIMS children | gast_nogap | 0.511 [0.22-0.70] | 0.015-0.919 | 0.310 | 0.795 | 20/60 | 6.16 |
| AIIMS children | gast_maskonly | 0.475 [0.25-0.64] | 0.043-0.835 | 0.153 | 0.785 | 11/60 | 6.46 |
| AIIMS children | gast (3 days) | 0.477 [0.23-0.64] | 0.008-0.887 | 0.300 | 0.734 | 21/60 | 7.97 |
| AIIMS children | gast (14 days) | 0.456 [0.25-0.62] | 0.002-0.846 | 0.227 | 0.750 | 12/60 | 7.51 |
| AIIMS children | gast2 (both tasks) | 0.456 [0.21-0.63] | 0.048-0.803 | 0.010 | 0.970 | 1/60 | 0.90 |

always_no: AUROC 0.500, sensitivity 0, specificity 1, no episodes flagged, in every group.

### 6.3 By train:test ratio (all groups, mean over seeds 0-4)

| model | AUROC 90:10 | 80:20 | 70:30 | 60:40 | 50:50 | sensitivity 90:10 | 80:20 | 70:30 | 60:40 | 50:50 |
|---|---|---|---|---|---|---|---|---|---|---|
| always_no | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| xgboost | 0.469 | 0.462 | 0.478 | 0.503 | 0.511 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| catboost | 0.504 | 0.537 | 0.517 | 0.518 | 0.538 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rnn | 0.557 | 0.544 | 0.578 | 0.578 | 0.561 | 0.452 | 0.339 | 0.368 | 0.379 | 0.311 |
| lstm | 0.617 | 0.541 | 0.554 | 0.647 | 0.635 | 0.398 | 0.375 | 0.309 | 0.529 | 0.605 |
| timesfm3 | 0.526 | 0.549 | 0.552 | 0.555 | 0.576 | 0.055 | 0.073 | 0.035 | 0.050 | 0.030 |
| gast | 0.481 | 0.443 | 0.565 | 0.503 | 0.539 | 0.300 | 0.193 | 0.172 | 0.317 | 0.408 |
| gast_ssm | 0.556 | 0.486 | 0.541 | 0.532 | 0.557 | 0.231 | 0.335 | 0.413 | 0.400 | 0.456 |
| gast_attn | 0.572 | 0.513 | 0.540 | 0.542 | 0.565 | 0.259 | 0.329 | 0.276 | 0.343 | 0.421 |
| gast_nogap | 0.633 | 0.494 | 0.587 | 0.605 | 0.579 | 0.587 | 0.369 | 0.334 | 0.369 | 0.242 |
| gast_maskonly | 0.511 | 0.533 | 0.512 | 0.534 | 0.550 | 0.297 | 0.342 | 0.292 | 0.364 | 0.310 |
| gast (3 days) | 0.561 | 0.492 | 0.548 | 0.590 | 0.566 | 0.418 | 0.304 | 0.405 | 0.481 | 0.406 |
| gast (14 days) | 0.346 | 0.487 | 0.494 | 0.487 | 0.530 | 0.235 | 0.528 | 0.255 | 0.409 | 0.050 |
| gast2 (both tasks) | 0.502 | 0.459 | 0.505 | 0.532 | 0.530 | 0.015 | 0.022 | 0.028 | 0.031 | 0.040 |

### 6.4 Calibration (pooled over the 25 splits)

Each cell: days in the band / mean predicted probability / share actually positive. '-' = no day in the band.

| band | xgboost | catboost | rnn | lstm |
|---|---|---|---|---|
| 0.0-0.1 | 121,965 / 0.002 / 0.010 | 118,973 / 0.014 / 0.010 | 18,013 / 0.044 / 0.010 | 7,027 / 0.064 / 0.009 |
| 0.1-0.2 | 60 / 0.134 / 0.017 | 2,463 / 0.135 / 0.013 | 21,747 / 0.149 / 0.007 | 11,227 / 0.156 / 0.006 |
| 0.2-0.3 | 6 / 0.255 / 0.000 | 455 / 0.237 / 0.013 | 26,344 / 0.246 / 0.007 | 31,529 / 0.259 / 0.006 |
| 0.3-0.4 | 4 / 0.345 / 0.000 | 96 / 0.343 / 0.010 | 14,969 / 0.344 / 0.010 | 21,526 / 0.340 / 0.009 |
| 0.4-0.5 | - | 35 / 0.448 / 0.057 | 10,637 / 0.449 / 0.010 | 11,568 / 0.449 / 0.010 |
| 0.5-0.6 | - | 9 / 0.551 / 0.000 | 10,383 / 0.550 / 0.014 | 12,940 / 0.554 / 0.011 |
| 0.6-0.7 | - | 3 / 0.610 / 0.000 | 9,444 / 0.647 / 0.014 | 17,468 / 0.641 / 0.016 |
| 0.7-0.8 | - | - | 5,203 / 0.746 / 0.014 | 5,846 / 0.742 / 0.018 |
| 0.8-0.9 | - | 1 / 0.825 / 0.000 | 3,589 / 0.846 / 0.015 | 2,481 / 0.838 / 0.023 |
| 0.9-1.0 | - | - | 1,706 / 0.942 / 0.015 | 423 / 0.931 / 0.009 |

| band | timesfm3 |
|---|---|
| 0.0-0.1 | 110,387 / 0.005 / 0.010 |
| 0.1-0.2 | 2,910 / 0.143 / 0.012 |
| 0.2-0.3 | 1,608 / 0.247 / 0.012 |
| 0.3-0.4 | 1,138 / 0.347 / 0.014 |
| 0.4-0.5 | 969 / 0.449 / 0.013 |
| 0.5-0.6 | 798 / 0.552 / 0.005 |
| 0.6-0.7 | 748 / 0.650 / 0.012 |
| 0.7-0.8 | 737 / 0.750 / 0.004 |
| 0.8-0.9 | 809 / 0.852 / 0.014 |
| 0.9-1.0 | 1,931 / 0.967 / 0.011 |

| band | gast | gast_ssm | gast_attn | gast_nogap |
|---|---|---|---|---|
| 0.0-0.1 | 41,882 / 0.022 / 0.009 | 18,995 / 0.062 / 0.009 | 32,873 / 0.038 / 0.010 | 60,467 / 0.014 / 0.009 |
| 0.1-0.2 | 11,154 / 0.148 / 0.009 | 20,513 / 0.145 / 0.008 | 20,922 / 0.146 / 0.007 | 10,377 / 0.147 / 0.009 |
| 0.2-0.3 | 10,258 / 0.251 / 0.009 | 13,626 / 0.249 / 0.007 | 13,303 / 0.247 / 0.008 | 7,912 / 0.248 / 0.007 |
| 0.3-0.4 | 11,736 / 0.351 / 0.011 | 12,033 / 0.350 / 0.010 | 9,751 / 0.349 / 0.012 | 7,065 / 0.350 / 0.008 |
| 0.4-0.5 | 12,734 / 0.450 / 0.011 | 14,542 / 0.453 / 0.011 | 9,555 / 0.450 / 0.012 | 6,938 / 0.451 / 0.011 |
| 0.5-0.6 | 12,693 / 0.548 / 0.011 | 16,664 / 0.548 / 0.012 | 9,564 / 0.550 / 0.010 | 8,340 / 0.552 / 0.012 |
| 0.6-0.7 | 7,991 / 0.647 / 0.010 | 15,615 / 0.647 / 0.011 | 9,179 / 0.651 / 0.011 | 9,172 / 0.650 / 0.014 |
| 0.7-0.8 | 7,531 / 0.750 / 0.011 | 6,397 / 0.747 / 0.013 | 8,885 / 0.748 / 0.012 | 6,062 / 0.747 / 0.015 |
| 0.8-0.9 | 4,831 / 0.844 / 0.010 | 3,606 / 0.832 / 0.007 | 5,581 / 0.843 / 0.013 | 3,607 / 0.844 / 0.017 |
| 0.9-1.0 | 1,225 / 0.939 / 0.008 | 44 / 0.908 / 0.000 | 2,422 / 0.939 / 0.014 | 2,095 / 0.944 / 0.012 |

| band | gast_maskonly | gast (3 days) | gast (14 days) | gast2 (both tasks) |
|---|---|---|---|---|
| 0.0-0.1 | 46,034 / 0.018 / 0.009 | 41,208 / 0.019 / 0.009 | 51,476 / 0.016 / 0.010 | 53,838 / 0.031 / 0.009 |
| 0.1-0.2 | 10,552 / 0.148 / 0.009 | 9,322 / 0.148 / 0.007 | 11,630 / 0.147 / 0.011 | 24,319 / 0.151 / 0.009 |
| 0.2-0.3 | 8,938 / 0.249 / 0.010 | 8,931 / 0.250 / 0.008 | 9,319 / 0.249 / 0.009 | 23,595 / 0.245 / 0.013 |
| 0.3-0.4 | 9,345 / 0.351 / 0.009 | 9,871 / 0.352 / 0.008 | 8,762 / 0.350 / 0.009 | 11,046 / 0.346 / 0.010 |
| 0.4-0.5 | 11,940 / 0.452 / 0.013 | 12,475 / 0.452 / 0.010 | 9,534 / 0.451 / 0.011 | 4,813 / 0.441 / 0.008 |
| 0.5-0.6 | 12,441 / 0.551 / 0.011 | 14,429 / 0.549 / 0.013 | 9,040 / 0.549 / 0.012 | 2,245 / 0.546 / 0.004 |
| 0.6-0.7 | 9,441 / 0.644 / 0.012 | 9,181 / 0.648 / 0.010 | 9,566 / 0.648 / 0.011 | 1,150 / 0.644 / 0.015 |
| 0.7-0.8 | 5,773 / 0.749 / 0.010 | 9,106 / 0.751 / 0.014 | 6,541 / 0.745 / 0.009 | 481 / 0.744 / 0.004 |
| 0.8-0.9 | 5,248 / 0.849 / 0.008 | 5,960 / 0.841 / 0.014 | 4,374 / 0.844 / 0.006 | 317 / 0.849 / 0.013 |
| 0.9-1.0 | 2,323 / 0.932 / 0.013 | 1,552 / 0.933 / 0.014 | 1,793 / 0.940 / 0.006 | 231 / 0.947 / 0.017 |

### 6.5 Training

| model | time per split (median) | steps run: median (min-max) | held-back AUROC at the best check (mean) |
|---|---|---|---|
| xgboost | 49 s | - | - |
| catboost | 38 s | - | - |
| rnn | 33 s | 600 (600-1,500) | 0.594 |
| lstm | 35 s | 700 (600-2,000) | 0.634 |
| timesfm3 | 6 s | 900 (600-1,700) | 0.599 |
| gast | 208 s | 800 (600-3,000) | 0.614 |
| gast_ssm | 246 s | 1,000 (600-1,900) | 0.614 |
| gast_attn | 204 s | 800 (600-2,000) | 0.624 |
| gast_nogap | 154 s | 800 (600-2,400) | 0.613 |
| gast_maskonly | 211 s | 900 (600-2,000) | 0.609 |
| gast (3 days) | 102 s | 900 (600-2,500) | 0.609 |
| gast (14 days) | 496 s | 900 (600-2,800) | 0.602 |
| gast2 (both tasks) | 254 s | 900 (600-2,200) | 0.585 |


## 7. Results: onset prediction (new episode in the next 3 days)

### 7.1 All groups

| model | AUROC [95%] | AUROC range | AUPRC [95%] | sensitivity [95%] | specificity [95%] | PPV | Brier [95%] | onsets flagged | event sens. | false alarms / 30 d [95%] | lead time (days) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| always_no | 0.500 [0.50-0.50] | 0.500-0.500 | 0.010 [0.004-0.017] | 0.000 [0.00-0.00] | 1.000 [1.00-1.00] | - | 0.010 [0.004-0.017] | 0/439 | 0.000 | 0.00 [0.00-0.00] | - |
| xgboost | 0.424 [0.28-0.58] | 0.261-0.486 | 0.010 [0.004-0.027] | 0.000 [0.00-0.00] | 1.000 [1.00-1.00] | - | 0.010 [0.004-0.017] | 0/439 | 0.000 | 0.00 [0.00-0.00] | - |
| catboost | 0.458 [0.33-0.59] | 0.348-0.575 | 0.012 [0.005-0.030] | 0.000 [0.00-0.00] | 1.000 [1.00-1.00] | 0.000 | 0.012 [0.007-0.019] | 0/439 | 0.000 | 0.01 [0.00-0.02] | - |
| rnn | 0.504 [0.39-0.62] | 0.404-0.590 | 0.012 [0.005-0.026] | 0.304 [0.15-0.47] | 0.744 [0.71-0.78] | 0.012 | 0.181 [0.165-0.196] | 213/439 | 0.502 | 7.69 [6.56-8.84] | 2.46 |
| lstm | 0.554 [0.41-0.69] | 0.396-0.636 | 0.015 [0.006-0.035] | 0.438 [0.22-0.64] | 0.644 [0.58-0.71] | 0.013 | 0.220 [0.195-0.244] | 259/439 | 0.581 | 10.68 [8.72-12.63] | 2.90 |
| timesfm3 | 0.524 [0.40-0.63] | 0.397-0.643 | 0.013 [0.005-0.032] | 0.082 [0.02-0.16] | 0.925 [0.91-0.94] | 0.012 | 0.065 [0.049-0.083] | 55/439 | 0.160 | 2.25 [1.66-2.84] | 2.28 |
| gast | 0.525 [0.38-0.65] | 0.380-0.666 | 0.011 [0.005-0.024] | 0.232 [0.07-0.41] | 0.774 [0.71-0.83] | 0.010 | 0.146 [0.118-0.176] | 133/439 | 0.310 | 6.79 [5.07-8.70] | 2.69 |
| gast_ssm | 0.512 [0.36-0.65] | 0.294-0.639 | 0.012 [0.005-0.032] | 0.382 [0.18-0.58] | 0.624 [0.54-0.70] | 0.010 | 0.205 [0.180-0.232] | 210/439 | 0.484 | 11.28 [9.00-13.70] | 2.80 |
| gast_attn | 0.563 [0.39-0.71] | 0.373-0.783 | 0.014 [0.005-0.033] | 0.390 [0.18-0.58] | 0.696 [0.62-0.77] | 0.012 | 0.175 [0.142-0.210] | 200/439 | 0.466 | 9.13 [6.85-11.53] | 2.85 |
| gast_nogap | 0.555 [0.39-0.71] | 0.456-0.746 | 0.013 [0.005-0.031] | 0.276 [0.09-0.47] | 0.792 [0.72-0.85] | 0.012 | 0.133 [0.099-0.170] | 150/439 | 0.342 | 6.24 [4.39-8.30] | 2.69 |
| gast_maskonly | 0.529 [0.36-0.68] | 0.433-0.709 | 0.014 [0.005-0.039] | 0.309 [0.10-0.54] | 0.706 [0.64-0.77] | 0.011 | 0.180 [0.150-0.211] | 171/439 | 0.403 | 8.81 [6.80-10.89] | 2.78 |
| gast2 (both tasks) | 0.535 [0.38-0.67] | 0.328-0.728 | 0.013 [0.005-0.031] | 0.176 [0.07-0.29] | 0.857 [0.81-0.90] | 0.009 | 0.118 [0.097-0.140] | 113/439 | 0.250 | 4.28 [3.09-5.69] | 2.55 |

| model | accuracy | balanced accuracy | NPV | F1 |
|---|---|---|---|---|
| always_no | 0.990 | 0.500 | 0.990 | 0.000 |
| xgboost | 0.990 | 0.500 | 0.990 | 0.000 |
| catboost | 0.990 | 0.500 | 0.990 | 0.000 |
| rnn | 0.739 | 0.524 | 0.991 | 0.022 |
| lstm | 0.642 | 0.541 | 0.992 | 0.025 |
| timesfm3 | 0.917 | 0.503 | 0.990 | 0.016 |
| gast | 0.768 | 0.503 | 0.990 | 0.019 |
| gast_ssm | 0.622 | 0.503 | 0.990 | 0.020 |
| gast_attn | 0.693 | 0.543 | 0.992 | 0.023 |
| gast_nogap | 0.787 | 0.534 | 0.991 | 0.023 |
| gast_maskonly | 0.702 | 0.508 | 0.990 | 0.022 |
| gast2 (both tasks) | 0.851 | 0.517 | 0.991 | 0.017 |

Pooled over the 25 splits (every test day of every split):

| model | test days | positive days | highest probability | days flagged | flagged and positive |
|---|---|---|---|---|---|
| always_no | 118,250 | 1,141 | 0.000 | 0 | 0 |
| xgboost | 118,250 | 1,141 | 0.270 | 0 | 0 |
| catboost | 118,250 | 1,141 | 0.766 | 15 | 0 |
| rnn | 118,250 | 1,141 | 0.999 | 29,810 | 327 |
| lstm | 118,250 | 1,141 | 0.999 | 41,291 | 511 |
| timesfm3 | 118,250 | 1,141 | 1.000 | 6,674 | 72 |
| gast | 118,250 | 1,141 | 0.998 | 24,904 | 253 |
| gast_ssm | 118,250 | 1,141 | 0.961 | 43,265 | 435 |
| gast_attn | 118,250 | 1,141 | 0.999 | 35,679 | 436 |
| gast_nogap | 118,250 | 1,141 | 1.000 | 25,189 | 323 |
| gast_maskonly | 118,250 | 1,141 | 0.999 | 33,273 | 343 |
| gast2 (both tasks) | 118,250 | 1,141 | 1.000 | 18,305 | 209 |

### 7.2 Per group

Average test fold per split: COPD 388 days (9.2 positive) from 7.6 patients (1.8 with a positive day); AIIMS adults 2,808 days (28.4 positive) from 32.5 patients (7.4 with a positive day); AIIMS children 1,534 days (8.0 positive) from 16.2 patients (3.1 with a positive day).

| group | model | AUROC [95%] | AUROC range | sensitivity | specificity | onsets flagged | false alarms / 30 d | lead time |
|---|---|---|---|---|---|---|---|---|
| COPD | xgboost | 0.366 [0.24-0.55] | 0.059-0.652 | 0.000 | 1.000 | 0/81 | 0.00 | - |
| COPD | catboost | 0.391 [0.26-0.51] | 0.052-0.691 | 0.000 | 1.000 | 0/81 | 0.00 | - |
| COPD | rnn | 0.464 [0.35-0.55] | 0.263-0.603 | 0.228 | 0.772 | 38/81 | 6.83 | 2.17 |
| COPD | lstm | 0.497 [0.36-0.63] | 0.190-0.773 | 0.305 | 0.691 | 42/81 | 9.28 | 2.00 |
| COPD | timesfm3 | 0.548 [0.40-0.66] | 0.328-0.746 | 0.038 | 0.948 | 7/81 | 1.56 | 1.58 |
| COPD | gast | 0.568 [0.38-0.69] | 0.329-0.802 | 0.166 | 0.813 | 18/81 | 5.62 | 2.19 |
| COPD | gast_ssm | 0.617 [0.44-0.73] | 0.372-0.872 | 0.448 | 0.697 | 48/81 | 9.09 | 2.25 |
| COPD | gast_attn | 0.619 [0.44-0.75] | 0.322-0.954 | 0.409 | 0.773 | 40/81 | 6.82 | 2.33 |
| COPD | gast_nogap | 0.536 [0.33-0.69] | 0.220-0.927 | 0.239 | 0.851 | 24/81 | 4.48 | 2.46 |
| COPD | gast_maskonly | 0.598 [0.42-0.73] | 0.361-0.950 | 0.303 | 0.756 | 35/81 | 7.33 | 2.18 |
| COPD | gast2 (both tasks) | 0.634 [0.45-0.76] | 0.361-0.894 | 0.228 | 0.879 | 27/81 | 3.63 | 2.10 |
| AIIMS adults | xgboost | 0.514 [0.36-0.68] | 0.323-0.645 | 0.000 | 1.000 | 0/265 | 0.00 | - |
| AIIMS adults | catboost | 0.515 [0.35-0.68] | 0.378-0.701 | 0.000 | 1.000 | 0/265 | 0.01 | - |
| AIIMS adults | rnn | 0.529 [0.40-0.66] | 0.397-0.647 | 0.361 | 0.721 | 150/265 | 8.38 | 2.44 |
| AIIMS adults | lstm | 0.576 [0.43-0.73] | 0.424-0.734 | 0.504 | 0.604 | 180/265 | 11.87 | 2.88 |
| AIIMS adults | timesfm3 | 0.555 [0.40-0.67] | 0.420-0.636 | 0.108 | 0.912 | 42/265 | 2.63 | 2.34 |
| AIIMS adults | gast | 0.494 [0.32-0.66] | 0.109-0.765 | 0.252 | 0.746 | 88/265 | 7.62 | 2.71 |
| AIIMS adults | gast_ssm | 0.472 [0.29-0.65] | 0.155-0.662 | 0.345 | 0.600 | 116/265 | 12.01 | 2.81 |
| AIIMS adults | gast_attn | 0.513 [0.32-0.71] | 0.278-0.740 | 0.364 | 0.654 | 115/265 | 10.38 | 2.76 |
| AIIMS adults | gast_nogap | 0.545 [0.35-0.72] | 0.301-0.737 | 0.260 | 0.767 | 95/265 | 6.98 | 2.68 |
| AIIMS adults | gast_maskonly | 0.487 [0.29-0.69] | 0.120-0.752 | 0.308 | 0.662 | 98/265 | 10.15 | 2.88 |
| AIIMS adults | gast2 (both tasks) | 0.499 [0.32-0.67] | 0.174-0.714 | 0.157 | 0.842 | 61/265 | 4.75 | 2.47 |
| AIIMS children | xgboost | 0.330 [0.17-0.52] | 0.149-0.604 | 0.000 | 1.000 | 0/93 | 0.00 | - |
| AIIMS children | catboost | 0.445 [0.27-0.71] | 0.255-0.627 | 0.000 | 1.000 | 0/93 | 0.01 | - |
| AIIMS children | rnn | 0.435 [0.18-0.59] | 0.188-0.805 | 0.179 | 0.779 | 25/93 | 6.62 | 2.54 |
| AIIMS children | lstm | 0.503 [0.26-0.70] | 0.131-0.836 | 0.313 | 0.706 | 37/93 | 8.82 | 2.83 |
| AIIMS children | timesfm3 | 0.408 [0.12-0.53] | 0.212-0.770 | 0.041 | 0.945 | 6/93 | 1.65 | 2.00 |
| AIIMS children | gast | 0.510 [0.23-0.75] | 0.125-0.735 | 0.209 | 0.811 | 27/93 | 5.67 | 2.58 |
| AIIMS children | gast_ssm | 0.489 [0.19-0.71] | 0.188-0.726 | 0.377 | 0.646 | 46/93 | 10.63 | 2.79 |
| AIIMS children | gast_attn | 0.590 [0.25-0.84] | 0.074-0.908 | 0.409 | 0.750 | 45/93 | 7.50 | 2.93 |
| AIIMS children | gast_nogap | 0.534 [0.25-0.79] | 0.222-0.820 | 0.285 | 0.823 | 31/93 | 5.30 | 2.73 |
| AIIMS children | gast_maskonly | 0.558 [0.25-0.77] | 0.344-0.874 | 0.282 | 0.774 | 38/93 | 6.78 | 2.81 |
| AIIMS children | gast2 (both tasks) | 0.504 [0.19-0.76] | 0.089-0.788 | 0.184 | 0.877 | 25/93 | 3.69 | 2.77 |

always_no: AUROC 0.500, sensitivity 0, specificity 1, no onsets flagged, in every group.

### 7.3 By train:test ratio (all groups, mean over seeds 0-4)

| model | AUROC 90:10 | 80:20 | 70:30 | 60:40 | 50:50 | sensitivity 90:10 | 80:20 | 70:30 | 60:40 | 50:50 |
|---|---|---|---|---|---|---|---|---|---|---|
| always_no | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| xgboost | 0.382 | 0.414 | 0.455 | 0.434 | 0.436 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| catboost | 0.462 | 0.457 | 0.451 | 0.459 | 0.464 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rnn | 0.520 | 0.509 | 0.516 | 0.499 | 0.478 | 0.291 | 0.371 | 0.323 | 0.360 | 0.176 |
| lstm | 0.522 | 0.592 | 0.523 | 0.549 | 0.584 | 0.351 | 0.579 | 0.327 | 0.510 | 0.424 |
| timesfm3 | 0.473 | 0.547 | 0.531 | 0.534 | 0.536 | 0.095 | 0.152 | 0.105 | 0.052 | 0.005 |
| gast | 0.522 | 0.471 | 0.520 | 0.550 | 0.562 | 0.288 | 0.168 | 0.283 | 0.246 | 0.174 |
| gast_ssm | 0.461 | 0.517 | 0.546 | 0.520 | 0.516 | 0.434 | 0.260 | 0.498 | 0.377 | 0.340 |
| gast_attn | 0.536 | 0.572 | 0.598 | 0.550 | 0.558 | 0.486 | 0.319 | 0.404 | 0.398 | 0.342 |
| gast_nogap | 0.557 | 0.574 | 0.544 | 0.558 | 0.542 | 0.200 | 0.424 | 0.157 | 0.305 | 0.291 |
| gast_maskonly | 0.504 | 0.545 | 0.511 | 0.535 | 0.549 | 0.259 | 0.375 | 0.310 | 0.319 | 0.282 |
| gast2 (both tasks) | 0.527 | 0.508 | 0.539 | 0.553 | 0.550 | 0.052 | 0.221 | 0.231 | 0.240 | 0.138 |

### 7.4 Calibration (pooled over the 25 splits)

Each cell: days in the band / mean predicted probability / share actually positive. '-' = no day in the band.

| band | xgboost | catboost | rnn | lstm |
|---|---|---|---|---|
| 0.0-0.1 | 118,175 / 0.003 / 0.010 | 112,691 / 0.023 / 0.010 | 25,532 / 0.022 / 0.011 | 20,953 / 0.025 / 0.009 |
| 0.1-0.2 | 67 / 0.132 / 0.030 | 4,696 / 0.133 / 0.009 | 10,956 / 0.155 / 0.011 | 8,232 / 0.146 / 0.009 |
| 0.2-0.3 | 8 / 0.237 / 0.000 | 671 / 0.235 / 0.012 | 14,467 / 0.255 / 0.009 | 7,379 / 0.256 / 0.009 |
| 0.3-0.4 | - | 136 / 0.340 / 0.029 | 23,002 / 0.346 / 0.007 | 19,473 / 0.366 / 0.007 |
| 0.4-0.5 | - | 41 / 0.435 / 0.000 | 14,480 / 0.446 / 0.009 | 20,913 / 0.441 / 0.008 |
| 0.5-0.6 | - | 10 / 0.535 / 0.000 | 10,891 / 0.548 / 0.010 | 18,123 / 0.562 / 0.012 |
| 0.6-0.7 | - | 3 / 0.652 / 0.000 | 8,753 / 0.647 / 0.011 | 9,668 / 0.644 / 0.011 |
| 0.7-0.8 | - | 2 / 0.747 / 0.000 | 5,461 / 0.746 / 0.011 | 9,538 / 0.751 / 0.014 |
| 0.8-0.9 | - | - | 2,945 / 0.845 / 0.014 | 3,424 / 0.834 / 0.013 |
| 0.9-1.0 | - | - | 1,763 / 0.944 / 0.010 | 547 / 0.951 / 0.009 |

| band | timesfm3 |
|---|---|
| 0.0-0.1 | 102,337 / 0.006 / 0.009 |
| 0.1-0.2 | 4,126 / 0.144 / 0.014 |
| 0.2-0.3 | 2,291 / 0.247 / 0.011 |
| 0.3-0.4 | 1,609 / 0.348 / 0.007 |
| 0.4-0.5 | 1,213 / 0.448 / 0.016 |
| 0.5-0.6 | 1,068 / 0.550 / 0.015 |
| 0.6-0.7 | 1,021 / 0.649 / 0.006 |
| 0.7-0.8 | 1,068 / 0.750 / 0.010 |
| 0.8-0.9 | 1,152 / 0.853 / 0.010 |
| 0.9-1.0 | 2,365 / 0.965 / 0.012 |

| band | gast | gast_ssm | gast_attn | gast_nogap |
|---|---|---|---|---|
| 0.0-0.1 | 57,609 / 0.014 / 0.009 | 19,287 / 0.051 / 0.008 | 50,568 / 0.017 / 0.008 | 67,619 / 0.011 / 0.009 |
| 0.1-0.2 | 8,150 / 0.147 / 0.008 | 13,016 / 0.146 / 0.009 | 7,380 / 0.146 / 0.010 | 7,976 / 0.146 / 0.010 |
| 0.2-0.3 | 7,510 / 0.252 / 0.011 | 9,949 / 0.250 / 0.009 | 6,673 / 0.251 / 0.012 | 6,241 / 0.248 / 0.010 |
| 0.3-0.4 | 9,282 / 0.350 / 0.011 | 11,818 / 0.353 / 0.010 | 8,254 / 0.352 / 0.008 | 5,448 / 0.350 / 0.009 |
| 0.4-0.5 | 10,790 / 0.452 / 0.011 | 20,902 / 0.456 / 0.011 | 9,694 / 0.452 / 0.009 | 5,775 / 0.450 / 0.008 |
| 0.5-0.6 | 9,060 / 0.546 / 0.012 | 21,909 / 0.545 / 0.011 | 12,471 / 0.551 / 0.013 | 6,317 / 0.551 / 0.012 |
| 0.6-0.7 | 5,926 / 0.648 / 0.011 | 14,112 / 0.641 / 0.009 | 9,511 / 0.647 / 0.012 | 6,980 / 0.651 / 0.014 |
| 0.7-0.8 | 4,919 / 0.749 / 0.008 | 5,125 / 0.744 / 0.009 | 7,662 / 0.749 / 0.012 | 5,175 / 0.745 / 0.014 |
| 0.8-0.9 | 3,649 / 0.845 / 0.009 | 2,083 / 0.835 / 0.009 | 4,461 / 0.841 / 0.011 | 4,029 / 0.849 / 0.012 |
| 0.9-1.0 | 1,355 / 0.934 / 0.006 | 49 / 0.916 / 0.041 | 1,576 / 0.936 / 0.008 | 2,690 / 0.942 / 0.010 |

| band | gast_maskonly | gast2 (both tasks) |
|---|---|---|
| 0.0-0.1 | 47,806 / 0.011 / 0.009 | 38,866 / 0.025 / 0.009 |
| 0.1-0.2 | 5,493 / 0.148 / 0.010 | 13,327 / 0.149 / 0.008 |
| 0.2-0.3 | 5,462 / 0.252 / 0.011 | 13,698 / 0.253 / 0.008 |
| 0.3-0.4 | 7,907 / 0.354 / 0.009 | 18,644 / 0.353 / 0.010 |
| 0.4-0.5 | 18,292 / 0.457 / 0.011 | 15,406 / 0.445 / 0.012 |
| 0.5-0.6 | 13,662 / 0.544 / 0.008 | 9,989 / 0.546 / 0.013 |
| 0.6-0.7 | 7,494 / 0.647 / 0.012 | 4,832 / 0.641 / 0.012 |
| 0.7-0.8 | 6,509 / 0.749 / 0.012 | 2,095 / 0.746 / 0.005 |
| 0.8-0.9 | 4,026 / 0.845 / 0.011 | 975 / 0.841 / 0.006 |
| 0.9-1.0 | 1,599 / 0.938 / 0.008 | 418 / 0.945 / 0.012 |

### 7.5 Training

| model | time per split (median) | steps run: median (min-max) | held-back AUROC at the best check (mean) |
|---|---|---|---|
| xgboost | 48 s | - | - |
| catboost | 38 s | - | - |
| rnn | 30 s | 600 (600-2,900) | 0.557 |
| lstm | 39 s | 800 (600-2,000) | 0.638 |
| timesfm3 | 5 s | 700 (600-2,000) | 0.591 |
| gast | 231 s | 900 (600-2,600) | 0.622 |
| gast_ssm | 264 s | 1,000 (600-2,300) | 0.588 |
| gast_attn | 254 s | 1,000 (600-2,700) | 0.619 |
| gast_nogap | 181 s | 1,000 (600-2,400) | 0.648 |
| gast_maskonly | 229 s | 1,000 (600-2,700) | 0.625 |
| gast2 (both tasks) | 256 s | 900 (600-2,200) | 0.585 |


Notes on the training tables. "Steps run" is where training stopped; 600 means the best held-back check
was the first one (step 100), followed by 5 checks without improvement. Every split had at least one
positive day held back, so early stopping ran in every split. gast2 is one model trained on both tasks:
its training row is the same in both tables, and its held-back AUROC is the mean over its two outputs.
TimesFM 3: the first detection split (80:20, seed 0) also computed the frozen outputs for all 15,691
sample days (2,521 s); every other split only trains the logistic layer. Times: one RTX A5500 GPU, 18 CPU
cores per run, two runs at once (the first TimesFM 3 split ran alone on 36 cores).

## 8. Where everything is

| what | where |
|---|---|
| per split: every test day's probability and flag | `results/exac/<task>/combined/<model>/test<NN>_seed<S>/predictions.csv.gz` |
| per split: every score, per group, intervals, calibration, slices by wear coverage / longest gap / days of history, training details, the patient lists | `.../metrics.json` |
| one line per split and group | `results/exac/summary_detect.csv`, `results/exac/summary_onset.csv` |
| run logs | `logs/classical/`, `logs/gast/`, `logs/timesfm3/` |
| code | `run_exac.py`, `models/rnn.py`, `models/gast_v1.py`, `models/timesfm3.py` (and the copy in `exac/`) |
| design decisions | `docs/EXAC_DESIGN.md`, `docs/GAST_DESIGN.md` |
