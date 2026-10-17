# Exacerbation classification: design log

Started 16-17 Oct 2026, planned with the user one decision at a time. **Locked** = the user answered.
**Proposed** = Claude's suggestion, not yet confirmed. Code is written only from locked decisions.

## 1. Task (locked 17 Oct 2026)

Two experiments, in this order (the user: "both, as two experiments"):

- **A. Detection, first:** for each patient-day (a calendar day, as exported), is it an exacerbation day?
- **B. Prediction, second:** at the end of a day, will a new exacerbation begin in the next few days?
  The onset of an episode is its **first recorded day**, for every episode: adult ranges and single
  dates, the children's onset, and the COPD datasheet dates. Designed in 6.7.

An exacerbation is an episode of one day or a few, and it comes back in the same patient. COPD and
AIIMS are two cohorts of patients (AIIMS: children and adults; COPD: adults). They are treated alike as
patients: no disease-specific rules; the groups are only used to report scores separately.

An outside review the user shared (17 Oct 2026) raised: detection vs prediction (taken: both), reliable
onset dates (taken: first recorded day), no future leakage, split by patient and scoring whole episodes
(already in the design), calibration (taken: section 5), disease-specific labels (not taken: the user
treats all as patients), a personal baseline as an input (not taken: the user ruled out baselines).

## 2. Data and labels (locked)

- Cohorts: COPD (this repo's data) and AIIMS = a-series (adults) + p-series (children). ACHE is left
  out: no exacerbation record.
- COPD: each dated exacerbation in the datasheet is one exacerbation day.
- AIIMS adults: every day of each episode in the adult sheet (one date, or a range).
- AIIMS children: the onset day only (`PAED_DAYS = 1`). The visit date only checks that the dates were
  read correctly. (Was "onset + next day" until 17 Oct 2026.)
- An episode with no recorded end (adults' single dates, COPD dates, children's onsets) is that one
  day; the days after it are normal days. Nothing is invented and no day is marked uncertain (locked
  17 Oct 2026; an outside review suggested leaving the next days out as uncertain - not taken).
- An a- or p-series patient in neither sheet had no exacerbation: every day is a normal day.
- A patient with an exacerbation whose date cannot be read is left out (c005, c032, c037: undated in
  the datasheet; one child: one row reads as a year between onset and visit).
- Sheet quirks (date ranges in several spellings; paediatric Excel cells with day and month swapped)
  are handled in `14_exac_data.py` and documented there.

## 3. What the models see (locked)

- **Every data point, no summaries.** No means, medians, baselines or other hand-made features. No
  interpolation: a gap stays a gap.
- **Minutes:** only minutes with a heart-rate reading of 0-300 (the user's pass-1 rule). Every other
  signal is used only on those minutes; all other minutes are blank.
- **Signals per minute:** HRV, HR, temperature, SpO2, steps, inside a sleep record (0/1), inside a
  step record (0/1). Sleep-stage numbers are left out (one total per sleep record, not per minute).
- **Samples:** a day is a sample if it has at least one HRV reading (pass 2). Kept on 17 Oct 2026 after
  an outside review suggested also using labelled days without HRV (up to 143 more exacerbation days).
- **Window:** the day being classified plus the 6 days before it = 7 x 1,440 = 10,080 minutes. The 6
  earlier days count as context even when they have no HRV (their HR minutes are real data). Missing
  days or minutes stay blank.
- **Tree models:** the same 7 days, every raw value side by side: 10,080 minutes x 7 signals = 70,560
  columns per day, blanks left blank (about 4.6 GB for the combined set; CatBoost will be slow).
- **Networks:** each minute's 7 values plus 7 "reading present" flags (a blank becomes 0 with its flag
  0), then a **learned compression layer** in front of RNN, LSTM and gast alike: it reads blocks of
  consecutive minutes with weights it learns (a 1-D convolution), not an average, and shortens the
  sequence so the networks fit in memory.
  - Blocks of **10 minutes** (10,080 -> 1,008 steps; HRV comes about every 10 minutes, so a block
    usually holds one HRV reading). Locked 17 Oct 2026.
  - Each signal **divided by its largest value in the training patients** (a change of units only, so
    the networks get numbers near 0-1; no mean or median). Locked 17 Oct 2026.

## 4. Why not the 180-minute segments here (locked: no segments for this task)

The imputation/forecasting work cut each patient's HRV into segments wherever HRV was missing for more
than 180 minutes. That suits rebuilding or extending a continuous HRV curve. It does not suit this task:

1. The label belongs to a calendar day; a segment does not. Segments start and stop at gaps, so a day
   can be split across segments or fall outside every segment.
2. Segments drop data: in COPD about 60% of HRV readings fell inside segments. An exacerbation day with
   patchy wear would vanish, and the model would learn only from well-worn days.
3. The gaps may carry information (wear and sleep may change during an exacerbation; not known). With
   "reading present" flags the model sees every gap without anything being filled in.

## 5. Split and scoring (locked)

- Split by patient: no patient is in both train and test. Within each stratum (group: COPD, AIIMS
  adult, AIIMS child, x has an exacerbation day or not) the test share is drawn at random with a seed.
- Train:test 80:20, 90:10, 70:30, 60:40, 50:50; **5 seeds per ratio** (0-4; locked 17 Oct 2026).
- Per day: accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, F1, AUROC, AUPRC.
- Per episode: caught if any of its days is flagged; false alarms per 30 days without an exacerbation.
- Calibration (locked 17 Oct 2026): the Brier score (the average squared difference between the
  probability and what happened, 0 = perfect) and a table of predicted vs observed exacerbation rates
  in probability bands (0-0.1, 0.1-0.2, ..., 0.9-1.0).
- Overall and per group (COPD, AIIMS adults, AIIMS children).
- From the user's GAST brief (17 Oct 2026), for every model: lead time for onset prediction (days from
  the first flag to the onset); 95% intervals by resampling patients; scores by wear coverage, after long
  gaps, by how much history a day has, per group and per fold.
- Sets: `combined` (COPD + AIIMS, watch data), `copd` (watch data), `copd_clinical` (watch data + the
  COPD clinical table, as recorded, minus the count of exacerbations during monitoring).

## 6. The models

### 6.1 always_no (reference, locked)
Says "no exacerbation" for every day. Shows the accuracy of never raising an alarm.

### 6.2 XGBoost (locked 17 Oct 2026)
Answers: weight exacerbation days up; flag at a fixed 0.5; fixed settings, no tuning. Class balance and
threshold are decided per model (the user chose not to apply one rule to every model).

- **Input:** one row per sample day: the 70,560 raw values of its 7-day window (section 3), blanks left
  blank. For `copd_clinical`, the clinical values are added as extra columns.
- **How it works:** a sum of many small decision trees, each one adding a correction to the previous
  ones. At every split XGBoost learns which side a blank goes to, so nothing is filled in.
- **Class balance:** `scale_pos_weight` = training normal days / training exacerbation days, so the
  exacerbation days together weigh as much as the normal days.
- **Settings:** 400 trees, depth 4, learning rate 0.05, row subsample 0.8, column subsample 0.8 per
  tree, histogram method; GPU when available. Same as the earlier exacerbation code.
- **Output:** a probability per day; the day is flagged if it is 0.5 or more.

### 6.3 CatBoost (locked 17 Oct 2026)
Answers: weight exacerbation days up; flag at a fixed 0.5; the same fixed settings as XGBoost.

- **Input:** the same row as XGBoost (70,560 raw values, blanks left blank; plus clinical values for
  `copd_clinical`).
- **How it works:** a sum of symmetric decision trees (every node at one depth uses the same split).
  CatBoost does not learn where a blank goes: it treats a blank as lower than every reading, so a
  split can still separate "no reading" from readings.
- **Class balance:** `auto_class_weights="Balanced"`: exacerbation days together weigh as much as
  normal days.
- **Settings:** 400 trees, depth 4, learning rate 0.05; GPU when available. Slow on CPU with 70,560
  columns.
- **Output:** a probability per day; flagged if 0.5 or more.

### 6.4 RNN (locked 17 Oct 2026)
Answers: forward only; half-exacerbation batches; flag at a fixed 0.5; the size in `models/rnn.py`;
read from the last step; stop early on held-back training patients (20%, AUROC, check every 100 steps,
stop after 5 checks without gain, at most 3,000 steps, keep the best weights).

- **Input:** the 7-day window as 10,080 minutes x 14 numbers (7 scaled readings + 7 "reading present"
  flags; for `copd_clinical` the scaled clinical values are added to every minute).
- **Compression layer:** a 1-D convolution over non-overlapping 10-minute blocks, 32 numbers per block,
  weights learned with the rest of the network: 10,080 minutes -> 1,008 steps.
- **Network:** `torch.nn.RNN` from `models/rnn.py`, 2 layers x 64 units, reading forward in time from
  7 days ago to the end of the classified day (nothing after the day).
- **Read-out:** the output at the last step (after the day's final 10-minute block) -> a probability.
- **Training:** Adam, learning rate 0.001, batches of 32 days, every batch half exacerbation days and
  half normal days drawn at random from the training patients; binary cross-entropy; gradients clipped
  at 1.0.
- **Early stopping:** 20% of the training patients are held back, drawn like the main split (by group
  and by having an exacerbation), and never trained on. Every 100 steps the AUROC on their days is
  checked; training stops after 5 checks without improvement or at 3,000 steps, and the weights from
  the best check are kept (not retrained on all training patients).
  Fallback (locked): if the held-back days have no exacerbation day, AUROC cannot be computed, and the
  model trains the full 3,000 steps.
- **Output:** a probability per day; flagged if 0.5 or more.

### 6.5 LSTM (locked 17 Oct 2026)
Answer: the RNN's design (6.4) with only the cell changed to `torch.nn.LSTM` (gated cells that can keep
a longer memory). Same input, 10-minute learned compression, forward only, 2 layers x 64, last-step
read-out, half-exacerbation batches, flag at 0.5, early stopping (20% held back, AUROC, every 100 steps,
stop after 5 without gain, at most 3,000 steps, best weights kept).

### 6.6 gast
GAST v1 (locked 17 Oct 2026): `GAST_DESIGN.md`. It replaces the placeholder `models/gast.py` in these
experiments (the placeholder answers given earlier the same day no longer apply).

- **Input and compression:** as 6.4 (10,080 minutes x 14 numbers -> learned 10-minute blocks -> 1,008
  steps).
- **Network:** `models/gast.py`, 64 wide, 2 blocks, each a gap-aware decay layer (a decaying memory of
  the last real readings and how stale they are) followed by self-attention with 4 heads. Each step may
  only look at earlier steps (causal), like RNN and LSTM.
- **Read-out and training:** as 6.4: last step, half-exacerbation batches, flag at 0.5, early stopping
  (20% held back, AUROC, every 100 steps, stop after 5 without gain, at most 3,000 steps, best weights
  kept).

### 6.7 Experiment B: prediction (locked 17 Oct 2026)
Answers: the same six models, settings and 7-day input as detection; only the label and the samples
change.

- **Question:** at the end of day d, does a new episode begin on d+1, d+2 or d+3? The onset is the
  first recorded day of every episode.
- **Samples:** days with at least one HRV reading (pass 2) that are **not inside an episode** (the
  question is about a new onset). The last 3 days of each patient's watch data are dropped: their next
  3 days are not fully observed.
- **Input:** the 7-day window ending at the end of day d (section 3); nothing after day d.
- **Scoring:** per day as in section 5, plus calibration; an onset counts as **caught** if any of the 3
  days before it was flagged; false alarms per 30 days.

### 6.8 TimesFM 3 (locked 17 Oct 2026, workstation session)
Answers: frozen weights + a learned head; all 7 signals as variates; run after the other models.

- **Input:** the 7-day window, every minute, each of the 7 signals one TimesFM variate (7 x 10,080
  minutes = 315 patches of 32), values as recorded (TimesFM normalises each series itself). A blank is
  masked with TimesFM's own per-point masks: nothing is filled.
- **Network:** TimesFM 3.0 (331 M parameters, local weights `TIMESFM_CHECKPOINT`), frozen. Its
  attention is causal, so the last patch has seen the whole window.
- **Read-out:** each variate's transformer output at the last patch (7 x 1,280 numbers) -> one learned
  logistic layer (+ the scaled clinical values for `copd_clinical`).
- **Training:** only the logistic layer learns, exactly as 6.4: half-exacerbation batches, Adam 0.001,
  early stopping on held-back patients, flag at 0.5. The frozen outputs are computed once per day and
  cached (`model_data_exac/timesfm3_out.npy`); they are the same in every split.
- **Order:** the user's instruction (17 Oct 2026): run always_no, XGBoost, CatBoost, RNN and LSTM first
  on every set and both tasks, then GAST and TimesFM 3.

### Not in this task
hmm, nlssm, pf, rsdpf, gru_ode_bayes, cd_gamma_dglm, ossa, pinode: they rebuild or forecast
HRV values and give no exacerbation output. The logistic-regression reference was dropped (it means
little on 70,560 raw columns).

## 7. Still open

- The other 8 models (TimesFM 3 was added on 17 Oct 2026, 6.8): left out of the first runs; the user designs them after the first results
  (locked 17 Oct 2026).
- Class balance and threshold were decided per model: weight up / 0.5 for the tree models;
  half-exacerbation batches / 0.5 for the networks.

## 8. Status of the code

Built 17 Oct 2026 from this log: `14_exac_data.py` (minutes and both label sets), `run_exac.py` (both
tasks, every model, scores, intervals, calibration, slices), `models/gast_v1.py` (GAST_DESIGN.md).

The data build: 20,050 stored days (0.81 GB of minutes), 15,691 with HRV.

| group | detection days | exacerbation days | episodes | prediction days | days before an onset | onsets |
|---|---|---|---|---|---|---|
| AIIMS adults | 9,338 | 125 | 38 | 9,009 | 97 | 36 |
| AIIMS children | 4,981 | 8 | 8 | 4,881 | 25 | 10 |
| COPD | 1,372 | 11 | 11 | 1,306 | 29 | 10 |

Tested on the Mac: the date parsers; the data self-checks; AUROC and AUPRC against scikit-learn (equal
to 1e-16 on 500 random cases with ties); `run_exac.py check` on the COPD set for both tasks (every model
except XGBoost and CatBoost, which have no working library on the Mac); a full save. Not yet run: any
real experiment.
