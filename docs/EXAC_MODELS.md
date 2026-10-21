# Exacerbation experiments: what every module takes in, does and gives out

A reference for each part of the pipeline, with the parameters as decided (17 Oct 2026). Why each decision
was made: `EXAC_DESIGN.md`; GAST's background: `GAST_DESIGN.md`. Values marked **(set by Claude)** were not
asked about: they are defaults chosen while building and can be changed.

## The pipeline at a glance

```text
watch minute files + exacerbation sheets
        |
        v
14_exac_data.py  ->  model_data_exac/   (every minute with heart rate, per day; labels for both tasks)
        |
        v
run_exac.py: pick task + set -> split patients -> build each model's input -> train -> score test days
        |
        v
results/exac/<task>/<set>/<model>/test<NN>_seed<S>/   +   results/exac/summary_<task>.csv
```

## 1. Data module: `14_exac_data.py`

| | |
|---|---|
| **Inputs** | COPD 1-minute files (`data/wearable/`), AIIMS 1-minute files (`../final_data/aiims/`, a-series and p-series), the COPD datasheet (exacerbation sheet), the AIIMS adult and paediatric exacerbation sheets, `clinical_table.csv` |
| **What it keeps** | Only minutes with a heart-rate reading of 0-300 (pass 1). On those minutes, 7 signals exactly as exported: HRV, HR, temperature, SpO2, steps, inside a sleep record (0/1), inside a step record (0/1). Nothing averaged, filled, moved or cut into segments. |
| **Outputs** | `minutes.npy`: (days, 1,440 minutes, 7 signals), blank = empty. `days.csv`: one row per stored day with its labels. `static.csv`: the COPD clinical table. `episodes.csv`, `dropped.csv`: every parsed and every rejected sheet entry. |
| **Runs** | Once, on the Mac (its inputs are not in git). About 40 seconds. Self-checks at the end. |

**Labels it writes**

| | detection | onset prediction |
|---|---|---|
| question | is this day inside an exacerbation episode? | does a new episode begin on one of the next 3 days? |
| sample days | every day with at least 1 HRV reading (pass 2) | days with HRV, not inside an episode, not among the patient's last 3 days of watch data |
| positive | the day is an episode day | an episode's first day falls on d+1, d+2 or d+3 |
| event id | the episode | the upcoming onset (the nearest one, if two fall within 3 days) |

**Episode rules:** adult ranges count every day; adult single dates, COPD dates and children's onsets
count as one day; a- or p-series patients in neither sheet have only normal days; patients with an
unreadable date are left out (c005, c032, c037 and one child). Paediatric Excel dates are read with day
and month swapped back; a children's row counts only if onset <= visit <= onset + 45 days.

**Size (17 Oct build):** 20,050 stored days (0.81 GB), 15,691 with HRV.

| group | detection days | exacerbation days (episodes) | onset days | days before an onset (onsets) |
|---|---|---|---|---|
| AIIMS adults | 9,338 | 125 (38) | 9,009 | 97 (36) |
| AIIMS children | 4,981 | 8 (8) | 4,881 | 25 (10) |
| COPD | 1,372 | 11 (11) | 1,306 | 29 (10) |

## 2. Shared by every model: `run_exac.py`

**One sample** = one sample day of one patient, with a **7-day window**: that day plus the 6 calendar
days before it = 10,080 minutes, oldest first, ending at 23:59 of the classified day. Nothing after the
day is ever used. Days with no data inside the window are blank (they are not dropped).

| choice | value |
|---|---|
| task | `--task detect` or `--task onset` |
| data set | `combined` (COPD + AIIMS), `copd`, `copd_clinical` (COPD + its clinical table) |
| split | by patient: in each stratum (COPD / AIIMS adult / AIIMS child x has an exacerbation day or not), the test share is drawn at random. Same patients for every model and both tasks. |
| train:test | 80:20, 90:10, 70:30, 60:40, 50:50 |
| seeds | 0, 1, 2, 3, 4 per ratio (25 runs per model per set per task) |
| history | 7 days for every model; GAST is also run with 3 and 14 (`--days`) |
| threshold | a day is flagged if its probability is 0.5 or more (every model) |

**Training setup for the networks (RNN, LSTM, every GAST):**

| | |
|---|---|
| class balance | every batch is half positive days, half negative days, drawn at random from the training patients |
| optimiser | Adam, learning rate 0.001, batches of 32 days, gradient clipping at 1.0 **(clipping set by Claude)** |
| loss | binary cross-entropy on the output for the classified day |
| early stopping | 20% of the training patients held back (drawn by the same strata) and never trained on; their AUROC is checked every 100 steps; stop after 5 checks without improvement or at 3,000 steps; keep the best weights; if the held-back days hold no positive, train all 3,000 steps |
| input scale | each signal divided by its largest value in the training patients (a change of units, not an average); clinical values the same way |

## 3. The models

### 3.1 always_no (reference)
| | |
|---|---|
| sees | nothing |
| outputs | probability 0 for every day: never flags |
| purpose | the floor: shows the accuracy of never raising an alarm (about 99%, because positives are rare) and AUROC 0.5 |

### 3.2 XGBoost
| | |
|---|---|
| sees | one row per sample day: every raw value of the 7-day window side by side, 10,080 minutes x 7 signals = **70,560 columns** (minute 0 HRV, minute 0 HR, ... minute 10,079 step record). Blanks stay blank. `copd_clinical` adds the 28 clinical values as extra columns. |
| how it runs | 400 boosted trees, depth 4, learning rate 0.05, 80% of rows and 80% of columns per tree, histogram method, GPU if present. Exacerbation days weighted up (weight = normal days / positive days in training). At each split it learns which side a blank goes to. |
| outputs | a probability per test day; flagged at 0.5 |
| note | each column is one clock minute of one window day: the trees cannot see that 10:00 and 10:01 are neighbours. Input for the combined set is about 4.6 GB. |

### 3.3 CatBoost
| | |
|---|---|
| sees | the same 70,560 raw values per day (+ clinical for `copd_clinical`) |
| how it runs | 400 symmetric trees, depth 4, learning rate 0.05, balanced class weights, GPU if present. A blank is treated as lower than every reading (CatBoost does not learn it). |
| outputs | a probability per day; flagged at 0.5 |
| note | slow on CPU with this many columns |

### 3.4 RNN
| | |
|---|---|
| sees | the 10,080 minutes in time order; per minute 14 numbers: the 7 scaled readings (blank = 0) and 7 "reading present" flags (+ the scaled clinical values on every minute for `copd_clinical`) |
| how it runs | (1) a learned compression layer: a 1-D convolution reading each block of 10 consecutive minutes and turning it into 32 numbers (weights learned, not an average): 10,080 minutes -> 1,008 steps. (2) `torch.nn.RNN`, 2 layers x 64 units, dropout 0.1 between layers, reading forward only (7 days ago -> end of the day). |
| outputs | the network's output after the last step (the day's final 10 minutes) -> sigmoid -> probability; flagged at 0.5 |

### 3.5 LSTM
Exactly the RNN (3.4) with `torch.nn.LSTM` cells, which have gates that can hold memory longer.

### 3.6 GAST v1 (`gast`)
| | |
|---|---|
| sees | **tokens**: one per minute in the window that has a heart-rate reading, at its real time (no grid, no blocks). Per token 24 numbers: the 7 scaled readings (blank = 0), 7 "reading present" flags, 7 x log(1 + minutes since that signal was last seen in the window), log(1 + minutes since the previous token), time of day as sine and cosine. **(Time of day set by Claude; it can be switched off.)** For `copd_clinical` the clinical values set the memory's starting state. |
| step 1: memory | a gap-aware state space with 32 channels. Between tokens each channel only fades: exp(-rate x real elapsed minutes); when a token arrives it is written in through a gate that depends on the token. Rates start spread from 1/(10 minutes) to 1/(the window), so some channels remember minutes and others days, then are learned. Computed exactly in one pass (associative scan). |
| step 2: hourly reading | the memory is read at the end of every hour of the window (168 readings for 7 days), faded from the last token before that hour; nothing is invented for hours without data |
| step 3: attention | one causal self-attention layer over the 168 hourly readings (2 heads): each hour may look only at earlier hours, with a learned bias per head that depends on how many hours apart they are |
| outputs | the reading at the end of the classified day -> one logistic output -> probability; flagged at 0.5 |
| size **(set by Claude)** | width 32, 32 state channels, 2 heads, dropout 0.1, attention time bias started at about -0.05 per hour |
| training | as section 2 (half-positive batches, early stopping) |

### 3.7 GAST ablations (same input, training and output as 3.6)
| model | what changes | question it answers |
|---|---|---|
| `gast_ssm` | no attention: the end-of-day memory reading goes straight to the output | does attention add anything? |
| `gast_attn` | no state space: each hourly reading attends directly to the tokens before it, with learned time embeddings (as in mTAN), then the same causal attention | is the gap-aware memory needed? |
| `gast_nogap` | no flags, no times since, no elapsed times; the memory steps by token count, not real time | does knowing the gaps help? |
| `gast_maskonly` | the readings are removed: only flags, times since, elapsed times and time of day | is the model only learning when the watch was worn? |
| `gast --days 3` / `--days 14` | 3 or 14 days of history instead of 7 | does history length matter? |

### 3.8 Two-head GAST (`gast2`)
| | |
|---|---|
| sees | as `gast`, on every day with HRV |
| how it runs | one shared GAST, two outputs: detection and onset. Loss = onset loss + 1 x detection loss (alpha = 1); each output only learns from days where its label is valid (onset: outside episodes and not the last 3 days). Half-positive batches use days positive for either task. Early stopping watches the mean held-back AUROC of the two. |
| outputs | two probabilities per day; scored on both tasks in one run |

### 3.9 Not in these experiments
hmm, nlssm, pf, rsdpf, gru_ode_bayes, cd_gamma_dglm, ossa, pinode, timesfm3 (they fill in or forecast HRV;
you design their exacerbation versions after the first results). The logistic reference was dropped.

## 4. Scoring module (for every model, on the test patients' days)

Reported for all test days together and for each group present (COPD, AIIMS adults, AIIMS children).

| score | meaning |
|---|---|
| accuracy | share of days called correctly (misleading alone: never flagging scores ~99%) |
| balanced accuracy | mean of sensitivity and specificity |
| sensitivity | positive days flagged / all positive days |
| specificity | negative days not flagged / all negative days |
| PPV | flagged days that were positive / all flagged days |
| NPV | unflagged days that were negative / all unflagged days |
| F1 | combines PPV and sensitivity |
| AUROC | how well probabilities rank positive days above negative ones (0.5 = chance); no threshold |
| AUPRC | the same focused on the rare positives (chance = share of positive days); no threshold |
| Brier score | mean squared difference between probability and outcome (0 = perfect) |
| events caught | detection: an episode counts if any of its days is flagged. Onset: an onset counts if any of the 3 days before it is flagged |
| false alarms per 30 days | flagged negative days per 30 negative days |
| lead time (onset) | days between the first flag and the onset (1-3), median |
| 95% intervals | for AUROC, AUPRC, sensitivity, specificity, PPV, Brier, event sensitivity and false alarms: 1,000 resamples of test patients (a patient's days stay together) **(1,000 set by Claude)** |
| calibration table | predicted vs observed rate in 10 probability bands (0-0.1 ... 0.9-1.0) |
| slices | scores in thirds of the test days by wear coverage of the window, longest stretch without heart rate in the day, and number of window days with data (thirds from the test days' own values) |

## 5. Outputs on disk

| file | content |
|---|---|
| `results/exac/<task>/<set>/<model>/test<NN>_seed<S>/predictions.csv.gz` | one row per test day: patient, group, date, label, event id, probability, flag |
| `.../metrics.json` | every score with its interval, per group; calibration table; slices; training record (steps, held-back AUROC, seconds, train / held-back / test patients) |
| `results/exac/summary_<task>.csv` | one line per run and group, all scores and intervals |
| console | at the end, tables of each score by model and train:test ratio (mean over seeds) |

## 6. Commands

```bash
python -m models.gast_v1                                  # GAST self-tests
python run_exac.py --list                                 # the models
python run_exac.py check [--task onset]                   # every model, tiny training on a small subset
python run_exac.py all --task detect                      # every single-task model, 5 ratios x 5 seeds
python run_exac.py all --task onset --set copd_clinical
python run_exac.py gast2                                  # both tasks
python run_exac.py gast --days 14
# options: --test 20 10 --seed 0 1 --cores N --device cpu|cuda --quick
```

## 7. Things worth knowing (not asked, but they matter)

- **Very few positives.** 144 exacerbation days in 57 episodes for detection, 151 days before 56 onsets
  for prediction. A 90:10 split may hold only one COPD patient and one child with an exacerbation, so
  single runs swing; read the 95% intervals and the mean over the 5 seeds.
- **Children are almost all negatives for detection** (8 positive days out of 4,981), so their detection
  scores will be the least stable; onset has more (25 days).
- **The threshold is fixed at 0.5** for every model. AUROC and AUPRC do not depend on it; sensitivity, PPV,
  false alarms and events caught do.
- **Missingness can be learned as a shortcut** (for example, the watch worn less during an exacerbation).
  `gast_maskonly` and the coverage slices show whether that is happening.
- **No retraining after early stopping:** a network keeps the weights from its best held-back check,
  trained on 80% of the training patients.
- **Onset events:** if two onsets fall within 3 days of one sample day, that day is tied to the nearer one.
- **Run time and memory:** tree inputs for the combined set are about 4.6 GB; CatBoost on CPU is slow;
  GAST holds up to 10,080 tokens per sample. Estimate from the first runs before the full grid
  (11 models x 5 ratios x 5 seeds x 3 sets x 2 tasks).
- **Not yet run:** everything above is built and checked; XGBoost and CatBoost have never run (the Mac had no
  working library for them).
