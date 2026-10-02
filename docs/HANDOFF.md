# Handoff: everything from the first Claude session (28-29 Sep 2026)

This project was built in one long Claude Code session on the user's Mac
(`analysis_final/copd_clinical`, then `~/Desktop/copd-hrv`, now `analysis_final/copd-clinical`).
That session cannot be opened on the GPU workstation, so this file carries what it knew.
CLAUDE.md has the rules. This file has the history, the reasons behind them, and the details.

---

## 1. The project in one paragraph

This is a COPD pilot cohort of 41 patients (c001–c041 in the datasheet; export of 29 Sep 2026; c034 has no watch data), with a clinical datasheet from enrolment and a smartwatch
(GOQii device) worn for weeks to months. The target signal is HRV: the watch reports SDNN about
every 10 minutes, with no unit. The EDA and the HTML report (light and dark) are finished. The
modelling data is built. 14 models plus 3 references are coded behind one runner. The models ran once on the GPU workstation on 29 Sep; those results were deleted on 2 Oct and the code fixed (see CLAUDE.md Status), so the next step is a full rerun.

This cohort is separate from the other cohorts in the parent `analysis_final/` repo (AIIMS a/p-series,
and the GOQii cohort the user calls "ACHE Cohort"). Two findings from comparing them:
- COPD does **not** have AIIMS's low/mid HRV split. All 26 classifiable COPD patients are the "low"
  type, so don't port that split here.
- Device facts do carry over: a hard cap at 129, a jump in counts at 120, and a step-count cliff at 10.

## 2. Timeline of what the user asked for, and what was done

1. **"understand this fully"**: read the whole `copd_clinical` folder.
2. **Report rewrite.** The user first asked for full explanations: Spearman with the GFG link, an
   hours-studied example, every term defined, and HRV = SDNN every ~10 min with no unit.
   **Then reversed:** "you have written too much ... keep it minimalist and jargon free."
   Other feedback in the same message:
   - The report must use **exactly** the PNGs in `figs/`, and every PNG there.
   - Story order: cohort EDA → pipeline → imputation → forecasting.
   - "we have to make this report ready and not write this is left ... if there is something
     left we leave space for it."
   - Keep the repo.
3. **"why you are removing the age 7 do not assume anything go according to data."** c025's
   age of 7 is shown as recorded. BODE is used as recorded. Nothing is "cleaned" unless the sheet's
   own coding requires it.
4. Added **stage-based sleep analysis** (08), a **datasheet review** (09) and **two reports**,
   light and dark.
5. **Modelling plan (core, do not change):**
   - "first we will do imputation we will artificially remove hrv and try to fill it with
     bidirectional methods then check it against the removed values."
   - "randomly remove 20 percent hrv and on the basis of hrv trajectory plus other vitals try to
     reconstruct."
   - "then we will do the forecasting ... remove the later part to predict, not from middle."
     80:20. Also **day-based forecasting**.
   - Segments: "if consecutively x no of hrv readings are missing we will make it a segment."
     x is decided from cohort-level and individual temporal missingness, is not a hard-coded
     cohort rule, and will become patient-level as more data arrives.
   - "if you feel like making it better do it but do not change the core first imputation and
     then forecasting."
6. **"the report shouldn't look AI generated."** It was restyled plain: serif body, "Figure N."
   captions, a "Summary." paragraph, and no cards, badges, gradients or bold lead-ins.
7. **results.md.** The user asked for a raw, human-looking notes file ("like i ran each py file
   and then written the result"). It is plain lowercase notes, one section per script, with the
   command and what it printed. Claude declined to add fake typos to disguise authorship. It is
   simply plain, and that should stay true.
8. **Report change round:**
   - `clin_vs_wearable`: the two axes in two colours (clinic = orange on x, watch = blue on y).
   - The gap graph, drawn the user's way. Take the timestamps where an HRV value
     arrived, subtract the previous arrival's timestamp to get the "minute difference"
     (10, 10, 10, 20, ...), plot it as a histogram and look for the bend. The report has a
     worked-example table of this.
   - Model set plus the rationale: 10 HRV characteristics from the literature (H1–H10), tested
     on this data, 8 held. Then a corpus of methods with their assumptions, strengths and
     weaknesses, keeping only models that don't violate the 8, natively or through a published
     modification. "the ssm + transformer approach is my own creation."
   - Research doc (characteristics in the first tab):
     https://docs.google.com/document/d/1V8rGaJFxRt5ccTJUKq4JcynNGbu9lqcs7h0J0KMzWDA/edit?usp=sharing
9. **"H9 obviously means for forecasting not for bidirectional, this is pilot cohort."**
10. **Threshold plot.** The first request was 12 h / 6 h / 3 h resolutions with no cutoff line and
    x left blank for physiology experts. The user then said "lets keep it 180 only but still show
    the graphs" and "keep this version only just add these 3 resolution graphs." Result: fig 05e
    has three stacked histograms with no cutoff lines, and the text says x = 180 min, set for this
    pilot and to be reviewed with experts. The top byline and the "Generated ..." footer were removed.
11. **Build the models** "one by one ... look online for implementations ... use it, don't
    reinvent the wheel ... shortest, simplest." Plus a run file, and the **core policy**:
    "whenever writing modelling code it shouldn't use all the cores ... figure out how many are
    empty, tell you and ask how many to use, and if not always leave 2 cores for UI."
    - "do not download the models here, just code; I will download the foundation model on my
      heavy workstation."
    - "don't run tests too." The model check was stopped after the three references passed.
12. Moved to `~/Desktop/copd-hrv`, ran `git init` and made the first commit. The user zipped it and copied it to the
    workstation through Google Drive. No GitHub remote existed at the time.

## 3. Data

Nothing here is in git. Copy `data/`, `model_data/` and `clinical_table.csv` by hand.

```
data/COPDAI_DATASHEET_01.xls     sheets: Baseline, Coding, exacerbation
data/copd/<pid>/<vital>/...      per-vital export of 29 Sep 2026, 40 patients (the one in use)
data/raw/<pid>/<vital>/...       older export (to 18 Sep), 33 patients, no longer used
data/wearable/<pid>.csv.gz       1-minute master per patient (one row per recorded minute)
clinical_table.csv               written by 02; one row per patient, values as recorded
```

**Datasheet traps** (all handled in `codings.py`, which self-tests with `--check`):

| Trap | How it is handled |
|---|---|
| Coded Yes=1, No=2, None=0 | Remapped to 1/0/0 |
| ND / NK / NA (not done / not known / not available) | Blank, not categories |
| BP_mmHg_SYS and BP_mmHg_DIA swapped (systolic < diastolic in 32/32 rows) | Un-swapped |
| mMRC written as roman text with half-grades | Mapped to midpoints 0.5 … 3 |
| GOLD A/B/E | Not ordinal: one-hot plus an E flag |
| c025 age 7 (63 kg, 160 cm, BMI 24.6) | `load_baseline()` puts `age_clean` = NaN, but **the reports and model data use AGE as recorded (7)**, on the user's instruction |
| BODE doesn't match its own components (agrees in 4/18 rows) | Used as recorded; `bode_reconstructed()` exists but is not used in the report |
| AVG_DUR_EACH_EP_DAYS mixes minutes and days | Minute entries dropped |
| Constant or empty columns | Constant: SMART_WATCH_PRV_0, TRTMNT_PLAN_0. Empty: AEC_0, EBC_0, INTRPTN_SPIROMETRY_0 |

- **Cohort = every patient in the datasheet (41).** The user was explicit: report the full cohort,
  never shrink the headline to those with data; list what is missing per patient instead. Watch
  analyses use the 40 with a watch file (`codings.wearable_pids()`); c034 has no export folder and is
  shown as "no watch data" (data-presence figures, HRV grid, text). data/copd holds 41 entries, but
  one is processed_users.txt.
- data/wearable/ is rebuilt from the export with `curate_dataset.master(pid, folder)` from the parent
  analysis_final repo (the same function that built the first files; verified to reproduce them).
- The 29 Sep export moved c001's and c027's sleep blocks 5 h 30 min earlier than the 18 Sep export
  (same blocks); the new times fall at night far more often, so they look corrected. Nothing else moved.
- No HRV at all for c002, c019, c031–c033 and c035–c041, which leaves 28 patients with HRV.
  c035–c041 send very little of anything (2–116 heart-rate readings each).
- **Exacerbations:** 17 dated episodes in 12 patients. 3 more patients (c005, c032, c037) have an event
  with no date. 12 of the 17 dates have HRV within 2 weeks, and 16 of 17 fall inside watch coverage.

**Open, not decided:** `02_clinical_eda.py` still uses **post**-BD FEV1 % (`fev1_pct`) and
`smoke_index` in the correlation plots. The earlier session planned to switch to pre-BD FEV1,
because post-BD is only 48% filled, but that edit never landed. The model data already uses
pre-BD FEV1 (`c_fev1_pre`). Ask the user before changing 02.

**Enrolment** (added 29 Sep): the sheet has no enrolment-date field, so
`codings.load_enrolment()` uses the date the smart watch was provided (the DATE after
`SMART_WATCH_PRV_0`). c022's entry is `01-06-20260`, so the treatment-plan date beside it is used
(1 Jun 2026). Figures 03c-03e count only data from enrolment to the export end (18 Sep 2026). Figure 07b
also shows the day's median step count and hours of recorded sleep around each exacerbation.

## 4. Key numbers (from results.md, and they match the report)

- **Readings (29 Sep export):** HR 1,332,839 · HRV 106,457 · temperature 189,815 · steps 56,468 · SpO2 2,044 (too few to use).
- **Cohort:** 37 men and 4 women. Age 7–83 as recorded, median 65. GOLD A 19, B 19, E 3.
- **HRV values** have two groups: a peak in the low 30s, and a block at 120–129, where 129 is the
  device maximum and 23% of readings are ≥ 120. Per-patient medians run from 41 to 107.
- **Gaps between HRV readings:** 86.7% are exactly 10 min. 98.7% are ≤ 3 h and 99.4% are ≤ 6 h.
  Median coverage is 35%.
- **Sleep:** HRV is lower inside recorded sleep for 21 of 27 patients. Deep vs light sleep blocks
  show no consistent effect (HRV lower in deep for 13 of 25). Sleep stages come per block, not per
  minute, so this analysis is block-level.
- **Clinic vs watch:** clinic pulse against watch median HR, r = +0.40, with the watch about 10 bpm lower.

## 5. Modelling data (`11_model_data.py` → `model_data/`)

**Settings** are at the top of the script and are echoed in `model_data/config.json`:

- `X = 18` slots, i.e. 180 min. A segment ends where HRV is missing for more than 18 consecutive
  10-minute slots. In minute-difference terms, a new segment starts when the difference is
  ≥ 200 min.
- `MIN_SEG_HOURS = 24`.
- `MASK_PCT = 20`.
- `TRAIN_PCT = 80`.
- `SEED = 0`.
- `CANDIDATES` = [1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 72, 144]. These are only for the scan table and fig 11a.

Setting `X = None` writes everything that doesn't depend on x and leaves the segment and mask
columns blank.

**Outputs:**
- `model_10min.csv.gz`: 339,350 rows, one per patient per 10-min slot from the first to the last HRV reading.
  - Watch columns: pid, time, hour, hrv (blank = no reading, **nothing is filled**), hrv_n, hr, hr_n,
    temp, spo2, steps, sleep_frac, steps_active_frac.
  - Labels: exac_day, missing_run, segment, split (train/test), mask_random, mask_block.
  - Clinical columns (as recorded): c_age, c_sex, c_bmi, c_mmrc, c_gold, c_cat, c_fev1_pre,
    c_walk_dist, c_spo2, c_exac_12m.
- `model_daily.csv`: one row per patient-day, split 80/20 per patient: 1,233 train and 319 test days.
- `segments.csv`, `patients.csv`, `threshold_scan.csv`.

**Results of the build:**
- 259 segments across 26 patients. 60% of HRV readings fall inside segments, and 29% of slots
  inside segments are missing.
- `mask_random` hides 12,677 of 63,381 observed readings inside segments. `mask_block` hides
  13,478, in whole runs whose lengths are drawn from the real gaps.
- The forecast split gives 71,206 train and 17,926 test slots. Test always comes after train.
- The script's self-checks assert three things: masks only fall on observed HRV inside segments,
  test comes after train, and no run longer than X exists inside a segment.

## 6. The ten HRV characteristics (research doc, first tab)

| # | Characteristic | Status |
|---|---|---|
| H1 | Bounded and positive | kept |
| H2 | Two modes (day state / night state) | kept |
| H3 | Strong lag-1 memory | **dropped**: lag-1 autocorrelation is about 0.41, which is moderate |
| H4 | Drifting baseline | kept |
| H5 | 24-hour (circadian) rhythm | kept |
| H6 | Heavy-tailed spikes | **dropped**: the watch drops bad readings, so they show up as gaps |
| H7 | Irregular timing | kept |
| H8 | Two kinds of missingness (short random gaps / long watch-off gaps) | kept |
| H9 | Real time: no data after t | kept, **for forecasting only**; imputation may use both sides |
| H10 | Per-patient baseline | kept |

## 7. The model code

**Runner:** `run_models.py`
- `--list`
- `impute|forecast|daily <model|all>`
- `check`
- Options: `--mask random|block`, `--imputer linear` (fills the training gaps before forecasting),
  `--cores`, `--device`, `--seed`, `--quick` (first 6 series, tiny training).
- Output: `results/<task>/<model>__<mask|imp-imputer>/predictions.csv.gz` and `metrics.json`, plus
  one line per run in `results/summary.csv`.

**The contract every model implements** (`models/data.py`):
- `impute(series, ctx)` returns a list of full-length arrays with every blank filled.
- `forecast(history, horizons, ctx)` returns a list of arrays, `horizons[i]` values each.
- `Series(id, pid, x, truth, score)`. `x` has hrv, hrv_obs, the covariates, t and time/date.
- `ctx` holds device, cores, quick, seed, period (144 for 10-min data, 7 for daily) and covs.
- Helpers: `lin_fill`, `zstats`/`to_z`/`from_z` (per-patient log standardisation, H10), `phase`,
  `future_t`, `cov_matrix`.
- `COVS = hr, temp, steps, sleep_frac, steps_active_frac`.

**Scoring** (`data.score`):
- MAE, RMSE, `mae_below_120`, `mae_120_up`, `mae_per_patient_median` and `n_missing_pred`.
- Scores come only from real readings: the hidden ones for imputation, the test part for forecasting.

**References:** `linear` (impute), `last_value` and `patient_median` (forecast).

| Model | File | Built on | Notes |
|---|---|---|---|
| xgboost, catboost | gbm.py | the libraries | One global model on per-patient z-scored log HRV. **Impute:** K=6 neighbours each side, the nearest real value and its distance on each side, vitals, time of day. **Forecast:** direct multi-step over (origin, h) pairs. |
| rnn, lstm | rnn.py + seq.py | torch | Window 288 slots. Extra 20% hiding during imputation training. Bidirectional for impute, causal rollout for forecast. |
| hmm | hmm.py | hmmlearn GaussianHMM, 3 states | Own forward-backward that skips blanks, since hmmlearn can't take NaN. |
| nlssm | nlssm.py | filterpy UKF + RTS | State [level, c1, c2]. Bounded sigmoid observation between LOW and HIGH. |
| pf | pf.py | `particles` bootstrap filter + `backward_sampling` | Log-AR around a daily rhythm. Uses `np.random.seed` because SMC has no seed argument. |
| rsdpf | rsdpf.py | plain torch | After Li et al. 2023 (arXiv:2302.10319). K=2 regimes, soft resampling ALPHA=0.5. Written by hand because PyDPF has no regime switching and needs Python ≥ 3.12. |
| gru_ode_bayes | grude.py + vendor/gru_ode_bayes.py | authors' code (MIT), vendored | ipdb removed, Euler solver. Forward model plus a reversed-time model for imputation. |
| cd_gamma_dglm | dglm.py | numpy/scipy | Gamma DGLM, log link, discount DELTA=0.98, trigamma conjugate update, smoother. Hand-written because PyBATS has no Gamma DGLM. |
| ossa | ossa.py | numpy | Iterative SSA gap fill. Trailing-window SSA plus linear recurrence for forecasting. |
| pinode | pinode.py | torchdiffeq (rk4) | dx/dt = −k(x − c(t)) + g(x, time, vitals). Integrates forward and backward for imputation. |
| gast | gast.py + seq.py | torch | **Placeholder**: FFT-decayed "gap-aware SSM" plus a TransformerEncoder. The user designed this model, so ask for their architecture before touching it. |
| timesfm3 | timesfm3.py | `timesfm` 3.0.2 (module `timesfm3`) | `TimesFM3Evaluator(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=32, device))`, then `predict_batch(...)`. Weights download from Hugging Face on first run and are **non-commercial**. Imputation blends forward and backward forecasts over each gap. `USE_TIME_COVARIATE=False`. `check` skips it. |

**Core policy:** `models/cores.py` shows total, busy and free cores and asks how many to use. The
default is free cores capped at total − 2. It sets the OMP/MKL/OPENBLAS/NUMEXPR/VECLIB thread
variables before numpy is imported, and sets torch threads. Every new modelling script must go
through it. `--cores N` skips the question.

**Tested so far:** only linear, last_value and patient_median passed `check`, on the Mac. Everything else
is written but has never run end to end. Expect some shape and API bugs.

## 8. Gotchas already hit

- **macOS:** importing torch before xgboost segfaults (two OpenMP runtimes). `run_models.py` imports
  xgboost first. Linux is probably fine, but keep the order.
- A crash that prints nothing when piped: run `python -u -X faulthandler run_models.py check`.
- `particles`: there is no `backward_sampling_ON2`; use `pf.hist.backward_sampling(M)`. `SMC()` takes no `seed`.
- **Python 3.10 f-strings** can't hold backslashes or nested triple quotes. `10_report.py` is one big
  f-string, so use typographic quotes or lift the block out.
- `git init -b main` failed on the Mac's old git. Use `git init && git checkout -b main`.
- On the GPU box, install torch for the right CUDA version from pytorch.org **before**
  `pip install -r requirements-models.txt`.

## 9. Report (`10_report.py`)

- **Output:** `COPD_EDA_Report.html` (light, from `figs/`) and `COPD_EDA_Report_dark.html` (from `figs_dark/`).
  Figures are embedded as base64. 28 figures each.
- **The 1:1 rule:** the script asserts that the set of figures used equals the set of PNGs in the
  folder. A new figure therefore needs a PNG in **both** `figs/` and `figs_dark/`
  (made with `THEME=dark`), plus a `fig(...)` call.
- **Sections:**
  1. The data
  2. What the datasheet holds
  3. Who the patients are
  4. Lung function, tests and treatment
  5. How clinical measures relate
  6. How much watch data there is
  7. Gaps in HRV
  8. HRV values
  9. Sleep
  10. Exacerbations
  11. Modelling data
  12. Models (H1–H10 table and model-set table)
  13. Imputation
  14. Forecasting
  15. Conclusions

  Then a Terms table.
- **Empty slots waiting for results:**
  - §13: Method, a Results table (8 empty rows: Method / Random MAE / Random RMSE / Block MAE /
    Notes), a Figure and a Takeaway.
  - §14: the same, with columns Model / Level / Horizon / MAE / Notes.
  - §15: a "Model results" slot.
- **Filling them:** the `slot(...)` calls and the `[("", ...)] * 8` rows in `10_report.py` are what
  get replaced. Read the numbers from `results/summary.csv` or `metrics.json`; don't type them in.
  A results figure should come from a small script (e.g. `12_model_results.py`) that writes
  into `figs/` and `figs_dark/` like the others. Then add a `## imputation` / `## forecasting`
  section to `results.md` in the same plain style.
- **Style:** minimal, no jargon, no "this is left / will be done" wording, no byline or date,
  no footer. Captions give the title, what the figure shows and the takeaway. HRV numbers never
  carry "ms".

## 9b. Model results

The 29 Sep run (18 Sep model data, 25 patients, classification on 31 patients) was deleted on 2 Oct on
the user's instruction; everything is rerun from scratch. Its findings that drove the fixes: pinode
imputation and cd_gamma_dglm imputation unusable, nlssm 10-min forecast negative, ossa daily infinite, rnn/lstm/gast
forecasts drifting (all fixed in code, untested), and `check` too weak (hardened). Only block-mask
imputation clearly beat its reference. The user wants report sections 13-15 written as process / input /
output / results only: no interpretation, no claims, no '% better'. Section 16 is the conclusions.

## 10. What to do next, in order

1. Copy `data/`, `model_data/` and `clinical_table.csv` into the repo on the workstation.
2. Set up the environment: `python -m venv .venv && source .venv/bin/activate`, install torch for
   the workstation's CUDA, then `pip install -r requirements.txt -r requirements-models.txt`.
3. Run `python run_models.py check` (now fails on impossible predictions) and fix failures one model at a time,
   keeping each fix minimal and in the model's own file.
4. TimesFM: weights are already on the workstation; `export TIMESFM_CHECKPOINT=<dir>` so nothing downloads.
5. **Imputation:** `impute all`, then `impute all --mask block`.
6. **Forecasting:** `forecast all` (train gaps filled by `linear`, or by the best imputer via
   `--imputer`), then `daily all`.
7. `python 12_exac_classify.py`, then `12_model_results.py` and `10_report.py` (both themes); write the results into `results.md`.
8. Later: the user's gast architecture; per-patient x; review of x = 180 with physiology experts.

## 11. The original session

The full transcript is on the Mac at
`~/.claude/projects/-Users-jatindangi-Desktop-TSB-KCDH-analysis-final/dba33b67-7321-49ea-93fb-af5e7bbcf774.jsonl`
(28 MB). It contains patient-data output, so treat it like the data.
