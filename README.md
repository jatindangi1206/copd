# COPD clinical + wearable analysis

Self-contained. Everything it reads is under `data/`, everything it writes goes to
`figs/` (and `figs_dark/` when run with THEME=dark) and `numbers/`.

```
data/
  COPDAI_DATASHEET_01.xls   clinical datasheet: Baseline, Coding, exacerbation
  raw/<pid>/<vital>/...     per-vital export, 33 patients
  wearable/<pid>.csv.gz     1-minute master per patient, one row per recorded minute
codings.py                  the datasheet's coding scheme and its repairs  (--check)
common.py                   paths and plot style
02_clinical_eda.py          baseline clinical EDA
03_data_presence.py         how much data each vital has
04_hrv_per_patient.py       raw HRV over time, per patient
05_missingness.py           HRV gaps and coverage
06_hrv_distributions.py     HRV distributions
07_sleep_and_events.py      HRV in sleep, HRV around each exacerbation
08_sleep_stages.py          deep / light / almost-awake sleep and HRV (per sleep block)
09_clinical_profile.py      datasheet completeness, symptoms/history, lung tests, imaging, treatment
11_model_data.py            model_data/: 10-min modelling table, segments, imputation masks, splits (self-checks)
10_report.py                COPD_EDA_Report.html (light) + COPD_EDA_Report_dark.html
results.md                  what each script found, script by script
run_models.py               runs the imputation and forecasting models (see below)
models/                     one file per model + shared data/scoring (models/data.py)
```

`data/`, `model_data/`, `results/` and `clinical_table.csv` hold patient-level data
and are in `.gitignore`: copy them to a new machine separately.

Setup (Python >= 3.10):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # EDA and report
pip install -r requirements-models.txt    # models (on a GPU box, install torch for its CUDA first)
```

Run in order:

```bash
python codings.py --check
for T in light dark; do for s in 02_clinical_eda 03_data_presence 04_hrv_per_patient 05_missingness 06_hrv_distributions 07_sleep_and_events 08_sleep_stages 09_clinical_profile 11_model_data; do
  THEME=$T python $s.py
done; done
python 10_report.py
```

## What the data is

33 patients with both a datasheet row and wearable data. Every wearable patient
has a datasheet row; 8 further datasheet patients (c034–c041) have no wearable
data and are excluded.

17 exacerbation episodes are dated, across 12 patients; 16 of the 17 fall inside
that patient's wearable coverage. Three patients (c005, c032, c037) have an event
recorded with no date.

## Known problems in the datasheet

All verified, all handled in `codings.py`:

| problem | evidence |
|---|---|
| systolic and diastolic BP columns swapped | systolic < diastolic in 32 of 32 rows |
| c025 age recorded as 7 | 63 kg, 160 cm, BMI 24.6 (flagged in `codings.py`; the reports show it as recorded) |
| BODE does not match its own components | agrees in 4 of 18 rows; BODE vs 6MWD r = +0.01 |
| `AVG_DUR_EACH_EP_DAYS` mixes units | 8 of 28 entries are minutes in a days column |
| `SMART_WATCH_PRV_0`, `TRTMNT_PLAN_0` constant | no information |
| `AEC_0`, `EBC_0`, `INTRPTN_SPIROMETRY_0` empty | 0% populated |

Codes are not model-ready as shipped: the sheet uses **Yes=1, No=2, None=0**, so raw
values imply No is twice Yes. `codings.py` remaps them and treats ND / NK / NA as
missing rather than as categories.

## Models

Imputation first, then forecasting, on the segments in `model_data/` (x = 180 min).

```bash
python run_models.py --list                        # models, families, libraries, tasks
python run_models.py impute xgboost                # hide mask_random, rebuild, score (MAE, RMSE)
python run_models.py impute lstm --mask block      # harder test: whole runs hidden
python run_models.py forecast catboost             # last 20% of each segment; train gaps filled first
python run_models.py forecast lstm --imputer xgboost
python run_models.py daily cd_gamma_dglm           # day-level forecasting
python run_models.py impute all                    # every model for the task, one by one
python run_models.py check                         # every installed model on 6 series, tiny training
```

Cores: it shows how many cores are free and asks how many to use; with no answer
it leaves 2 free for the UI. `--cores N` skips the question. GPU is used when
available (`--device cpu` to force CPU). Results: `results/<task>/<run>/` and one
line per run in `results/summary.csv`.

| model | built on |
|---|---|
| xgboost, catboost | xgboost, catboost |
| rnn, lstm | torch.nn.RNN / LSTM |
| hmm | hmmlearn (+ missing-aware forward-backward) |
| nlssm | filterpy unscented Kalman filter + RTS smoother |
| rsdpf | torch, after Li et al. 2023 (no library has regime switching) |
| pf | `particles` (bootstrap filter + backward sampling) |
| gru_ode_bayes | authors' code, vendored in `models/vendor/` (MIT) |
| cd_gamma_dglm | numpy/scipy (PyBATS has no Gamma DGLM) |
| ossa | numpy |
| pinode | torchdiffeq |
| gast | torch, **first version** - replace with the team's design |
| timesfm3 | `timesfm` 3.0.2; weights download on first run, non-commercial licence |

Every model returns full series (imputation) or horizons (forecasting) through
the same two functions, so all are scored identically (`models/data.py`).
