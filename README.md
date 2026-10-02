# COPD clinical + wearable analysis

Self-contained. Everything it reads is under `data/`, everything it writes goes to
`figs/` (and `figs_dark/` when run with THEME=dark) and `numbers/`.

```
data/
  COPDAI_DATASHEET_01.xls   clinical datasheet: Baseline, Coding, exacerbation
  copd/<pid>/<vital>/...    per-vital export (29 Sep 2026), 40 patients (no c034)
  raw/<pid>/<vital>/...     older export (18 Sep), 33 patients, not used
  wearable/<pid>.csv.gz     1-minute master per patient, one row per recorded minute
codings.py                  the datasheet's coding scheme and its repairs  (--check)
common.py                   paths and plot style
02_clinical_eda.py          baseline clinical EDA
03_data_presence.py         how much data each vital has, and since enrolment
04_hrv_per_patient.py       raw HRV over time, per patient
05_missingness.py           HRV gaps and coverage
06_hrv_distributions.py     HRV distributions
07_sleep_and_events.py      HRV in sleep; HRV, steps and sleep around each exacerbation
08_sleep_stages.py          deep / light / almost-awake sleep and HRV (per sleep block)
09_clinical_profile.py      datasheet completeness, symptoms/history, lung tests, imaging, treatment
11_model_data.py            model_data/: 10-min modelling table, segments, imputation masks, splits (self-checks)
12_exac_classify.py         exacerbation classification from enrolment tests -> results/exac_monitoring/
12_model_results.py         results/ -> numbers/model_results.csv, figs 12a-c
tune_models.py              hyperparameter tuning (Optuna) on a practice set cut from the training data -> results/tuning/
shap_analysis.py            SHAP for the tree models and classifiers, drop-a-vital ablation for the rest -> figs 14a-e
13_method_figures.py        diagrams: how imputation, forecasting and classification work, one row per model family (needs results/)
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
# models, in this order (see Models below), then:
python 12_exac_classify.py                                                       # exacerbation classification
THEME=light python 12_model_results.py; THEME=dark python 12_model_results.py   # results/ -> numbers/, figs 12a-c
THEME=light python 13_method_figures.py;  THEME=dark python 13_method_figures.py    # figs 13a-e (uses the best model of each test)
python 10_report.py
```

## What the data is

41 patients in the datasheet (c001–c041), all analysed. 40 have wearable data
(export of 29 Sep 2026); c034 has none and is reported as such, not dropped.
Watch analyses use whoever has a file in data/wearable/.

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
python tune_models.py impute xgboost               # tune one model (40 tries; 8 for the slow ones); --list shows the budget
python run_models.py impute xgboost --tuned        # run with the tuned settings -> results/summary_tuned.csv
python shap_analysis.py all --tuned                # SHAP, ablation, figures 14a-e
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
