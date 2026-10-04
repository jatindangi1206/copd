# COPD clinical + wearable analysis

From a raw smartwatch export and a clinical datasheet to a report: data checks, HRV imputation and
forecasting with 14 models, exacerbation classification, tuning and SHAP.

* **Run it:** `./run_pipeline.sh` (every stage, in order) - [docs/REPRODUCE.md](docs/REPRODUCE.md) has setup,
  stage times, what is repeatable, and how to run another cohort.
* **Understand a model or change a setting:** [docs/MODELS.md](docs/MODELS.md) - the maths of every model as
  implemented, and each setting's name, default and tuning range. One run with other settings:
  `python run_models.py impute xgboost --mask block --params '{"K": 8}'`.
* **What each script found:** [results.md](results.md). **The report:** `COPD_EDA_Report.html` (and `_dark`).

```
run_pipeline.sh             the whole pipeline (stages: provenance wearable eda modeldata models classify tune tuned explain report)
data/                       patient data, not in git
  COPDAI_DATASHEET_01.xls   clinical datasheet: Baseline, Coding, exacerbation
  copd/<pid>/<vital>/...    raw watch export (29 Sep 2026), 40 patients (no c034)
  wearable/<pid>.csv.gz     1-minute table per patient, every minute from first to last reading (built by 01)
common.py                   where the data is (COHORT_* env vars), patient-ID pattern, plot style
codings.py                  the datasheet's coding scheme and its repairs  (--check)
01_build_wearable.py        raw export -> data/wearable/  (--check: rebuild equals the files on disk)
02-09_*.py                  EDA: clinical, data presence, HRV per patient, gaps, distributions, sleep, datasheet profile
11_model_data.py            model_data/: 10-min table, segments, imputation masks, splits (self-checks)
run_models.py               one model on one test (--tuned, --params); scripts/sweep.sh runs every model
models/                     one file per model; models/data.py: shared inputs, scoring, saving
12_exac_classify.py         exacerbation classification from enrolment tests -> results/exac_monitoring/
tune_models.py              tuning (Optuna) on a practice set cut from the training data -> results/tuning/
shap_analysis.py            SHAP for the tree models and classifiers, drop-a-vital ablation -> figs 14a-e
12_model_results.py         results/ -> numbers/model_results*.csv, figs 12a-d
13_method_figures.py        diagrams of how each test and model family works, figs 13a-f
10_report.py                COPD_EDA_Report.html + COPD_EDA_Report_dark.html
requirements-lock.txt       exact package versions behind the committed results
```

`data/`, `model_data/`, `results/` and `clinical_table.csv` hold patient-level data and are in `.gitignore`.

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
