# Reproducing the pipeline, and running it on another cohort

What each model does, with the maths, is in [MODELS.md](MODELS.md).

## 1 Set up

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130   # match your CUDA; CPU works, slower
pip install -r requirements-lock.txt          # the exact versions that produced the committed results
```

`requirements.txt` + `requirements-models.txt` list the same packages without pinned versions.
TimesFM 3 weights download from Hugging Face on first use (non-commercial licence). To use a local copy:
`export TIMESFM_CHECKPOINT=/path/to/timesfm-3.0-pytorch`.

Patient data is not in git (`.gitignore`): put the cohort's files under `data/` (section 4).

## 2 Run

```bash
./run_pipeline.sh                                  # everything, raw export -> report
./run_pipeline.sh eda modeldata                    # some stages only
PY=.venv/bin/python CORES=16 ./run_pipeline.sh     # which python, how many cores (default: all but 2)
```

| stage | does | writes | time here* |
|---|---|---|---|
| `provenance` | records the git commit, Python, `pip freeze` and a SHA-256 of every input file | `results/provenance/` | seconds |
| `wearable` | raw export -> 1-minute table per patient (`01_build_wearable.py`) | `data/wearable/` | ~5 min |
| `eda` | datasheet check, then scripts 02-09 in light and dark | `figs/`, `figs_dark/`, `numbers/`, `clinical_table.csv` | ~10 min |
| `modeldata` | 10-minute grid, segments, masks, splits (`11_model_data.py`) | `model_data/` | ~2 min |
| `models` | every model on the 4 tests, default settings | `results/<task>/`, `results/summary.csv` | ~4 h |
| `classify` | exacerbation classification (`12_exac_classify.py`) | `results/exac_monitoring/` | ~15 min |
| `tune` | tuning, 42 studies (`tune_models.py`) | `results/tuning/` | ~12-15 h** |
| `tuned` | every model on the 4 tests with its tuned setting | `results/summary_tuned.csv` | ~4 h |
| `explain` | SHAP, ablation (`shap_analysis.py`) | `numbers/shap_*.csv`, `numbers/ablation.csv`, figs 14a-e | ~40 min |
| `report` | result tables and figures, diagrams, the HTML report | `numbers/`, figs 12-13, `COPD_EDA_Report*.html` | ~5 min |

\* one RTX A5500, 40 cores. \** run one study at a time; several at once on one GPU each run about 3x slower.

**From scratch.** Start with no `results/` folder (move the old one aside). Tuning studies resume from
`results/tuning/` and the summaries are appended to (the last row for a model wins), so a leftover `results/`
mixes old and new.

**Checks along the way.** `codings.py --check` tests the datasheet decoders; `11_model_data.py` asserts its masks and
splits; `01_build_wearable.py --check` compares a rebuild with `data/wearable/`; `run_models.py check` runs every model
on 6 segments with 20-step training (a smoke test, not a correctness test); `10_report.py` asserts that each sentence
claiming a pattern still matches the numbers, and stops if one does not.

## 3 What is repeatable
Checked on 4 Oct 2026, on the machine that produced the results:

* **1-minute tables:** `01_build_wearable.py --check` rebuilds every patient from the raw export; 40 of 40 are identical
  to the tables the analysis used (built earlier by an outside script, now replaced by this one).
* **Data stages, run twice in a row:** `./run_pipeline.sh eda modeldata` twice, then 114 outputs compared - every file
  in `model_data/`, every EDA table in `numbers/`, `clinical_table.csv`, and all 86 figures pixel by pixel. All
  identical. (Compressed files are written with a fixed timestamp and every sort breaks ties, so the bytes match too.)
* **Models:** XGBoost and LSTM block imputation rerun on the GPU gave the official MAE to every digit
  (28.865138690360595 and 27.695605889823543). Training uses fixed seeds (masks 0, models 0, tuning 0, practice set 12345).
* **No date dependence:** no script reads today's date; only the `finished` timestamp in each run's `metrics.json` does.
* **Not guaranteed:** a different GPU, driver or package version can change the last digits of GPU-trained models
  (RNN, LSTM, gast, RS-DPF, GRU-ODE-Bayes, the neural ODE, XGBoost), and through them which setting tuning picks.
  Use `requirements-lock.txt`, and compare `results/provenance/` between two runs before comparing their results.
* **Not rerun in full:** the `models`, `tune` and `tuned` stages (about a day) were run once; the spot checks above are the
  evidence for them.

## 4 Another cohort

Use one copy of the repository per cohort: every output (figures, tables, results, report) is written inside it.

### 4.1 The data the pipeline expects

```
data/
  <export>/<pid>/<vital>/<pid>_<vital>.csv     watch export, one folder per patient
  <datasheet>.xls                              clinical datasheet (sheets: Baseline, exacerbation)
```

Set the names in `common.py` or with environment variables: `COHORT_DATA` (folder, default `data/`),
`COHORT_EXPORT` (default `copd`), `COHORT_SHEET` (default `COPDAI_DATASHEET_01.xls`), and `PID`, the pattern of a
patient ID (default `^c\d{3}$`).

**Watch export.** `01_build_wearable.py` reads these files and columns (edit `POINT` / `INTERVAL` there if yours differ):

| file | columns |
|---|---|
| `heartrate` | `logDateTime`, `lastRate` |
| `hrv` | `createdTime`, `hrvValue` |
| `temperature` | `createdTime`, `temperature` |
| `spo2` | `createdTime`, `spo2Value` |
| `sleep` | `logDateTime`, `logEndTime`, `deepSleep`, `lightSleep`, `almostAwake` |
| `steps` | `logDateTime`, `logEndTime`, `steps` |

A missing file is fine (that signal is blank). Set `EARLIEST` to a date before the cohort's first real reading.

**Clinical datasheet.** `codings.py` reads the `Baseline` sheet with section names in row 2, column names in row 3 and
patients from row 4 (ID column `PARTCPNT_ID`), and the `exacerbation` sheet (`Subject ID`, `Date-*` columns). The
datasheet's coding (Yes=1/No=2, ND/NK/NA as missing, the swapped blood-pressure labels) is decoded there, and
`load_enrolment()` takes the enrolment date from the date after `SMART_WATCH_PRV_0`. The EDA scripts 02 and 09 and the
classifier use named columns (age, sex, GOLD, mMRC, CAT, BODE, spirometry, oscillometry, walk test, treatment). A
datasheet with other column names needs those names changed in `codings.py`, `02_clinical_eda.py`,
`09_clinical_profile.py` and `12_exac_classify.py`; run the `eda` stage and a `KeyError` names the first missing one.

### 4.2 What depends on this watch or this cohort

| item | where | why |
|---|---|---|
| `HRV_MAX = 129` | `models/data.py` | this watch's HRV ceiling; the plausibility checks use it |
| error split at 120 | `models/data.py: score` | this watch's top band (120-129) |
| `X = 18` (180 min) | `11_model_data.py` | set for this cohort's gaps (report figure 05e); review for a new device |
| the report's text | `10_report.py` | numbers are computed; sentences that state a pattern are guarded by asserts |
| enrolment rule | `codings.py: load_enrolment` | this sheet has no enrolment-date field |

### 4.3 Checklist

1. Data under `data/`, names set in `common.py`.
2. `python 01_build_wearable.py`, then open one `data/wearable/<pid>.csv.gz` and check the spans look right.
3. `python codings.py --check`, then `./run_pipeline.sh eda modeldata`; read `results.md`-style output for counts.
4. Look at figure 05e (gaps between readings) and decide `X`; rerun `modeldata`.
5. `python run_models.py check`, then `./run_pipeline.sh models classify tune tuned explain report`.

## 5 Changing a model setting

```bash
python run_models.py impute xgboost --mask block --params '{"K": 8, "lr": 0.1}'   # one run -> results/summary_custom.csv
python tune_models.py impute xgboost                                               # search -> results/tuning/
python run_models.py impute xgboost --mask block --tuned                           # use the search's best
```

Every setting's name, default and tuning range is in [MODELS.md](MODELS.md), section 6; section 9 there explains
changing a default permanently.
