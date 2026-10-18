# exac/ - the exacerbation experiments, one job per file

A copy of `run_exac.py` + the model code it uses, split so each part is easy to find and edit. The code is
copied, not rewritten. Only references to settings and data were changed (`MAX_STEPS` -> `C.MAX_STEPS`,
`D` -> `data.D`). `run_exac.py` and `models/` are unchanged. Every decision behind the code is in
`docs/EXAC_DESIGN.md` and `docs/GAST_DESIGN.md`.

## Run (from the repo root)

```bash
.venv/bin/python -m exac.run --list                          # every model and its file
.venv/bin/python -m exac.run check                           # every model, tiny training: does the code run?
.venv/bin/python -m exac.run all --task detect               # every single-task model, 5 ratios x 5 seeds
.venv/bin/python -m exac.run rnn lstm --task onset --set copd
.venv/bin/python -m exac.run gast --days 14
.venv/bin/python -m exac.run gast2                           # two heads: both tasks
.venv/bin/python -m exac.models.gast_v1                      # GAST self-tests
```

Options: `--task detect|onset`, `--set combined|copd|copd_clinical`, `--test 20 10 ...`, `--seed 0 1 ...`,
`--days N` (GAST only), `--cores N`, `--device cpu|cuda`, `--gpu-gb G`, `--quick`.
Two runs in parallel: give each `--cores` and `--gpu-gb` so the total leaves room for the GUI.
Model names go before the options.

## Where to change what

| to change | edit |
|---|---|
| any setting: splits, seeds, window, batch, learning rate, steps, early stopping, threshold, bootstrap, tree settings, RNN/GAST size, TimesFM batch, paths, cores left free | `config.py` |
| how the data is loaded, the 7-day window, the tree input row, scaling | `data.py` |
| how patients are split into train / held-back / test | `split.py` |
| the network training loop (batches, loss, early stopping) | `train.py` |
| scores, intervals, calibration, slices | `scoring.py` |
| one model | `models/<model>.py` |
| add a model | a new file in `models/` + one line in `models/__init__.py` |
| the command line, saving results | `run.py` |

## Files

```
exac/
  config.py      every setting
  cores.py       how many CPU cores (asks; default leaves KEEP_FREE free)
  data.py        load the set, windows, tree rows, scales
  split.py       patient split, --quick subset
  train.py       the shared network training loop
  scoring.py     AUROC, AUPRC, per-day / per-event scores, 95% intervals, calibration, slices
  run.py         command line, run, save
  models/
    __init__.py  the list of models
    always_no.py
    xgboost.py   catboost.py      trees on every raw value
    rnn.py       lstm.py          10-minute learned compression + RNN / LSTM
    gast_v1.py                    GAST v1, its variants, its self-tests
    timesfm3.py                   frozen TimesFM 3 + a learned logistic layer
```

Results go to the same place as `run_exac.py`: `results/exac/<task>/<set>/<model>/test<NN>_seed<S>/` and
`results/exac/summary_<task>.csv`. Don't run both runners on the same task at the same time: they append
to the same summary file.
