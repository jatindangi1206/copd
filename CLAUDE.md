# CLAUDE.md

COPD pilot cohort (41 patients in the datasheet, 40 with watch data; export of 29 Sep 2026): clinical datasheet + smartwatch HRV. EDA and report
are done; the models are fixed and waiting for a full rerun. Read README.md (layout, setup, commands) and
results.md (what each script found) first.

The project was built in an earlier Claude session on the user's Mac. Everything that session
knew - the user's instructions and why, data traps, numbers, per-model notes, gotchas, next
steps - is in docs/HANDOFF.md, loaded here:

@docs/HANDOFF.md

On a new machine: patient data (data/, model_data/, clinical_table.csv) is not in git - check it
is there before running anything.

## Decisions already made - don't reopen them

- **Order is fixed: imputation first, then forecasting.** Imputation hides the 20% `mask_random`
  (and the harder `mask_block`) and rebuilds it; forecasting predicts only the last 20% of each
  segment, never the middle. Daily forecasting: last 20% of each patient's days.
- **Segments: x = 18 missing readings in a row (180 min)**, set for this sparse pilot after looking
  at the minute-difference graphs (fig 05e). Not a data-derived rule; to be reviewed with physiology
  experts and made per-patient later. It is `X` in 11_model_data.py.
- **HRV is SDNN, one value about every 10 minutes, no unit.** Never write "ms".
- **Values are used as recorded** (c025's age of 7, BODE as in the sheet). Only the sheet's own
  coding is converted (Yes=1/No=2, ND/NK/NA blank) and the swapped BP labels.
- **H1-H10** (research doc, first tab): 8 kept, H3 and H6 dropped. **H9 (no data after t) applies
  to forecasting only**; imputation may use both sides of a gap.
- The model set is final for now (report section 12). `gast` (gap-aware SSM transformer) is the
  user's own design: `models/gast.py` is only a placeholder - ask for the architecture.

## Conventions

- **Cores:** modelling code must show free cores, ask how many to use, and default to leaving 2
  free for the UI (`models/cores.py`, called before numpy/torch are imported).
- **Report style:** minimal, plain, jargon-free, not "AI-looking" (no cards, badges, bold lead-ins).
  Every figure in a report is a PNG in figs/ (figs_dark/ for the dark report) and every PNG there is
  in the report - 10_report.py asserts it. Figures: title, labelled axes, one-line caption.
  Don't invent definitions, units, thresholds or results; say when something is unknown.
- **results.md** is plain notes, one section per script with its command and output.
- Don't download foundation-model weights unless asked (TimesFM 3.0 weights: non-commercial licence).

## Status (2026-10-02)

- Old model results (29 Sep run, 18 Sep data, 31 patients) were deleted on purpose: everything is to be rerun
  from scratch on the current model_data (29 Sep export: 259 segments, 26 patients; 28 patients have any
  HRV, and only those can be modelled - c034 and the 12 patients with watch data but no HRV are left out).
  No model has been run since the fixes below.
- Fixes made in this copy (not yet run): the four workstation fixes (seq.py `_windows`, nlssm.py
  `_batch_filter`, run_models.py, timesfm3.py `TIMESFM_CHECKPOINT`) plus the six failures from the 29 Sep run:
  pinode step size, cd_gamma_dglm singular smoother, nlssm sigma-point weights (negative HRV), ossa
  recurrence and range clamp, and rnn/lstm/gast forecast drift (seq.py `fit_forecast` is now direct
  multi-step, no feedback). `run_models.py check` now also fails on non-finite / <= 0 predictions, more than 5%
  above 129, or MAE >= 100, and uses the 6 longest series. It is still only a smoke test.
- 12_exac_classify.py is now in this repo. It includes every datasheet patient with a dated outcome
  (all except c005, c032, c037 and c034) and writes results/exac_monitoring/.
- To rerun: use /home/kcdha/timesfm3/.venv/bin/python (the default python cannot import pandas) and
  `export TIMESFM_CHECKPOINT=/home/kcdha/timesfm3/timesfm-3.0-pytorch` (local weights, no download). Then
  `run_models.py check`, `impute all` (random, then `--mask block`), `forecast all`, `daily all`,
  `12_exac_classify.py`, `12_model_results.py` (light and dark), `10_report.py`.
- COPD_EDA_Report*.html in git still hold the old results until 10_report.py is rerun.
- On macOS, xgboost crashed if torch was imported first (two OpenMP runtimes); run_models.py
  imports xgboost first.
- Patient data (data/, model_data/, results/, clinical_table.csv) is gitignored; copy it separately.
