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

- 2 Oct: everything was rerun from scratch on the current model_data (259 segments, 26 patients; only the
  28 patients with any HRV can be modelled - c034 and the 12 patients with watch data but no HRV are left out).
  Old results were deleted. Results are in results/ (gitignored), tables in numbers/, report rebuilt.
- Fixes made before the rerun: seq.py `_windows`, nlssm `_batch_filter`, timesfm3 `TIMESFM_CHECKPOINT`, plus pinode
  (reversed-time pass, one-slot steps), cd_gamma_dglm (pinv smoother), nlssm (sigma-point weights), ossa
  (range clamp, safe fallback) and rnn/lstm/gast forecasting (seq.py `fit_forecast` is direct multi-step).
  `run_models.py check` also fails on non-finite / <= 0 predictions, more than 5% above 129, or MAE >= 100.
  Still only a smoke test: it passed 45 of 47 at 20 training steps, and the two it flagged were fine at full training.
- gast forecast gave all NaN once and worked on an identical rerun; cause unknown. If NaNs return, look there first.
- 12_exac_classify.py is in this repo and covers every datasheet patient with a dated outcome
  (all except c005, c032, c037 and c034). It writes results/exac_monitoring/.
- 3 Oct: all 14 models tuned (tune_models.py, 42 studies) and rerun with `run_models.py --tuned`
  (results/summary_tuned.csv); SHAP and ablation in shap_analysis.py. Report sections 16-17. Tuning uses a
  practice set cut from the training data, never the official test - keep it that way.
- models/gast.py got a numerical guard (clamp of the FFT count at 0) that fixed its NaN runs. It is still the
  placeholder architecture; ask before changing the design itself.
- CatBoost runs on CPU by default (it stalled on the shared GPU). One GPU: 5 jobs at once made the slow models
  3x slower; run slow studies on their own.
- To rerun (the commands that were used): use /home/kcdha/timesfm3/.venv/bin/python (the default python cannot import pandas) and
  `export TIMESFM_CHECKPOINT=/home/kcdha/timesfm3/timesfm-3.0-pytorch` (local weights, no download). Then
  `run_models.py check`, `impute all` (random, then `--mask block`), `forecast all`, `daily all`,
  `12_exac_classify.py`, `12_model_results.py` (light and dark), `10_report.py`.
- 13_method_figures.py draws the flow diagrams (imputation, forecasting, classification) and the model-family
  figures; run it in both themes after the models and before 10_report.py (the report asserts every PNG is used).
- logs/sweep.sh runs one task for every model, one process each, so a crash does not stop the rest.
- On macOS, xgboost crashed if torch was imported first (two OpenMP runtimes); run_models.py
  imports xgboost first.
- Patient data (data/, model_data/, results/, clinical_table.csv) is gitignored; copy it separately.
