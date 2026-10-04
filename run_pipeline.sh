#!/bin/bash
# The whole pipeline, in order, from the raw export to the report.
#
#   ./run_pipeline.sh                    every stage
#   ./run_pipeline.sh eda modeldata      only these stages (names below)
#   PY=/path/to/python CORES=16 ./run_pipeline.sh
#
# Start from an empty results/ to reproduce from scratch: tuning studies resume from results/tuning/
# and the summaries are appended to. docs/REPRODUCE.md has the times, the data layout and a new-cohort guide.
set -euo pipefail
cd "$(dirname "$0")"
export PY=${PY:-python}
export CORES=${CORES:-$(( $(nproc) - 2 ))}                 # never the whole machine: 2 cores stay free
export TIMESFM_CHECKPOINT=${TIMESFM_CHECKPOINT:-google/timesfm-3.0-pytorch}
EXPORT_DIR=$($PY -c "from common import RAW; print(RAW)")
SHEET=$($PY -c "from common import SHEET; print(SHEET)")
MODELS_ALL="xgboost catboost rnn lstm gast nlssm hmm pinode cd_gamma_dglm ossa timesfm3 pf rsdpf gru_ode_bayes"
both() { for T in light dark; do THEME=$T $PY "$@"; done; }
tests() { scripts/sweep.sh impute --mask random "$@"; scripts/sweep.sh impute --mask block "$@"
          scripts/sweep.sh forecast "$@"; scripts/sweep.sh daily "$@"; }

for stage in ${*:-provenance wearable eda modeldata models classify tune tuned explain report}; do
  echo "##### $(date '+%F %T') $stage"
  case $stage in
    provenance)   # what produced this run: code, environment, input data
      P=results/provenance; mkdir -p $P
      [ -f results/summary.csv ] && echo "note: results/ already has runs; new rows are appended (last one wins)"
      git rev-parse HEAD > $P/git_commit.txt; git status --porcelain > $P/git_uncommitted.txt
      { $PY --version; $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"; } > $P/python.txt 2>&1
      $PY -m pip freeze > $P/pip_freeze.txt
      { sha256sum "$SHEET"; find "$EXPORT_DIR" -type f | sort | xargs sha256sum; } > $P/inputs.sha256
      echo "inputs: $(wc -l < $P/inputs.sha256) files hashed in $P/inputs.sha256" ;;
    wearable)  $PY 01_build_wearable.py ;;                                 # export -> data/wearable/
    eda)       $PY codings.py --check
               for s in 02_clinical_eda 03_data_presence 04_hrv_per_patient 05_missingness 06_hrv_distributions \
                        07_sleep_and_events 08_sleep_stages 09_clinical_profile; do both $s.py; done ;;
    modeldata) both 11_model_data.py ;;                                    # -> model_data/
    models)    tests ;;                                                    # every model, default settings
    classify)  $PY 12_exac_classify.py --cores "$CORES" ;;
    tune)      scripts/tune_sweep.sh "$MODELS_ALL" "$CORES" ;;             # long: hours (see docs/REPRODUCE.md)
    tuned)     MODELS="$MODELS_ALL" tests --tuned ;;
    explain)   $PY shap_analysis.py all --tuned --cores "$CORES"
               THEME=dark $PY shap_analysis.py figures --cores "$CORES" ;;
    report)    both 12_model_results.py; both 13_method_figures.py; $PY 10_report.py ;;
    *) echo "unknown stage: $stage"; exit 1 ;;
  esac
done
echo "##### $(date '+%F %T') done"
