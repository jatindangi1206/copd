#!/bin/bash
# usage: sweep.sh "<task args>" ; runs every model for the task in its own process
cd /home/kcdha/Desktop/JD/copd-clinical/copd-clinical
export TIMESFM_CHECKPOINT=/home/kcdha/timesfm3/timesfm-3.0-pytorch
PY=/home/kcdha/timesfm3/.venv/bin/python
task=$1; shift
case $task in
  impute) models="linear xgboost catboost rnn lstm gast nlssm hmm pinode cd_gamma_dglm ossa timesfm3 pf rsdpf gru_ode_bayes";;
  *) models="last_value patient_median xgboost catboost rnn lstm gast nlssm hmm pinode cd_gamma_dglm ossa timesfm3 pf rsdpf gru_ode_bayes";;
esac
for m in $models; do
  echo "=== $(date +%T) $task $m $*"
  $PY -u run_models.py $task $m --cores 19 "$@" 2>&1 | grep -v "^    step" || true
  echo "=== exit ${PIPESTATUS[0]} $m"
done
echo "SWEEP DONE $task $*"
