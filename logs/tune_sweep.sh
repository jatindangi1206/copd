#!/bin/bash
# usage: tune_sweep.sh "<models>" <cores> ; tunes each model for impute, forecast, daily, one process each
cd /home/kcdha/Desktop/JD/copd-clinical/copd-clinical
export TIMESFM_CHECKPOINT=/home/kcdha/timesfm3/timesfm-3.0-pytorch
PY=/home/kcdha/timesfm3/.venv/bin/python
for task in impute forecast daily; do
  for m in $1; do
    echo "=== $(date +%T) $task $m"
    $PY -u tune_models.py $task $m --cores $2 2>&1 | grep --line-buffered -v "^    step" || true
  done
done
echo "TUNE DONE $1"
