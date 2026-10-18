#!/bin/bash
# Waits for both GAST chains, then runs TimesFM 3 (cache alone first, then detect and onset in parallel).
cd /home/kcdha/Desktop/JD/prediction/copd
export TIMESFM_CHECKPOINT=/home/kcdha/timesfm3/timesfm-3.0-pytorch
while [ ! -f logs/gast/detect.done ] || [ ! -f logs/gast/onset.done ]; do sleep 60; done
R=".venv/bin/python -u run_exac.py timesfm3"; L=logs/timesfm3
echo "start $(date)" > $L/started
$R --task detect --set combined --test 20 --seed 0 --cores 36 --device cuda --gpu-gb 6 > $L/detect_combined_first.log 2>&1
O="--cores 18 --device cuda --gpu-gb 6"
( $R --task detect --set combined --test 20 --seed 1 2 3 4 $O > $L/detect_combined_t20.log 2>&1
  $R --task detect --set combined --test 10 30 40 50 $O > $L/detect_combined_rest.log 2>&1
  $R --task detect --set copd $O > $L/detect_copd.log 2>&1
  $R --task detect --set copd_clinical $O > $L/detect_copd_clinical.log 2>&1
  echo done > $L/detect.done ) &
( $R --task onset --set combined $O > $L/onset_combined.log 2>&1
  $R --task onset --set copd $O > $L/onset_copd.log 2>&1
  $R --task onset --set copd_clinical $O > $L/onset_copd_clinical.log 2>&1
  echo done > $L/onset.done ) &
wait
