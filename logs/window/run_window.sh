#!/bin/bash
# Window ablation (EXAC_DESIGN 9). Phase 1: every model except GAST, 3 streams split by window length.
# Phase 2 (GAST) starts only when all three phase-1 streams are done. gast, detection, 3 days: reused from the main run.
cd /home/kcdha/Desktop/JD/prediction/copd
W=".venv/bin/python -u -m exac.window"; O="--cores 12 --device cuda --gpu-gb 6"; L=logs/window
OTHER="always_no xgboost catboost rnn lstm timesfm3"; GAST="gast gast_ssm gast_attn gast_nogap gast_maskonly"
wait_p1() { while [ ! -f $L/p1_A.done ] || [ ! -f $L/p1_B.done ] || [ ! -f $L/p1_C.done ]; do sleep 60; done; }
( $W $OTHER --days 5 $O > $L/p1_A_d5.log 2>&1; touch $L/p1_A.done; wait_p1
  $W $GAST --days 5 $O > $L/p2_A_d5.log 2>&1; touch $L/p2_A.done ) &
( $W $OTHER --days 4 $O > $L/p1_B_d4.log 2>&1; touch $L/p1_B.done; wait_p1
  $W $GAST --days 4 $O > $L/p2_B_d4.log 2>&1; touch $L/p2_B.done ) &
( $W $OTHER --days 3 2 $O > $L/p1_C_d3_d2.log 2>&1; touch $L/p1_C.done; wait_p1
  $W gast_ssm gast_attn gast_nogap gast_maskonly --days 3 2 $O > $L/p2_C_d3_d2.log 2>&1
  $W gast --task detect --days 2 $O > $L/p2_C_gast_detect_d2.log 2>&1
  $W gast --task onset --days 3 2 $O > $L/p2_C_gast_onset_d3_d2.log 2>&1; touch $L/p2_C.done ) &
wait
$W --table > $L/tables.log 2>&1; touch $L/all.done
