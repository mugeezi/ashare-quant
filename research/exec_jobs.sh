#!/bin/sh
cd "$(dirname "$0")"
L=../logs
for h in X01 X02 X03 X04 X05; do python3 -u -W ignore run_exec.py train $h > $L/x_${h}_train.log 2>&1; done
for h in H80 H81; do
  python3 -u -W ignore cache_scores.py $h base > $L/ap_${h}_cache.log 2>&1
  python3 -u -W ignore run_leak.py $h base > $L/ap_${h}_leak.log 2>&1
done
python3 -u -W ignore run_exec.py train X06 > $L/x_X06_train.log 2>&1
python3 -u -W ignore run_exec.py train X07 > $L/x_X07_train.log 2>&1
echo done > $L/exec_jobs.done
