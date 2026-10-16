#!/bin/sh
cd "$(dirname "$0")"
while kill -0 1650 2>/dev/null; do sleep 30; done
python3 -u -W ignore leaderboard.py > ../logs/leaderboard_nf.log 2>&1
python3 -u -W ignore diagnose.py nf > ../logs/diag_nf.log 2>&1
AP_TRAIN_ONLY=1 python3 -u autopilot.py E01 E02 E04 E05 E03 E06 E07 E08
echo done > ../logs/overnight.done
