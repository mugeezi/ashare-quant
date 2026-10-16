#!/bin/sh
# 数据修复重跑（2026-10-16）：去掉业绩报表数值特征的 _nf 版本
cd "$(dirname "$0")"
L=../logs
AP_NOSCREEN=1 python3 -u autopilot.py H67_nf H68_nf H60s_nf
for h in H60_nf H53_nf H67s_nf H71_nf; do
  python3 -u -W ignore cache_scores.py $h base > $L/ap_${h}_cache.log 2>&1
  python3 -u -W ignore run_leak.py $h base > $L/ap_${h}_leak.log 2>&1
  python3 -u -W ignore cache_scores.py $h base valid > $L/ap_${h}_vcache.log 2>&1
done
AP_NOSCREEN=1 python3 -u autopilot.py H73_nf
python3 -u -W ignore run_exec.py pipeline X08_H60s_nf:vk3 X08_H71_nf:vk3 X08_H53_nf:zb X12_H68_nf:dip1zb X12_H67s_nf:dip1zb > $L/nf_x_pipeline.log 2>&1
echo done > $L/nf_jobs.done
