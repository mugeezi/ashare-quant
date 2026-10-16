#!/bin/sh
# 数据修复重跑续跑（容器重启后可重复执行：已有输出的步骤跳过）
cd "$(dirname "$0")"
L=../logs
S=/tmp/claude-0/-home-claude/47e34d9e-ea3d-5e5a-bd04-7c13816a5033/scratchpad
for h in H60_nf H53_nf H67s_nf H71_nf; do
  [ -f $S/score_${h}_base_train.npy ] || python3 -u -W ignore cache_scores.py $h base > $L/ap_${h}_cache.log 2>&1
  [ -f ../reports/leak_${h}_base.json ] || python3 -u -W ignore run_leak.py $h base > $L/ap_${h}_leak.log 2>&1
  [ -f $S/score_${h}_base_valid.npy ] || python3 -u -W ignore cache_scores.py $h base valid > $L/ap_${h}_vcache.log 2>&1
done
grep -q "H73_nf_dsr" $L/autopilot.log || AP_NOSCREEN=1 python3 -u autopilot.py H73_nf
[ -f $L/nf_x_pipeline.done ] || { python3 -u -W ignore run_exec.py pipeline X08_H60s_nf:vk3 X08_H71_nf:vk3 X08_H53_nf:zb X12_H68_nf:dip1zb X12_H67s_nf:dip1zb > $L/nf_x_pipeline.log 2>&1 && touch $L/nf_x_pipeline.done; }
for h in H67 H68 H60 H53 H71; do
  python3 -c "import json,sys; d=json.load(open('../reports/importance_original.json')); sys.exit(0 if '$h' in d else 1)" 2>/dev/null || IMP_OUT=1 python3 -u -W ignore importance.py $h >> $L/importance.log 2>&1
done
echo done > $L/nf_jobs.done
