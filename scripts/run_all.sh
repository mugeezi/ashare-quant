#!/usr/bin/env bash
# 阶段 1 全流程（约 1 小时）
set -euo pipefail
cd "$(dirname "$0")"
PY="python3 -u -W ignore"
$PY download_meta.py "$@"                    # 交易日历、股票池、交易所退市名单（首次约 5 分钟穷举探测；可加 --skip-probe）
$PY download_daily.py --workers 6 || $PY download_daily.py --workers 3
$PY finalize_universe.py                     # 退市日、重复序列、上市日
$PY cninfo_names.py post_cutoff              # 2026-07-06 起 ST 简称法所需的逐只公告简称
$PY download_st.py                           # ST 重建 + 验证（需要先跑 cninfo_names.py validation_sh 才有沪市验证）
$PY cninfo_names.py validation_sh
$PY download_st.py --rebuild
$PY download_special.py                      # 恢复上市、股改公告
$PY download_index.py
$PY build_panel.py
$PY crosscheck_sample.py
$PY check_delist.py
$PY st_diagnose.py
$PY quality_check.py
