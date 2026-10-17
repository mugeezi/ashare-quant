"""最终测试运行前检查 4.3-1：基准指数（国证 2000、中证全指等）2024-01-01 至截止日的日线质量检查，口径同研究期 quality_check.check_index：
交易日完整（对照交易日历）、无重复日期、收盘价为正；另列出单日涨跌幅绝对值超过 9.5% 的日期供核对（只列日期，不做收益统计）。
任何一项不满足即以非零状态退出。用法：python final_index_check.py --cutoff 2026-10-16"""
import argparse
import sys

import pandas as pd

from build_panel import trading_days
from common import INDEX, load_df, log_test_access


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True)
    a = ap.parse_args()
    log_test_access("最终测试运行前检查 4.3-1：基准指数完整性检查（只列异常日期，不计算收益统计）", f"data/index gz2000 等 2024-01-01 ~ {a.cutoff}", "integrity")
    td = trading_days()
    span = td[(td >= pd.Timestamp("2024-01-01")) & (td <= pd.Timestamp(a.cutoff))]
    ok = True
    for name in ("gz2000", "zzqz", "zz1000", "hs300", "zz500"):
        d = load_df(INDEX / f"{name}.parquet").sort_values("date")
        d = d[(d.date >= "2024-01-01") & (d.date <= a.cutoff)]
        miss = span.difference(pd.DatetimeIndex(d.date))
        extra = pd.DatetimeIndex(d.date).difference(span)
        big = d.loc[d.close.pct_change().abs() > 0.095, "date"].dt.date.tolist()
        row = dict(指数=name, 起=d.date.min().date(), 止=d.date.max().date(), 行数=len(d), 应有交易日=len(span),
                   缺失交易日=len(miss), 多余日期=len(extra), 重复日期=int(d.date.duplicated().sum()),
                   非正收盘=int((d.close <= 0).sum()), 单日超95pct日期=big)
        good = row["缺失交易日"] == 0 and row["多余日期"] == 0 and row["重复日期"] == 0 and row["非正收盘"] == 0 \
            and str(row["止"]) == str(span.max().date())
        ok &= good if name in ("gz2000", "zzqz") else True
        print(("通过 " if good else "不通过 ") + str(row), flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
