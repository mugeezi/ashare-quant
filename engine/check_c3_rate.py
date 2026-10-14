"""C3 补充：追涨停检验 A 的"开盘涨停买不进"比例。
run_checks.py 中 20 万本金很快亏到不足一手，大量尝试因资金不足失败，比例被稀释；
这里用充足资金重跑，并与全市场直接统计对照。"""
import pickle, sys
import numpy as np
from backtest import Backtest, Config
from test_strategies import LimitUpChase

md = pickle.load(open(sys.argv[1], "rb"))
for cash in [2e5, 1e9]:
    r = Backtest(md, LimitUpChase(0), Config(init_cash=cash, hold_days=5)).run()
    s = r.stat
    blk = s["buy_block"].get("limit_up_open", 0)
    print(f"本金 {cash:,.0f}: 尝试 {s['buy_attempts']}, 开盘涨停买不进 {blk} ({blk / s['buy_attempts']:.1%}), "
          f"停牌 {s['buy_block'].get('suspended', 0)}, 资金不足一手 {s['insufficient_lot']}, 成交 {s['buys']}, "
          f"剔除资金不足后买不进比例 {blk / (s['buy_attempts'] - s['insufficient_lot']):.1%}")
# 全市场直接统计（引擎同口径过滤：非 ST/S/整理期、上市满 60 日、20 日均额 ≥ 5000 万）
a = md.a
t0 = int(md.dates.searchsorted(np.datetime64("2014-01-01")))
sig = ((a["tradestatus"] == 1) & (a["no_limit"] == 0) & (a["close"] >= a["up_limit"] - 0.005) & (a["isST"] == 0)
       & (a["isS"] == 0) & (a["delist_period"] == 0) & (a["days_listed"] >= 60) & (np.nan_to_num(a["amt20"]) >= 5e7))[t0:-1]
nxt_blk = ((a["tradestatus"] == 1) & (a["no_limit"] == 0) & (a["open"] >= a["up_limit"] - 0.005))[t0 + 1:]
print(f"全市场：t 日收盘涨停 {sig.sum():,} 次，t+1 开盘即涨停 {(sig & nxt_blk).sum():,} 次（{(sig & nxt_blk).sum() / sig.sum():.1%}）")
