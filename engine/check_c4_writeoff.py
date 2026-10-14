"""C4 补充：逐笔列出退市按 0 计的持仓，核对原因（整理期最后交易日买入 / 面值退市连续一字跌停）。"""
import pickle, sys
import numpy as np, pandas as pd
from backtest import Backtest, Config
from test_strategies import DelistCheat
md = pickle.load(open(sys.argv[1], "rb"))
r = Backtest(md, DelistCheat(md), Config(start="2014-01-01", end="2023-12-31", hold_days=10, filters=False, n_target=10)).run()
rt = r.roundtrips
w = rt[rt.how.str.startswith("writeoff")].copy()
a = md.a; ci = {c: i for i, c in enumerate(md.codes)}
rows = []
for _, x in w.iterrows():
    j = ci[x.code]; t0 = md.dates.get_loc(pd.Timestamp(x.entry_date))
    dl = md.universe.loc[x.code, "delist_date"]; t1 = md.dates.searchsorted(dl)
    seg = slice(t0, min(t1, md.T))
    dp = a["delist_period"][seg, j]; ts = a["tradestatus"][seg, j]
    o, dn = a["open"][seg, j], a["down_limit"][seg, j]
    rows.append(dict(code=x.code, entry=str(x.entry_date)[:10], delist=str(dl)[:10], cls=md.exits.loc[x.code, "exit_class"],
                     days=t1 - t0, dp_days=int(dp.sum()), trade_days=int((ts == 1).sum()),
                     ld_open_days=int(((ts == 1) & (o <= dn + 0.005)).sum()), ret=x.ret))
print(pd.DataFrame(rows).to_string())
dp_rt = rt[rt.how == "delist_period"]
print("delist_period exits ret mean", dp_rt.ret.mean(), "n", len(dp_rt))
