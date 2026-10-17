"""A01 对照：用同一缓存重跑 10 只版本（H73_nf 原设定），只作对照、不计试验；核对与已登记 E0789 / E0796 一致。"""
import sys
import numpy as np
import r2
from r2 import Backtest, Config, F
import ml
from stock_strats import cached

r2.check_pause()
lab = F.Lab()
md = lab.md
pct = r2.fmv_percentile(md)
out = {}
for sp, s0 in (("train", "2016-01-01"), ("valid", "2021-01-01")):
    s1 = r2.SPLITS[sp][1]
    res = Backtest(md, ml.RegimeStrategy(cached("H73_nf", sp)), Config(start=s0, end=s1, n_target=10, renew=True, hold_days=10)).run()
    st = r2.nav_stats(res.nav.nav, md.index_close, lab.smb)
    st["conc"] = r2.concentration(md, res.roundtrips, res.nav.nav, lab.bench, lab.smb, fmv_pct=pct)
    out[sp] = st
    print(sp, round(st["cagr"], 4), round(st["excess_cagr"], 4), round(st["max_drawdown"], 4), flush=True)
r2.save_json("compare_H73_nf_n10.json", out)
